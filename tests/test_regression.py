import contextlib
import io
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from yolo_regression.__main__ import main
from yolo_regression.diagnostics import diagnose
from yolo_regression.evaluation import compare
from yolo_regression.inputs import Dataset, Detection, Image, Label, load_dataset, load_predictions

PROJECT = Path(__file__).resolve().parents[1]
DEMO = PROJECT / "demo"


class RegressionTests(unittest.TestCase):
    def setUp(self):
        self.dataset = load_dataset(DEMO / "annotations.json")
        self.baseline = load_predictions(DEMO / "baseline.json", self.dataset)
        self.candidate = load_predictions(DEMO / "candidate.json", self.dataset)

    def test_tracks_new_misses_even_when_a_different_object_is_recovered(self):
        comparison = compare(self.dataset, self.baseline, self.candidate, 0.3, 0.5)
        washers, bolts = comparison.classes
        self.assertEqual((washers.baseline_tp, washers.candidate_tp, washers.total), (3, 2, 4))
        self.assertEqual((washers.new_misses, washers.recovered, washers.recall_drop_pp), (2, 1, 25))
        self.assertEqual((bolts.baseline_fp, bolts.candidate_fp), (0, 1))
        self.assertEqual([item.image_id for item in comparison.images], [3, 1])
        size, overlaps = diagnose(comparison, None).findings
        self.assertEqual(size.rows[0], ("小", 3, 2, 1, 2))
        self.assertEqual([row[1:] for row in overlaps.rows], [(1, 2), (1, 2)])

    def test_matches_one_to_one_by_score_and_ignores_crowd_overlap(self):
        image = Image("sample", "sample.jpg", 100, 100)
        box = (0.0, 0.0, 10.0, 10.0)
        crowd = (40.0, 40.0, 20.0, 20.0)
        dataset = Dataset({image.id: image}, {0: "item"}, [
            Label(0, image.id, 0, box, False),
            Label(1, image.id, 0, crowd, True),
        ])
        lower = Detection(image.id, 0, box, 0.4)
        higher = Detection(image.id, 0, box, 0.7)
        ignored = Detection(image.id, 0, (42.0, 42.0, 7.0, 7.0), 0.5)
        result = compare(dataset, [], [lower, ignored, higher], 0.3, 0.5)
        self.assertEqual(result.candidate.matches, {0: higher})
        self.assertEqual(result.candidate.false_positives, {image.id: [lower]})
        self.assertEqual(result.classes[0].total, 1)

    def test_replaced_false_positive_is_visible_even_when_counts_match(self):
        image = Image(1, "sample.jpg", 100, 100)
        truth = (5.0, 5.0, 10.0, 10.0)
        dataset = Dataset({1: image}, {0: "item"}, [Label(0, 1, 0, truth, False)])
        correct = Detection(1, 0, truth, 0.9)
        previous = Detection(1, 0, (25.0, 25.0, 10.0, 10.0), 0.8)
        current = Detection(1, 0, (75.0, 75.0, 10.0, 10.0), 0.8)
        result = compare(dataset, [correct, previous], [correct, current], 0.3, 0.5)
        row = result.classes[0]
        self.assertEqual((row.baseline_fp, row.candidate_fp), (1, 1))
        self.assertEqual((row.added_fp, row.resolved_fp, row.persistent_fp), (1, 1, 0))
        self.assertEqual(result.images[0].false_positives.added, (current,))
        self.assertEqual(result.images[0].false_positives.resolved, (previous,))

    def test_false_positive_matching_reassigns_a_pair_to_preserve_both_matches(self):
        image = Image(1, "sample.jpg", 100, 100)
        truth = (70.0, 70.0, 10.0, 10.0)
        dataset = Dataset({1: image}, {0: "item"}, [Label(0, 1, 0, truth, False)])
        left = Detection(1, 0, (10.0, 10.0, 10.0, 10.0), 0.9)
        right = Detection(1, 0, (14.0, 10.0, 10.0, 10.0), 0.8)
        middle = Detection(1, 0, (11.0, 10.0, 10.0, 10.0), 0.95)
        result = compare(dataset, [left, right], [middle, left], 0.3, 0.5)
        row = result.classes[0]
        self.assertEqual((row.added_fp, row.resolved_fp, row.persistent_fp), (0, 0, 2))
        self.assertEqual(result.images, [])

    def test_ci_writes_report_even_if_recall_gate_fails(self):
        with tempfile.TemporaryDirectory(dir=PROJECT) as directory:
            path = Path(directory) / "report.html"
            args = ["--annotations", str(DEMO / "annotations.json"), "--baseline", str(DEMO / "baseline.json"),
                    "--candidate", str(DEMO / "candidate.json"), "--output", str(path)]
            with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
                self.assertEqual(main([*args, "--max-recall-drop-pp", "5"]), 1)
                failed = path.read_text(encoding="utf-8")
                self.assertEqual(main([*args, "--max-recall-drop-pp", "25"]), 0)
                passed = path.read_text(encoding="utf-8")
            self.assertIn("未通过", failed)
            self.assertIn("75.0% → 50.0%", failed)
            self.assertIn('class="box weak"', failed)
            self.assertIn("通过。", passed)

    def test_new_miss_gate_blocks_equal_recall_and_preserves_both_reports(self):
        with tempfile.TemporaryDirectory(dir=PROJECT) as directory:
            report = Path(directory) / "report.html"
            data = Path(directory) / "report.json"
            args = ["--annotations", str(DEMO / "annotations.json"), "--baseline", str(DEMO / "baseline.json"),
                    "--candidate", str(DEMO / "candidate-equal-recall.json"), "--output", str(report),
                    "--json-output", str(data), "--max-recall-drop-pp", "0"]
            with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
                self.assertEqual(main(args), 0)
                self.assertEqual(main([*args, "--gate-class", "0", "--max-new-misses", "0"]), 1)
            result = json.loads(data.read_text(encoding="utf-8"))
            self.assertEqual(result["schema_version"], 1)
            self.assertEqual(result["gate_status"], "failed")
            self.assertEqual(result["new_misses"], 1)
            self.assertEqual(result["recovered"], 1)
            self.assertEqual(result["class_changes"][0]["recall_drop_pp"], 0)
            self.assertEqual(result["sample_previews"][0]["file_name"], "line-001.jpg")
            failure, = result["gate_failures"]
            self.assertEqual((failure["category_id"], failure["metric"], failure["actual"], failure["limit"]),
                             (0, "new_misses", 1, 0))
            self.assertIn("未通过", report.read_text(encoding="utf-8"))

    def test_false_positive_gate_checks_only_selected_classes(self):
        with tempfile.TemporaryDirectory(dir=PROJECT) as directory:
            report = Path(directory) / "report.html"
            args = ["--annotations", str(DEMO / "annotations.json"), "--baseline", str(DEMO / "baseline.json"),
                    "--candidate", str(DEMO / "candidate.json"), "--output", str(report),
                    "--max-new-false-positives", "0"]
            with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
                self.assertEqual(main([*args, "--gate-class", "0"]), 0)
                self.assertEqual(main([*args, "--gate-class", "1"]), 1)
                self.assertEqual(main([*args, "--gate-class", "999"]), 2)
            self.assertIn('class="box new-fp"', report.read_text(encoding="utf-8"))

    def test_rejects_mismatched_image_id_type(self):
        with tempfile.TemporaryDirectory(dir=PROJECT) as directory:
            path = Path(directory) / "predictions.json"
            path.write_text(json.dumps([{"image_id": "1", "category_id": 0, "bbox": [1, 1, 2, 2], "score": 0.7}]),
                            encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "未知图片 ID"):
                load_predictions(path, self.dataset)

    def test_model_can_finish_after_one_check_and_sees_only_aggregate_evidence(self):
        comparison = compare(self.dataset, self.baseline, self.candidate, 0.3, 0.5)
        with patch("yolo_regression.diagnostics.build_opener") as opener:
            connection = opener.return_value.open.return_value.__enter__.return_value
            connection.read.side_effect = [
                json.dumps({"message": {"content": json.dumps({"choice": choice})}}).encode("utf-8")
                for choice in ("overlap", "finish")
            ]
            result = diagnose(comparison, "local-test")
            self.assertEqual(len(result.findings), 1)
            self.assertIn("选择了检查步骤", result.message)
            requests = opener.return_value.open.call_args_list
            second_request = requests[1].args[0]
            self.assertEqual(second_request.full_url, "http://127.0.0.1:11434/api/chat")
            body = json.loads(second_request.data)
            contents = json.loads(body["messages"][1]["content"])
            self.assertEqual(contents["allowed"], ["by_size", "finish"])
            self.assertIn("overlap", contents["results"])
            self.assertNotIn("line-001.jpg", second_request.data.decode())

    def test_unavailable_model_falls_back_to_both_checks(self):
        comparison = compare(self.dataset, self.baseline, self.candidate, 0.3, 0.5)
        with patch("yolo_regression.diagnostics._choose", side_effect=OSError):
            result = diagnose(comparison, "local-test")
        self.assertEqual(result.findings, diagnose(comparison, None).findings)
        self.assertIn("固定规则", result.message)
