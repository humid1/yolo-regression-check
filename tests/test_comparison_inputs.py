import contextlib
import io
import json
import tempfile
import unittest
from pathlib import Path

from yolo_regression.__main__ import main
from yolo_regression.manifest import main as create_manifest

PROJECT = Path(__file__).resolve().parents[1]
DEMO = PROJECT / "demo"


class ComparisonInputTests(unittest.TestCase):
    def setUp(self):
        directory = tempfile.TemporaryDirectory(dir=PROJECT)
        self.addCleanup(directory.cleanup)
        self.root = Path(directory.name)
        self.html = self.root / "report.html"
        self.summary = self.root / "report.json"

    def write_json(self, filename, data):
        path = self.root / filename
        path.write_text(json.dumps(data, ensure_ascii=False), encoding="utf-8")
        return path

    def compare(self, annotations, baseline, candidate, *options):
        args = ["--annotations", str(annotations), "--baseline", str(baseline), "--candidate", str(candidate),
                "--output", str(self.html), "--json-output", str(self.summary), *options]
        errors = io.StringIO()
        with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(errors):
            code = main(args)
        return code, errors.getvalue()

    def manifest(self, predictions, revision):
        output = io.StringIO()
        with contextlib.redirect_stdout(output):
            create_manifest(["--annotations", str(DEMO / "annotations.json"), "--predictions", str(predictions),
                             "--coverage", str(DEMO / "coverage.json"), "--revision", revision])
        return json.loads(output.getvalue())

    def test_manifest_covers_images_with_zero_predictions(self):
        baseline = DEMO / "baseline.json"
        candidate = DEMO / "candidate-equal-recall.json"
        old = self.write_json("baseline-manifest.json", self.manifest(baseline, "production"))
        new = self.write_json("candidate-manifest.json", self.manifest(candidate, "candidate"))
        self.assertNotIn(4, {row["image_id"] for row in json.loads(baseline.read_text(encoding="utf-8"))})
        code, errors = self.compare(DEMO / "annotations.json", baseline, candidate,
                                    "--baseline-manifest", str(old), "--candidate-manifest", str(new))
        self.assertEqual((code, errors), (0, ""))
        summary = json.loads(self.summary.read_text(encoding="utf-8"))
        self.assertTrue(summary["provenance"]["coverage_verified"])
        self.assertEqual(summary["provenance"]["baseline"]["revision"], "production")
        self.assertIn("已核对两份运行清单", self.html.read_text(encoding="utf-8"))

    def test_incomplete_or_inconsistent_manifest_is_an_input_error(self):
        baseline = DEMO / "baseline.json"
        candidate = DEMO / "candidate-equal-recall.json"
        old = self.write_json("baseline-manifest.json", self.manifest(baseline, "production"))
        complete = self.manifest(candidate, "candidate")
        swapped = [{**row, "name": "incorrect"} if row["id"] == 0 else row for row in complete["categories"]]
        for change, message in (({"image_ids": [1, 2, 3, 4]}, "缺少 1 张（5）"),
                                ({"predictions_sha256": "0" * 64}, "预测 SHA-256 不匹配"),
                                ({"annotations_sha256": "0" * 64}, "标注 SHA-256 不匹配"),
                                ({"categories": swapped}, "类别映射与标注不一致")):
            with self.subTest(change=change):
                new = self.write_json("candidate-manifest.json", {**complete, **change})
                code, errors = self.compare(DEMO / "annotations.json", baseline, candidate,
                                            "--baseline-manifest", str(old), "--candidate-manifest", str(new))
                self.assertEqual(code, 2)
                self.assertIn(message, errors)
                self.assertFalse(self.html.exists())
                self.assertFalse(self.summary.exists())

    def test_gate_evidence_is_first_in_html_and_json(self):
        labels = [
            {"image_id": 1, "category_id": 1, "bbox": [5, 5, 10, 10]},
            {"image_id": 1, "category_id": 1, "bbox": [25, 25, 10, 10]},
            {"image_id": 2, "category_id": 0, "bbox": [5, 5, 10, 10]},
        ]
        annotations = self.write_json("annotations.json", {
            "images": [{"id": index, "file_name": name, "width": 100, "height": 100}
                       for index, name in ((1, "general.jpg"), (2, "critical.jpg"))],
            "categories": [{"id": 0, "name": "critical"}, {"id": 1, "name": "general"}],
            "annotations": labels,
        })
        before = self.write_json("baseline.json", [{**row, "score": 0.9} for row in labels])
        after = self.write_json("candidate.json", [])
        code, _ = self.compare(annotations, before, after, "--gate-class", "0", "--max-new-misses", "0", "--top", "1")
        self.assertEqual(code, 1)
        html = self.html.read_text(encoding="utf-8")
        self.assertIn("<h3>critical.jpg</h3>", html)
        self.assertNotIn("<h3>general.jpg</h3>", html)
        rows = json.loads(self.summary.read_text(encoding="utf-8"))["sample_previews"]
        self.assertEqual([(row["file_name"], row["gate_evidence"]) for row in rows],
                         [("critical.jpg", True), ("general.jpg", False)])

    def test_localization_is_reported_when_both_boxes_match(self):
        truth = {"image_id": 1, "category_id": 0, "bbox": [5, 5, 10, 10]}
        annotations = self.write_json("annotations.json", {
            "images": [{"id": 1, "file_name": "shift.jpg", "width": 100, "height": 100}],
            "categories": [{"id": 0, "name": "item"}], "annotations": [truth],
        })
        before = self.write_json("baseline.json", [{**truth, "score": 0.9}])
        after = self.write_json("candidate.json", [{**truth, "bbox": [8, 5, 10, 10], "score": 0.9}])
        code, _ = self.compare(annotations, before, after, "--max-new-misses", "0")
        self.assertEqual(code, 0)
        summary = json.loads(self.summary.read_text(encoding="utf-8"))
        self.assertEqual((summary["new_misses"], summary["localization_regressions"]), (0, 1))
        change, = summary["sample_previews"][0]["localization_changes"]
        self.assertEqual(change["baseline_iou"], 1)
        self.assertAlmostEqual(change["candidate_iou"], 7 / 13)
        self.assertIn('class="box localization"', self.html.read_text(encoding="utf-8"))
        self.assertEqual(self.compare(annotations, before, after, "--min-localization-drop", "0.5")[0], 0)
        self.assertEqual(json.loads(self.summary.read_text(encoding="utf-8"))["affected_images"], 0)
