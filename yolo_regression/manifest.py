import argparse
import json
from dataclasses import dataclass
from pathlib import Path

from .inputs import Dataset, Id, _image_id, _integer, _nonempty_string, _rows, load_dataset, load_predictions
from .inputs import read_json_document


@dataclass(frozen=True)
class Sources:
    baseline_manifest: Path | None = None
    candidate_manifest: Path | None = None
    baseline_revision: str | None = None
    candidate_revision: str | None = None

    def __post_init__(self):
        if (self.baseline_manifest is None) != (self.candidate_manifest is None):
            raise ValueError("需要同时提供基准与候选运行清单")
        for revision in (self.baseline_revision, self.candidate_revision):
            if revision is not None:
                _nonempty_string(revision, "revision")


@dataclass(frozen=True)
class RunSource:
    predictions_sha256: str
    revision: str | None


@dataclass(frozen=True)
class Provenance:
    annotations_sha256: str
    baseline: RunSource
    candidate: RunSource
    coverage_verified: bool


def _read_coverage(raw, context) -> tuple[list[Id], dict[int, str]]:
    if not isinstance(raw, dict) or not isinstance(raw.get("image_ids"), list):
        raise ValueError(f"{context} 需要 image_ids 数组")
    image_ids = [_image_id(item, f"{context} image_ids") for item in raw["image_ids"]]
    if len(image_ids) != len(set(image_ids)):
        raise ValueError(f"{context} 的 image_ids 有重复项")

    categories = {}
    for row, location in _rows(raw.get("categories"), f"{context} categories"):
        category_id = _integer(row.get("id"), f"{location} id")
        if category_id in categories:
            raise ValueError(f"{context} 有重复类别 ID {category_id}")
        categories[category_id] = _nonempty_string(row.get("name"), f"{location} name")
    return image_ids, categories


def _check_coverage(dataset: Dataset, image_ids: list[Id], categories: dict[int, str], context):
    expected = dataset.images.keys()
    actual = set(image_ids)
    missing, unexpected = expected - actual, actual - expected
    if missing or unexpected:
        missing_examples = ", ".join(sorted(map(repr, missing))[:3]) or "—"
        unexpected_examples = ", ".join(sorted(map(repr, unexpected))[:3]) or "—"
        raise ValueError(f"{context} 图片覆盖与标注不一致，缺少 {len(missing)} 张（{missing_examples}），"
                         f"多出 {len(unexpected)} 张（{unexpected_examples}）")
    if categories != dataset.categories:
        changed = [key for key in categories.keys() | dataset.categories.keys()
                   if categories.get(key) != dataset.categories.get(key)]
        examples = ", ".join(map(str, sorted(changed)[:5]))
        raise ValueError(f"{context} 类别映射与标注不一致，涉及类别 ID {examples}")


def _verify_run(dataset: Dataset, manifest: Path, annotations_sha256: str,
                predictions_sha256: str, revision: str | None) -> RunSource:
    raw = read_json_document(manifest).value
    if not isinstance(raw, dict) or type(raw.get("schema_version")) is not int or raw["schema_version"] != 1:
        raise ValueError(f"{manifest} 需要 schema_version 为 1")
    if raw.get("annotations_sha256") != annotations_sha256:
        raise ValueError(f"{manifest} 标注 SHA-256 不匹配")
    if raw.get("predictions_sha256") != predictions_sha256:
        raise ValueError(f"{manifest} 预测 SHA-256 不匹配")
    image_ids, categories = _read_coverage(raw, manifest)
    _check_coverage(dataset, image_ids, categories, manifest)
    recorded = raw.get("revision")
    if recorded is not None:
        _nonempty_string(recorded, f"{manifest} revision")
        if revision is not None and recorded != revision:
            raise ValueError(f"{manifest} 的 revision 与传入版本不一致")
    return RunSource(predictions_sha256, revision or recorded)


def verify_sources(dataset: Dataset, annotations_sha256: str, baseline_sha256: str,
                   candidate_sha256: str, sources: Sources) -> Provenance:
    if sources.baseline_manifest is not None and sources.candidate_manifest is not None:
        baseline = _verify_run(dataset, sources.baseline_manifest, annotations_sha256,
                               baseline_sha256, sources.baseline_revision)
        candidate = _verify_run(dataset, sources.candidate_manifest, annotations_sha256,
                                candidate_sha256, sources.candidate_revision)
        return Provenance(annotations_sha256, baseline, candidate, coverage_verified=True)

    return Provenance(annotations_sha256, RunSource(baseline_sha256, sources.baseline_revision),
                      RunSource(candidate_sha256, sources.candidate_revision), coverage_verified=False)


def main(argv: list[str] | None = None):
    parser = argparse.ArgumentParser(description="Create a run manifest from evaluator-reported coverage")
    parser.add_argument("--annotations", required=True, type=Path)
    parser.add_argument("--predictions", required=True, type=Path)
    parser.add_argument("--coverage", required=True, type=Path, help="JSON with the processed image_ids and categories")
    parser.add_argument("--revision", help="Optional model, commit, or evaluation job identifier")
    args = parser.parse_args(argv)
    try:
        annotations = read_json_document(args.annotations)
        predictions = read_json_document(args.predictions)
        dataset = load_dataset(annotations)
        load_predictions(predictions, dataset)
        image_ids, categories = _read_coverage(read_json_document(args.coverage).value, args.coverage)
        _check_coverage(dataset, image_ids, categories, args.coverage)
        if args.revision is not None:
            _nonempty_string(args.revision, "revision")
    except ValueError as error:
        parser.error(str(error))
    print(json.dumps({
        "schema_version": 1,
        "annotations_sha256": annotations.sha256,
        "predictions_sha256": predictions.sha256,
        "image_ids": image_ids,
        "categories": [{"id": category_id, "name": name} for category_id, name in categories.items()],
        "revision": args.revision,
    }, indent=2))


if __name__ == "__main__":
    main()
