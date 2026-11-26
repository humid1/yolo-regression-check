import math
from dataclasses import dataclass
from pathlib import Path
from tempfile import NamedTemporaryFile

from .diagnostics import Diagnosis, diagnose
from .evaluation import ClassChange, Comparison, compare
from .inputs import load_dataset, load_predictions
from .report import render_report


@dataclass(frozen=True)
class ComparisonOptions:
    confidence: float = 0.3
    iou: float = 0.5
    top: int = 20
    max_recall_drop_pp: float | None = None
    ollama_model: str | None = None

    def __post_init__(self):
        if not 0 <= self.confidence <= 1:
            raise ValueError("--confidence 应在 0 到 1 之间")
        if not 0 < self.iou <= 1:
            raise ValueError("--iou 应大于 0 且小于或等于 1")
        if self.top < 0:
            raise ValueError("--top 应为非负整数")
        if self.max_recall_drop_pp is not None and not 0 <= self.max_recall_drop_pp <= 100:
            raise ValueError("--max-recall-drop-pp 应在 0 到 100 之间")
        if self.ollama_model is not None and not self.ollama_model.strip():
            raise ValueError("--ollama-model 需要非空模型名")


@dataclass(frozen=True)
class ComparisonRun:
    comparison: Comparison
    diagnosis: Diagnosis
    failures: list[ClassChange]
    report_path: Path


def _write_report(path: Path, html: str, overwrite: bool):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = None
    try:
        with NamedTemporaryFile(mode="w", encoding="utf-8", dir=path.parent, suffix=".tmp", delete=False) as handle:
            temporary = Path(handle.name)
            handle.write(html)
        if overwrite:
            temporary.replace(path)
        else:
            # Linking publishes the completed file and fails atomically if the destination exists.
            path.hardlink_to(temporary)
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


def run_comparison(annotations: Path, baseline: Path, candidate: Path, output: Path, *,
                   options: ComparisonOptions, images_dir: Path | None = None,
                   overwrite: bool = True) -> ComparisonRun:
    output = output.resolve()
    if output in (annotations.resolve(), baseline.resolve(), candidate.resolve()):
        raise ValueError("报告输出路径与输入文件相同")
    if images_dir is not None:
        images_dir = images_dir.resolve()
        if not images_dir.is_dir():
            raise ValueError(f"{images_dir} 应为样本图片目录")

    dataset = load_dataset(annotations)
    baseline_predictions = load_predictions(baseline, dataset)
    candidate_predictions = load_predictions(candidate, dataset)
    comparison = compare(dataset, baseline_predictions, candidate_predictions, options.confidence, options.iou)
    diagnosis = diagnose(comparison, options.ollama_model)
    failures = []
    limit = options.max_recall_drop_pp
    if limit is not None:
        failures = [row for row in comparison.classes if row.recall_drop_pp > limit
                    and not math.isclose(row.recall_drop_pp, limit, rel_tol=0, abs_tol=1e-10)]
    html = render_report(comparison, diagnosis, images_dir, options.top, limit, failures)
    _write_report(output, html, overwrite)
    return ComparisonRun(comparison, diagnosis, failures, output)
