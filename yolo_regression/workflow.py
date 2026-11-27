import json
from dataclasses import asdict, dataclass
from pathlib import Path
from tempfile import NamedTemporaryFile

from .diagnostics import Diagnosis, diagnose
from .evaluation import Comparison, compare
from .gates import GatePolicy, GateResult, evaluate_gate
from .inputs import load_dataset, load_predictions
from .report import render_report
from .summary import build_summary


@dataclass(frozen=True)
class ComparisonOptions:
    confidence: float = 0.3
    iou: float = 0.5
    top: int = 20
    gate: GatePolicy = GatePolicy()
    ollama_model: str | None = None

    def __post_init__(self):
        if not 0 <= self.confidence <= 1:
            raise ValueError("--confidence 应在 0 到 1 之间")
        if not 0 < self.iou <= 1:
            raise ValueError("--iou 应大于 0 且小于或等于 1")
        if self.top < 0:
            raise ValueError("--top 应为非负整数")
        if self.ollama_model is not None and not self.ollama_model.strip():
            raise ValueError("--ollama-model 需要非空模型名")


@dataclass(frozen=True)
class ComparisonRun:
    comparison: Comparison
    diagnosis: Diagnosis
    gate: GateResult
    report_path: Path


def _write_text(path: Path, content: str, overwrite: bool):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = None
    try:
        with NamedTemporaryFile(mode="w", encoding="utf-8", dir=path.parent, suffix=".tmp", delete=False) as handle:
            temporary = Path(handle.name)
            handle.write(content)
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
                   json_output: Path | None = None, overwrite: bool = True) -> ComparisonRun:
    output = output.resolve()
    input_paths = {annotations.resolve(), baseline.resolve(), candidate.resolve()}
    if output in input_paths:
        raise ValueError("报告输出路径与输入文件相同")
    if json_output is not None:
        json_output = json_output.resolve()
        if json_output in input_paths or json_output == output:
            raise ValueError("JSON 输出路径与输入文件或 HTML 报告相同")
    if images_dir is not None:
        images_dir = images_dir.resolve()
        if not images_dir.is_dir():
            raise ValueError(f"{images_dir} 应为样本图片目录")

    dataset = load_dataset(annotations)
    baseline_predictions = load_predictions(baseline, dataset)
    candidate_predictions = load_predictions(candidate, dataset)
    comparison = compare(dataset, baseline_predictions, candidate_predictions, options.confidence, options.iou)
    gate = evaluate_gate(comparison, options.gate)
    diagnosis = diagnose(comparison, options.ollama_model)
    html = render_report(comparison, diagnosis, images_dir, options.top, gate)
    _write_text(output, html, overwrite)
    if json_output is not None:
        summary = build_summary(comparison, diagnosis, gate, output)
        _write_text(json_output, json.dumps(asdict(summary), ensure_ascii=False, indent=2) + "\n", overwrite)
    return ComparisonRun(comparison, diagnosis, gate, output)
