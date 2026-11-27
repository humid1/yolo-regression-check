from dataclasses import dataclass
from pathlib import Path

from .diagnostics import Diagnosis, Finding
from .evaluation import Comparison
from .gates import GateFailure, GateResult, GateStatus, Metric
from .inputs import Id


@dataclass(frozen=True)
class ClassSummary:
    category_id: int
    name: str
    ground_truth: int
    baseline_tp: int
    candidate_tp: int
    baseline_fp: int
    candidate_fp: int
    recall_drop_pp: float | None
    new_misses: int
    recovered: int
    added_false_positives: int
    resolved_false_positives: int
    persistent_false_positives: int


@dataclass(frozen=True)
class SampleSummary:
    image_id: Id
    file_name: str
    new_misses: int
    added_false_positives: int
    resolved_false_positives: int
    changed_category_ids: list[int]


@dataclass(frozen=True)
class ComparisonSummary:
    schema_version: int
    confidence: float
    iou: float
    report_path: str
    gate_status: GateStatus
    gate_limits: dict[Metric, int | float]
    gate_class_ids: list[int]
    gate_failures: list[GateFailure]
    failed_category_ids: list[int]
    validation_images: int
    affected_images: int
    new_misses: int
    recovered: int
    added_false_positives: int
    resolved_false_positives: int
    persistent_false_positives: int
    class_changes: list[ClassSummary]
    sample_previews: list[SampleSummary]
    diagnostic_summary: str
    findings: list[Finding]


def build_summary(comparison: Comparison, diagnosis: Diagnosis, gate: GateResult, report_path: Path, *,
                  class_limit: int | None = None, sample_limit: int | None = None) -> ComparisonSummary:
    classes = comparison.classes
    samples = []
    for row in comparison.images[:sample_limit]:
        changes = row.false_positives
        category_ids = {label.category_id for label in row.new_misses}
        category_ids.update(box.category_id for box in changes.added + changes.resolved)
        samples.append(SampleSummary(
            row.image_id, comparison.dataset.images[row.image_id].file_name, len(row.new_misses), row.added_fp,
            len(changes.resolved), sorted(category_ids),
        ))
    return ComparisonSummary(
        schema_version=1,
        confidence=comparison.confidence,
        iou=comparison.iou,
        report_path=str(report_path),
        gate_status=gate.status,
        gate_limits=gate.limits,
        gate_class_ids=list(gate.category_ids),
        gate_failures=gate.failures,
        failed_category_ids=list(dict.fromkeys(failure.category_id for failure in gate.failures)),
        validation_images=len(comparison.dataset.images),
        affected_images=len(comparison.images),
        new_misses=sum(row.new_misses for row in classes),
        recovered=sum(row.recovered for row in classes),
        added_false_positives=sum(row.added_fp for row in classes),
        resolved_false_positives=sum(row.resolved_fp for row in classes),
        persistent_false_positives=sum(row.persistent_fp for row in classes),
        class_changes=[ClassSummary(
            row.category_id, row.name, row.total, row.baseline_tp, row.candidate_tp,
            row.baseline_fp, row.candidate_fp, row.recall_drop_pp if row.total else None,
            row.new_misses, row.recovered, row.added_fp, row.resolved_fp, row.persistent_fp,
        ) for row in classes[:class_limit]],
        sample_previews=samples,
        diagnostic_summary=diagnosis.message,
        findings=diagnosis.findings,
    )
