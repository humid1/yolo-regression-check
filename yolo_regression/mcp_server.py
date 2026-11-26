import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import Literal
from uuid import uuid4

from .diagnostics import Finding
from .inputs import Id
from .workflow import ComparisonOptions, run_comparison


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


@dataclass(frozen=True)
class SampleSummary:
    image_id: Id
    file_name: str
    new_misses: int
    added_false_positives: int


@dataclass(frozen=True)
class ComparisonSummary:
    report_path: str
    gate_status: Literal["not_configured", "passed", "failed"]
    failed_category_ids: list[int]
    validation_images: int
    affected_images: int
    new_misses: int
    recovered: int
    added_false_positives: int
    class_changes: list[ClassSummary]
    sample_previews: list[SampleSummary]
    diagnostic_summary: str
    findings: list[Finding]


def _workspace_path(workspace: Path, value: str, field: str) -> Path:
    path = Path(value)
    if path.anchor and (not path.is_absolute() or not path.is_relative_to(workspace)):
        raise ValueError(f"{field} must be inside the configured workspace")
    resolved = (workspace / path).resolve()
    if not resolved.is_relative_to(workspace):
        raise ValueError(f"{field} must be inside the configured workspace")
    return resolved


def _json_file(workspace: Path, value: str, field: str) -> Path:
    path = _workspace_path(workspace, value, field)
    if path.suffix.lower() != ".json" or not path.is_file():
        raise ValueError(f"{field} must point to an existing .json file")
    return path


def create_server(workspace: Path):
    from mcp.server.fastmcp import FastMCP
    from mcp.types import ToolAnnotations

    workspace = workspace.resolve()
    if not workspace.is_dir():
        raise ValueError("--workspace must point to a directory")
    server = FastMCP("YOLO Regression Check")

    @server.tool(annotations=ToolAnnotations(
        title="Compare YOLO predictions", readOnlyHint=False, destructiveHint=False,
        idempotentHint=False, openWorldHint=False,
    ))
    def compare_yolo_predictions(
        annotations: str,
        baseline: str,
        candidate: str,
        output_html: str | None = None,
        images_dir: str | None = None,
        confidence: float = ComparisonOptions.confidence,
        iou: float = ComparisonOptions.iou,
        top: int = ComparisonOptions.top,
        max_recall_drop_pp: float | None = None,
    ) -> ComparisonSummary:
        """Compare two COCO prediction JSON arrays on the same COCO annotation JSON.

        Provide paths relative to the configured workspace or absolute paths inside it.
        Use images_dir to embed sample images. Set confidence (0–1), iou (>0 to 1),
        and top (maximum report previews) as needed. Set max_recall_drop_pp in
        percentage points to evaluate a per-class recall gate. A failed gate still
        produces a report. Set output_html to an unused .html path or let the tool
        create a new report in reports/. The summary includes up to ten classes and
        five sample previews. Read the HTML report for the full comparison.
        """
        annotation_file = _json_file(workspace, annotations, "annotations")
        baseline_file = _json_file(workspace, baseline, "baseline")
        candidate_file = _json_file(workspace, candidate, "candidate")
        if output_html is None:
            output_html = f"reports/yolo-regression-{uuid4().hex}.html"
        output = _workspace_path(workspace, output_html, "output_html")
        if output.suffix.lower() != ".html":
            raise ValueError("output_html must end in .html")
        if output.exists():
            raise ValueError("output_html already exists. Choose another path.")
        images = _workspace_path(workspace, images_dir, "images_dir") if images_dir is not None else None

        options = ComparisonOptions(confidence=confidence, iou=iou, top=top, max_recall_drop_pp=max_recall_drop_pp)
        result = run_comparison(annotation_file, baseline_file, candidate_file, output,
                                options=options, images_dir=images, overwrite=False)
        comparison = result.comparison
        if max_recall_drop_pp is None:
            gate_status = "not_configured"
        else:
            gate_status = "failed" if result.failures else "passed"
        return ComparisonSummary(
            report_path=str(result.report_path),
            gate_status=gate_status,
            failed_category_ids=[row.category_id for row in result.failures],
            validation_images=len(comparison.dataset.images),
            affected_images=len(comparison.images),
            new_misses=sum(row.new_misses for row in comparison.classes),
            recovered=sum(row.recovered for row in comparison.classes),
            added_false_positives=sum(row.added_fp for row in comparison.images),
            class_changes=[ClassSummary(
                row.category_id, row.name, row.total, row.baseline_tp, row.candidate_tp,
                row.baseline_fp, row.candidate_fp, row.recall_drop_pp if row.total else None,
                row.new_misses, row.recovered,
            ) for row in comparison.classes[:10]],
            sample_previews=[SampleSummary(
                row.image_id, comparison.dataset.images[row.image_id].file_name, len(row.new_misses), row.added_fp,
            ) for row in comparison.images[:min(top, 5)]],
            diagnostic_summary=result.diagnosis.message,
            findings=result.diagnosis.findings,
        )

    return server


def main(argv: list[str] | None = None):
    parser = argparse.ArgumentParser(description="Run the YOLO regression MCP server over stdio")
    parser.add_argument("--workspace", required=True, type=Path, help="Directory containing the inputs and reports")
    args = parser.parse_args(argv)
    try:
        server = create_server(args.workspace)
    except ModuleNotFoundError as error:
        if error.name == "mcp":
            parser.error('Install MCP support with python -m pip install -e ".[mcp]"')
        raise
    except (OSError, RuntimeError, ValueError) as error:
        parser.error(str(error))
    server.run(transport="stdio")


if __name__ == "__main__":
    main()
