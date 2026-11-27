import argparse
from pathlib import Path
from uuid import uuid4

from .gates import GatePolicy
from .manifest import Sources
from .summary import ComparisonSummary, build_summary
from .workflow import ComparisonOptions, run_comparison


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
        max_new_misses: int | None = None,
        max_new_false_positives: int | None = None,
        gate_class_ids: list[int] | None = None,
        min_localization_drop: float = ComparisonOptions.min_localization_drop,
        baseline_manifest: str | None = None,
        candidate_manifest: str | None = None,
        baseline_revision: str | None = None,
        candidate_revision: str | None = None,
    ) -> ComparisonSummary:
        """Compare two COCO prediction JSON arrays on the same COCO annotation JSON.

        Provide paths relative to the configured workspace or absolute paths inside it.
        Use images_dir to embed sample images. Set confidence (0–1), iou (>0 to 1),
        and top (maximum report previews) as needed. Limits apply per class. Use
        percentage points for max_recall_drop_pp and counts for max_new_misses and
        max_new_false_positives. Use gate_class_ids to check specific classes.
        Provide both run manifests to check processed image IDs, categories and file hashes.
        Revisions are optional provenance labels. min_localization_drop is an absolute IoU difference.
        A failed gate still produces a report. Set output_html to an unused .html
        path or let the tool create a new report in reports/. The summary includes
        up to ten classes and five sample previews. Read the HTML report for the
        full comparison.
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
        sources = Sources(
            _json_file(workspace, baseline_manifest, "baseline_manifest") if baseline_manifest is not None else None,
            _json_file(workspace, candidate_manifest, "candidate_manifest") if candidate_manifest is not None else None,
            baseline_revision, candidate_revision,
        )

        gate = GatePolicy(max_recall_drop_pp, max_new_misses, max_new_false_positives, tuple(gate_class_ids or ()))
        options = ComparisonOptions(confidence=confidence, iou=iou, top=top, gate=gate,
                                    min_localization_drop=min_localization_drop)
        result = run_comparison(annotation_file, baseline_file, candidate_file, output,
                                options=options, images_dir=images, overwrite=False, sources=sources)
        return build_summary(result.comparison, result.diagnosis, result.gate, result.report_path, result.provenance,
                             class_limit=10, sample_limit=min(top, 5))

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
