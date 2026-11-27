---
name: yolo-regression-check
description: Compare baseline and candidate YOLO detection predictions on the same labeled validation set, diagnose new misses and false positives, and check per-class release limits. Use when reviewing a YOLO version or investigating a detection regression from exported COCO-style JSON.
---

# YOLO Regression Check

- Identify the annotation JSON, baseline prediction JSON, and candidate prediction JSON. Keep the baseline and candidate order as requested. The three files must refer to the same validation set and share image and category IDs and their JSON types. Use the bundled demo only when the user asks for a demo.
- Call the MCP tool `compare_yolo_predictions` when available. Pass the three paths as `annotations`, `baseline`, and `candidate`. Paths are relative to the configured workspace or absolute paths inside it. Pass `images_dir` when sample images are available. Let the server generate a new report path unless the user provides an unused `.html` path.
- Pass both run manifests when available. They must come from evaluator-reported coverage, including images with zero detections. Use `baseline_manifest` and `candidate_manifest` in MCP, or `--baseline-manifest` and `--candidate-manifest` in the CLI. If verification fails, report the input mismatch.
- Use the user's confidence and IoU thresholds when specified. The defaults are 0.3 confidence and 0.5 IoU. Configure only the limits requested. Use percentage points for `max_recall_drop_pp` and counts for `max_new_misses` and `max_new_false_positives`. Pass annotation category IDs in `gate_class_ids` when the checks are scoped to particular classes.
- When MCP is unavailable, run the installed CLI or run it from this repository. Choose a fresh report path. Use `--max-recall-drop-pp`, `--max-new-misses`, and `--max-new-false-positives` for the corresponding limits. Repeat `--gate-class` for each selected category ID and use `--json-output` when a machine-readable report is requested.

```sh
python -m yolo_regression --annotations annotations.json --baseline baseline.json --candidate candidate.json --output yolo-regression-report.html
```

Report the largest recall drops, new misses or false positives, affected samples, and the HTML report path. Use `gate_failures` to identify which rules failed and cite samples marked `gate_evidence`. For localization regressions, report both IoUs for the same annotation. Treat low-score boxes and overlapping boxes of another class as clues to review. MCP can return `gate_status: failed` after generating the report. CLI exit code `1` also indicates a failed gate with reports available, and code `2` indicates an error.
