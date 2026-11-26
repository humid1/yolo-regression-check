---
name: yolo-regression-check
description: Compare baseline and candidate YOLO detection predictions on the same labeled validation set, diagnose newly missed objects, and report whether a per-class recall gate passes. Use when reviewing a YOLO version or investigating a detection regression from exported COCO-style JSON.
---

# YOLO Regression Check

- Identify the annotation JSON, baseline prediction JSON, and candidate prediction JSON. Keep the baseline and candidate order as requested. The three files must refer to the same validation set and share image and category IDs and their JSON types. Use the bundled demo only when the user asks for a demo.
- Call the MCP tool `compare_yolo_predictions` when available. Pass the three paths as `annotations`, `baseline`, and `candidate`. Paths are relative to the configured workspace or absolute paths inside it. Pass `images_dir` when sample images are available. Let the server generate a new report path unless the user provides an unused `.html` path.
- Use the user's confidence and IoU thresholds when specified. The defaults are 0.3 confidence and 0.5 IoU. Set `max_recall_drop_pp` only when the user supplies a recall gate. This value is in percentage points.
- When MCP is unavailable, run the installed CLI or run it from this repository. Choose a fresh report path and add `--images-dir`, `--confidence`, `--iou`, or `--max-recall-drop-pp` to match the request.

```sh
python -m yolo_regression --annotations annotations.json --baseline baseline.json --candidate candidate.json --output yolo-regression-report.html
```

Report the largest recall drops and the affected samples, along with the HTML report path. An object counts as newly missed when the baseline matched its annotation and the candidate did not. Treat low-score boxes and overlapping boxes of another class as clues to review. With MCP, a completed run can return `gate_status: failed` and still produce a report. With the CLI, exit code `1` has the same meaning when a gate was configured, and code `2` indicates an error.
