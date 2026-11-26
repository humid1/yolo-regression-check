# YOLO Regression Check

**English** · [简体中文](README.zh-CN.md)

Compare a baseline and a candidate YOLO model on the same labeled validation set. The HTML report shows changes in recall, precision, and false positives for each class and puts ground truth, baseline predictions, and candidate predictions side by side for affected images.

Use the report to review newly missed objects, investigate low-confidence predictions or possible class mix-ups, and catch unacceptable recall drops in CI. The CLI uses Python's standard library and runs offline by default.

## Quick start

Clone the repository and run the demo with Python 3.10 or later. The demo includes synthetic annotations and predictions, so you can run it without a GPU, model weights, or an API key.

```sh
git clone https://github.com/humid1/yolo-regression-check.git
cd yolo-regression-check
python -m yolo_regression --annotations demo/annotations.json --baseline demo/baseline.json --candidate demo/candidate.json --output demo-report.html
```

Open `demo-report.html` in your browser. In the demo, recall for the washer class (`垫片`) drops from 75% to 50%. The candidate misses two previously detected objects and recovers one previous miss. One of the new misses has a low-confidence prediction of the right class. The other has an overlapping prediction of a different class.

The demo shows bounding boxes on a grid. With your own validation set, add `--images-dir` to embed JPEG, PNG, or WebP images of up to 1 MB each. The CLI and generated report currently display their labels and messages in Chinese.

## Compare your own models

Generate predictions for both models on the same labeled validation set. The CLI accepts three COCO-style JSON files.

| Input | Required fields |
| --- | --- |
| Annotations | An object containing `images`, `categories`, and `annotations`. Each image needs `id`, `file_name`, `width`, and `height`. Each category needs `id` and `name`. Each annotation needs `image_id`, `category_id`, and `bbox`. |
| Baseline and candidate predictions | One array per model. Each prediction needs `image_id`, `category_id`, `bbox`, and a `score` between 0 and 1. |

Use original-image pixel coordinates in `[x, y, width, height]` format, with `x` and `y` at the top-left of the box. Image and category IDs must match across the files, including the JSON type of each ID. If you export predictions with Ultralytics `save_json=True`, check the ID mappings and export at a confidence threshold below the one you use for comparison if you want to inspect low-scoring boxes. [Ultralytics prediction export documentation](https://docs.ultralytics.com/reference/models/yolo/detect/val/#ultralytics.models.yolo.detect.val.DetectionValidator.pred_to_json)

```sh
python -m yolo_regression --annotations instances_val.json --baseline baseline/predictions.json --candidate candidate/predictions.json --images-dir images/val --confidence 0.3 --iou 0.5 --max-recall-drop-pp 5 --output report.html
```

For each image and class, the CLI processes predictions in descending confidence order. Each prediction is matched to the unmatched annotation with the highest IoU, provided both thresholds are met. Crowd annotations are excluded from the recall denominator. An unmatched prediction is excluded from the false-positive count when enough of its area overlaps a crowd annotation of the same class. Metrics are calculated at the selected confidence and IoU thresholds. Use `--top` to limit sample previews, which are ordered by newly missed objects and then by increases in false positives.

## CI checks

Set `--max-recall-drop-pp` to the maximum allowed recall drop in **percentage points** for any class. For example, a drop from 75% to 50% is 25 percentage points. The report is written even when the check fails, so CI can keep it as a build artifact.

| Exit code | Meaning |
| --- | --- |
| `0` | The report was generated and any configured check passed. |
| `1` | At least one class exceeded the configured recall drop. |
| `2` | An argument, input, or output error occurred. |

## Optional local Ollama diagnostics

By default, the CLI investigates the class with the most newly missed objects. It groups annotations by bounding-box area, then checks whether new misses overlap a same-class prediction below the confidence threshold or a prediction of a different class at or above it.

Add `--ollama-model YOUR_LOCAL_MODEL` to let a model running in Ollama choose the next check based on the aggregate results. Requests go to `127.0.0.1:11434` and include only aggregate findings and predefined check names. If Ollama is unavailable or returns an invalid response, the CLI completes the remaining checks using its built-in rules. The CI outcome is determined by the configured recall threshold. [Ollama structured outputs documentation](https://docs.ollama.com/capabilities/structured-outputs)

## Use with Claude Code, Codex, or another MCP client

Install the optional MCP extra from this repository. Use a virtual environment if you want to keep the dependencies isolated.

```sh
python -m pip install -e ".[mcp]"
```

In the commands below, replace `/absolute/path/to/python` with the interpreter you used to install the extra, and replace `/absolute/path/to/workspace` with the directory that contains your validation data. To try the demo, use the root of this repository as the workspace.

```sh
claude mcp add --transport stdio yolo-regression -- "/absolute/path/to/python" -m yolo_regression.mcp_server --workspace "/absolute/path/to/workspace"
codex mcp add yolo-regression -- "/absolute/path/to/python" -m yolo_regression.mcp_server --workspace "/absolute/path/to/workspace"
```

Run the line for your client. Use `claude mcp list` or `codex mcp list` to verify registration, then start a new session. Other local MCP clients can use the same Python interpreter and arguments with the `stdio` transport. [Claude Code MCP setup](https://code.claude.com/docs/en/mcp) · [Codex MCP setup](https://developers.openai.com/codex/mcp)

Ask the client to compare `demo/baseline.json` and `demo/candidate.json` using `demo/annotations.json` and a 5-percentage-point recall gate. The `compare_yolo_predictions` tool returns a structured summary, the gate status, and a path to the full HTML report. A failed gate still returns a report. Paths must resolve inside `--workspace`. By default, each call creates a new file under `reports/` and leaves existing reports intact.

### Use the Skill

The repository includes a [Claude Code Skill](.claude/skills/yolo-regression-check/SKILL.md) and a [Codex Skill](.agents/skills/yolo-regression-check/SKILL.md). Start the client in this checkout or copy the corresponding `yolo-regression-check` skill folder into the same skill location in your own project. Invoke `/yolo-regression-check` in Claude Code or `$yolo-regression-check` in Codex. The Skill uses the MCP tool when available and can run the CLI directly otherwise. [Claude Code Skills](https://code.claude.com/docs/en/skills) · [Codex Skills](https://developers.openai.com/codex/skills)

## Run the tests

```sh
python -m unittest discover -s tests -v
```

Install the optional MCP extra to include the stdio integration test.

## License

[MIT](LICENSE)
