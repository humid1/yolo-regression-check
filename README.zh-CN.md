# YOLO 模型回归诊断

[English](README.md) · **简体中文**

在同一份带标注的验证集上对比基准 YOLO 模型与候选模型。HTML 报告展示各类别召回率、精确率和误检数的变化，并把受影响图片的标注框、基准预测框和候选预测框并排展示。

你可以用报告复查新出现的漏检，找出低置信度预测或可能的类别混淆，也可以在 CI 中拦截召回率下降过多的模型版本。命令行工具只使用 Python 标准库，默认离线运行。

## 快速开始

克隆仓库并使用 Python 3.10 或更新版本运行示例。示例自带合成标注和预测数据，无需 GPU、模型权重或 API Key。

```sh
git clone https://github.com/humid1/yolo-regression-check.git
cd yolo-regression-check
python -m yolo_regression --annotations demo/annotations.json --baseline demo/baseline.json --candidate demo/candidate.json --output demo-report.html
```

在浏览器中打开 `demo-report.html`。示例里，垫片类别的召回率从 75% 降到 50%。候选模型漏掉了两个原来能检出的目标，同时恢复检出一个原来漏掉的目标。两处新增漏检各有一条线索，一处有置信度偏低的同类预测框，另一处有位置重叠但类别不同的预测框。

示例用网格显示框的位置。使用自己的验证集时，可以加上 `--images-dir` 将 JPEG、PNG 或 WebP 图片嵌入报告，每张图最多 1 MB。命令行提示和报告界面目前使用简体中文。

## 对比自己的模型

先在同一份带标注的验证集上导出两个模型的预测。工具接收三份 COCO 风格的 JSON 文件。

| 输入 | 必填字段 |
| --- | --- |
| 标注 | 一个包含 `images`、`categories` 和 `annotations` 的对象。每张图片包含 `id`、`file_name`、`width`、`height`，每个类别包含 `id` 和 `name`，每条标注包含 `image_id`、`category_id` 和 `bbox`。 |
| 基准预测和候选预测 | 每个模型对应一个数组。每条预测包含 `image_id`、`category_id`、`bbox` 和取值范围为 0 到 1 的 `score`。 |

`bbox` 使用原图的像素坐标 `[x, y, width, height]`，其中 `x` 和 `y` 是框左上角的坐标。三份文件的图片 ID 和类别 ID（包括 JSON 类型）必须一致。如果用 Ultralytics 的 `save_json=True` 导出预测，请核对 ID 映射。需要复查低置信度预测时，导出阈值应低于诊断所用阈值。 [Ultralytics 预测导出文档](https://docs.ultralytics.com/reference/models/yolo/detect/val/#ultralytics.models.yolo.detect.val.DetectionValidator.pred_to_json)

```sh
python -m yolo_regression --annotations instances_val.json --baseline baseline/predictions.json --candidate candidate/predictions.json --images-dir images/val --confidence 0.3 --iou 0.5 --max-recall-drop-pp 5 --output report.html
```

在同一张图的同一类别内，工具按置信度从高到低处理预测，同时满足置信度和 IoU 阈值时，会匹配 IoU 最大且尚未匹配的标注。`iscrowd` 标注不计入召回率的分母。未匹配的同类预测有足够面积与 `iscrowd` 标注重叠时，不计为误检。报告指标基于选定的置信度和 IoU 阈值。可用 `--top` 限制展示样本数量，样本优先按新增漏检数排序，其次按误检增加数排序。

## CI 检查

用 `--max-recall-drop-pp` 设置每个类别允许的最大召回降幅，单位为**百分点**。例如，从 75% 降到 50% 是下降 25 个百分点。超出阈值时仍会写出报告，可在 CI 中将它作为构建产物保留。

| 退出码 | 含义 |
| --- | --- |
| `0` | 报告生成成功，已配置的检查通过。 |
| `1` | 至少一个类别的召回降幅超过阈值。 |
| `2` | 参数、输入或输出有误。 |

## 可选的本地 Ollama 诊断

默认检查新增漏检最多的类别。工具会按标注框面积分组，并检查新增漏检附近是否有置信度低于阈值的同类别预测，或者置信度达到阈值的其他类别预测。

加上 `--ollama-model YOUR_LOCAL_MODEL`，即可让本地 Ollama 模型根据已收集的统计选择下一项检查。请求发往 `127.0.0.1:11434`，只包含汇总结果和预定义的检查名称。Ollama 不可用或响应无效时，工具会使用内置规则完成剩余检查。CI 结果由配置的召回阈值决定。 [Ollama 结构化输出文档](https://docs.ollama.com/capabilities/structured-outputs)

## 接入 Claude Code、Codex 或其他 MCP 客户端

在本仓库目录下安装可选的 MCP 依赖。需要隔离依赖时，可以先激活 Python 虚拟环境。

```sh
python -m pip install -e ".[mcp]"
```

把下面的 `/absolute/path/to/python` 换成安装依赖时使用的 Python 解释器，把 `/absolute/path/to/workspace` 换成存放验证数据的目录。试跑示例时，将工作目录设为本仓库根目录即可。

```sh
claude mcp add --transport stdio yolo-regression -- "/absolute/path/to/python" -m yolo_regression.mcp_server --workspace "/absolute/path/to/workspace"
codex mcp add yolo-regression -- "/absolute/path/to/python" -m yolo_regression.mcp_server --workspace "/absolute/path/to/workspace"
```

执行所用客户端对应的命令，通过 `claude mcp list` 或 `codex mcp list` 确认注册后，开启新会话。其他本地 MCP 客户端可用同一个解释器和参数，通过 `stdio` 连接。 [Claude Code MCP 配置](https://code.claude.com/docs/en/mcp) · [Codex MCP 配置](https://developers.openai.com/codex/mcp)

试着让客户端用 `demo/annotations.json` 对比 `demo/baseline.json` 和 `demo/candidate.json`，并设置 5 个百分点的召回降幅阈值。`compare_yolo_predictions` 工具会返回结构化诊断摘要、阈值检查结果和完整 HTML 报告路径。超出阈值时仍会生成报告。所有路径解析后都须位于 `--workspace` 目录内，每次调用默认在 `reports/` 下生成一份新报告，已有报告会保留。

### 使用 Skill

仓库自带 [Claude Code Skill](.claude/skills/yolo-regression-check/SKILL.md) 和 [Codex Skill](.agents/skills/yolo-regression-check/SKILL.md)。可以在本仓库中启动客户端，也可以把对应的 `yolo-regression-check` 目录复制到自己项目的同一 Skill 位置。在 Claude Code 中调用 `/yolo-regression-check`，在 Codex 中调用 `$yolo-regression-check`。Skill 优先使用 MCP 工具，也支持直接运行 CLI。 [Claude Code Skills](https://code.claude.com/docs/en/skills) · [Codex Skills](https://developers.openai.com/codex/skills)

## 运行测试

```sh
python -m unittest discover -s tests -v
```

安装可选的 MCP 依赖后，测试会包含 `stdio` 集成测试。

## 许可证

[MIT](LICENSE)
