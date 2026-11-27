# YOLO 模型回归诊断

[English](README.md) · **简体中文**

在同一份带标注的验证集上对比基准 YOLO 模型与候选模型。HTML 报告展示各类别召回率、精确率和误检数的变化，并把受影响图片的标注框、基准预测框和候选预测框并排展示。

你可以用报告复查新出现的漏检，找出低置信度预测或可能的类别混淆，也可以在 CI 中拦截超出阈值的模型版本。命令行工具只使用 Python 标准库，默认离线运行。

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

在同一张图的同一类别内，工具按置信度从高到低处理预测，同时满足置信度和 IoU 阈值时，会匹配 IoU 最大且尚未匹配的标注。`iscrowd` 标注不计入召回率的分母。未匹配的同类预测有足够面积与 `iscrowd` 标注重叠时，不计为误检。

工具再按图片、类别和所选的 IoU 阈值，对两版误检框做一对一匹配。配对的是持续误检，未配对的候选框是新增误检，未配对的基准框是消除的误检。因此，误检框换到图片的另一处时，即使总数相同，也可能同时记录一个新增和一个消除的误检。

两版都检出同一标注时，报告会突出定位退步并显示各自的 IoU。`--min-localization-drop` 控制纳入报告的最小绝对降幅，默认是在 0 到 1 的 IoU 刻度上下降 `0.1`。可用 `--top` 限制 HTML 预览数，触发门禁规则的样本优先展示。

## 核对验证集覆盖范围

让评估程序记录每张实际处理完成的图片 ID，包括零检出的图片，并保存本次评估使用的类别映射。覆盖记录包含 `image_ids` 数组和由 `{id, name}` 对象组成的 `categories` 数组。[示例覆盖记录](demo/coverage.json) 对应仓库中的合成数据。对比自己的模型时，应使用每次评估实际产生的记录。

```sh
python -m yolo_regression.manifest --annotations demo/annotations.json --predictions demo/baseline.json --coverage demo/coverage.json --revision baseline-v1 > baseline-manifest.json
python -m yolo_regression.manifest --annotations demo/annotations.json --predictions demo/candidate-equal-recall.json --coverage demo/coverage.json --revision candidate-v2 > candidate-manifest.json
python -m yolo_regression --annotations demo/annotations.json --baseline demo/baseline.json --candidate demo/candidate-equal-recall.json --baseline-manifest baseline-manifest.json --candidate-manifest candidate-manifest.json --output reports/verified.html --json-output reports/verified.json
```

提供两份运行清单后，如果图片覆盖、类别映射或文件哈希对不上，程序在比较前返回退出码 `2`。两份清单应对应本次比较使用的同一份标注文件，哈希按原始字节计算。报告会记录三个输入文件的 SHA-256。可直接传入 `--baseline-revision` 和 `--candidate-revision`，也可在生成清单时记录来源版本。

## CI 检查

可以按类别启用一项或多项阈值。指标超过阈值时检查失败，仍会写出报告。

| 参数 | 检查内容 |
| --- | --- |
| `--max-recall-drop-pp` | 允许的净召回降幅，单位为**百分点**。从 75% 降到 50% 是下降 25 个百分点。 |
| `--max-new-misses` | 允许的新增漏检数，对应基准模型检出而候选模型漏掉的标注。 |
| `--max-new-false-positives` | 允许的新增误检框数，对应候选模型中未配对的误检框。 |
| `--gate-class ID` | 仅检查指定的标注类别 ID。可重复指定多个类别。默认检查全部类别。 |

加上 `--json-output` 可同时保存机器可读的报告。JSON 的 `schema_version` 为 `1`，记录失败类别 ID、指标、实测值和阈值。JSON 包含全部受影响样本，HTML 的预览数量可单独设置。

```sh
python -m yolo_regression --annotations demo/annotations.json --baseline demo/baseline.json --candidate demo/candidate-equal-recall.json --gate-class 0 --max-new-misses 0 --output reports/yolo-regression.html --json-output reports/yolo-regression.json
```

垫片类别的召回率仍为 75%，但一处新增漏检会使上面的命令失败。把本项目中的 [GitHub Actions 示例](examples/github-actions/yolo-regression.yml) 复制到 `.github/workflows/yolo-regression.yml`，即可试跑同样的检查。PR 检查从目标提交读取基准预测和标注，从检出的合并提交读取候选预测。手动运行时，两次 checkout 都使用所选版本。任务摘要会记录双方版本并列出触发规则的图片，门禁失败时仍会保留 HTML 和 JSON 报告。用于生产时，替换成自己的数据路径和阈值。 [GitHub 任务摘要](https://docs.github.com/en/actions/reference/workflows-and-actions/workflow-commands#adding-a-job-summary) · [工作流产物](https://docs.github.com/en/actions/tutorials/store-and-share-data)

| 退出码 | 含义 |
| --- | --- |
| `0` | 报告生成成功，已配置的检查通过。 |
| `1` | 至少一个受检类别的指标超过配置的阈值。 |
| `2` | 参数、输入或输出有误。 |

## 可选的本地 Ollama 诊断

默认检查新增漏检最多的类别。工具会按标注框面积分组，并检查新增漏检附近是否有置信度低于阈值的同类别预测，或者置信度达到阈值的其他类别预测。

加上 `--ollama-model YOUR_LOCAL_MODEL`，即可让本地 Ollama 模型根据已收集的统计选择下一项检查。请求发往 `127.0.0.1:11434`，只包含汇总结果和预定义的检查名称。Ollama 不可用或响应无效时，工具会使用内置规则完成剩余检查。CI 结果由配置的门禁阈值决定。 [Ollama 结构化输出文档](https://docs.ollama.com/capabilities/structured-outputs)

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

MCP 工具也接受 `max_new_misses`、`max_new_false_positives` 和 `gate_class_ids`。如果某类指标超过阈值，可从 `gate_failures` 获取具体规则和实测值。传入 `baseline_manifest` 和 `candidate_manifest` 可核对输入覆盖，通过 `min_localization_drop` 设置定位退步的报告阈值。

### 使用 Skill

仓库自带 [Claude Code Skill](.claude/skills/yolo-regression-check/SKILL.md) 和 [Codex Skill](.agents/skills/yolo-regression-check/SKILL.md)。可以在本仓库中启动客户端，也可以把对应的 `yolo-regression-check` 目录复制到自己项目的同一 Skill 位置。在 Claude Code 中调用 `/yolo-regression-check`，在 Codex 中调用 `$yolo-regression-check`。Skill 优先使用 MCP 工具，也支持直接运行 CLI。 [Claude Code Skills](https://code.claude.com/docs/en/skills) · [Codex Skills](https://developers.openai.com/codex/skills)

## 运行测试

```sh
python -m unittest discover -s tests -v
```

安装可选的 MCP 依赖后，测试会包含 `stdio` 集成测试。

## 许可证

[MIT](LICENSE)
