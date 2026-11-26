import argparse
import sys
from pathlib import Path

from .workflow import ComparisonOptions, run_comparison


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="比较同一验证集上的两份目标检测预测并生成 HTML 报告")
    parser.add_argument("--annotations", required=True, type=Path, help="COCO 风格的标注 JSON")
    parser.add_argument("--baseline", required=True, type=Path, help="基准模型预测 JSON")
    parser.add_argument("--candidate", required=True, type=Path, help="候选模型预测 JSON")
    parser.add_argument("--output", type=Path, default=Path("report.html"), help="报告输出路径（默认 report.html）")
    parser.add_argument("--images-dir", type=Path, help="可选样本图目录，按标注 file_name 查找")
    parser.add_argument("--confidence", type=float, default=ComparisonOptions.confidence, help="最低预测置信度（默认 0.3）")
    parser.add_argument("--iou", type=float, default=ComparisonOptions.iou, help="匹配所需的最小 IoU（默认 0.5）")
    parser.add_argument("--top", type=int, default=ComparisonOptions.top, help="报告中的最多样本数（默认 20）")
    parser.add_argument("--max-recall-drop-pp", type=float, help="类别召回率允许下降的百分点数，用于 CI")
    parser.add_argument("--ollama-model", help="可选的本地 Ollama 模型名，用于选择诊断检查")
    args = parser.parse_args(argv)

    try:
        options = ComparisonOptions(confidence=args.confidence, iou=args.iou, top=args.top,
                                    max_recall_drop_pp=args.max_recall_drop_pp, ollama_model=args.ollama_model)
    except ValueError as error:
        parser.error(str(error))

    try:
        result = run_comparison(args.annotations, args.baseline, args.candidate, args.output,
                                options=options, images_dir=args.images_dir)
    except (ValueError, OSError, RuntimeError) as error:
        print(f"诊断失败 — {error}", file=sys.stderr)
        return 2

    print(f"报告已写入 {result.report_path}")
    if result.failures:
        print(f"有 {len(result.failures)} 个类别的召回降幅超过 {options.max_recall_drop_pp:g} 个百分点", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
