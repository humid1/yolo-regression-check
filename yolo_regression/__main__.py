import argparse
import math
import sys
from pathlib import Path
from tempfile import NamedTemporaryFile

from .diagnostics import diagnose
from .evaluation import compare
from .inputs import load_dataset, load_predictions
from .report import render_report


def _write_report(path: Path, html: str):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = None
    try:
        with NamedTemporaryFile(mode="w", encoding="utf-8", dir=path.parent, suffix=".tmp", delete=False) as handle:
            temporary = Path(handle.name)
            handle.write(html)
        temporary.replace(path)
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="比较同一验证集上的两份目标检测预测并生成 HTML 报告")
    parser.add_argument("--annotations", required=True, type=Path, help="COCO 风格的标注 JSON")
    parser.add_argument("--baseline", required=True, type=Path, help="基准模型预测 JSON")
    parser.add_argument("--candidate", required=True, type=Path, help="候选模型预测 JSON")
    parser.add_argument("--output", type=Path, default=Path("report.html"), help="报告输出路径（默认 report.html）")
    parser.add_argument("--images-dir", type=Path, help="可选样本图目录，按标注 file_name 查找")
    parser.add_argument("--confidence", type=float, default=0.3, help="最低预测置信度（默认 0.3）")
    parser.add_argument("--iou", type=float, default=0.5, help="匹配所需的最小 IoU（默认 0.5）")
    parser.add_argument("--top", type=int, default=20, help="报告中的最多样本数（默认 20）")
    parser.add_argument("--max-recall-drop-pp", type=float, help="类别召回率允许下降的百分点数，用于 CI")
    parser.add_argument("--ollama-model", help="可选的本地 Ollama 模型名，用于选择诊断检查")
    args = parser.parse_args(argv)

    if not 0 <= args.confidence <= 1:
        parser.error("--confidence 应在 0 到 1 之间")
    if not 0 < args.iou <= 1:
        parser.error("--iou 应大于 0 且小于或等于 1")
    if args.top < 0:
        parser.error("--top 应为非负整数")
    limit = args.max_recall_drop_pp
    if limit is not None and not 0 <= limit <= 100:
        parser.error("--max-recall-drop-pp 应在 0 到 100 之间")
    if args.ollama_model is not None and not args.ollama_model.strip():
        parser.error("--ollama-model 需要非空模型名")

    try:
        output = args.output.resolve()
        if output in (args.annotations.resolve(), args.baseline.resolve(), args.candidate.resolve()):
            raise ValueError("报告输出路径与输入文件相同")
        images_dir = args.images_dir.resolve() if args.images_dir is not None else None
        if images_dir is not None and not images_dir.is_dir():
            raise ValueError(f"{images_dir} 应为样本图片目录")
        dataset = load_dataset(args.annotations)
        baseline = load_predictions(args.baseline, dataset)
        candidate = load_predictions(args.candidate, dataset)
        comparison = compare(dataset, baseline, candidate, args.confidence, args.iou)
        diagnosis = diagnose(comparison, args.ollama_model)
        failures = []
        if limit is not None:
            failures = [row for row in comparison.classes if row.recall_drop_pp > limit
                        and not math.isclose(row.recall_drop_pp, limit, rel_tol=0, abs_tol=1e-10)]
        _write_report(output, render_report(comparison, diagnosis, images_dir, args.top, limit, failures))
    except (ValueError, OSError, RuntimeError) as error:
        print(f"诊断失败 — {error}", file=sys.stderr)
        return 2

    print(f"报告已写入 {output}")
    if failures:
        print(f"有 {len(failures)} 个类别的召回降幅超过 {limit:g} 个百分点", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
