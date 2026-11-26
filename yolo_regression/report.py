from base64 import b64encode
from html import escape
from pathlib import Path

from .diagnostics import Diagnosis
from .evaluation import ClassChange, Comparison, ImageChange, overlap
from .inputs import Box, Image

STYLE = """
:root { color-scheme: light; font: 16px/1.55 system-ui, sans-serif; color: #17242e; background: #f2f5f5; }
* { box-sizing: border-box; }
body { margin: 0; }
main { width: min(1120px, 94%); margin: 2.5rem auto 4rem; }
header, section { padding: clamp(1rem, 3vw, 2rem); background: #fff; border: 1px solid #d6e1e0; border-radius: 12px; margin-bottom: 1rem; }
header { border-top: 5px solid #087e72; }
h1, h2, h3 { line-height: 1.2; overflow-wrap: anywhere; }
h1 { font-size: clamp(1.8rem, 4vw, 2.8rem); margin: 0 0 .5rem; }
h2 { font-size: 1.4rem; margin: 0 0 .85rem; }
h3 { font-size: 1.05rem; margin: .65rem 0; }
p { margin: .5rem 0; }
.muted { color: #51636b; }
.summary { display: flex; flex-wrap: wrap; gap: 1.5rem; margin-top: 1.2rem; }
.summary strong { font-size: 1.7rem; display: block; font-variant-numeric: tabular-nums; }
.table-scroll { overflow-x: auto; }
table { border-collapse: collapse; width: 100%; }
th, td { border-bottom: 1px solid #e1e9e9; text-align: left; padding: .6rem; }
thead { background: #edf4f3; }
td { font-variant-numeric: tabular-nums; }
.bad { color: #a52d42; font-weight: 700; }
.sample { border-top: 1px solid #d6e1e0; padding: 1rem 0; }
.sample:first-of-type { margin-top: 1rem; }
.panels { display: grid; grid-template-columns: repeat(auto-fit, minmax(min(230px, 100%), 1fr)); gap: .85rem; }
figure { margin: 0; min-width: 0; }
figcaption { font-weight: 600; margin: .25rem 0; }
.plane { position: relative; overflow: hidden; border-radius: 6px; background: #21333e; }
.plane::before { content: ''; position: absolute; inset: 0; background: repeating-linear-gradient(0deg, #ffffff15 0 1px, transparent 1px 20%), repeating-linear-gradient(90deg, #ffffff15 0 1px, transparent 1px 20%); }
.plane img { position: absolute; inset: 0; width: 100%; height: 100%; }
.box { position: absolute; border: 2px solid currentColor; }
.label { color: #edf4f3; }
.missed { color: #f6a4da; border-style: dashed; }
.match { color: #5de0c9; }
.fp { color: #ff8f7f; border-style: dashed; }
.weak { color: #ffd574; border-style: dotted; }
.legend { display: flex; flex-wrap: wrap; gap: .4rem 1rem; }
.legend span::before { content: ''; display: inline-block; width: .65rem; height: .65rem; margin-right: .3rem; border: 2px solid currentColor; vertical-align: -.08em; border-radius: 2px; background: #21333e; }
.legend .label { color: #3b5765; } .legend .missed { color: #a43080; }
.legend .match { color: #087e72; } .legend .fp { color: #a52d42; } .legend .weak { color: #8d5e0d; }
"""


def _table(headers, rows):
    head = "".join(f'<th scope="col">{escape(str(value))}</th>' for value in headers)
    body = "".join("<tr>" + "".join(f"<td>{escape(str(value))}</td>" for value in row) + "</tr>" for row in rows)
    return f'<div class="table-scroll"><table><thead><tr>{head}</tr></thead><tbody>{body}</tbody></table></div>'


def _rate(count, total):
    return f"{count / total:.1%}" if total else "—"


def _class_row(change: ClassChange):
    before_recall = _rate(change.baseline_tp, change.total)
    after_recall = _rate(change.candidate_tp, change.total)
    before_precision = _rate(change.baseline_tp, change.baseline_tp + change.baseline_fp)
    after_precision = _rate(change.candidate_tp, change.candidate_tp + change.candidate_fp)
    delta = f"{-change.recall_drop_pp:+.1f}" if change.recall_drop_pp else "0.0"
    return (change.name, change.total, f"{before_recall} → {after_recall}", delta if change.total else "—",
            f"{before_precision} → {after_precision}", f"{change.baseline_fp} → {change.candidate_fp}",
            change.new_misses, change.recovered)


def _image_url(root: Path | None, file_name: str) -> str:
    if root is None:
        return ""
    try:
        relative_path = Path(file_name)
        if relative_path.anchor or ".." in relative_path.parts:
            return ""
        image_path = (root / relative_path).resolve()
        if not image_path.is_relative_to(root) or not image_path.is_file():
            return ""
        with image_path.open("rb") as image_file:
            content = image_file.read(1_000_001)
    except (OSError, RuntimeError, ValueError):
        return ""
    if len(content) > 1_000_000:
        return ""
    if content.startswith(b"\x89PNG\r\n\x1a\n"):
        media_type = "png"
    elif content.startswith(b"\xff\xd8\xff"):
        media_type = "jpeg"
    elif content.startswith(b"RIFF") and content[8:12] == b"WEBP":
        media_type = "webp"
    else:
        return ""
    return f"data:image/{media_type};base64,{b64encode(content).decode('ascii')}"


def _box(image: Image, box: Box, style: str, title: str):
    x, y, width, height = box
    left, right = max(x, 0), min(x + width, image.width)
    top, bottom = max(y, 0), min(y + height, image.height)
    if left >= right or top >= bottom:
        return ""
    position = (f"left:{left / image.width:.3%};top:{top / image.height:.3%};"
                f"width:{(right - left) / image.width:.3%};height:{(bottom - top) / image.height:.3%}")
    return f'<span class="box {style}" style="{position}" title="{escape(title)}"></span>'


def _sample(comparison: Comparison, change: ImageChange, images_dir: Path | None):
    dataset = comparison.dataset
    image = dataset.images[change.image_id]
    labels = comparison.labels_by_image.get(image.id, ())
    missed = {label.index for label in change.new_misses}
    file_name = escape(image.file_name)
    source = _image_url(images_dir, image.file_name)
    background = f'<img alt="" src="{source}">' if source else ""
    label_boxes = [_box(image, label.box, "missed" if label.index in missed else "label",
                        dataset.categories[label.category_id]) for label in labels]
    panels = [("标注", label_boxes)]
    for caption, evaluation in (("基准预测", comparison.baseline), ("候选预测", comparison.candidate)):
        matches = [evaluation.matches[label.index] for label in labels if label.index in evaluation.matches]
        false_positives = evaluation.false_positives.get(image.id, ())
        boxes = [
            _box(image, prediction.box, style, f"{dataset.categories[prediction.category_id]} {prediction.score:.2f}")
            for style, predictions in (("match", matches), ("fp", false_positives))
            for prediction in predictions
        ]
        if evaluation is comparison.candidate:
            boxes.extend(
                _box(image, prediction.box, "weak", f"{dataset.categories[prediction.category_id]} {prediction.score:.2f}")
                for prediction in comparison.candidate_by_image.get(image.id, ())
                if prediction.score < comparison.confidence and any(
                    prediction.category_id == label.category_id and overlap(prediction.box, label.box) >= comparison.iou
                    for label in change.new_misses
                )
            )
        panels.append((caption, boxes))
    figures = "".join(
        f'<figure><figcaption>{caption}</figcaption><div class="plane" '
        f'style="aspect-ratio:{image.width} / {image.height}">{background}{"".join(boxes)}</div></figure>'
        for caption, boxes in panels
    )
    return (f'<article class="sample"><h3>{file_name}</h3><p class="muted">图片 ID {escape(str(image.id))} · '
            f'新增漏检 {len(change.new_misses)} · 误检增加 {change.added_fp}</p><div class="panels">{figures}</div></article>')


def render_report(comparison: Comparison, diagnosis: Diagnosis, images_dir: Path | None, top: int,
                  max_recall_drop_pp: float | None, failures: list[ClassChange]) -> str:
    if failures:
        failed_labels = "、".join(escape(item.name) for item in failures)
        gate = f'<p class="bad">未通过。召回降幅超过 {max_recall_drop_pp:g} 个百分点的类别为 {failed_labels}。</p>'
    elif max_recall_drop_pp is not None:
        gate = f'<p>通过。各类别召回降幅均在 {max_recall_drop_pp:g} 个百分点以内。</p>'
    else:
        gate = ""
    classes = _table(("类别", "标注数", "召回率", "召回变化（百分点）", "精确率", "误检数", "新增漏检", "恢复检出"),
                     (_class_row(change) for change in comparison.classes))
    findings = "".join(f"<h3>{escape(item.title)}</h3>{_table(item.columns, item.rows)}<p>{escape(item.next_step)}</p>"
                       for item in diagnosis.findings)
    shown = comparison.images[:top]
    samples = "".join(_sample(comparison, image, images_dir) for image in shown)
    sample_summary = f"展示 {len(shown)} / {len(comparison.images)} 张受影响图片。"
    misses = sum(change.new_misses for change in comparison.classes)
    added_fp = sum(change.added_fp for change in comparison.images)
    return f"""<!doctype html>
<html lang="zh-CN"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1">
<meta http-equiv="Content-Security-Policy" content="default-src 'none'; style-src 'unsafe-inline'; img-src data:">
<title>YOLO 模型回归诊断</title><style>{STYLE}</style></head><body><main>
<header><h1>YOLO 模型回归诊断</h1>
<p class="muted">基准模型与候选模型使用同一份标注 · 置信度 ≥ {comparison.confidence:g} · IoU ≥ {comparison.iou:g}</p>
{gate}<div class="summary"><div><strong>{len(comparison.dataset.images)}</strong>验证集图片</div>
<div><strong>{misses}</strong>新增漏检</div><div><strong>{added_fp}</strong>增加的误检</div></div></header>
<section><h2>类别变化</h2>{classes}
<p class="muted">指标顺序为基准 → 候选。新增漏检按标注实例计算。增加的误检按图片和类别的计数增量汇总。</p></section>
<section><h2>漏检诊断</h2><p>{escape(diagnosis.message)}</p>{findings}</section>
<section><h2>样本复查</h2><p>{sample_summary}</p>
<div class="legend"><span class="label">标注</span><span class="missed">新增漏检</span>
<span class="match">正确匹配</span><span class="fp">误检</span><span class="weak">低分同类别预测</span></div>
{samples}</section></main></body></html>
"""
