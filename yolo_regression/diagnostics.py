import json
from dataclasses import dataclass
from http.client import HTTPException
from urllib.request import ProxyHandler, Request, build_opener

from .evaluation import Comparison, overlap


@dataclass(frozen=True)
class Finding:
    title: str
    columns: tuple[str, ...]
    rows: list[tuple[str | int, ...]]
    next_step: str


@dataclass(frozen=True)
class Diagnosis:
    message: str
    findings: list[Finding]


def _by_size(comparison, labels, missed):
    rows = []
    for name, lower, upper in (("小", 0, 32**2), ("中", 32**2, 96**2), ("大", 96**2, float("inf"))):
        group = [label for label in labels if lower <= label.box[2] * label.box[3] < upper]
        rows.append((name, len(group), sum(label.index in comparison.baseline.matches for label in group),
                     sum(label.index in comparison.candidate.matches for label in group),
                     sum(label.index in missed for label in group)))
    return Finding("按标注框面积分组", ("尺寸", "标注数", "基准命中", "候选命中", "新增漏检"), rows,
                   "优先复查新增漏检比例高、且有足够标注的尺寸组。")


def _overlapping_predictions(comparison, labels, missed):
    lower_score = wrong_class = 0
    for label in labels:
        if label.index not in missed:
            continue
        nearby = [prediction for prediction in comparison.candidate_by_image.get(label.image_id, ())
                  if overlap(prediction.box, label.box) >= comparison.iou]
        lower_score += any(prediction.category_id == label.category_id and prediction.score < comparison.confidence
                           for prediction in nearby)
        wrong_class += any(prediction.category_id != label.category_id and prediction.score >= comparison.confidence
                           for prediction in nearby)
    return Finding("新增漏检附近的候选框", ("信号", "涉及标注数", "新增漏检总数"), [
        ("同类别框低于置信度阈值", lower_score, len(missed)),
        ("其他类别框达到置信度阈值", wrong_class, len(missed)),
    ], "逐图复核这些线索。一处漏检可能同时计入两项，也可能没有可用线索。")


CHECKS = {"by_size": _by_size, "overlap": _overlapping_predictions}


def _choose(model, context, remaining, results):
    choices = list(remaining)
    if results:
        choices.append("finish")
    schema = {"type": "object", "properties": {"choice": {"type": "string", "enum": choices}},
              "required": ["choice"], "additionalProperties": False}
    request = Request("http://127.0.0.1:11434/api/chat", data=json.dumps({
        "model": model,
        "stream": False,
        "format": schema,
        "messages": [
            {"role": "system", "content": "Choose the next object detection regression check. by_size groups ground "
             "truth boxes by size. overlap checks weak same-class and confident other-class predictions near new "
             "misses. Choose finish only if the existing evidence is sufficient. Return only the requested JSON."},
            {"role": "user", "content": json.dumps({"regression": context, "allowed": choices, "results": results})},
        ],
    }).encode("utf-8"), headers={"Content-Type": "application/json"})
    with build_opener(ProxyHandler({})).open(request, timeout=20) as response:
        result = json.loads(response.read(65_537))
    choice = json.loads(result["message"]["content"])["choice"]
    if choice not in choices:
        raise ValueError("Ollama 返回了不受支持的检查")
    return choice


def diagnose(comparison: Comparison, model: str | None) -> Diagnosis:
    focus = max(comparison.classes, key=lambda row: row.new_misses)
    if not focus.new_misses:
        return Diagnosis("本次没有新增漏检。", [])

    labels = [label for label in comparison.dataset.labels if not label.crowd and label.category_id == focus.category_id]
    missed = {label.index for label in labels if label.index in comparison.baseline.matches
              and label.index not in comparison.candidate.matches}
    message = f"优先检查新增漏检最多的类别 {focus.name}，共 {len(missed)} 个。"
    remaining = dict(CHECKS)
    findings = []
    if model:
        context = {"ground_truth": focus.total, "new_misses": focus.new_misses, "recall_drop_pp": focus.recall_drop_pp}
        results = {}
        try:
            while remaining:
                choice = _choose(model, context, remaining, results)
                if choice == "finish":
                    break
                finding = remaining.pop(choice)(comparison, labels, missed)
                findings.append(finding)
                # The model sees counts and fixed labels from the checks.
                results[choice] = finding.rows
            return Diagnosis(message + " 本地 Ollama 选择了检查步骤。", findings)
        except (ValueError, OSError, KeyError, TypeError, HTTPException):
            message += " 本地 Ollama 未能完成选择，已用固定规则完成剩余检查。"
    findings.extend(check(comparison, labels, missed) for check in remaining.values())
    return Diagnosis(message, findings)
