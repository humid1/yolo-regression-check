from dataclasses import dataclass
from math import isclose
from typing import Literal

from .evaluation import Comparison

Metric = Literal["recall_drop_pp", "new_misses", "new_false_positives"]
GateStatus = Literal["not_configured", "passed", "failed"]


@dataclass(frozen=True)
class GatePolicy:
    max_recall_drop_pp: float | None = None
    max_new_misses: int | None = None
    max_new_false_positives: int | None = None
    category_ids: tuple[int, ...] = ()

    def __post_init__(self):
        if self.max_recall_drop_pp is not None and not 0 <= self.max_recall_drop_pp <= 100:
            raise ValueError("--max-recall-drop-pp 应在 0 到 100 之间")
        if self.max_new_misses is not None and self.max_new_misses < 0:
            raise ValueError("--max-new-misses 应为非负整数")
        if self.max_new_false_positives is not None and self.max_new_false_positives < 0:
            raise ValueError("--max-new-false-positives 应为非负整数")
        if any(category_id < 0 for category_id in self.category_ids):
            raise ValueError("--gate-class 应为非负类别 ID")
        if self.category_ids and not self.limits:
            raise ValueError("--gate-class 需要至少一个门禁阈值")

    @property
    def limits(self) -> dict[Metric, float | int]:
        configured: dict[Metric, float | int] = {}
        if self.max_recall_drop_pp is not None:
            configured["recall_drop_pp"] = self.max_recall_drop_pp
        if self.max_new_misses is not None:
            configured["new_misses"] = self.max_new_misses
        if self.max_new_false_positives is not None:
            configured["new_false_positives"] = self.max_new_false_positives
        return configured


@dataclass(frozen=True)
class GateFailure:
    category_id: int
    category_name: str
    metric: Metric
    actual: int | float
    limit: int | float


@dataclass(frozen=True)
class GateResult:
    status: GateStatus
    limits: dict[Metric, float | int]
    category_ids: tuple[int, ...]
    failures: list[GateFailure]


def evaluate_gate(comparison: Comparison, policy: GatePolicy) -> GateResult:
    limits = policy.limits
    if not limits:
        return GateResult("not_configured", {}, (), [])

    category_ids = tuple(sorted(set(policy.category_ids)))
    unknown_ids = set(category_ids) - comparison.dataset.categories.keys()
    if unknown_ids:
        raise ValueError(f"门禁引用了未知类别 ID {', '.join(map(str, sorted(unknown_ids)))}")

    failures = []
    for row in comparison.classes:
        if category_ids and row.category_id not in category_ids:
            continue
        values = {"recall_drop_pp": row.recall_drop_pp, "new_misses": row.new_misses,
                  "new_false_positives": row.added_fp}
        for metric, limit in limits.items():
            actual = values[metric]
            if actual <= limit:
                continue
            if metric == "recall_drop_pp" and isclose(actual, limit, rel_tol=0, abs_tol=1e-10):
                continue
            failures.append(GateFailure(row.category_id, row.name, metric, actual, limit))
    return GateResult("failed" if failures else "passed", limits, category_ids, failures)
