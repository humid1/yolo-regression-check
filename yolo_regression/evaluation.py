from collections import Counter, defaultdict, deque
from dataclasses import dataclass

from .inputs import Box, Dataset, Detection, Id, Label


@dataclass
class Evaluation:
    matches: dict[int, Detection]
    false_positives: dict[Id, list[Detection]]


@dataclass(frozen=True)
class ClassChange:
    category_id: int
    name: str
    total: int
    baseline_tp: int
    candidate_tp: int
    baseline_fp: int
    candidate_fp: int
    new_misses: int
    recovered: int
    added_fp: int
    resolved_fp: int

    @property
    def recall_drop_pp(self) -> float:
        if not self.total:
            return 0.0
        return 100 * (self.baseline_tp - self.candidate_tp) / self.total

    @property
    def persistent_fp(self) -> int:
        return self.baseline_fp - self.resolved_fp


@dataclass(frozen=True)
class FalsePositiveChanges:
    added: tuple[Detection, ...] = ()
    resolved: tuple[Detection, ...] = ()
    persistent: tuple[tuple[Detection, Detection], ...] = ()


@dataclass(frozen=True)
class ImageChange:
    image_id: Id
    new_misses: list[Label]
    false_positives: FalsePositiveChanges

    @property
    def added_fp(self) -> int:
        return len(self.false_positives.added)


@dataclass
class Comparison:
    dataset: Dataset
    baseline: Evaluation
    candidate: Evaluation
    classes: list[ClassChange]
    images: list[ImageChange]
    labels_by_image: dict[Id, list[Label]]
    candidate_by_image: dict[Id, list[Detection]]
    confidence: float
    iou: float


def overlap(prediction: Box, label: Box, crowd: bool = False) -> float:
    px, py, pw, ph = prediction
    lx, ly, lw, lh = label
    width = max(0.0, min(px + pw, lx + lw) - max(px, lx))
    height = max(0.0, min(py + ph, ly + lh) - max(py, ly))
    # Cap intersection at the smaller area to absorb arithmetic rounding.
    intersection = min(width * height, pw * ph, lw * lh)
    union = pw * ph if crowd else pw * ph + lw * lh - intersection
    return intersection / union


def _evaluate(labels_by_group, predictions, confidence, iou) -> Evaluation:
    matches = {}
    false_positives = defaultdict(list)
    for prediction in sorted(predictions, key=lambda item: item.score, reverse=True):
        if prediction.score < confidence:
            break
        labels = labels_by_group.get((prediction.image_id, prediction.category_id), ())
        best_label = None
        best_overlap = 0.0
        for label in labels:
            if label.crowd or label.index in matches:
                continue
            similarity = overlap(prediction.box, label.box)
            if similarity > best_overlap:
                best_overlap = similarity
                best_label = label
        if best_label is not None and best_overlap >= iou:
            matches[best_label.index] = prediction
        elif not any(label.crowd and overlap(prediction.box, label.box, crowd=True) >= iou for label in labels):
            false_positives[prediction.image_id].append(prediction)
    return Evaluation(matches, dict(false_positives))


def _pair_false_positives(baseline: list[Detection], candidate: list[Detection],
                          iou: float) -> FalsePositiveChanges:
    if not baseline or not candidate:
        return FalsePositiveChanges(added=tuple(candidate), resolved=tuple(baseline))

    baseline_by_class = defaultdict(list)
    for index, box in enumerate(baseline):
        baseline_by_class[box.category_id].append(index)

    choices = []
    for box in candidate:
        overlaps = [(overlap(box.box, baseline[index].box), index)
                    for index in baseline_by_class[box.category_id]]
        choices.append([index for score, index in sorted(overlaps, key=lambda item: -item[0]) if score >= iou])

    baseline_matches = {}
    candidate_matches = {}
    for start in range(len(candidate)):
        # Search for a free baseline box, reassigning existing pairs along the path when necessary.
        queue = deque([start])
        previous = {}
        while queue:
            candidate_index = queue.popleft()
            for baseline_index in choices[candidate_index]:
                if baseline_index in previous:
                    continue
                previous[baseline_index] = candidate_index
                if baseline_index in baseline_matches:
                    queue.append(baseline_matches[baseline_index])
                    continue
                while baseline_index is not None:
                    candidate_index = previous[baseline_index]
                    baseline_matches[baseline_index] = candidate_index
                    old_index = candidate_matches.get(candidate_index)
                    candidate_matches[candidate_index] = baseline_index
                    baseline_index = old_index
                queue.clear()
                break

    return FalsePositiveChanges(
        added=tuple(box for index, box in enumerate(candidate) if index not in candidate_matches),
        resolved=tuple(box for index, box in enumerate(baseline) if index not in baseline_matches),
        persistent=tuple((baseline[candidate_matches[index]], box)
                         for index, box in enumerate(candidate) if index in candidate_matches),
    )


def compare(dataset: Dataset, baseline: list[Detection], candidate: list[Detection],
            confidence: float, iou: float) -> Comparison:
    labels_by_group = defaultdict(list)
    labels_by_image = defaultdict(list)
    for label in dataset.labels:
        labels_by_group[label.image_id, label.category_id].append(label)
        if not label.crowd:
            labels_by_image[label.image_id].append(label)

    before = _evaluate(labels_by_group, baseline, confidence, iou)
    after = _evaluate(labels_by_group, candidate, confidence, iou)
    new_miss_ids = before.matches.keys() - after.matches.keys()
    recovered_ids = after.matches.keys() - before.matches.keys()
    total = Counter(label.category_id for label in dataset.labels if not label.crowd)
    before_tp = Counter(item.category_id for item in before.matches.values())
    after_tp = Counter(item.category_id for item in after.matches.values())
    before_fp = Counter(item.category_id for group in before.false_positives.values() for item in group)
    after_fp = Counter(item.category_id for group in after.false_positives.values() for item in group)
    missed = Counter(dataset.labels[index].category_id for index in new_miss_ids)
    recovered = Counter(dataset.labels[index].category_id for index in recovered_ids)

    images = []
    added = Counter()
    resolved = Counter()
    for image_id in dataset.images:
        image_misses = [label for label in labels_by_image.get(image_id, ()) if label.index in new_miss_ids]
        fp_changes = _pair_false_positives(before.false_positives.get(image_id, []),
                                           after.false_positives.get(image_id, []), iou)
        added.update(box.category_id for box in fp_changes.added)
        resolved.update(box.category_id for box in fp_changes.resolved)
        if image_misses or fp_changes.added or fp_changes.resolved:
            images.append(ImageChange(image_id, image_misses, fp_changes))
    images.sort(key=lambda item: (-len(item.new_misses), -item.added_fp, -len(item.false_positives.resolved)))

    classes = [
        ClassChange(category_id, name, total[category_id], before_tp[category_id], after_tp[category_id],
                    before_fp[category_id], after_fp[category_id], missed[category_id], recovered[category_id],
                    added[category_id], resolved[category_id])
        for category_id, name in dataset.categories.items()
        if total[category_id] or before_fp[category_id] or after_fp[category_id]
    ]
    classes.sort(key=lambda item: (-item.recall_drop_pp, -item.new_misses, -item.added_fp, item.category_id))

    candidate_by_image = defaultdict(list)
    for detection in candidate:
        candidate_by_image[detection.image_id].append(detection)
    return Comparison(dataset, before, after, classes, images, dict(labels_by_image),
                      dict(candidate_by_image), confidence, iou)
