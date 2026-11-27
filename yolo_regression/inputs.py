import json
import math
from dataclasses import dataclass
from hashlib import sha256
from pathlib import Path
from typing import Any

Id = int | str
Box = tuple[float, float, float, float]


@dataclass(frozen=True)
class Image:
    id: Id
    file_name: str
    width: int
    height: int


@dataclass(frozen=True)
class Label:
    index: int
    image_id: Id
    category_id: int
    box: Box
    crowd: bool


@dataclass(frozen=True)
class Detection:
    image_id: Id
    category_id: int
    box: Box
    score: float


@dataclass(frozen=True)
class Dataset:
    images: dict[Id, Image]
    categories: dict[int, str]
    labels: list[Label]


@dataclass(frozen=True)
class JsonDocument:
    path: Path
    value: Any
    sha256: str


def read_json_document(path: Path) -> JsonDocument:
    try:
        content = path.read_bytes()
        return JsonDocument(path, json.loads(content.decode("utf-8-sig")), sha256(content).hexdigest())
    except (ValueError, OSError) as error:
        raise ValueError(f"{path} — {error}") from error


def _rows(value, context):
    if not isinstance(value, list):
        raise ValueError(f"{context} 应为 JSON 数组")
    for number, row in enumerate(value, 1):
        location = f"{context} 第 {number} 条"
        if not isinstance(row, dict):
            raise ValueError(f"{location}应为 JSON 对象")
        yield row, location


def _integer(value, field, minimum=0):
    if type(value) is not int or value < minimum:
        raise ValueError(f"{field} 应为大于或等于 {minimum} 的整数")
    return value


def _nonempty_string(value, field):
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{field} 应为非空字符串")
    return value


def _image_id(value, field):
    if type(value) is int and value >= 0:
        return value
    if isinstance(value, str) and value.strip():
        return value
    raise ValueError(f"{field} 应为非负整数或非空字符串")


def _number(value, field):
    if type(value) not in (float, int) or abs(value) > 1e100 or not math.isfinite(value):
        raise ValueError(f"{field} 应为有效的有限数值")
    return float(value)


def _box(value, field) -> Box:
    if not isinstance(value, list) or len(value) != 4:
        raise ValueError(f"{field} 应为 [x, y, width, height]")
    x, y, width, height = (_number(item, field) for item in value)
    if width <= 0 or height <= 0 or width * height == 0:
        raise ValueError(f"{field} 的宽高和面积应大于零")
    return x, y, width, height


def _references(row, context, dataset):
    image_id = _image_id(row.get("image_id"), f"{context} image_id")
    category_id = _integer(row.get("category_id"), f"{context} category_id")
    if image_id not in dataset.images:
        raise ValueError(f"{context} 引用了未知图片 ID {image_id!r}")
    if category_id not in dataset.categories:
        raise ValueError(f"{context} 引用了未知类别 ID {category_id}")
    return image_id, category_id


def load_dataset(path: Path | JsonDocument) -> Dataset:
    document = read_json_document(path) if isinstance(path, Path) else path
    path, raw = document.path, document.value
    if not isinstance(raw, dict):
        raise ValueError(f"{path} 应为包含 images、categories、annotations 的 JSON 对象")

    images = {}
    for row, context in _rows(raw.get("images"), f"{path} images"):
        image_id = _image_id(row.get("id"), f"{context} id")
        if image_id in images:
            raise ValueError(f"{context} 包含重复图片 ID {image_id!r}")
        width = _integer(row.get("width"), f"{context} width", 1)
        height = _integer(row.get("height"), f"{context} height", 1)
        if max(width, height) > 1_000_000:
            raise ValueError(f"{context} 的图片宽高应小于或等于 1000000")
        file_name = _nonempty_string(row.get("file_name"), f"{context} file_name")
        images[image_id] = Image(image_id, file_name, width, height)

    categories = {}
    for row, context in _rows(raw.get("categories"), f"{path} categories"):
        category_id = _integer(row.get("id"), f"{context} id")
        if category_id in categories:
            raise ValueError(f"{context} 包含重复类别 ID {category_id}")
        categories[category_id] = _nonempty_string(row.get("name"), f"{context} name")

    dataset = Dataset(images, categories, [])
    for index, (row, context) in enumerate(_rows(raw.get("annotations"), f"{path} annotations")):
        image_id, category_id = _references(row, context, dataset)
        crowd = row.get("iscrowd", 0)
        if type(crowd) not in (int, bool) or crowd not in (0, 1):
            raise ValueError(f"{context} iscrowd 应为 0 或 1")
        dataset.labels.append(Label(index, image_id, category_id, _box(row.get("bbox"), context), bool(crowd)))

    if not images or not categories or not any(not label.crowd for label in dataset.labels):
        raise ValueError(f"{path} 需要至少一张图片、一个类别和一个非 crowd 标注框")
    return dataset


def load_predictions(path: Path | JsonDocument, dataset: Dataset) -> list[Detection]:
    document = read_json_document(path) if isinstance(path, Path) else path
    predictions = []
    for row, context in _rows(document.value, str(document.path)):
        image_id, category_id = _references(row, context, dataset)
        score = _number(row.get("score"), f"{context} score")
        if not 0 <= score <= 1:
            raise ValueError(f"{context} score 应在 0 到 1 之间")
        predictions.append(Detection(image_id, category_id, _box(row.get("bbox"), context), score))
    return predictions
