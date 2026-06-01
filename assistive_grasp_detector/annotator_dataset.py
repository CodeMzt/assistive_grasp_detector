"""Reader and validator for AssistiveGraspAnnotator datasets."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from PIL import Image

from assistive_grasp_detector.schema import (
    ALLOWED_DIFFICULTIES,
    ALLOWED_SPLITS,
    EXPECTED_VGA_SIZE,
    IMAGE_EXTENSIONS,
    Issue,
    ClassInfo,
    classes_by_id,
    load_classes,
    normalize_path,
)


@dataclass(frozen=True)
class ImageRecord:
    key: str
    image_path: Path
    annotation_path: Path


@dataclass
class DatasetValidationReport:
    dataset_root: str
    image_count: int = 0
    annotation_count: int = 0
    object_count: int = 0
    grasp_count: int = 0
    issues: list[Issue] = field(default_factory=list)

    @property
    def errors(self) -> list[Issue]:
        return [issue for issue in self.issues if issue.level == "error"]

    @property
    def warnings(self) -> list[Issue]:
        return [issue for issue in self.issues if issue.level == "warning"]

    @property
    def ok(self) -> bool:
        return not self.errors

    def add(self, level: str, code: str, message: str, path: str | Path = "") -> None:
        self.issues.append(Issue(level=level, code=code, message=message, path=str(path)))

    def to_dict(self) -> dict[str, Any]:
        return {
            "dataset_root": self.dataset_root,
            "ok": self.ok,
            "image_count": self.image_count,
            "annotation_count": self.annotation_count,
            "object_count": self.object_count,
            "grasp_count": self.grasp_count,
            "error_count": len(self.errors),
            "warning_count": len(self.warnings),
            "issues": [issue.to_dict() for issue in self.issues],
        }


def images_root(dataset_root: str | Path) -> Path:
    return Path(dataset_root) / "images"


def annotations_root(dataset_root: str | Path) -> Path:
    return Path(dataset_root) / "annotations"


def classes_path(dataset_root: str | Path) -> Path:
    return Path(dataset_root) / "classes.yaml"


def iter_image_records(dataset_root: str | Path) -> list[ImageRecord]:
    root = Path(dataset_root)
    img_root = images_root(root)
    records: list[ImageRecord] = []
    if not img_root.is_dir():
        return records

    for path in sorted(img_root.rglob("*")):
        if not path.is_file() or path.suffix.lower() not in IMAGE_EXTENSIONS:
            continue
        key = normalize_path(path.relative_to(img_root))
        ann_path = annotations_root(root) / Path(key).with_suffix(".json")
        records.append(ImageRecord(key=key, image_path=path, annotation_path=ann_path))
    return records


def read_annotation(path: str | Path) -> dict[str, Any]:
    with Path(path).open("r", encoding="utf-8") as file:
        data = json.load(file)
    if not isinstance(data, dict):
        raise ValueError(f"annotation root must be a JSON object: {path}")
    return data


def stable_split_for_key(key: str) -> str:
    bucket = int(hashlib.sha1(key.encode("utf-8")).hexdigest()[:8], 16) % 100
    return "train" if bucket < 80 else "val"


def split_for_annotation(annotation: dict[str, Any], key: str) -> str:
    split = annotation.get("split")
    if isinstance(split, str) and split in ALLOWED_SPLITS:
        return split
    return stable_split_for_key(key)


def validate_self_dataset(dataset_root: str | Path) -> DatasetValidationReport:
    root = Path(dataset_root)
    report = DatasetValidationReport(dataset_root=str(root))

    if not root.is_dir():
        report.add("error", "dataset_missing", "dataset root does not exist", root)
        return report

    if not images_root(root).is_dir():
        report.add("error", "images_missing", "images/ directory is required", images_root(root))

    classes: list[ClassInfo] = []
    cls_path = classes_path(root)
    if not cls_path.is_file():
        report.add("error", "classes_missing", "classes.yaml is required", cls_path)
    else:
        try:
            classes = load_classes(cls_path)
        except Exception as exc:
            report.add("error", "classes_invalid", str(exc), cls_path)
    class_map = classes_by_id(classes)

    records = iter_image_records(root)
    report.image_count = len(records)
    if not records and images_root(root).is_dir():
        report.add("error", "images_empty", "images/ contains no supported image files", images_root(root))

    seen_annotations: set[Path] = set()
    for record in records:
        _validate_image_record(record, report, class_map)
        seen_annotations.add(record.annotation_path.resolve())

    ann_root = annotations_root(root)
    if ann_root.is_dir():
        for ann_path in sorted(ann_root.rglob("*.json")):
            if ann_path.resolve() not in seen_annotations:
                report.add("warning", "orphan_annotation", "annotation has no matching image", ann_path)

    return report


def _validate_image_record(
    record: ImageRecord,
    report: DatasetValidationReport,
    class_map: dict[int, ClassInfo],
) -> None:
    try:
        with Image.open(record.image_path) as image:
            image_size = image.size
    except Exception as exc:
        report.add("error", "image_unreadable", str(exc), record.image_path)
        return

    if image_size != EXPECTED_VGA_SIZE:
        report.add(
            "warning",
            "image_not_vga",
            f"image size is {image_size[0]}x{image_size[1]}, expected 640x480 for board-domain data",
            record.image_path,
        )

    if not record.annotation_path.is_file():
        report.add("error", "annotation_missing", "annotation JSON is missing", record.annotation_path)
        return

    try:
        annotation = read_annotation(record.annotation_path)
    except Exception as exc:
        report.add("error", "annotation_invalid_json", str(exc), record.annotation_path)
        return

    report.annotation_count += 1
    _validate_annotation(annotation, record, image_size, report, class_map)


def _validate_annotation(
    annotation: dict[str, Any],
    record: ImageRecord,
    image_size: tuple[int, int],
    report: DatasetValidationReport,
    class_map: dict[int, ClassInfo],
) -> None:
    width = annotation.get("width")
    height = annotation.get("height")
    if width != image_size[0] or height != image_size[1]:
        report.add(
            "error",
            "annotation_size_mismatch",
            f"annotation width/height {width}x{height} does not match image {image_size[0]}x{image_size[1]}",
            record.annotation_path,
        )

    image_path = annotation.get("image_path")
    expected_image_path = normalize_path(Path("images") / record.key)
    if isinstance(image_path, str) and image_path and normalize_path(image_path) != expected_image_path:
        report.add(
            "warning",
            "annotation_image_path_mismatch",
            f"annotation image_path is {image_path!r}, expected {expected_image_path!r}",
            record.annotation_path,
        )

    split = annotation.get("split")
    if split is None or split == "":
        report.add("warning", "split_missing", "split missing; deterministic 80/20 fallback will be used", record.annotation_path)
    elif split not in ALLOWED_SPLITS:
        report.add("error", "split_invalid", f"split must be one of {sorted(ALLOWED_SPLITS)}, got {split!r}", record.annotation_path)

    objects = annotation.get("objects")
    if not isinstance(objects, list):
        report.add("error", "objects_invalid", "objects must be a list", record.annotation_path)
        return

    seen_instances: set[int] = set()
    for index, obj in enumerate(objects):
        if not isinstance(obj, dict):
            report.add("error", "object_invalid", f"object #{index} must be a JSON object", record.annotation_path)
            continue
        _validate_object(obj, record, image_size, report, class_map, seen_instances)


def _validate_object(
    obj: dict[str, Any],
    record: ImageRecord,
    image_size: tuple[int, int],
    report: DatasetValidationReport,
    class_map: dict[int, ClassInfo],
    seen_instances: set[int],
) -> None:
    report.object_count += 1
    prefix = f"object {obj.get('instance_id', '?')}"

    instance_id = obj.get("instance_id")
    if not isinstance(instance_id, int):
        report.add("error", "instance_id_invalid", f"{prefix}: instance_id must be an integer", record.annotation_path)
    elif instance_id in seen_instances:
        report.add("error", "instance_id_duplicate", f"{prefix}: duplicate instance_id", record.annotation_path)
    else:
        seen_instances.add(instance_id)

    class_id = obj.get("class_id")
    if not isinstance(class_id, int):
        report.add("error", "class_id_invalid", f"{prefix}: class_id must be an integer", record.annotation_path)
    elif class_id not in class_map:
        report.add("error", "class_id_unknown", f"{prefix}: class_id {class_id} is not in classes.yaml", record.annotation_path)
    else:
        expected_name = class_map[class_id].name
        class_name = obj.get("class_name")
        if isinstance(class_name, str) and class_name and class_name != expected_name:
            report.add(
                "warning",
                "class_name_mismatch",
                f"{prefix}: class_name {class_name!r} does not match classes.yaml {expected_name!r}",
                record.annotation_path,
            )

    bbox = obj.get("bbox_xyxy")
    if not _valid_bbox(bbox, image_size, report, record.annotation_path, prefix):
        return

    grasps = obj.get("grasps", [])
    if not isinstance(grasps, list):
        report.add("error", "grasps_invalid", f"{prefix}: grasps must be a list", record.annotation_path)
        return
    if obj.get("graspable") is False and grasps:
        report.add("warning", "non_graspable_has_grasps", f"{prefix}: non-graspable object has grasps", record.annotation_path)

    for grasp_index, grasp in enumerate(grasps):
        if not isinstance(grasp, dict):
            report.add("error", "grasp_invalid", f"{prefix}: grasp #{grasp_index} must be a JSON object", record.annotation_path)
            continue
        _validate_grasp(grasp, record, image_size, report, prefix)


def _valid_bbox(
    bbox: Any,
    image_size: tuple[int, int],
    report: DatasetValidationReport,
    path: Path,
    prefix: str,
) -> bool:
    if not isinstance(bbox, list) or len(bbox) != 4:
        report.add("error", "bbox_invalid", f"{prefix}: bbox_xyxy must have 4 values", path)
        return False
    try:
        x1, y1, x2, y2 = [float(value) for value in bbox]
    except (TypeError, ValueError):
        report.add("error", "bbox_invalid", f"{prefix}: bbox_xyxy values must be numeric", path)
        return False
    width, height = image_size
    if x1 >= x2 or y1 >= y2:
        report.add("error", "bbox_invalid", f"{prefix}: bbox must satisfy x1 < x2 and y1 < y2", path)
        return False
    if x1 < 0 or y1 < 0 or x2 > width or y2 > height:
        report.add("error", "bbox_out_of_bounds", f"{prefix}: bbox exceeds image bounds {width}x{height}", path)
        return False
    return True


def _validate_grasp(
    grasp: dict[str, Any],
    record: ImageRecord,
    image_size: tuple[int, int],
    report: DatasetValidationReport,
    object_prefix: str,
) -> None:
    report.grasp_count += 1
    prefix = f"{object_prefix} grasp {grasp.get('grasp_id', '?')}"
    points = grasp.get("points")
    if not isinstance(points, list) or len(points) != 4:
        report.add("error", "grasp_points_invalid", f"{prefix}: points must contain 4 points", record.annotation_path)
        return

    width, height = image_size
    for index, point in enumerate(points):
        if not isinstance(point, list) or len(point) != 2:
            report.add("error", "grasp_point_invalid", f"{prefix}: point {index} must have 2 values", record.annotation_path)
            continue
        try:
            x, y = float(point[0]), float(point[1])
        except (TypeError, ValueError):
            report.add("error", "grasp_point_invalid", f"{prefix}: point {index} values must be numeric", record.annotation_path)
            continue
        if not (0 <= x <= width and 0 <= y <= height):
            report.add("error", "grasp_point_out_of_bounds", f"{prefix}: point {index} exceeds image bounds", record.annotation_path)

    difficulty = grasp.get("difficulty", "easy")
    if difficulty not in ALLOWED_DIFFICULTIES:
        report.add("error", "difficulty_invalid", f"{prefix}: invalid difficulty {difficulty!r}", record.annotation_path)

    quality = grasp.get("quality", 1.0)
    try:
        quality_value = float(quality)
    except (TypeError, ValueError):
        report.add("error", "quality_invalid", f"{prefix}: quality must be numeric", record.annotation_path)
        return
    if not (0.0 <= quality_value <= 1.0):
        report.add("error", "quality_invalid", f"{prefix}: quality must be in [0, 1]", record.annotation_path)
