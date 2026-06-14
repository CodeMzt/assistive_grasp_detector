"""EthosSafeDet-A v1 manifest conversion and calibration sampling."""

from __future__ import annotations

import json
import random
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

from PIL import Image

from assistive_grasp_detector.annotator_dataset import (
    iter_image_records,
    read_annotation,
    split_for_annotation,
    stable_split_for_key,
)
from assistive_grasp_detector.schema import (
    ETHOSSAFEDET_CLASS_NAMES,
    ETHOSSAFEDET_MAX_CALIBRATION_IMAGES,
    ETHOSSAFEDET_MIN_CALIBRATION_IMAGES,
    ETHOSSAFEDET_SCHEMA_VERSION,
    EXPECTED_VGA_SIZE,
    IMAGE_EXTENSIONS,
    Issue,
    classes_by_id,
    load_classes,
    validate_ethossafedet_classes,
)


@dataclass
class ManifestBuildResult:
    output_path: str
    record_count: int = 0
    object_count: int = 0
    negative_count: int = 0
    excluded_count: int = 0
    split_counts: dict[str, int] = field(default_factory=dict)
    class_counts: dict[int, int] = field(default_factory=dict)
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
            "output_path": self.output_path,
            "ok": self.ok,
            "record_count": self.record_count,
            "object_count": self.object_count,
            "negative_count": self.negative_count,
            "excluded_count": self.excluded_count,
            "split_counts": self.split_counts,
            "class_counts": {str(k): v for k, v in sorted(self.class_counts.items())},
            "error_count": len(self.errors),
            "warning_count": len(self.warnings),
            "issues": [issue.to_dict() for issue in self.issues],
        }


def prepare_ethossafedet_manifest_from_export(
    dataset_root: str | Path,
    output_path: str | Path,
    negative_image_ids: list[str] | tuple[str, ...] | None = None,
    image_subdir: str = "camera_1",
    label_subdir: str = "camera_1",
    strict_classes: bool = True,
) -> ManifestBuildResult:
    """Convert the current exported board-camera dataset into JSONL records.

    The source label text is treated as a legacy normalized bbox format:
    `class_id cx cy w h`, with all coordinates normalized to the VGA image.
    The output manifest stores only VGA `bbox_xyxy_vga` boxes.
    """

    root = Path(dataset_root)
    out = Path(output_path)
    result = ManifestBuildResult(output_path=str(out))
    class_path = root / "classes.yaml"
    image_root = root / "images" / image_subdir
    label_root = root / label_subdir
    negative_ids = {_normalize_image_id(image_id) for image_id in (negative_image_ids or [])}

    if not root.is_dir():
        result.add("error", "dataset_missing", "dataset root does not exist", root)
        return result
    if not class_path.is_file():
        result.add("error", "classes_missing", "classes.yaml is required", class_path)
    if not image_root.is_dir():
        result.add("error", "images_missing", "source image directory is required", image_root)
    if not label_root.is_dir():
        result.add("error", "labels_missing", "source label directory is required", label_root)
    if result.errors:
        return result

    classes = load_classes(class_path)
    if strict_classes:
        try:
            validate_ethossafedet_classes(classes)
        except ValueError as exc:
            result.add("error", "classes_not_ethossafedet_v1", str(exc), class_path)
            return result
    class_map = classes_by_id(classes)

    records: list[dict[str, Any]] = []
    images = sorted(path for path in image_root.iterdir() if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS)
    image_by_id = {path.stem: path for path in images}
    for image_id in sorted(negative_ids):
        if image_id not in image_by_id:
            result.add("error", "negative_image_missing", "negative image id has no matching image", image_id)
    if result.errors:
        return result

    for image_path in images:
        label_path = label_root / f"{image_path.stem}.txt"
        split = stable_split_for_key(f"{image_subdir}/{image_path.name}")
        image_record_base = _image_record_base(root, image_path, split, result)
        if image_record_base is None:
            continue

        if label_path.is_file():
            objects = _parse_normalized_bbox_label(label_path, class_map, result)
            record = {**image_record_base, "negative": len(objects) == 0, "objects": objects}
            records.append(record)
            continue

        if image_path.stem in negative_ids:
            records.append({**image_record_base, "negative": True, "objects": []})
            continue

        result.excluded_count += 1

    if result.errors:
        return result
    _write_jsonl(out, records)
    _write_summary(out, result, records)
    _accumulate_result(result, records)
    return result


def prepare_ethossafedet_manifest_from_self_dataset(
    dataset_root: str | Path,
    output_path: str | Path,
    strict_classes: bool = True,
) -> ManifestBuildResult:
    root = Path(dataset_root)
    out = Path(output_path)
    result = ManifestBuildResult(output_path=str(out))
    class_path = root / "classes.yaml"

    if not root.is_dir():
        result.add("error", "dataset_missing", "dataset root does not exist", root)
        return result
    if not class_path.is_file():
        result.add("error", "classes_missing", "classes.yaml is required", class_path)
        return result

    classes = load_classes(class_path)
    if strict_classes:
        try:
            validate_ethossafedet_classes(classes)
        except ValueError as exc:
            result.add("error", "classes_not_ethossafedet_v1", str(exc), class_path)
            return result
    class_map = classes_by_id(classes)

    records: list[dict[str, Any]] = []
    for image_record in iter_image_records(root):
        if not image_record.annotation_path.is_file():
            result.add("error", "annotation_missing", "annotation JSON is missing", image_record.annotation_path)
            continue
        try:
            annotation = read_annotation(image_record.annotation_path)
        except Exception as exc:
            result.add("error", "annotation_invalid_json", str(exc), image_record.annotation_path)
            continue
        split = split_for_annotation(annotation, image_record.key)
        image_record_base = _image_record_base(root, image_record.image_path, split, result)
        if image_record_base is None:
            continue
        objects: list[dict[str, Any]] = []
        for obj in annotation.get("objects", []):
            if not isinstance(obj, dict):
                continue
            class_id = int(obj.get("class_id", -1))
            class_info = class_map.get(class_id)
            bbox = obj.get("bbox_xyxy")
            if class_info is None:
                result.add("error", "class_id_unknown", f"class_id {class_id} is not in classes.yaml", image_record.annotation_path)
                continue
            if not _valid_bbox_xyxy(bbox):
                result.add("error", "bbox_invalid", "bbox_xyxy must be numeric x1< x2, y1< y2", image_record.annotation_path)
                continue
            objects.append(
                {
                    "class_id": class_id,
                    "class_name": class_info.name,
                    "bbox_xyxy_vga": [round(float(v), 6) for v in bbox],
                }
            )
        records.append({**image_record_base, "negative": len(objects) == 0, "objects": objects})

    if result.errors:
        return result
    _write_jsonl(out, records)
    _write_summary(out, result, records)
    _accumulate_result(result, records)
    return result


def load_manifest_records(path: str | Path) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    manifest_path = Path(path)
    for line_number, line in enumerate(manifest_path.read_text(encoding="utf-8").splitlines(), start=1):
        stripped = line.strip()
        if not stripped:
            continue
        record = json.loads(stripped)
        if record.get("schema_version") != ETHOSSAFEDET_SCHEMA_VERSION:
            raise ValueError(f"line {line_number}: unsupported schema_version {record.get('schema_version')!r}")
        records.append(record)
    return records


def resolve_record_image(record: dict[str, Any]) -> Path:
    image = Path(str(record.get("image", "")))
    if image.is_absolute():
        return image
    root_value = str(record.get("dataset_root", ""))
    if not root_value:
        raise ValueError("manifest record has relative image path but no dataset_root")
    root = Path(root_value)
    return root / image


def build_calibration_manifest(
    source_manifest: str | Path,
    output_path: str | Path,
    target_count: int = 320,
    seed: int = 0,
    min_count: int = ETHOSSAFEDET_MIN_CALIBRATION_IMAGES,
    max_count: int = ETHOSSAFEDET_MAX_CALIBRATION_IMAGES,
) -> dict[str, Any]:
    if target_count < min_count or target_count > max_count:
        raise ValueError(f"target_count must be in [{min_count}, {max_count}], got {target_count}")
    records = load_manifest_records(source_manifest)
    usable = [record for record in records if _record_has_real_vga_image(record)]
    if len(usable) < target_count:
        raise ValueError(f"need {target_count} real VGA camera images, found {len(usable)}")
    rng = random.Random(seed)
    selected = rng.sample(usable, target_count)
    items = [
        {
            "image": resolve_record_image(record).as_posix(),
            "split": record.get("split", ""),
            "negative": bool(record.get("negative", False)),
            "class_ids": sorted({int(obj["class_id"]) for obj in record.get("objects", [])}),
        }
        for record in selected
    ]
    manifest = {
        "schema_version": "ethossafedet_calibration_v1",
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "source_manifest": Path(source_manifest).resolve().as_posix(),
        "seed": seed,
        "target_count": target_count,
        "min_count": min_count,
        "max_count": max_count,
        "items": items,
    }
    out = Path(output_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    return manifest


def _parse_normalized_bbox_label(
    label_path: Path,
    class_map: dict[int, Any],
    result: ManifestBuildResult,
) -> list[dict[str, Any]]:
    objects: list[dict[str, Any]] = []
    for line_number, line in enumerate(label_path.read_text(encoding="utf-8", errors="ignore").splitlines(), start=1):
        stripped = line.strip()
        if not stripped:
            continue
        parts = stripped.split()
        if len(parts) != 5:
            result.add("error", "label_invalid", f"line {line_number}: expected 5 fields, got {len(parts)}", label_path)
            continue
        try:
            class_id = int(parts[0])
            cx, cy, width, height = [float(value) for value in parts[1:]]
        except ValueError:
            result.add("error", "label_invalid", f"line {line_number}: non-numeric field", label_path)
            continue
        if class_id not in class_map:
            result.add("error", "class_id_unknown", f"line {line_number}: unknown class_id {class_id}", label_path)
            continue
        if not (0.0 <= cx <= 1.0 and 0.0 <= cy <= 1.0 and 0.0 < width <= 1.0 and 0.0 < height <= 1.0):
            result.add("error", "label_out_of_range", f"line {line_number}: normalized bbox is out of range", label_path)
            continue
        x1 = (cx - width / 2.0) * EXPECTED_VGA_SIZE[0]
        y1 = (cy - height / 2.0) * EXPECTED_VGA_SIZE[1]
        x2 = (cx + width / 2.0) * EXPECTED_VGA_SIZE[0]
        y2 = (cy + height / 2.0) * EXPECTED_VGA_SIZE[1]
        bbox = [
            round(max(0.0, min(float(EXPECTED_VGA_SIZE[0]), x1)), 6),
            round(max(0.0, min(float(EXPECTED_VGA_SIZE[1]), y1)), 6),
            round(max(0.0, min(float(EXPECTED_VGA_SIZE[0]), x2)), 6),
            round(max(0.0, min(float(EXPECTED_VGA_SIZE[1]), y2)), 6),
        ]
        if not _valid_bbox_xyxy(bbox):
            result.add("error", "bbox_invalid", f"line {line_number}: bbox collapsed after clipping", label_path)
            continue
        info = class_map[class_id]
        objects.append({"class_id": class_id, "class_name": info.name, "bbox_xyxy_vga": bbox})
    return objects


def _image_record_base(
    dataset_root: Path,
    image_path: Path,
    split: str,
    result: ManifestBuildResult,
) -> dict[str, Any] | None:
    try:
        with Image.open(image_path) as image:
            width, height = image.size
    except Exception as exc:
        result.add("error", "image_unreadable", str(exc), image_path)
        return None
    if (width, height) != EXPECTED_VGA_SIZE:
        result.add("error", "image_not_vga", f"expected 640x480 board camera image, got {width}x{height}", image_path)
        return None
    return {
        "schema_version": ETHOSSAFEDET_SCHEMA_VERSION,
        "dataset_root": dataset_root.resolve().as_posix(),
        "image": image_path.relative_to(dataset_root).as_posix(),
        "split": split,
        "width": width,
        "height": height,
    }


def _write_jsonl(path: Path, records: Iterable[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as file:
        for record in records:
            file.write(json.dumps(record, ensure_ascii=False, sort_keys=True) + "\n")


def _write_summary(path: Path, result: ManifestBuildResult, records: list[dict[str, Any]]) -> None:
    summary_path = path.with_suffix(path.suffix + ".summary.json")
    temp = ManifestBuildResult(output_path=result.output_path, excluded_count=result.excluded_count, issues=result.issues.copy())
    _accumulate_result(temp, records)
    summary_path.write_text(json.dumps(temp.to_dict(), indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def _accumulate_result(result: ManifestBuildResult, records: list[dict[str, Any]]) -> None:
    result.record_count = len(records)
    result.object_count = 0
    result.negative_count = 0
    result.split_counts = {}
    result.class_counts = {}
    for record in records:
        split = str(record.get("split", ""))
        result.split_counts[split] = result.split_counts.get(split, 0) + 1
        if record.get("negative"):
            result.negative_count += 1
        for obj in record.get("objects", []):
            result.object_count += 1
            class_id = int(obj["class_id"])
            result.class_counts[class_id] = result.class_counts.get(class_id, 0) + 1


def _record_has_real_vga_image(record: dict[str, Any]) -> bool:
    if int(record.get("width", 0)) != EXPECTED_VGA_SIZE[0] or int(record.get("height", 0)) != EXPECTED_VGA_SIZE[1]:
        return False
    image = resolve_record_image(record)
    if not image.is_file():
        return False
    try:
        with Image.open(image) as opened:
            return opened.size == EXPECTED_VGA_SIZE
    except Exception:
        return False


def _normalize_image_id(image_id: str) -> str:
    return Path(str(image_id)).stem


def _valid_bbox_xyxy(value: Any) -> bool:
    if not isinstance(value, list) or len(value) != 4:
        return False
    try:
        x1, y1, x2, y2 = [float(v) for v in value]
    except (TypeError, ValueError):
        return False
    return x1 < x2 and y1 < y2
