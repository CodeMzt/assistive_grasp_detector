"""Model A V2 manifest builder for EthosSafeDetV2 formal training."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import sys
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from PIL import Image

from assistive_grasp_detector.annotator_dataset import annotations_root, classes_path, images_root, read_annotation
from assistive_grasp_detector.schema import ETHOSSAFEDET_CLASS_NAMES, EXPECTED_VGA_SIZE, Issue, classes_by_id, load_classes

ETHOSSAFEDET_V2_SCHEMA_VERSION = "ethossafedet_v2_manifest_v1"
SPLIT_RULE = "sha1(annotation-relative-key) % 100: train<80, val<90, test>=90"


@dataclass
class V2ManifestResult:
    output_path: str
    dataset_root: str
    records: int = 0
    objects: int = 0
    split_counts: dict[str, int] = field(default_factory=dict)
    class_counts: dict[str, int] = field(default_factory=dict)
    class_by_split: dict[str, dict[str, int]] = field(default_factory=dict)
    theta_valid_counts: dict[str, int] = field(default_factory=dict)
    yaw_source_counts: dict[str, dict[str, int]] = field(default_factory=dict)
    issues: list[Issue] = field(default_factory=list)

    @property
    def errors(self) -> list[Issue]:
        return [issue for issue in self.issues if issue.level == "error"]

    @property
    def ok(self) -> bool:
        return not self.errors

    def add(self, level: str, code: str, message: str, path: str | Path = "") -> None:
        self.issues.append(Issue(level=level, code=code, message=message, path=str(path)))

    def to_dict(self) -> dict[str, Any]:
        return {
            "ok": self.ok,
            "output_path": self.output_path,
            "dataset_root": self.dataset_root,
            "schema_version": ETHOSSAFEDET_V2_SCHEMA_VERSION,
            "split_rule": SPLIT_RULE,
            "record_count": self.records,
            "object_count": self.objects,
            "split_counts": self.split_counts,
            "class_counts": self.class_counts,
            "class_by_split": self.class_by_split,
            "theta_valid_counts": self.theta_valid_counts,
            "yaw_source_counts": self.yaw_source_counts,
            "error_count": len(self.errors),
            "warning_count": len([issue for issue in self.issues if issue.level == "warning"]),
            "issues": [issue.to_dict() for issue in self.issues],
        }


def stable_split_for_key(key: str) -> str:
    bucket = int(hashlib.sha1(key.encode("utf-8")).hexdigest()[:8], 16) % 100
    if bucket < 80:
        return "train"
    if bucket < 90:
        return "val"
    return "test"


def prepare_ethossafedet_v2_manifest(dataset_root: str | Path, output_path: str | Path) -> V2ManifestResult:
    root = Path(dataset_root)
    out = Path(output_path)
    result = V2ManifestResult(output_path=str(out), dataset_root=str(root))
    if not root.is_dir():
        result.add("error", "dataset_missing", "dataset root does not exist", root)
        return result
    if not images_root(root).is_dir():
        result.add("error", "images_missing", "images directory is missing", images_root(root))
    if not annotations_root(root).is_dir():
        result.add("error", "annotations_missing", "annotations directory is missing", annotations_root(root))
    if not classes_path(root).is_file():
        result.add("error", "classes_missing", "classes.yaml is missing", classes_path(root))
        return result

    try:
        classes = load_classes(classes_path(root))
    except Exception as exc:
        result.add("error", "classes_invalid", str(exc), classes_path(root))
        return result
    actual = tuple(cls.name for cls in classes)
    if actual != ETHOSSAFEDET_CLASS_NAMES:
        result.add("error", "classes_not_object_vocab_v1", f"expected {ETHOSSAFEDET_CLASS_NAMES}, got {actual}", classes_path(root))
        return result
    class_map = classes_by_id(classes)

    records: list[dict[str, Any]] = []
    for annotation_path in sorted(annotations_root(root).rglob("*.json")):
        try:
            annotation = read_annotation(annotation_path)
        except Exception as exc:
            result.add("error", "annotation_invalid_json", str(exc), annotation_path)
            continue
        image_path = _resolve_annotation_image(root, annotation, annotation_path, result)
        if image_path is None:
            continue
        image_info = _image_record_base(root, image_path, annotation_path, result)
        if image_info is None:
            continue
        split = stable_split_for_key(annotation_path.relative_to(annotations_root(root)).with_suffix("").as_posix())
        objects: list[dict[str, Any]] = []
        for obj in annotation.get("objects", []):
            converted = _convert_object(obj, class_map, annotation_path, result)
            if converted is not None:
                objects.append(converted)
        records.append(
            {
                "schema_version": ETHOSSAFEDET_V2_SCHEMA_VERSION,
                "dataset_root": root.resolve().as_posix(),
                "image_id": str(annotation.get("image_id") or annotation_path.stem),
                "image_path": image_path.resolve().as_posix(),
                "annotation_path": annotation_path.resolve().as_posix(),
                "split": split,
                "width": image_info["width"],
                "height": image_info["height"],
                "negative": len(objects) == 0,
                "objects": objects,
            }
        )

    if result.errors:
        return result
    _validate_split_coverage(records, result)
    if result.errors:
        return result
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text("\n".join(json.dumps(record, ensure_ascii=False, sort_keys=True) for record in records) + "\n", encoding="utf-8")
    _accumulate_result(result, records)
    summary_path = out.with_suffix(out.suffix + ".summary.json")
    summary_path.write_text(json.dumps(result.to_dict(), indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    return result


def load_v2_manifest_records(path: str | Path) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    for line_number, line in enumerate(Path(path).read_text(encoding="utf-8").splitlines(), start=1):
        stripped = line.strip()
        if not stripped:
            continue
        record = json.loads(stripped)
        if record.get("schema_version") != ETHOSSAFEDET_V2_SCHEMA_VERSION:
            raise ValueError(f"line {line_number}: unsupported schema_version {record.get('schema_version')!r}")
        records.append(record)
    return records


def resolve_v2_record_image(record: dict[str, Any]) -> Path:
    path = Path(str(record.get("image_path", "")))
    if path.is_absolute():
        return path
    return Path(str(record["dataset_root"])) / path


def _resolve_annotation_image(root: Path, annotation: dict[str, Any], annotation_path: Path, result: V2ManifestResult) -> Path | None:
    raw_path = annotation.get("image_path")
    candidates: list[Path] = []
    if isinstance(raw_path, str) and raw_path:
        candidates.append(Path(raw_path))
        candidates.append(root / raw_path)
    rel = annotation_path.relative_to(annotations_root(root)).with_suffix(".png")
    candidates.append(images_root(root) / rel)
    for suffix in (".jpg", ".jpeg", ".bmp"):
        candidates.append((images_root(root) / rel).with_suffix(suffix))
    for candidate in candidates:
        if candidate.is_file():
            return candidate
    result.add("error", "image_missing", "annotation has no readable matching image", annotation_path)
    return None


def _image_record_base(root: Path, image_path: Path, annotation_path: Path, result: V2ManifestResult) -> dict[str, int] | None:
    try:
        with Image.open(image_path) as image:
            width, height = image.size
    except Exception as exc:
        result.add("error", "image_unreadable", str(exc), image_path)
        return None
    if (width, height) != EXPECTED_VGA_SIZE:
        result.add("error", "image_not_vga", f"expected {EXPECTED_VGA_SIZE}, got {(width, height)}", image_path)
        return None
    expected_ann = annotations_root(root) / image_path.relative_to(images_root(root)).with_suffix(".json")
    if expected_ann.resolve() != annotation_path.resolve():
        result.add("warning", "annotation_path_unexpected", f"annotation path differs from expected {expected_ann}", annotation_path)
    return {"width": width, "height": height}


def _convert_object(
    obj: Any,
    class_map: dict[int, Any],
    annotation_path: Path,
    result: V2ManifestResult,
) -> dict[str, Any] | None:
    if not isinstance(obj, dict):
        result.add("error", "object_invalid", "object entry must be a JSON object", annotation_path)
        return None
    try:
        class_id = int(obj.get("class_id"))
    except Exception:
        result.add("error", "class_id_invalid", f"class_id is invalid: {obj.get('class_id')!r}", annotation_path)
        return None
    class_info = class_map.get(class_id)
    if class_info is None:
        result.add("error", "class_id_unknown", f"class_id {class_id} is not in object_vocab_v1", annotation_path)
        return None
    bbox = obj.get("bbox_xyxy")
    if not _valid_bbox_xyxy(bbox):
        result.add("error", "bbox_invalid", f"invalid bbox for object {obj.get('instance_id')}", annotation_path)
        return None
    theta, yaw_source = _theta_from_object(obj)
    theta_valid = theta is not None
    item = {
        "instance_id": int(obj.get("instance_id", -1)) if isinstance(obj.get("instance_id"), int) else -1,
        "class_id": class_id,
        "class_name": class_info.name,
        "bbox_xyxy_vga": [round(float(v), 6) for v in bbox],
        "theta_valid": bool(theta_valid),
        "yaw_source": yaw_source,
    }
    if theta_valid:
        item["orientation_rad"] = round(float(theta), 8)
        item["orientation_sin2theta"] = round(float(math.sin(2.0 * float(theta))), 8)
        item["orientation_cos2theta"] = round(float(math.cos(2.0 * float(theta))), 8)
    return item


def _theta_from_object(obj: dict[str, Any]) -> tuple[float | None, str]:
    status = str(obj.get("yaw_label_status") or "missing")
    if status != "valid":
        return None, status
    grasp_yaw = obj.get("grasp_yaw")
    if isinstance(grasp_yaw, (int, float)):
        return float(grasp_yaw), "grasp_yaw"
    pts = obj.get("main_axis_points")
    if (
        isinstance(pts, list)
        and len(pts) == 2
        and all(isinstance(p, list) and len(p) == 2 for p in pts)
        and all(isinstance(v, (int, float)) for p in pts for v in p)
    ):
        return math.atan2(float(pts[1][1]) - float(pts[0][1]), float(pts[1][0]) - float(pts[0][0])), "main_axis_points"
    return None, "valid_missing_geometry"


def _valid_bbox_xyxy(value: Any) -> bool:
    if not isinstance(value, list) or len(value) != 4:
        return False
    try:
        x1, y1, x2, y2 = [float(v) for v in value]
    except Exception:
        return False
    return x1 < x2 and y1 < y2


def _validate_split_coverage(records: list[dict[str, Any]], result: V2ManifestResult) -> None:
    split_counts = Counter(str(record.get("split")) for record in records)
    for split in ("train", "val", "test"):
        if split_counts.get(split, 0) <= 0:
            result.add("error", "split_empty", f"split {split!r} has no records")
    by_split: dict[str, Counter[int]] = defaultdict(Counter)
    for record in records:
        split = str(record.get("split"))
        for obj in record.get("objects", []):
            by_split[split][int(obj["class_id"])] += 1
    for split in ("train", "val", "test"):
        for class_id, name in enumerate(ETHOSSAFEDET_CLASS_NAMES):
            if by_split[split].get(class_id, 0) <= 0:
                result.add("error", "class_split_empty", f"class {name!r} has no objects in split {split!r}")


def _accumulate_result(result: V2ManifestResult, records: list[dict[str, Any]]) -> None:
    result.records = len(records)
    split_counts: Counter[str] = Counter()
    class_counts: Counter[str] = Counter()
    class_by_split: dict[str, Counter[str]] = defaultdict(Counter)
    theta_valid_counts: Counter[str] = Counter()
    yaw_source_counts: dict[str, Counter[str]] = defaultdict(Counter)
    object_count = 0
    for record in records:
        split = str(record.get("split"))
        split_counts[split] += 1
        for obj in record.get("objects", []):
            object_count += 1
            name = str(obj["class_name"])
            class_counts[name] += 1
            class_by_split[name][split] += 1
            if obj.get("theta_valid"):
                theta_valid_counts[name] += 1
            yaw_source_counts[name][str(obj.get("yaw_source", ""))] += 1
    result.objects = object_count
    result.split_counts = dict(split_counts)
    result.class_counts = dict(class_counts)
    result.class_by_split = {name: dict(counter) for name, counter in class_by_split.items()}
    result.theta_valid_counts = dict(theta_valid_counts)
    result.yaw_source_counts = {name: dict(counter) for name, counter in yaw_source_counts.items()}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Build a Model A V2 EthosSafeDetV2 manifest.")
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args(argv)
    result = prepare_ethossafedet_v2_manifest(args.dataset, args.out)
    payload = result.to_dict()
    if args.json:
        print(json.dumps(payload, indent=2, ensure_ascii=False))
    else:
        print(f"ok={result.ok} records={result.records} objects={result.objects} out={args.out}")
        if result.issues:
            for issue in result.issues[:20]:
                print(f"{issue.level}: {issue.code}: {issue.message} {issue.path}", file=sys.stderr)
    return 0 if result.ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
