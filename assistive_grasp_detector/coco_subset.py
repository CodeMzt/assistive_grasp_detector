"""Prepare a local COCO detection subset for auxiliary experiments."""

from __future__ import annotations

import json
import shutil
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from assistive_grasp_detector.schema import (
    Issue,
    classes_by_name,
    contiguous_class_names,
    load_classes,
    load_yaml,
    write_yaml,
)


@dataclass
class CocoSubsetResult:
    output_root: str
    dataset_yaml: str
    split_counts: dict[str, int] = field(default_factory=dict)
    label_counts: dict[str, int] = field(default_factory=dict)
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
            "output_root": self.output_root,
            "dataset_yaml": self.dataset_yaml,
            "ok": self.ok,
            "split_counts": self.split_counts,
            "label_counts": self.label_counts,
            "error_count": len(self.errors),
            "warning_count": len([issue for issue in self.issues if issue.level == "warning"]),
            "issues": [issue.to_dict() for issue in self.issues],
        }


def prepare_coco_subset(
    coco_root: str | Path,
    config_path: str | Path,
    output_root: str | Path,
) -> CocoSubsetResult:
    coco_root = Path(coco_root)
    out = Path(output_root)
    config = load_yaml(config_path)
    target_classes_path = _resolve_config_path(config_path, config["target_classes"])
    classes = load_classes(target_classes_path)
    class_by_name = classes_by_name(classes)

    category_map = config.get("category_map", {})
    exclude_categories = set(config.get("exclude_categories", []))
    max_per_class = config.get("max_per_class", {})
    split_configs = config.get("splits", {})

    result = CocoSubsetResult(output_root=str(out), dataset_yaml=str(out / "dataset.yaml"))
    if not coco_root.is_dir():
        result.add("error", "coco_root_missing", "COCO root directory does not exist", coco_root)
        return result

    for split, split_config in split_configs.items():
        _prepare_coco_split(
            split=split,
            split_config=split_config,
            coco_root=coco_root,
            output_root=out,
            category_map=category_map,
            exclude_categories=exclude_categories,
            max_count=int(max_per_class.get(split, 0) or 0),
            class_by_name=class_by_name,
            result=result,
        )

    write_yaml(
        out / "dataset.yaml",
        {
            "path": out.resolve().as_posix(),
            "train": "images/train",
            "val": "images/val",
            "names": contiguous_class_names(classes),
        },
    )
    manifest = {
        "source_coco_root": coco_root.resolve().as_posix(),
        "config": Path(config_path).resolve().as_posix(),
        "split_counts": result.split_counts,
        "label_counts": result.label_counts,
        "note": "COCO subset for Model A only. Do not use as ROIContourNet mask labels.",
    }
    (out / "manifest.json").write_text(json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8")
    return result


def _prepare_coco_split(
    split: str,
    split_config: dict[str, Any],
    coco_root: Path,
    output_root: Path,
    category_map: dict[str, str],
    exclude_categories: set[str],
    max_count: int,
    class_by_name: dict[str, Any],
    result: CocoSubsetResult,
) -> None:
    ann_path = coco_root / split_config["annotations"]
    images_dir = coco_root / split_config["images"]
    if not ann_path.is_file():
        result.add("error", "coco_annotations_missing", "COCO annotations file is missing", ann_path)
        return
    if not images_dir.is_dir():
        result.add("error", "coco_images_missing", "COCO images directory is missing", images_dir)
        return

    with ann_path.open("r", encoding="utf-8") as file:
        coco = json.load(file)

    categories = {int(cat["id"]): str(cat["name"]) for cat in coco.get("categories", [])}
    images = {int(image["id"]): image for image in coco.get("images", [])}
    labels_by_image: dict[int, list[str]] = {}
    per_target_counts: dict[str, int] = {}

    for annotation in coco.get("annotations", []):
        category_name = categories.get(int(annotation.get("category_id", -1)))
        if category_name is None or category_name in exclude_categories:
            continue
        target_name = category_map.get(category_name)
        if target_name is None:
            continue
        target_class = class_by_name.get(target_name)
        if target_class is None:
            result.add("error", "target_class_missing", f"target class {target_name!r} is not in target_classes", ann_path)
            continue
        if max_count > 0 and per_target_counts.get(target_name, 0) >= max_count:
            continue

        image = images.get(int(annotation.get("image_id", -1)))
        if image is None:
            result.add("warning", "coco_image_missing_from_json", "annotation references missing image", ann_path)
            continue

        bbox = annotation.get("bbox", [])
        line = _coco_bbox_to_normalized_bbox_line(bbox, int(image["width"]), int(image["height"]), int(target_class.id))
        if line is None:
            result.add("warning", "coco_bbox_invalid", "skipped invalid COCO bbox", ann_path)
            continue

        labels_by_image.setdefault(int(image["id"]), []).append(line)
        per_target_counts[target_name] = per_target_counts.get(target_name, 0) + 1

    image_count = 0
    label_count = 0
    for image_id, label_lines in sorted(labels_by_image.items()):
        if not label_lines:
            continue
        image = images[image_id]
        file_name = image["file_name"]
        src = images_dir / file_name
        if not src.is_file():
            result.add("warning", "coco_image_file_missing", "COCO image file is missing", src)
            continue
        dst_image = output_root / "images" / split / file_name
        dst_label = output_root / "labels" / split / Path(file_name).with_suffix(".txt")
        dst_image.parent.mkdir(parents=True, exist_ok=True)
        dst_label.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(src, dst_image)
        dst_label.write_text("\n".join(label_lines) + "\n", encoding="utf-8")
        image_count += 1
        label_count += len(label_lines)

    result.split_counts[split] = image_count
    result.label_counts[split] = label_count


def _coco_bbox_to_normalized_bbox_line(
    bbox: list[Any],
    image_width: int,
    image_height: int,
    class_id: int,
) -> str | None:
    if not isinstance(bbox, list) or len(bbox) != 4:
        return None
    x, y, width, height = [float(value) for value in bbox]
    if width <= 0 or height <= 0 or image_width <= 0 or image_height <= 0:
        return None
    cx = (x + width / 2.0) / image_width
    cy = (y + height / 2.0) / image_height
    bw = width / image_width
    bh = height / image_height
    return f"{class_id} {cx:.6f} {cy:.6f} {bw:.6f} {bh:.6f}"


def _resolve_config_path(config_path: str | Path, value: str | Path) -> Path:
    path = Path(value)
    if path.is_absolute():
        return path
    config_relative = Path(config_path).parent / path
    if config_relative.exists():
        return config_relative
    return path
