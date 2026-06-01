"""Build Ultralytics-compatible YOLO data from self-collected annotations."""

from __future__ import annotations

import json
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from PIL import Image

from assistive_grasp_detector.annotator_dataset import (
    classes_path,
    iter_image_records,
    read_annotation,
    split_for_annotation,
    validate_self_dataset,
)
from assistive_grasp_detector.schema import load_classes, write_yaml, yolo_names


@dataclass(frozen=True)
class YoloBuildResult:
    output_root: str
    dataset_yaml: str
    split_counts: dict[str, int]
    label_count: int

    def to_dict(self) -> dict[str, Any]:
        return {
            "output_root": self.output_root,
            "dataset_yaml": self.dataset_yaml,
            "split_counts": self.split_counts,
            "label_count": self.label_count,
        }


def normalize_bbox_xyxy(
    bbox_xyxy: list[float],
    image_width: int,
    image_height: int,
) -> tuple[float, float, float, float]:
    x1, y1, x2, y2 = [float(value) for value in bbox_xyxy]
    cx = ((x1 + x2) / 2.0) / image_width
    cy = ((y1 + y2) / 2.0) / image_height
    width = (x2 - x1) / image_width
    height = (y2 - y1) / image_height
    return cx, cy, width, height


def build_model_a_yolo(
    dataset_root: str | Path,
    output_root: str | Path,
    class_config: str | Path | None = None,
) -> YoloBuildResult:
    root = Path(dataset_root)
    out = Path(output_root)

    validation = validate_self_dataset(root)
    if validation.errors:
        messages = "; ".join(f"{issue.code}: {issue.message}" for issue in validation.errors[:5])
        raise ValueError(f"dataset validation failed: {messages}")

    cls_path = Path(class_config) if class_config is not None else classes_path(root)
    classes = load_classes(cls_path)

    split_counts: dict[str, int] = {"train": 0, "val": 0, "test": 0}
    label_count = 0
    for record in iter_image_records(root):
        annotation = read_annotation(record.annotation_path)
        split = split_for_annotation(annotation, record.key)
        split_counts[split] = split_counts.get(split, 0) + 1

        image_out = out / "images" / split / Path(record.key)
        label_out = out / "labels" / split / Path(record.key).with_suffix(".txt")
        image_out.parent.mkdir(parents=True, exist_ok=True)
        label_out.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(record.image_path, image_out)

        with Image.open(record.image_path) as image:
            image_width, image_height = image.size
        label_lines: list[str] = []
        for obj in annotation.get("objects", []):
            cx, cy, bw, bh = normalize_bbox_xyxy(obj["bbox_xyxy"], image_width, image_height)
            label_lines.append(f"{int(obj['class_id'])} {cx:.6f} {cy:.6f} {bw:.6f} {bh:.6f}")
        label_count += len(label_lines)
        label_out.write_text("\n".join(label_lines) + ("\n" if label_lines else ""), encoding="utf-8")

    dataset_yaml = out / "dataset.yaml"
    yaml_data: dict[str, Any] = {
        "path": out.resolve().as_posix(),
        "train": "images/train",
        "val": "images/val",
        "names": yolo_names(classes),
    }
    if split_counts.get("test", 0):
        yaml_data["test"] = "images/test"
    write_yaml(dataset_yaml, yaml_data)

    manifest = {
        "source_dataset": root.resolve().as_posix(),
        "class_config": cls_path.resolve().as_posix(),
        "split_counts": split_counts,
        "label_count": label_count,
        "note": "Generated from AssistiveGraspAnnotator master annotations.",
    }
    (out / "manifest.json").write_text(json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8")

    return YoloBuildResult(
        output_root=str(out),
        dataset_yaml=str(dataset_yaml),
        split_counts=split_counts,
        label_count=label_count,
    )
