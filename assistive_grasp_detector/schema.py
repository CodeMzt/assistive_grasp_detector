"""Shared schema helpers for dataset infrastructure."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".tif"}
VGA_WIDTH = 640
VGA_HEIGHT = 480
EXPECTED_VGA_SIZE = (VGA_WIDTH, VGA_HEIGHT)
ALLOWED_SPLITS = {"train", "val", "test"}
ALLOWED_DIFFICULTIES = {"easy", "medium", "hard", "invalid"}
TARGET_MAP_KEYS = ("q_map", "sin2theta_map", "cos2theta_map", "width_map")


@dataclass(frozen=True)
class ClassInfo:
    """One object class from classes.yaml."""

    id: int
    name: str
    graspable: bool = True
    policy: str = "grasp_rect"


@dataclass(frozen=True)
class Issue:
    """Validation issue emitted by data tools."""

    level: str
    code: str
    message: str
    path: str = ""

    def to_dict(self) -> dict[str, str]:
        return {
            "level": self.level,
            "code": self.code,
            "message": self.message,
            "path": self.path,
        }


def normalize_path(path: str | Path) -> str:
    return Path(path).as_posix()


def load_yaml(path: str | Path) -> dict[str, Any]:
    with Path(path).open("r", encoding="utf-8") as file:
        data = yaml.safe_load(file) or {}
    if not isinstance(data, dict):
        raise ValueError(f"YAML root must be a mapping: {path}")
    return data


def write_yaml(path: str | Path, data: dict[str, Any]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as file:
        yaml.safe_dump(data, file, allow_unicode=True, sort_keys=False)


def load_classes(path: str | Path) -> list[ClassInfo]:
    data = load_yaml(path)
    raw_classes = data.get("classes", [])
    if not isinstance(raw_classes, list):
        raise ValueError(f"classes must be a list: {path}")

    classes: list[ClassInfo] = []
    seen_ids: set[int] = set()
    seen_names: set[str] = set()
    for item in raw_classes:
        if not isinstance(item, dict):
            raise ValueError(f"class entries must be mappings: {path}")
        class_id = int(item["id"])
        name = str(item["name"])
        if class_id in seen_ids:
            raise ValueError(f"duplicate class id {class_id}: {path}")
        if name in seen_names:
            raise ValueError(f"duplicate class name {name}: {path}")
        seen_ids.add(class_id)
        seen_names.add(name)
        classes.append(
            ClassInfo(
                id=class_id,
                name=name,
                graspable=bool(item.get("graspable", True)),
                policy=str(item.get("policy", "grasp_rect")),
            )
        )
    return sorted(classes, key=lambda cls: cls.id)


def classes_by_id(classes: list[ClassInfo]) -> dict[int, ClassInfo]:
    return {cls.id: cls for cls in classes}


def classes_by_name(classes: list[ClassInfo]) -> dict[str, ClassInfo]:
    return {cls.name: cls for cls in classes}


def yolo_names(classes: list[ClassInfo]) -> dict[int, str]:
    if not classes:
        return {}
    class_map = {cls.id: cls.name for cls in classes}
    return {class_id: class_map.get(class_id, f"unused_{class_id}") for class_id in range(max(class_map) + 1)}
