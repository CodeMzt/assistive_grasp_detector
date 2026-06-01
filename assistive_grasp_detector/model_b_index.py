"""Index and validate Model B target maps exported by AssistiveGraspAnnotator."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

from assistive_grasp_detector.schema import Issue, TARGET_MAP_KEYS


@dataclass
class TargetIndexResult:
    target_maps_root: str
    output_path: str
    record_count: int = 0
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
            "target_maps_root": self.target_maps_root,
            "output_path": self.output_path,
            "ok": self.ok,
            "record_count": self.record_count,
            "error_count": len(self.errors),
            "issues": [issue.to_dict() for issue in self.issues],
        }


def index_model_b_targets(
    target_maps_root: str | Path,
    output_path: str | Path,
) -> TargetIndexResult:
    root = Path(target_maps_root)
    out = Path(output_path)
    result = TargetIndexResult(target_maps_root=str(root), output_path=str(out))

    if not root.is_dir():
        result.add("error", "target_maps_missing", "target maps directory does not exist", root)
        return result

    records: list[dict[str, Any]] = []
    for npz_path in sorted(root.rglob("*.npz")):
        record = _validate_target_npz(root, npz_path, result)
        if record is not None:
            records.append(record)

    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", encoding="utf-8") as file:
        for record in records:
            file.write(json.dumps(record, ensure_ascii=False) + "\n")

    result.record_count = len(records)
    if not records:
        result.add("warning", "no_target_maps", "no valid target map records were indexed", root)
    return result


def _validate_target_npz(
    root: Path,
    npz_path: Path,
    result: TargetIndexResult,
) -> dict[str, Any] | None:
    png_path = npz_path.with_suffix(".png")
    json_path = npz_path.with_suffix(".json")

    if not png_path.is_file():
        result.add("error", "roi_image_missing", "ROI image sibling is missing", png_path)
        return None
    if not json_path.is_file():
        result.add("error", "target_metadata_missing", "metadata JSON sibling is missing", json_path)
        return None

    try:
        metadata = json.loads(json_path.read_text(encoding="utf-8"))
    except Exception as exc:
        result.add("error", "target_metadata_invalid", str(exc), json_path)
        return None

    try:
        with np.load(npz_path) as data:
            missing = [key for key in TARGET_MAP_KEYS if key not in data.files]
            if missing:
                result.add("error", "target_map_keys_missing", f"missing keys: {missing}", npz_path)
                return None
            arrays = {key: data[key] for key in TARGET_MAP_KEYS}
    except Exception as exc:
        result.add("error", "target_map_npz_invalid", str(exc), npz_path)
        return None

    q_map = arrays["q_map"]
    shape = q_map.shape
    if len(shape) != 2 or shape[0] != shape[1]:
        result.add("error", "target_map_shape_invalid", f"q_map must be square 2D, got {shape}", npz_path)
        return None

    for key, array in arrays.items():
        if array.shape != shape:
            result.add("error", "target_map_shape_mismatch", f"{key} shape {array.shape} does not match {shape}", npz_path)
            return None
        if array.dtype != np.float32:
            result.add("error", "target_map_dtype_invalid", f"{key} dtype must be float32, got {array.dtype}", npz_path)
            return None

    map_size = metadata.get("map_size")
    if map_size is not None and int(map_size) != int(shape[0]):
        result.add("error", "target_map_metadata_mismatch", "metadata map_size does not match array shape", json_path)
        return None

    return {
        "target_npz": npz_path.relative_to(root).as_posix(),
        "roi_image": png_path.relative_to(root).as_posix(),
        "metadata_json": json_path.relative_to(root).as_posix(),
        "source_image": metadata.get("source_image", ""),
        "source_bbox": metadata.get("source_bbox", []),
        "padded_bbox": metadata.get("padded_bbox", []),
        "instance_id": metadata.get("instance_id"),
        "class_id": metadata.get("class_id"),
        "class_name": metadata.get("class_name", ""),
        "map_size": int(shape[0]),
        "positive_pixels": int((q_map > 0).sum()),
        "max_quality": float(q_map.max()) if q_map.size else 0.0,
    }
