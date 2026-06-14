"""CPU reference post-processing for EthosSafeDet-A v1 raw outputs."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

import numpy as np

from assistive_grasp_detector.coords import make_letterbox_transform, bboxes_model_to_vga_xyxy
from assistive_grasp_detector.schema import EXPECTED_VGA_SIZE


@dataclass(frozen=True)
class DetectionCandidate:
    class_id: int
    score: float
    bbox_xyxy_vga: tuple[float, float, float, float]
    bbox_xyxy_model: tuple[float, float, float, float]
    grid_y: int
    grid_x: int

    def to_dict(self) -> dict[str, Any]:
        data = asdict(self)
        data["bbox_xyxy_vga"] = [round(float(v), 6) for v in self.bbox_xyxy_vga]
        data["bbox_xyxy_model"] = [round(float(v), 6) for v in self.bbox_xyxy_model]
        data["score"] = round(float(self.score), 8)
        return data


def sigmoid(value: np.ndarray) -> np.ndarray:
    value = np.asarray(value, dtype=np.float32)
    return 1.0 / (1.0 + np.exp(-value))


def decode_ltrb_outputs(
    cls_logits: np.ndarray,
    box_ltrb: np.ndarray,
    input_size: int = 320,
    source_size: tuple[int, int] = EXPECTED_VGA_SIZE,
    score_threshold: float = 0.25,
    pre_nms_top_k: int = 100,
    nms_iou_threshold: float = 0.5,
) -> list[DetectionCandidate]:
    cls = _canonical_chw(cls_logits)
    box = _canonical_chw(box_ltrb)
    if cls.shape[1:] != box.shape[1:]:
        raise ValueError(f"cls and box spatial shapes differ: {cls.shape} vs {box.shape}")
    if box.shape[0] != 4:
        raise ValueError(f"box_ltrb must have 4 channels, got {box.shape[0]}")

    class_count, height, width = cls.shape
    stride_y = float(input_size) / float(height)
    stride_x = float(input_size) / float(width)
    scores = sigmoid(cls)
    candidates: list[DetectionCandidate] = []
    for class_id in range(class_count):
        ys, xs = np.where(scores[class_id] >= score_threshold)
        for y, x in zip(ys.tolist(), xs.tolist()):
            center_x = (float(x) + 0.5) * stride_x
            center_y = (float(y) + 0.5) * stride_y
            left, top, right, bottom = [float(v) for v in box[:, y, x]]
            model_box = (
                max(0.0, center_x - max(0.0, left)),
                max(0.0, center_y - max(0.0, top)),
                min(float(input_size), center_x + max(0.0, right)),
                min(float(input_size), center_y + max(0.0, bottom)),
            )
            if model_box[0] >= model_box[2] or model_box[1] >= model_box[3]:
                continue
            candidates.append(
                DetectionCandidate(
                    class_id=class_id,
                    score=float(scores[class_id, y, x]),
                    bbox_xyxy_model=model_box,
                    bbox_xyxy_vga=(0.0, 0.0, 0.0, 0.0),
                    grid_y=int(y),
                    grid_x=int(x),
                )
            )

    candidates.sort(key=lambda item: item.score, reverse=True)
    if pre_nms_top_k > 0:
        candidates = candidates[:pre_nms_top_k]
    candidates = _attach_vga_boxes(candidates, input_size, source_size)
    return classwise_nms(candidates, iou_threshold=nms_iou_threshold)


def classwise_nms(candidates: list[DetectionCandidate], iou_threshold: float = 0.5) -> list[DetectionCandidate]:
    kept: list[DetectionCandidate] = []
    for class_id in sorted({candidate.class_id for candidate in candidates}):
        group = [candidate for candidate in candidates if candidate.class_id == class_id]
        group.sort(key=lambda item: item.score, reverse=True)
        while group:
            best = group.pop(0)
            kept.append(best)
            group = [candidate for candidate in group if bbox_iou(best.bbox_xyxy_vga, candidate.bbox_xyxy_vga) < iou_threshold]
    kept.sort(key=lambda item: item.score, reverse=True)
    return kept


def bbox_iou(a: tuple[float, float, float, float] | list[float], b: tuple[float, float, float, float] | list[float]) -> float:
    ax1, ay1, ax2, ay2 = [float(v) for v in a]
    bx1, by1, bx2, by2 = [float(v) for v in b]
    ix1 = max(ax1, bx1)
    iy1 = max(ay1, by1)
    ix2 = min(ax2, bx2)
    iy2 = min(ay2, by2)
    iw = max(0.0, ix2 - ix1)
    ih = max(0.0, iy2 - iy1)
    inter = iw * ih
    area_a = max(0.0, ax2 - ax1) * max(0.0, ay2 - ay1)
    area_b = max(0.0, bx2 - bx1) * max(0.0, by2 - by1)
    denom = area_a + area_b - inter
    return float(inter / denom) if denom > 0 else 0.0


def compare_main_detection(
    reference: list[dict[str, Any]] | list[DetectionCandidate],
    candidate: list[dict[str, Any]] | list[DetectionCandidate],
    min_iou: float = 0.85,
) -> dict[str, Any]:
    ref = _as_detection_dicts(reference)
    cand = _as_detection_dicts(candidate)
    if not ref or not cand:
        return {"ok": False, "reason": "missing_detection", "iou": 0.0, "class_match": False}
    ref_top = ref[0]
    cand_top = cand[0]
    class_match = int(ref_top["class_id"]) == int(cand_top["class_id"])
    iou = bbox_iou(ref_top["bbox_xyxy_vga"], cand_top["bbox_xyxy_vga"])
    return {
        "ok": bool(class_match and iou >= min_iou),
        "class_match": bool(class_match),
        "iou": float(iou),
        "min_iou": float(min_iou),
        "reference": ref_top,
        "candidate": cand_top,
    }


def candidates_to_json(candidates: list[DetectionCandidate], limit: int = 20) -> list[dict[str, Any]]:
    return [candidate.to_dict() for candidate in candidates[:limit]]


def _canonical_chw(array: np.ndarray) -> np.ndarray:
    arr = np.asarray(array)
    if arr.ndim == 4 and arr.shape[0] == 1:
        arr = arr[0]
    if arr.ndim != 3:
        raise ValueError(f"expected CHW/HWC or BCHW/BHWC output, got shape {arr.shape}")
    if arr.shape[0] <= 16:
        return arr.astype(np.float32, copy=False)
    if arr.shape[-1] <= 16:
        return np.transpose(arr, (2, 0, 1)).astype(np.float32, copy=False)
    raise ValueError(f"cannot infer channel axis for shape {arr.shape}")


def _attach_vga_boxes(
    candidates: list[DetectionCandidate],
    input_size: int,
    source_size: tuple[int, int],
) -> list[DetectionCandidate]:
    if not candidates:
        return []
    transform = make_letterbox_transform(source_size[0], source_size[1], input_size, input_size)
    model_boxes = np.asarray([candidate.bbox_xyxy_model for candidate in candidates], dtype=np.float32)
    vga_boxes = bboxes_model_to_vga_xyxy(model_boxes, transform)
    result: list[DetectionCandidate] = []
    for candidate, vga_box in zip(candidates, vga_boxes):
        result.append(
            DetectionCandidate(
                class_id=candidate.class_id,
                score=candidate.score,
                bbox_xyxy_model=candidate.bbox_xyxy_model,
                bbox_xyxy_vga=tuple(float(v) for v in vga_box),
                grid_y=candidate.grid_y,
                grid_x=candidate.grid_x,
            )
        )
    return result


def _as_detection_dicts(items: list[dict[str, Any]] | list[DetectionCandidate]) -> list[dict[str, Any]]:
    result: list[dict[str, Any]] = []
    for item in items:
        if isinstance(item, DetectionCandidate):
            result.append(item.to_dict())
        else:
            result.append(item)
    result.sort(key=lambda row: float(row.get("score", 0.0)), reverse=True)
    return result
