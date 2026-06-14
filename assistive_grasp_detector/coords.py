from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from PIL import Image


@dataclass(frozen=True)
class LetterboxTransform:
    src_w: int
    src_h: int
    dst_w: int
    dst_h: int
    scale: float
    pad_x: float
    pad_y: float


def make_letterbox_transform(
    src_w: int,
    src_h: int,
    dst_w: int,
    dst_h: int,
) -> LetterboxTransform:
    scale = min(dst_w / src_w, dst_h / src_h)
    new_w = src_w * scale
    new_h = src_h * scale
    return LetterboxTransform(
        src_w=src_w,
        src_h=src_h,
        dst_w=dst_w,
        dst_h=dst_h,
        scale=scale,
        pad_x=(dst_w - new_w) / 2.0,
        pad_y=(dst_h - new_h) / 2.0,
    )


def bbox_model_to_vga(
    bbox_xyxy_model: tuple[float, float, float, float],
    transform: LetterboxTransform,
) -> tuple[float, float, float, float]:
    x1, y1, x2, y2 = bbox_xyxy_model
    mapped = (
        (x1 - transform.pad_x) / transform.scale,
        (y1 - transform.pad_y) / transform.scale,
        (x2 - transform.pad_x) / transform.scale,
        (y2 - transform.pad_y) / transform.scale,
    )
    return (
        min(max(mapped[0], 0.0), float(transform.src_w)),
        min(max(mapped[1], 0.0), float(transform.src_h)),
        min(max(mapped[2], 0.0), float(transform.src_w)),
        min(max(mapped[3], 0.0), float(transform.src_h)),
    )


def bbox_vga_to_model(
    bbox_xyxy_vga: tuple[float, float, float, float],
    transform: LetterboxTransform,
) -> tuple[float, float, float, float]:
    x1, y1, x2, y2 = bbox_xyxy_vga
    return (
        x1 * transform.scale + transform.pad_x,
        y1 * transform.scale + transform.pad_y,
        x2 * transform.scale + transform.pad_x,
        y2 * transform.scale + transform.pad_y,
    )


def bboxes_model_to_vga_xyxy(
    bboxes_xyxy_model: np.ndarray,
    transform: LetterboxTransform,
) -> np.ndarray:
    bboxes = np.asarray(bboxes_xyxy_model, dtype=np.float32)
    if bboxes.ndim != 2 or bboxes.shape[1] != 4:
        raise ValueError(f"bboxes_xyxy_model must have shape (N,4), got {bboxes.shape}")
    bboxes = bboxes.copy()
    bboxes[:, [0, 2]] = (bboxes[:, [0, 2]] - float(transform.pad_x)) / float(transform.scale)
    bboxes[:, [1, 3]] = (bboxes[:, [1, 3]] - float(transform.pad_y)) / float(transform.scale)
    bboxes[:, [0, 2]] = np.clip(bboxes[:, [0, 2]], 0.0, float(transform.src_w))
    bboxes[:, [1, 3]] = np.clip(bboxes[:, [1, 3]], 0.0, float(transform.src_h))
    return bboxes


def bboxes_vga_to_model_xyxy(
    bboxes_xyxy_vga: np.ndarray,
    transform: LetterboxTransform,
) -> np.ndarray:
    bboxes = np.asarray(bboxes_xyxy_vga, dtype=np.float32)
    if bboxes.ndim != 2 or bboxes.shape[1] != 4:
        raise ValueError(f"bboxes_xyxy_vga must have shape (N,4), got {bboxes.shape}")
    bboxes = bboxes.copy()
    bboxes[:, [0, 2]] = bboxes[:, [0, 2]] * float(transform.scale) + float(transform.pad_x)
    bboxes[:, [1, 3]] = bboxes[:, [1, 3]] * float(transform.scale) + float(transform.pad_y)
    return bboxes


def letterbox_rgb_image(image: Image.Image, dst_w: int, dst_h: int, fill: int = 114) -> Image.Image:
    source = image.convert("RGB")
    transform = make_letterbox_transform(source.width, source.height, dst_w, dst_h)
    resized = source.resize((int(round(source.width * transform.scale)), int(round(source.height * transform.scale))))
    canvas = Image.new("RGB", (dst_w, dst_h), color=(fill, fill, fill))
    canvas.paste(resized, (int(round(transform.pad_x)), int(round(transform.pad_y))))
    return canvas
