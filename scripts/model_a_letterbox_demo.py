"""Minimal coordinate-contract demo for Model A.

The project treats VGA 640x480 as the coordinate source of truth. Model A may
run on a 416x416 letterboxed image, but exported detections must be mapped back
to VGA before they leave the visual module.
"""

from __future__ import annotations

from dataclasses import dataclass


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
    src_w: int = 640,
    src_h: int = 480,
    dst_w: int = 416,
    dst_h: int = 416,
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


def main() -> None:
    transform = make_letterbox_transform()

    # This box corresponds to roughly x=200..440, y=100..340 in VGA coords.
    bbox_416 = (130.0, 117.0, 286.0, 273.0)
    bbox_vga = bbox_model_to_vga(bbox_416, transform)

    print("letterbox:")
    print(f"  src: {transform.src_w}x{transform.src_h}")
    print(f"  dst: {transform.dst_w}x{transform.dst_h}")
    print(f"  scale: {transform.scale:.6f}")
    print(f"  pad_x: {transform.pad_x:.3f}")
    print(f"  pad_y: {transform.pad_y:.3f}")
    print("semantic_det_raw_t sample:")
    print(f"  class_id: 0")
    print(f"  confidence: 0.900000")
    print(f"  bbox_x1: {bbox_vga[0]:.3f}")
    print(f"  bbox_y1: {bbox_vga[1]:.3f}")
    print(f"  bbox_x2: {bbox_vga[2]:.3f}")
    print(f"  bbox_y2: {bbox_vga[3]:.3f}")


if __name__ == "__main__":
    main()
