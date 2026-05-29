"""Run the Model A YOLOv8n PC smoke pipeline.

This script intentionally keeps generated artifacts under runs/ or the model
export location. Those outputs are ignored by git and should be summarized in
the experiment record instead of committed.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import sys
from pathlib import Path
from typing import Any


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def summarize_prediction(results: list[Any]) -> list[dict[str, Any]]:
    summary: list[dict[str, Any]] = []
    for result in results:
        boxes = getattr(result, "boxes", None)
        if boxes is None:
            continue
        names = getattr(result, "names", {})
        for box in boxes:
            class_id = int(box.cls.item())
            confidence = float(box.conf.item())
            xyxy = [float(v) for v in box.xyxy[0].tolist()]
            summary.append(
                {
                    "class_id": class_id,
                    "class_name": names.get(class_id, str(class_id)),
                    "confidence": confidence,
                    "bbox_xyxy_source": xyxy,
                    "semantic_det_raw_t": {
                        "class_id": class_id,
                        "confidence": confidence,
                        "bbox_x1": xyxy[0],
                        "bbox_y1": xyxy[1],
                        "bbox_x2": xyxy[2],
                        "bbox_y2": xyxy[3],
                    },
                }
            )
    summary.sort(key=lambda item: item["confidence"], reverse=True)
    return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--weights", default="yolov8n.pt")
    parser.add_argument("--data", default="coco8.yaml")
    parser.add_argument("--imgsz", type=int, default=416)
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--batch", type=int, default=2)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--source", required=True)
    parser.add_argument("--conf", type=float, default=0.25)
    parser.add_argument("--iou", type=float, default=0.7)
    parser.add_argument("--run-name", default="model_a_yolov8n_pc_export_run_001")
    parser.add_argument("--skip-train", action="store_true")
    parser.add_argument("--export-onnx", action="store_true")
    parser.add_argument("--export-nms-onnx", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()

    try:
        import torch
        import ultralytics
        from ultralytics import YOLO
    except ImportError as exc:
        print("Missing dependency. Install with: python -m pip install ultralytics", file=sys.stderr)
        print(str(exc), file=sys.stderr)
        return 2

    project_dir = (Path.cwd() / "runs" / args.run_name).resolve()
    model = YOLO(args.weights)
    active_weights = Path(args.weights)

    if not args.skip_train:
        train_result = model.train(
            data=args.data,
            epochs=args.epochs,
            imgsz=args.imgsz,
            batch=args.batch,
            device=args.device,
            project=str(project_dir),
            name="train",
            exist_ok=True,
            workers=0,
        )
        save_dir = getattr(train_result, "save_dir", None)
        if save_dir is None and getattr(model, "trainer", None) is not None:
            save_dir = getattr(model.trainer, "save_dir", None)
        if save_dir is None:
            raise RuntimeError("Ultralytics did not expose train save_dir; cannot locate best.pt")
        best = Path(save_dir) / "weights" / "best.pt"
        if best.exists():
            active_weights = best
            model = YOLO(str(active_weights))

    predict_results = model.predict(
        source=args.source,
        imgsz=args.imgsz,
        conf=args.conf,
        iou=args.iou,
        device=args.device,
        project=str(project_dir),
        name="predict",
        exist_ok=True,
        save=True,
    )

    exported: dict[str, dict[str, str]] = {}
    if args.export_onnx:
        onnx_path = Path(model.export(format="onnx", imgsz=args.imgsz, nms=False))
        raw_path = onnx_path.with_name(f"{onnx_path.stem}_raw.onnx")
        shutil.copyfile(onnx_path, raw_path)
        exported["onnx_raw"] = {"path": str(raw_path), "sha256": sha256_file(raw_path)}

    if args.export_nms_onnx:
        onnx_path = Path(model.export(format="onnx", imgsz=args.imgsz, nms=True))
        nms_path = onnx_path.with_name(f"{onnx_path.stem}_nms.onnx")
        shutil.copyfile(onnx_path, nms_path)
        exported["onnx_nms"] = {"path": str(nms_path), "sha256": sha256_file(nms_path)}

    report = {
        "weights": str(active_weights),
        "imgsz": args.imgsz,
        "device": args.device,
        "ultralytics_version": getattr(ultralytics, "__version__", "unknown"),
        "torch_version": getattr(torch, "__version__", "unknown"),
        "prediction_count": len(summarize_prediction(predict_results)),
        "top_predictions": summarize_prediction(predict_results)[:10],
        "exports": exported,
    }
    print(json.dumps(report, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
