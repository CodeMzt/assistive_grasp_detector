"""Export immutable Model-A R3 detections and separate GT match residuals.

The raw file is the deployment-facing cascade contract.  It intentionally does
not contain ground truth boxes.  A second file carries GT matching solely for
Model-B training jitter construction and offline cascade evaluation.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import numpy as np

from assistive_grasp_detector.ethossafedet_postprocess import bbox_iou
from assistive_grasp_detector.ethossafedet_v2_manifest import load_v2_manifest_records
from assistive_grasp_detector.ethossafedet_v2_model import EthosSafeDetV2Config, make_ethossafedet_v2
from assistive_grasp_detector.ethossafedet_v2_report import letterbox_rgb_image
from assistive_grasp_detector.ethossafedet_v2_train import decode_v2_outputs
from assistive_grasp_detector.schema import ETHOSSAFEDET_CLASS_NAMES


PREDICTION_SCHEMA = "ethossafedet_v2_r3_raw_predictions_v1"
MATCH_SCHEMA = "ethossafedet_v2_r3_gt_matches_v1"


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as file:
        for block in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def annotation_key(record: dict[str, Any]) -> str:
    """Return the stable camera/image key used by the R3 snapshot."""
    path = Path(str(record["annotation_path"])).with_suffix("")
    return f"{path.parent.name}/{path.name}"


def residual_from_boxes(prediction: list[float], target: list[float]) -> dict[str, float]:
    px1, py1, px2, py2 = (float(v) for v in prediction)
    tx1, ty1, tx2, ty2 = (float(v) for v in target)
    pw, ph = max(px2 - px1, 1e-6), max(py2 - py1, 1e-6)
    tw, th = max(tx2 - tx1, 1e-6), max(ty2 - ty1, 1e-6)
    return {
        "dx_over_gt_w": ((px1 + px2) - (tx1 + tx2)) / (2.0 * tw),
        "dy_over_gt_h": ((py1 + py2) - (ty1 + ty2)) / (2.0 * th),
        "log_w_over_gt_w": float(np.log(pw / tw)),
        "log_h_over_gt_h": float(np.log(ph / th)),
    }


def match_record_detections(
    detections: list[dict[str, Any]],
    targets: list[dict[str, Any]],
    *,
    roi_match_iou: float = 0.10,
    true_positive_iou: float = 0.50,
) -> tuple[list[dict[str, Any]], set[int]]:
    """One-to-one same-class assignment with explicit cascade availability."""
    pairs: list[tuple[float, int, int]] = []
    for target_index, target in enumerate(targets):
        class_id = int(target["class_id"])
        for detection_index, detection in enumerate(detections):
            if int(detection["class_id"]) != class_id:
                continue
            iou = bbox_iou(detection["bbox_xyxy_vga"], target["bbox_xyxy_vga"])
            if iou >= roi_match_iou:
                pairs.append((iou, target_index, detection_index))
    pairs.sort(key=lambda row: (-row[0], row[1], row[2]))
    target_assignment: dict[int, tuple[float, int]] = {}
    used_detections: set[int] = set()
    for iou, target_index, detection_index in pairs:
        if target_index in target_assignment or detection_index in used_detections:
            continue
        target_assignment[target_index] = (float(iou), detection_index)
        used_detections.add(detection_index)

    rows: list[dict[str, Any]] = []
    for target_index, target in enumerate(targets):
        assigned = target_assignment.get(target_index)
        detection = detections[assigned[1]] if assigned is not None else None
        iou = float(assigned[0]) if assigned is not None else 0.0
        row: dict[str, Any] = {
            "instance_id": int(target["instance_id"]),
            "class_id": int(target["class_id"]),
            "class_name": str(target["class_name"]),
            "gt_bbox_xyxy_vga": [float(v) for v in target["bbox_xyxy_vga"]],
            "det_index": int(assigned[1]) if assigned is not None else None,
            "best_iou": iou,
            "roi_input_available": bool(detection is not None and iou >= roi_match_iou),
            "match_iou50": bool(detection is not None and iou >= true_positive_iou),
        }
        if detection is not None:
            row["matched_bbox_xyxy_vga"] = [float(v) for v in detection["bbox_xyxy_vga"]]
            row["matched_score"] = float(detection["score"])
            row["residual"] = residual_from_boxes(row["matched_bbox_xyxy_vga"], row["gt_bbox_xyxy_vga"])
        rows.append(row)
    return rows, used_detections


def _predict_record_on_device(model, torch, record: dict[str, Any], input_size: int, score_threshold: float, nms_iou_threshold: float, device: str):  # type: ignore[no-untyped-def]
    from PIL import Image
    from assistive_grasp_detector.ethossafedet_v2_manifest import resolve_v2_record_image

    with Image.open(resolve_v2_record_image(record)) as image:
        model_image = letterbox_rgb_image(image.convert("RGB"), input_size, input_size)
    array = np.asarray(model_image, dtype=np.uint8)
    tensor = torch.from_numpy(np.ascontiguousarray(np.transpose(array, (2, 0, 1)))).to(device=device, dtype=torch.float32).div(255.0).unsqueeze(0)
    outputs = model(tensor)
    outputs_np = [output.detach().float().cpu().numpy() for output in outputs]
    return decode_v2_outputs(outputs_np, input_size=input_size, score_threshold=score_threshold, nms_iou_threshold=nms_iou_threshold)


def _load_run_model(run: Path, device: str):  # type: ignore[no-untyped-def]
    import torch

    report = json.loads((run / "train_report.json").read_text(encoding="utf-8"))
    checkpoint_path = Path(report["checkpoint"])
    model = make_ethossafedet_v2(
        EthosSafeDetV2Config(input_size=int(report["model"]["input_size"]), width=int(report["model"]["width"]))
    )
    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    model.load_state_dict(checkpoint.get("model_state", checkpoint))
    if device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA prediction export was requested but CUDA is unavailable")
    model.to(device)
    model.eval()
    return torch, report, checkpoint_path, model


def export_r3_predictions(
    run_dir: str | Path,
    out_dir: str | Path,
    *,
    splits: tuple[str, ...] = ("train", "val", "test", "real_scene_holdout"),
    roi_match_iou: float = 0.10,
    device: str = "cuda",
) -> dict[str, Any]:
    run = Path(run_dir).resolve()
    out = Path(out_dir).resolve()
    out.mkdir(parents=True, exist_ok=True)
    torch, train_report, checkpoint_path, model = _load_run_model(run, device)
    manifest = Path(train_report["data"]["manifest"])
    snapshot = manifest.with_name("r3_dataset_snapshot.json")
    if not snapshot.is_file():
        raise FileNotFoundError(f"R3 snapshot is required: {snapshot}")
    snapshot_payload = json.loads(snapshot.read_text(encoding="utf-8"))
    manifest_hash = sha256_file(manifest)
    if str(snapshot_payload.get("manifest_sha256")) != manifest_hash:
        raise ValueError("R3 snapshot manifest hash does not match the formal manifest")
    records = [record for record in load_v2_manifest_records(manifest) if str(record.get("split")) in set(splits)]
    input_size = int(train_report["model"]["input_size"])
    threshold = float(train_report["hyperparameters"]["eval_score_threshold"])
    nms = float(train_report["hyperparameters"]["nms_iou_threshold"])
    common = {
        "model_a_run": run.as_posix(),
        "checkpoint": checkpoint_path.resolve().as_posix(),
        "checkpoint_sha256": sha256_file(checkpoint_path),
        "manifest": manifest.resolve().as_posix(),
        "manifest_sha256": manifest_hash,
        "snapshot": snapshot.resolve().as_posix(),
        "snapshot_sha256": sha256_file(snapshot),
        "input_size": input_size,
        "score_threshold": threshold,
        "nms_iou_threshold": nms,
    }
    raw_path = out / "model_a_r3_predictions_v1.jsonl"
    match_path = out / "model_a_r3_matches_v1.jsonl"
    summary: dict[str, Any] = {"schema_version": MATCH_SCHEMA, **common, "by_split": {}}
    rows_by_split: dict[str, list[dict[str, Any]]] = defaultdict(list)
    with raw_path.open("w", encoding="utf-8", newline="\n") as raw_file, match_path.open("w", encoding="utf-8", newline="\n") as match_file:
        with torch.no_grad():
            for record in records:
                key = annotation_key(record)
                detections = _predict_record_on_device(model, torch, record, input_size, threshold, nms, device)
                raw_detections = [
                    {
                        "det_index": index,
                        "class_id": int(detection["class_id"]),
                        "class_name": ETHOSSAFEDET_CLASS_NAMES[int(detection["class_id"])],
                        "score": float(detection["score"]),
                        "bbox_xyxy_vga": [float(v) for v in detection["bbox_xyxy_vga"]],
                        "orientation_rad": detection.get("orientation_rad"),
                    }
                    for index, detection in enumerate(detections)
                ]
                raw_row = {
                    "schema_version": PREDICTION_SCHEMA,
                    **common,
                    "annotation_key": key,
                    "image_id": str(record["image_id"]),
                    "image_path": str(record["image_path"]),
                    "split": str(record["split"]),
                    "r3_tags": list(record.get("r3_tags", [])),
                    "detections": raw_detections,
                }
                raw_file.write(json.dumps(raw_row, ensure_ascii=False, sort_keys=True) + "\n")
                targets = [obj for obj in record.get("objects", []) if not record.get("negative")]
                matches, used = match_record_detections(raw_detections, targets, roi_match_iou=roi_match_iou)
                true_positive_indices = {int(item["det_index"]) for item in matches if item.get("match_iou50") and item.get("det_index") is not None}
                false_positive_count = sum(1 for index in range(len(raw_detections)) if index not in true_positive_indices)
                match_row = {
                    "schema_version": MATCH_SCHEMA,
                    **common,
                    "annotation_key": key,
                    "image_id": str(record["image_id"]),
                    "image_sha256": next(
                        row["image_sha256"] for row in snapshot_payload["records"] if row["annotation_key"] == key
                    ),
                    "split": str(record["split"]),
                    "r3_tags": list(record.get("r3_tags", [])),
                    "objects": matches,
                    "detector_false_positive_count": false_positive_count,
                    "detector_detection_count": len(raw_detections),
                }
                match_file.write(json.dumps(match_row, ensure_ascii=False, sort_keys=True) + "\n")
                rows_by_split[str(record["split"])].append(match_row)
    for split, rows in sorted(rows_by_split.items()):
        objects = [obj for row in rows for obj in row["objects"]]
        summary["by_split"][split] = {
            "image_count": len(rows),
            "object_count": len(objects),
            "roi_input_available_count": sum(bool(obj["roi_input_available"]) for obj in objects),
            "matched_iou50_count": sum(bool(obj["match_iou50"]) for obj in objects),
            "missed_count": sum(not bool(obj["roi_input_available"]) for obj in objects),
            "false_positive_count": sum(int(row["detector_false_positive_count"]) for row in rows),
            "by_class": dict(Counter(str(obj["class_name"]) for obj in objects)),
        }
    summary["raw_predictions"] = raw_path.as_posix()
    summary["raw_predictions_sha256"] = sha256_file(raw_path)
    summary["matches"] = match_path.as_posix()
    summary["matches_sha256"] = sha256_file(match_path)
    summary_path = out / "model_a_r3_prediction_export_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    return {"ok": True, "raw": raw_path.as_posix(), "matches": match_path.as_posix(), "summary": summary_path.as_posix()}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Export Model-A R3 raw detections and separate GT matches.")
    parser.add_argument("--run", required=True, type=Path)
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument("--splits", default="train,val,test,real_scene_holdout")
    parser.add_argument("--roi-match-iou", type=float, default=0.10)
    parser.add_argument("--device", default="cuda", choices=("cpu", "cuda"))
    args = parser.parse_args(argv)
    splits = tuple(item.strip() for item in args.splits.split(",") if item.strip())
    print(json.dumps(export_r3_predictions(args.run, args.out, splits=splits, roi_match_iou=args.roi_match_iou, device=args.device), ensure_ascii=False))
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
