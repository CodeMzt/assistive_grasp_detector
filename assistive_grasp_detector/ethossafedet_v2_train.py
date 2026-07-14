"""Training loop for Model A V2 / EthosSafeDetV2."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import platform
import random
import subprocess
import sys
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image, ImageEnhance

from assistive_grasp_detector.coords import bbox_vga_to_model, bboxes_model_to_vga_xyxy, letterbox_rgb_image, make_letterbox_transform
from assistive_grasp_detector.ethossafedet_postprocess import bbox_iou
from assistive_grasp_detector.ethossafedet_v2_manifest import load_v2_manifest_records, resolve_v2_record_image
from assistive_grasp_detector.ethossafedet_v2_model import EthosSafeDetV2Config, make_ethossafedet_v2, parameter_count
from assistive_grasp_detector.schema import ETHOSSAFEDET_CLASS_NAMES, ETHOSSAFEDET_NUM_CLASSES, EXPECTED_VGA_SIZE

V2_STRIDES = (8, 16)


def assign_v2_targets(
    objects: list[dict[str, Any]],
    input_size: int = 320,
    stride: int = 8,
    num_classes: int = ETHOSSAFEDET_NUM_CLASSES,
    source_size: tuple[int, int] = EXPECTED_VGA_SIZE,
    center_radius: float = 1.5,
) -> dict[str, np.ndarray]:
    grid = input_size // stride
    cls_target = np.zeros((num_classes, grid, grid), dtype=np.float32)
    box_target = np.zeros((4, grid, grid), dtype=np.float32)
    ori_target = np.zeros((2, grid, grid), dtype=np.float32)
    positive = np.zeros((grid, grid), dtype=bool)
    ori_mask = np.zeros((grid, grid), dtype=bool)
    owner_area = np.full((grid, grid), np.inf, dtype=np.float32)
    owner_center_distance = np.full((grid, grid), np.inf, dtype=np.float32)
    transform = make_letterbox_transform(source_size[0], source_size[1], input_size, input_size)

    for obj in objects:
        class_id = int(obj["class_id"])
        if class_id < 0 or class_id >= num_classes:
            continue
        x1, y1, x2, y2 = bbox_vga_to_model(tuple(float(v) for v in obj["bbox_xyxy_vga"]), transform)
        cx = (x1 + x2) / 2.0
        cy = (y1 + y2) / 2.0
        area = max(1.0, (x2 - x1) * (y2 - y1))
        radius = float(center_radius) * float(stride)
        gx0 = max(0, int(np.floor(x1 / stride)))
        gx1 = min(grid - 1, int(np.ceil(x2 / stride)))
        gy0 = max(0, int(np.floor(y1 / stride)))
        gy1 = min(grid - 1, int(np.ceil(y2 / stride)))
        for gy in range(gy0, gy1 + 1):
            center_y = (gy + 0.5) * stride
            if center_y < y1 or center_y > y2 or abs(center_y - cy) > radius:
                continue
            for gx in range(gx0, gx1 + 1):
                center_x = (gx + 0.5) * stride
                if center_x < x1 or center_x > x2 or abs(center_x - cx) > radius:
                    continue
                distance = float((center_x - cx) ** 2 + (center_y - cy) ** 2)
                current_area = owner_area[gy, gx]
                if area > current_area:
                    continue
                if np.isclose(area, current_area) and distance >= owner_center_distance[gy, gx]:
                    continue
                cls_target[:, gy, gx] = 0.0
                cls_target[class_id, gy, gx] = 1.0
                box_target[:, gy, gx] = np.asarray(
                    [
                        max(0.0, center_x - x1),
                        max(0.0, center_y - y1),
                        max(0.0, x2 - center_x),
                        max(0.0, y2 - center_y),
                    ],
                    dtype=np.float32,
                )
                positive[gy, gx] = True
                owner_area[gy, gx] = area
                owner_center_distance[gy, gx] = distance
                if bool(obj.get("theta_valid")):
                    ori_target[:, gy, gx] = np.asarray(
                        [
                            float(obj["orientation_sin2theta"]),
                            float(obj["orientation_cos2theta"]),
                        ],
                        dtype=np.float32,
                    )
                    ori_mask[gy, gx] = True
                else:
                    ori_target[:, gy, gx] = 0.0
                    ori_mask[gy, gx] = False
    return {"cls": cls_target, "box": box_target, "positive": positive, "orientation": ori_target, "orientation_mask": ori_mask}


class EthosSafeDetV2Dataset:
    def __init__(self, manifest_path: str | Path, split: str, input_size: int = 320, augment: bool = False, cache_images: bool = True) -> None:
        self.records = [record for record in load_v2_manifest_records(manifest_path) if record.get("split") == split]
        self.input_size = int(input_size)
        self.augment = bool(augment)
        self.cache_images = bool(cache_images)
        if not self.records:
            raise ValueError(f"manifest has no records for split {split!r}")
        self.cache = self._build_cache() if self.cache_images else None

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, index: int):  # type: ignore[no-untyped-def]
        torch, _ = _torch_modules()
        record = self.records[index]
        if self.cache is not None:
            entry = self.cache[index]
            arr_u8 = np.asarray(entry["image"], dtype=np.uint8)
            targets = entry["targets"]
            if self.augment and random.random() < 0.5:
                arr_u8 = np.ascontiguousarray(arr_u8[:, ::-1, :])
                targets = entry["targets_flip"]
            if self.augment:
                arr_u8 = _augment_cached_image(arr_u8)
        else:
            image_path = resolve_v2_record_image(record)
            objects = [dict(obj) for obj in record.get("objects", [])]
            with Image.open(image_path) as image:
                image = image.convert("RGB")
                if self.augment:
                    image, objects = _augment_image_and_objects(image, objects)
                model_image = letterbox_rgb_image(image, self.input_size, self.input_size)
            arr_u8 = np.asarray(model_image, dtype=np.uint8)
            targets = assign_targets_all_scales(objects, input_size=self.input_size)
        chw = np.ascontiguousarray(np.transpose(arr_u8, (2, 0, 1)))
        item = {
            "image": torch.from_numpy(chw).to(dtype=torch.float32).div_(255.0),
            "record_index": torch.tensor(index, dtype=torch.long),
        }
        for name, value in targets.items():
            item[name] = torch.from_numpy(value)
        return item

    def _build_cache(self) -> list[dict[str, Any]]:
        cache: list[dict[str, Any]] = []
        for record in self.records:
            image_path = resolve_v2_record_image(record)
            with Image.open(image_path) as image:
                model_image = letterbox_rgb_image(image.convert("RGB"), self.input_size, self.input_size)
            objects = [dict(obj) for obj in record.get("objects", [])]
            flipped_objects = _flip_objects(objects, width=int(record.get("width", EXPECTED_VGA_SIZE[0])))
            cache.append(
                {
                    "image": np.asarray(model_image, dtype=np.uint8),
                    "targets": assign_targets_all_scales(objects, input_size=self.input_size),
                    "targets_flip": assign_targets_all_scales(flipped_objects, input_size=self.input_size),
                }
            )
        return cache


def assign_targets_all_scales(objects: list[dict[str, Any]], input_size: int = 320) -> dict[str, np.ndarray]:
    result: dict[str, np.ndarray] = {}
    for stride in V2_STRIDES:
        prefix = f"s{stride}"
        targets = assign_v2_targets(objects, input_size=input_size, stride=stride)
        result[f"{prefix}_cls"] = targets["cls"]
        result[f"{prefix}_box"] = targets["box"]
        result[f"{prefix}_positive"] = targets["positive"]
        result[f"{prefix}_orientation"] = targets["orientation"]
        result[f"{prefix}_orientation_mask"] = targets["orientation_mask"]
    return result


def train_ethossafedet_v2(
    manifest_path: str | Path,
    output_dir: str | Path,
    input_size: int = 320,
    width: int = 40,
    epochs: int = 120,
    batch_size: int = 32,
    lr: float = 3e-4,
    weight_decay: float = 1e-4,
    device: str = "cuda",
    seed: int = 0,
    eval_score_threshold: float = 0.25,
    nms_iou_threshold: float = 0.5,
    eval_every: int = 1,
    eval_limit: int | None = None,
    num_workers: int = 0,
    amp: bool = True,
    cache_images: bool = True,
    min_int8_weight_bytes: int = 100 * 1024,
    max_int8_weight_bytes: int = 300 * 1024,
    init_checkpoint: str | Path | None = None,
) -> dict[str, Any]:
    torch, nn = _torch_modules()
    from torch.utils.data import DataLoader, WeightedRandomSampler

    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    manifest = Path(manifest_path)
    records = load_v2_manifest_records(manifest)
    _validate_records_for_training(records)
    _seed_everything(torch, seed)

    train_ds = EthosSafeDetV2Dataset(manifest, "train", input_size=input_size, augment=True, cache_images=cache_images)
    val_ds = EthosSafeDetV2Dataset(manifest, "val", input_size=input_size, augment=False, cache_images=cache_images)
    test_ds = EthosSafeDetV2Dataset(manifest, "test", input_size=input_size, augment=False, cache_images=cache_images)
    class_weights_np, sampler_weights_np = _class_and_sampler_weights(train_ds.records)
    class_weights = torch.tensor(class_weights_np, dtype=torch.float32, device=device)
    sampler = WeightedRandomSampler(
        torch.tensor(sampler_weights_np, dtype=torch.double),
        num_samples=len(train_ds),
        replacement=True,
        generator=torch.Generator().manual_seed(int(seed)),
    )
    train_loader = DataLoader(train_ds, batch_size=batch_size, sampler=sampler, num_workers=num_workers)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=num_workers)

    config = EthosSafeDetV2Config(input_size=input_size, num_classes=ETHOSSAFEDET_NUM_CLASSES, width=width)
    model = make_ethossafedet_v2(config).to(device)
    if init_checkpoint is not None:
        checkpoint = torch.load(Path(init_checkpoint), map_location=device)
        model.load_state_dict(checkpoint.get("model_state", checkpoint))
    params = parameter_count(model)
    estimated_int8_bytes = params
    if estimated_int8_bytes < int(min_int8_weight_bytes) or estimated_int8_bytes > int(max_int8_weight_bytes):
        raise ValueError(
            f"estimated int8 weight bytes {estimated_int8_bytes} outside target "
            f"{int(min_int8_weight_bytes)}..{int(max_int8_weight_bytes)}"
        )

    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    cls_loss_fn = nn.BCEWithLogitsLoss(reduction="none")
    box_loss_fn = nn.SmoothL1Loss(reduction="none")
    ori_loss_fn = nn.MSELoss(reduction="none")
    use_amp = bool(amp and str(device).startswith("cuda") and torch.cuda.is_available())
    scaler = _make_grad_scaler(torch, use_amp)

    history: list[dict[str, Any]] = []
    best_state: dict[str, Any] | None = None
    best_metric = -float("inf")
    best_metric_name = "val_class_recall_iou_theta"
    best_epoch = 0
    best_live_checkpoint = out / "ethossafedet_v2_best_live.pt"
    last_live_checkpoint = out / "ethossafedet_v2_last_live.pt"

    for epoch in range(int(epochs)):
        train_metrics = _run_epoch(
            model,
            train_loader,
            optimizer,
            cls_loss_fn,
            box_loss_fn,
            ori_loss_fn,
            class_weights=class_weights,
            device=device,
            train=True,
            use_amp=use_amp,
            scaler=scaler,
        )
        val_loss_metrics = _run_epoch(
            model,
            val_loader,
            None,
            cls_loss_fn,
            box_loss_fn,
            ori_loss_fn,
            class_weights=class_weights,
            device=device,
            train=False,
            use_amp=use_amp,
            scaler=None,
        )
        should_eval = int(eval_every) <= 1 or (epoch + 1) % int(eval_every) == 0 or epoch + 1 == int(epochs)
        val_det_metrics = (
            evaluate_v2_model(
                model,
                val_ds,
                device=device,
                input_size=input_size,
                score_threshold=eval_score_threshold,
                nms_iou_threshold=nms_iou_threshold,
                batch_size=batch_size,
                num_workers=num_workers,
                limit=eval_limit,
                use_amp=use_amp,
            )
            if should_eval
            else _empty_eval_metrics()
        )
        row = {
            "epoch": epoch + 1,
            **{f"train_{key}": value for key, value in train_metrics.items()},
            **{f"val_{key}": value for key, value in val_loss_metrics.items()},
            **val_det_metrics,
        }
        history.append(row)
        _write_history_csv(out / "train_history_live.csv", history)
        print(
            json.dumps(
                {
                    "epoch": row["epoch"],
                    "train_loss": row.get("train_loss"),
                    "val_loss": row.get("val_loss"),
                    "recall50": row.get("recall50"),
                    "best_iou_mean": row.get("best_iou_mean"),
                    "theta_abs_error_rad_mean": row.get("theta_abs_error_rad_mean"),
                },
                ensure_ascii=False,
            ),
            flush=True,
        )
        metric = _checkpoint_metric(row)
        if metric >= best_metric:
            best_metric = metric
            best_epoch = epoch + 1
            best_state = _clone_state_dict(model.state_dict())
            _save_live_checkpoint(
                torch,
                best_live_checkpoint,
                model.state_dict(),
                input_size=input_size,
                width=width,
                epoch=epoch + 1,
                metric=metric,
                metric_name=best_metric_name,
                row=row,
                history=history,
                class_weights=class_weights_np,
            )
        _save_live_checkpoint(
            torch,
            last_live_checkpoint,
            model.state_dict(),
            input_size=input_size,
            width=width,
            epoch=epoch + 1,
            metric=metric,
            metric_name=best_metric_name,
            row=row,
            history=history,
            class_weights=class_weights_np,
        )

    last_state = _clone_state_dict(model.state_dict())
    if best_state is None:
        best_state = last_state
        best_epoch = int(epochs)

    best_checkpoint = out / "ethossafedet_v2_best.pt"
    last_checkpoint = out / "ethossafedet_v2_last.pt"
    checkpoint_common = {
        "model_id": "EthosSafeDetV2",
        "config": {
            "input_size": int(input_size),
            "num_classes": ETHOSSAFEDET_NUM_CLASSES,
            "width": int(width),
        },
        "history": history,
        "best_epoch": best_epoch,
        "best_metric": best_metric,
        "best_metric_name": best_metric_name,
        "class_weights": class_weights_np.tolist(),
        "class_names": list(ETHOSSAFEDET_CLASS_NAMES),
    }
    torch.save({**checkpoint_common, "model_state": best_state}, best_checkpoint)
    torch.save({**checkpoint_common, "model_state": last_state}, last_checkpoint)

    model.load_state_dict(best_state)
    test_metrics = evaluate_v2_model(
        model,
        test_ds,
        device=device,
        input_size=input_size,
        score_threshold=eval_score_threshold,
        nms_iou_threshold=nms_iou_threshold,
        batch_size=batch_size,
        num_workers=num_workers,
        limit=None,
        use_amp=use_amp,
    )

    report_paths = {
        "json": str(out / "train_report.json"),
        "csv": str(out / "train_history.csv"),
        "markdown": str(out / "train_report.md"),
    }
    report = _build_train_report(
        manifest_path=manifest,
        records=records,
        output_dir=out,
        config=config,
        model=model,
        class_weights=class_weights_np,
        sampler_weights=sampler_weights_np,
        epochs=int(epochs),
        batch_size=int(batch_size),
        lr=float(lr),
        weight_decay=float(weight_decay),
        device=device,
        seed=int(seed),
        eval_score_threshold=float(eval_score_threshold),
        nms_iou_threshold=float(nms_iou_threshold),
        eval_every=int(eval_every),
        eval_limit=eval_limit,
        num_workers=int(num_workers),
        amp=use_amp,
        cache_images=cache_images,
        min_int8_weight_bytes=int(min_int8_weight_bytes),
        max_int8_weight_bytes=int(max_int8_weight_bytes),
        init_checkpoint=init_checkpoint,
        history=history,
        best_epoch=best_epoch,
        best_metric=best_metric,
        best_metric_name=best_metric_name,
        checkpoint=best_checkpoint,
        last_checkpoint=last_checkpoint,
        test_metrics=test_metrics,
    )
    report["report_paths"] = report_paths
    Path(report_paths["json"]).write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    _write_history_csv(Path(report_paths["csv"]), history)
    Path(report_paths["markdown"]).write_text(_format_train_markdown(report), encoding="utf-8")
    return report


def evaluate_v2_model(
    model,  # type: ignore[no-untyped-def]
    dataset: EthosSafeDetV2Dataset,
    device: str,
    input_size: int,
    score_threshold: float,
    nms_iou_threshold: float,
    batch_size: int,
    num_workers: int,
    limit: int | None,
    use_amp: bool,
) -> dict[str, Any]:
    torch, _ = _torch_modules()
    from torch.utils.data import DataLoader

    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    model.eval()
    total_gt = 0
    recall_hits = 0
    best_ious: list[float] = []
    primary_count = 0
    primary_class_matches = 0
    primary_ious: list[float] = []
    theta_errors: list[float] = []
    negative_image_count = 0
    negative_image_false_positive_count = 0
    negative_detection_count = 0
    negative_scores_by_class: dict[str, list[float]] = {name: [] for name in ETHOSSAFEDET_CLASS_NAMES}
    per_class: dict[str, dict[str, Any]] = {
        name: {"gt": 0, "recall50": 0, "best_iou_sum": 0.0, "theta_count": 0, "theta_abs_error_sum": 0.0}
        for name in ETHOSSAFEDET_CLASS_NAMES
    }
    count = 0
    with torch.no_grad():
        for batch in loader:
            images = batch["image"].to(device)
            with _autocast_context(torch, use_amp):
                outputs = model(images)
            outputs_np = [output.detach().float().cpu().numpy() for output in outputs]
            record_indices = batch["record_index"].detach().cpu().numpy().tolist()
            for batch_index, record_index in enumerate(record_indices):
                if limit is not None and int(limit) > 0 and count >= int(limit):
                    break
                record = dataset.records[int(record_index)]
                detections = decode_v2_outputs(
                    [value[batch_index : batch_index + 1] for value in outputs_np],
                    input_size=input_size,
                    score_threshold=score_threshold,
                    nms_iou_threshold=nms_iou_threshold,
                )
                objects = [obj for obj in record.get("objects", []) if not record.get("negative")]
                total_gt += len(objects)
                count += 1
                if not objects:
                    negative_image_count += 1
                    if detections:
                        negative_image_false_positive_count += 1
                    negative_detection_count += len(detections)
                    for detection in detections:
                        class_id = int(detection["class_id"])
                        if 0 <= class_id < len(ETHOSSAFEDET_CLASS_NAMES):
                            negative_scores_by_class[ETHOSSAFEDET_CLASS_NAMES[class_id]].append(float(detection["score"]))
                primary = _primary_object(objects)
                if primary is not None:
                    primary_count += 1
                    if detections:
                        top = detections[0]
                        if int(top["class_id"]) == int(primary["class_id"]):
                            primary_class_matches += 1
                        primary_ious.append(bbox_iou(top["bbox_xyxy_vga"], primary["bbox_xyxy_vga"]))
                    else:
                        primary_ious.append(0.0)
                for obj in objects:
                    class_id = int(obj["class_id"])
                    class_name = ETHOSSAFEDET_CLASS_NAMES[class_id]
                    per_class[class_name]["gt"] += 1
                    same_class = [det for det in detections if int(det["class_id"]) == class_id]
                    best = max(same_class, key=lambda det: bbox_iou(det["bbox_xyxy_vga"], obj["bbox_xyxy_vga"]), default=None)
                    best_iou = bbox_iou(best["bbox_xyxy_vga"], obj["bbox_xyxy_vga"]) if best is not None else 0.0
                    best_ious.append(best_iou)
                    per_class[class_name]["best_iou_sum"] += best_iou
                    if best_iou >= 0.5:
                        recall_hits += 1
                        per_class[class_name]["recall50"] += 1
                    if best is not None and best_iou >= 0.3 and obj.get("theta_valid") and best.get("orientation_rad") is not None:
                        err = orientation_abs_error(float(best["orientation_rad"]), float(obj["orientation_rad"]))
                        theta_errors.append(err)
                        per_class[class_name]["theta_count"] += 1
                        per_class[class_name]["theta_abs_error_sum"] += err
            if limit is not None and int(limit) > 0 and count >= int(limit):
                break
    per_class_out: dict[str, dict[str, Any]] = {}
    for name, row in per_class.items():
        gt = int(row["gt"])
        theta_count = int(row["theta_count"])
        per_class_out[name] = {
            "gt": gt,
            "recall50": (float(row["recall50"]) / float(gt)) if gt else None,
            "best_iou_mean": (float(row["best_iou_sum"]) / float(gt)) if gt else None,
            "theta_abs_error_rad_mean": (float(row["theta_abs_error_sum"]) / float(theta_count)) if theta_count else None,
            "theta_count": theta_count,
        }
    return {
        "eval_count": count,
        "gt_count": total_gt,
        "primary_class_acc": (primary_class_matches / primary_count) if primary_count else None,
        "primary_iou_mean": float(np.mean(primary_ious)) if primary_ious else None,
        "recall50": (recall_hits / total_gt) if total_gt else None,
        "best_iou_mean": float(np.mean(best_ious)) if best_ious else None,
        "theta_abs_error_rad_mean": float(np.mean(theta_errors)) if theta_errors else None,
        "theta_eval_count": len(theta_errors),
        "per_class": per_class_out,
        "negative_image_count": negative_image_count,
        "negative_image_false_positive_count": negative_image_false_positive_count,
        "negative_image_false_positive_rate": (
            float(negative_image_false_positive_count) / float(negative_image_count) if negative_image_count else None
        ),
        "negative_detection_count": negative_detection_count,
        "negative_per_class": {
            name: {
                "prediction_count": len(scores),
                "score_mean": float(np.mean(scores)) if scores else None,
                "score_max": float(np.max(scores)) if scores else None,
            }
            for name, scores in negative_scores_by_class.items()
        },
    }


def decode_v2_outputs(
    outputs: list[np.ndarray],
    input_size: int = 320,
    source_size: tuple[int, int] = EXPECTED_VGA_SIZE,
    score_threshold: float = 0.25,
    pre_nms_top_k: int = 160,
    nms_iou_threshold: float = 0.5,
) -> list[dict[str, Any]]:
    if len(outputs) != 6:
        raise ValueError(f"expected 6 V2 outputs, got {len(outputs)}")
    groups = [
        (8, outputs[0], outputs[1], outputs[2]),
        (16, outputs[3], outputs[4], outputs[5]),
    ]
    transform = make_letterbox_transform(source_size[0], source_size[1], input_size, input_size)
    candidates: list[dict[str, Any]] = []
    for stride, cls_logits, box_ltrb, orientation in groups:
        cls = _canonical_chw(cls_logits)
        box = _canonical_chw(box_ltrb)
        ori = _canonical_chw(orientation)
        scores = sigmoid(cls)
        flat = scores.reshape(-1)
        if flat.size == 0:
            continue
        top_k = min(pre_nms_top_k, int(flat.size))
        top_indices = np.argpartition(-flat, kth=top_k - 1)[:top_k]
        top_indices = top_indices[np.argsort(-flat[top_indices])]
        class_count, height, width = scores.shape
        boxes_model: list[tuple[float, float, float, float]] = []
        pending: list[dict[str, Any]] = []
        for flat_index in top_indices.tolist():
            score = float(flat[flat_index])
            if score < score_threshold:
                continue
            class_id = flat_index // (height * width)
            rem = flat_index % (height * width)
            y = rem // width
            x = rem % width
            if class_id < 0 or class_id >= class_count:
                continue
            center_x = (float(x) + 0.5) * float(stride)
            center_y = (float(y) + 0.5) * float(stride)
            left, top, right, bottom = [float(v) for v in box[:, y, x]]
            model_box = (
                max(0.0, center_x - max(0.0, left)),
                max(0.0, center_y - max(0.0, top)),
                min(float(input_size), center_x + max(0.0, right)),
                min(float(input_size), center_y + max(0.0, bottom)),
            )
            if model_box[0] >= model_box[2] or model_box[1] >= model_box[3]:
                continue
            sin2 = float(ori[0, y, x])
            cos2 = float(ori[1, y, x])
            orientation_rad = None
            if abs(sin2) + abs(cos2) > 1e-6:
                orientation_rad = 0.5 * math.atan2(sin2, cos2)
            boxes_model.append(model_box)
            pending.append(
                {
                    "class_id": int(class_id),
                    "score": score,
                    "bbox_xyxy_model": [float(v) for v in model_box],
                    "stride": int(stride),
                    "grid_y": int(y),
                    "grid_x": int(x),
                    "orientation_rad": orientation_rad,
                    "orientation_sin2theta": sin2,
                    "orientation_cos2theta": cos2,
                }
            )
        if pending:
            boxes_vga = bboxes_model_to_vga_xyxy(np.asarray(boxes_model, dtype=np.float32), transform)
            for item, bbox in zip(pending, boxes_vga):
                item["bbox_xyxy_vga"] = [float(v) for v in bbox]
                candidates.append(item)
    candidates.sort(key=lambda row: float(row["score"]), reverse=True)
    return _classwise_nms(candidates, nms_iou_threshold)


def orientation_abs_error(pred: float, target: float) -> float:
    return abs(((float(pred) - float(target) + math.pi / 2.0) % math.pi) - math.pi / 2.0)


def sigmoid(value: np.ndarray) -> np.ndarray:
    value = np.asarray(value, dtype=np.float32)
    return 1.0 / (1.0 + np.exp(-value))


def _run_epoch(
    model,  # type: ignore[no-untyped-def]
    loader,  # type: ignore[no-untyped-def]
    optimizer,  # type: ignore[no-untyped-def]
    cls_loss_fn,  # type: ignore[no-untyped-def]
    box_loss_fn,  # type: ignore[no-untyped-def]
    ori_loss_fn,  # type: ignore[no-untyped-def]
    class_weights,  # type: ignore[no-untyped-def]
    device: str,
    train: bool,
    use_amp: bool,
    scaler,  # type: ignore[no-untyped-def]
) -> dict[str, Any]:
    torch, _ = _torch_modules()
    model.train(mode=train)
    totals = {"loss": 0.0, "cls_loss": 0.0, "box_loss": 0.0, "ori_loss": 0.0, "samples": 0, "positive_cells": 0, "theta_cells": 0}
    context = torch.enable_grad() if train else torch.no_grad()
    with context:
        for batch in loader:
            images = batch["image"].to(device)
            with _autocast_context(torch, use_amp):
                outputs = model(images)
                loss, loss_parts = _batch_losses(outputs, batch, cls_loss_fn, box_loss_fn, ori_loss_fn, class_weights, device)
            if not torch.isfinite(loss):
                raise RuntimeError("non-finite loss encountered")
            if train:
                optimizer.zero_grad()
                if scaler is not None and use_amp:
                    scaler.scale(loss).backward()
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    loss.backward()
                    optimizer.step()
            batch_size = int(images.shape[0])
            totals["loss"] += float(loss.detach().cpu()) * batch_size
            for key in ("cls_loss", "box_loss", "ori_loss"):
                totals[key] += float(loss_parts[key].detach().cpu()) * batch_size
            totals["samples"] += batch_size
            totals["positive_cells"] += int(loss_parts["positive_cells"])
            totals["theta_cells"] += int(loss_parts["theta_cells"])
    samples = max(1, int(totals["samples"]))
    return {
        "loss": totals["loss"] / samples,
        "cls_loss": totals["cls_loss"] / samples,
        "box_loss": totals["box_loss"] / samples,
        "ori_loss": totals["ori_loss"] / samples,
        "positive_cells": int(totals["positive_cells"]),
        "theta_cells": int(totals["theta_cells"]),
    }


def _batch_losses(outputs, batch: dict[str, Any], cls_loss_fn, box_loss_fn, ori_loss_fn, class_weights, device: str):  # type: ignore[no-untyped-def]
    scale_map = {
        "s8": outputs[:3],
        "s16": outputs[3:],
    }
    total = None
    parts = {"cls_loss": 0.0, "box_loss": 0.0, "ori_loss": 0.0, "positive_cells": 0, "theta_cells": 0}
    for prefix, (cls_logits, box_ltrb, orientation) in scale_map.items():
        cls_target = batch[f"{prefix}_cls"].to(device)
        box_target = batch[f"{prefix}_box"].to(device)
        positive = batch[f"{prefix}_positive"].to(device)
        ori_target = batch[f"{prefix}_orientation"].to(device)
        ori_mask = batch[f"{prefix}_orientation_mask"].to(device)
        cls_loss = _hard_negative_focal_loss(cls_loss_fn(cls_logits, cls_target), cls_logits, cls_target, class_weights)
        box_loss = _box_loss(box_ltrb, box_target, positive, box_loss_fn, stride=int(prefix[1:]))
        ori_loss = _orientation_loss(orientation, ori_target, ori_mask, ori_loss_fn)
        scale_loss = cls_loss + box_loss + ori_loss
        total = scale_loss if total is None else total + scale_loss
        parts["cls_loss"] = parts["cls_loss"] + cls_loss
        parts["box_loss"] = parts["box_loss"] + box_loss
        parts["ori_loss"] = parts["ori_loss"] + ori_loss
        parts["positive_cells"] += int(positive.detach().cpu().sum())
        parts["theta_cells"] += int(ori_mask.detach().cpu().sum())
    return total, parts


def _hard_negative_focal_loss(loss, logits, targets, class_weights, gamma: float = 2.0, alpha: float = 0.25, hard_negative_ratio: int = 20):  # type: ignore[no-untyped-def]
    torch, _ = _torch_modules()
    probs = torch.sigmoid(logits)
    pt = torch.where(targets > 0.0, probs, 1.0 - probs)
    alpha_t = torch.where(targets > 0.0, alpha, 1.0 - alpha)
    focal = loss * alpha_t * torch.pow(1.0 - pt, gamma)
    weights = torch.ones_like(focal)
    weights = torch.where(targets > 0.0, class_weights.view(1, -1, 1, 1).to(focal.device), weights)
    focal = focal * weights
    positive = targets > 0.0
    positive_losses = focal[positive]
    negative_losses = focal[~positive].flatten()
    positive_count = int(positive_losses.numel())
    negative_count = min(int(negative_losses.numel()), max(256, positive_count * int(hard_negative_ratio)))
    if negative_count > 0:
        negative_losses = torch.topk(negative_losses, k=negative_count).values
        total = positive_losses.sum() + negative_losses.sum()
    else:
        total = positive_losses.sum()
    return total / float(max(1, positive_count + negative_count))


def _box_loss(pred_ltrb, target_ltrb, positive, box_loss_fn, stride: int):  # type: ignore[no-untyped-def]
    if not bool(positive.any()):
        return pred_ltrb.sum() * 0.0
    pos = positive.unsqueeze(1).expand_as(pred_ltrb)
    smooth_l1 = box_loss_fn(pred_ltrb[pos], target_ltrb[pos]).mean()
    return smooth_l1 + 10.0 * _positive_ltrb_iou_loss(pred_ltrb, target_ltrb, positive, stride=stride)


def _orientation_loss(pred, target, mask, ori_loss_fn):  # type: ignore[no-untyped-def]
    if not bool(mask.any()):
        return pred.sum() * 0.0
    pos = mask.unsqueeze(1).expand_as(pred)
    return ori_loss_fn(pred[pos], target[pos]).mean()


def _positive_ltrb_iou_loss(pred_ltrb, target_ltrb, positive, stride: int):  # type: ignore[no-untyped-def]
    torch, _ = _torch_modules()
    if not bool(positive.any()):
        return pred_ltrb.sum() * 0.0
    batch, _, height, width = pred_ltrb.shape
    ys = torch.arange(height, device=pred_ltrb.device, dtype=pred_ltrb.dtype)
    xs = torch.arange(width, device=pred_ltrb.device, dtype=pred_ltrb.dtype)
    grid_y, grid_x = torch.meshgrid(ys, xs, indexing="ij")
    center_x = ((grid_x + 0.5) * float(stride)).unsqueeze(0).expand(batch, -1, -1)
    center_y = ((grid_y + 0.5) * float(stride)).unsqueeze(0).expand(batch, -1, -1)
    pos = positive.bool()
    pred = torch.clamp(pred_ltrb, min=0.0)
    target = torch.clamp(target_ltrb, min=0.0)
    pred_boxes = torch.stack(
        [center_x[pos] - pred[:, 0][pos], center_y[pos] - pred[:, 1][pos], center_x[pos] + pred[:, 2][pos], center_y[pos] + pred[:, 3][pos]],
        dim=1,
    )
    target_boxes = torch.stack(
        [center_x[pos] - target[:, 0][pos], center_y[pos] - target[:, 1][pos], center_x[pos] + target[:, 2][pos], center_y[pos] + target[:, 3][pos]],
        dim=1,
    )
    inter_x1 = torch.maximum(pred_boxes[:, 0], target_boxes[:, 0])
    inter_y1 = torch.maximum(pred_boxes[:, 1], target_boxes[:, 1])
    inter_x2 = torch.minimum(pred_boxes[:, 2], target_boxes[:, 2])
    inter_y2 = torch.minimum(pred_boxes[:, 3], target_boxes[:, 3])
    inter = torch.clamp(inter_x2 - inter_x1, min=0.0) * torch.clamp(inter_y2 - inter_y1, min=0.0)
    pred_area = torch.clamp(pred_boxes[:, 2] - pred_boxes[:, 0], min=0.0) * torch.clamp(pred_boxes[:, 3] - pred_boxes[:, 1], min=0.0)
    target_area = torch.clamp(target_boxes[:, 2] - target_boxes[:, 0], min=0.0) * torch.clamp(target_boxes[:, 3] - target_boxes[:, 1], min=0.0)
    union = torch.clamp(pred_area + target_area - inter, min=1e-6)
    return (1.0 - inter / union).mean()


def _class_and_sampler_weights(records: list[dict[str, Any]]) -> tuple[np.ndarray, np.ndarray]:
    counts = Counter()
    for record in records:
        for obj in record.get("objects", []):
            counts[int(obj["class_id"])] += 1
    max_count = max(counts.values()) if counts else 1
    class_weights = np.ones((ETHOSSAFEDET_NUM_CLASSES,), dtype=np.float32)
    for class_id in range(ETHOSSAFEDET_NUM_CLASSES):
        count = max(1, counts.get(class_id, 0))
        class_weights[class_id] = float(np.sqrt(float(max_count) / float(count)))
    class_weights = np.clip(class_weights, 0.6, 2.5)
    class_weights = class_weights / float(np.mean(class_weights))
    sampler_weights: list[float] = []
    for record in records:
        class_ids = {int(obj["class_id"]) for obj in record.get("objects", [])}
        class_weight = float(max([class_weights[class_id] for class_id in class_ids], default=1.0))
        scene_multiplier = float(record.get("sampler_multiplier", 1.0))
        if not math.isfinite(scene_multiplier) or scene_multiplier <= 0.0:
            raise ValueError(f"invalid sampler_multiplier {scene_multiplier!r}")
        sampler_weights.append(min(3.0, class_weight * scene_multiplier))
    return class_weights.astype(np.float32), np.asarray(sampler_weights, dtype=np.float64)


def _save_live_checkpoint(
    torch,  # type: ignore[no-untyped-def]
    path: Path,
    state_dict: dict[str, Any],
    *,
    input_size: int,
    width: int,
    epoch: int,
    metric: float,
    metric_name: str,
    row: dict[str, Any],
    history: list[dict[str, Any]],
    class_weights: np.ndarray,
) -> None:
    torch.save(
        {
            "model_id": "EthosSafeDetV2",
            "config": {
                "input_size": int(input_size),
                "num_classes": ETHOSSAFEDET_NUM_CLASSES,
                "width": int(width),
            },
            "model_state": _clone_state_dict(state_dict),
            "epoch": int(epoch),
            "best_epoch": int(epoch),
            "best_metric": float(metric),
            "best_metric_name": metric_name,
            "last_row": row,
            "history": list(history),
            "class_weights": class_weights.tolist(),
            "class_names": list(ETHOSSAFEDET_CLASS_NAMES),
        },
        path,
    )


def _validate_records_for_training(records: list[dict[str, Any]]) -> None:
    split_counts = Counter(str(record.get("split")) for record in records)
    for split in ("train", "val", "test"):
        if split_counts.get(split, 0) <= 0:
            raise ValueError(f"manifest split {split!r} is empty")
    by_split: dict[str, Counter[int]] = defaultdict(Counter)
    for record in records:
        for obj in record.get("objects", []):
            by_split[str(record["split"])][int(obj["class_id"])] += 1
            if obj.get("theta_valid") and not {"orientation_sin2theta", "orientation_cos2theta", "orientation_rad"}.issubset(obj):
                raise ValueError("theta_valid object is missing orientation fields")
    for split in ("train", "val", "test"):
        for class_id, name in enumerate(ETHOSSAFEDET_CLASS_NAMES):
            if by_split[split].get(class_id, 0) <= 0:
                raise ValueError(f"class {name!r} has no objects in split {split!r}")


def _flip_objects(objects: list[dict[str, Any]], width: int) -> list[dict[str, Any]]:
    flipped: list[dict[str, Any]] = []
    for obj in objects:
        item = dict(obj)
        x1, y1, x2, y2 = [float(v) for v in item["bbox_xyxy_vga"]]
        item["bbox_xyxy_vga"] = [float(width) - x2, y1, float(width) - x1, y2]
        if item.get("theta_valid"):
            theta = math.pi - float(item["orientation_rad"])
            item["orientation_rad"] = theta
            item["orientation_sin2theta"] = math.sin(2.0 * theta)
            item["orientation_cos2theta"] = math.cos(2.0 * theta)
        flipped.append(item)
    return flipped


def _augment_image_and_objects(image: Image.Image, objects: list[dict[str, Any]]) -> tuple[Image.Image, list[dict[str, Any]]]:
    if random.random() < 0.5:
        image = image.transpose(Image.Transpose.FLIP_LEFT_RIGHT)
        objects = _flip_objects(objects, width=image.width)
    factor = random.uniform(0.85, 1.15)
    image = ImageEnhance.Brightness(image).enhance(factor)
    image = ImageEnhance.Contrast(image).enhance(random.uniform(0.90, 1.10))
    return image, objects


def _augment_cached_image(image: np.ndarray) -> np.ndarray:
    arr = image.astype(np.float32)
    arr *= random.uniform(0.85, 1.15)
    mean = arr.mean(axis=(0, 1), keepdims=True)
    arr = (arr - mean) * random.uniform(0.90, 1.10) + mean
    return np.clip(arr, 0.0, 255.0).astype(np.uint8)


def _primary_object(objects: list[dict[str, Any]]) -> dict[str, Any] | None:
    if not objects:
        return None
    return max(objects, key=lambda obj: _bbox_area(obj.get("bbox_xyxy_vga", [])))


def _bbox_area(bbox: Any) -> float:
    if not isinstance(bbox, list) or len(bbox) != 4:
        return 0.0
    x1, y1, x2, y2 = [float(v) for v in bbox]
    return max(0.0, x2 - x1) * max(0.0, y2 - y1)


def _classwise_nms(candidates: list[dict[str, Any]], iou_threshold: float) -> list[dict[str, Any]]:
    kept: list[dict[str, Any]] = []
    for class_id in sorted({int(candidate["class_id"]) for candidate in candidates}):
        group = [candidate for candidate in candidates if int(candidate["class_id"]) == class_id]
        group.sort(key=lambda item: float(item["score"]), reverse=True)
        while group:
            best = group.pop(0)
            kept.append(best)
            group = [candidate for candidate in group if bbox_iou(best["bbox_xyxy_vga"], candidate["bbox_xyxy_vga"]) < iou_threshold]
    kept.sort(key=lambda item: float(item["score"]), reverse=True)
    return kept


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


def _checkpoint_metric(row: dict[str, Any]) -> float:
    recall = row.get("recall50")
    iou = row.get("best_iou_mean")
    theta = row.get("theta_abs_error_rad_mean")
    theta_score = 0.0 if theta is None else max(0.0, 1.0 - float(theta) / math.pi)
    if recall is not None and iou is not None:
        return float(recall) + float(iou) + float(theta_score)
    val_loss = row.get("val_loss")
    return -float(val_loss) if val_loss is not None else -float(row.get("train_loss", 0.0))


def _empty_eval_metrics() -> dict[str, Any]:
    return {
        "eval_count": 0,
        "gt_count": 0,
        "primary_class_acc": None,
        "primary_iou_mean": None,
        "recall50": None,
        "best_iou_mean": None,
        "theta_abs_error_rad_mean": None,
        "theta_eval_count": 0,
        "per_class": {},
        "negative_image_count": 0,
        "negative_image_false_positive_count": 0,
        "negative_image_false_positive_rate": None,
        "negative_detection_count": 0,
        "negative_per_class": {},
    }


def _build_train_report(
    manifest_path: Path,
    records: list[dict[str, Any]],
    output_dir: Path,
    config: EthosSafeDetV2Config,
    model,  # type: ignore[no-untyped-def]
    class_weights: np.ndarray,
    sampler_weights: np.ndarray,
    epochs: int,
    batch_size: int,
    lr: float,
    weight_decay: float,
    device: str,
    seed: int,
    eval_score_threshold: float,
    nms_iou_threshold: float,
    eval_every: int,
    eval_limit: int | None,
    num_workers: int,
    amp: bool,
    cache_images: bool,
    min_int8_weight_bytes: int,
    max_int8_weight_bytes: int,
    init_checkpoint: str | Path | None,
    history: list[dict[str, Any]],
    best_epoch: int,
    best_metric: float,
    best_metric_name: str,
    checkpoint: Path,
    last_checkpoint: Path,
    test_metrics: dict[str, Any],
) -> dict[str, Any]:
    return {
        "schema_version": "ethossafedet_v2_train_report_v1",
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "model_id": "EthosSafeDetV2",
        "output_dir": output_dir.resolve().as_posix(),
        "git": _git_summary(),
        "environment": _environment(device),
        "data": _manifest_summary(manifest_path, records),
        "model": {
            "input_size": config.input_size,
            "num_classes": config.num_classes,
            "class_names": list(ETHOSSAFEDET_CLASS_NAMES),
            "width": config.width,
            "strides": list(V2_STRIDES),
            "parameter_count": parameter_count(model),
            "estimated_int8_weight_bytes": parameter_count(model),
            "estimated_fp32_weight_bytes": parameter_count(model) * 4,
            "output_names": [
                "s8_cls_logits",
                "s8_box_ltrb",
                "s8_orientation",
                "s16_cls_logits",
                "s16_box_ltrb",
                "s16_orientation",
            ],
        },
        "class_balance": {
            "class_weights": {name: float(class_weights[i]) for i, name in enumerate(ETHOSSAFEDET_CLASS_NAMES)},
            "sampler_weight_min": float(np.min(sampler_weights)) if sampler_weights.size else 0.0,
            "sampler_weight_max": float(np.max(sampler_weights)) if sampler_weights.size else 0.0,
            "strategy": "image_weighted_sampler_plus_inverse_sqrt_positive_class_loss_weights",
        },
        "hyperparameters": {
            "epochs": epochs,
            "batch_size": batch_size,
            "lr": lr,
            "weight_decay": weight_decay,
            "device": device,
            "seed": seed,
            "eval_score_threshold": eval_score_threshold,
            "nms_iou_threshold": nms_iou_threshold,
            "eval_every": eval_every,
            "eval_limit": eval_limit,
            "num_workers": num_workers,
            "amp": amp,
            "cache_images": cache_images,
            "min_int8_weight_bytes": int(min_int8_weight_bytes),
            "max_int8_weight_bytes": int(max_int8_weight_bytes),
            "init_checkpoint": str(Path(init_checkpoint).resolve().as_posix()) if init_checkpoint is not None else None,
            "classification_loss": "focal_bce_hard_negative_ratio_20_with_class_weights",
            "box_loss": "smooth_l1_plus_10x_iou_positive_cells",
            "orientation_loss": "mse_on_theta_valid_positive_cells_sin2_cos2",
        },
        "history": history,
        "best_epoch": best_epoch,
        "best_metric": best_metric,
        "best_metric_name": best_metric_name,
        "checkpoint": checkpoint.resolve().as_posix(),
        "checkpoint_sha256": _sha256_file(checkpoint),
        "last_checkpoint": last_checkpoint.resolve().as_posix(),
        "last_checkpoint_sha256": _sha256_file(last_checkpoint),
        "test_metrics": test_metrics,
        "limitations": [
            "Training evidence is host-side only.",
            "TFLite, host MERA, RUHMI dispatch, and board static golden are not proven by this report.",
        ],
    }


def _manifest_summary(path: Path, records: list[dict[str, Any]]) -> dict[str, Any]:
    split_counts: Counter[str] = Counter()
    class_counts: Counter[str] = Counter()
    theta_counts: Counter[str] = Counter()
    class_by_split: dict[str, Counter[str]] = defaultdict(Counter)
    for record in records:
        split = str(record.get("split", ""))
        split_counts[split] += 1
        for obj in record.get("objects", []):
            name = str(obj["class_name"])
            class_counts[name] += 1
            class_by_split[name][split] += 1
            if obj.get("theta_valid"):
                theta_counts[name] += 1
    return {
        "manifest": path.resolve().as_posix(),
        "manifest_sha256": _sha256_file(path),
        "record_count": len(records),
        "object_count": int(sum(class_counts.values())),
        "split_counts": dict(split_counts),
        "class_counts": dict(class_counts),
        "class_by_split": {name: dict(counter) for name, counter in class_by_split.items()},
        "theta_valid_counts": dict(theta_counts),
    }


def _environment(device: str) -> dict[str, Any]:
    torch, _ = _torch_modules()
    return {
        "python": sys.version.split()[0],
        "platform": platform.platform(),
        "torch": getattr(torch, "__version__", ""),
        "cuda_available": bool(torch.cuda.is_available()),
        "cuda_device": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "",
        "requested_device": device,
    }


def _git_summary() -> dict[str, Any]:
    try:
        head = subprocess.run(["git", "rev-parse", "--short", "HEAD"], text=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=False)
        branch = subprocess.run(["git", "branch", "--show-current"], text=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=False)
        status = subprocess.run(["git", "status", "--short"], text=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=False)
    except Exception:
        return {"head": "", "branch": "", "dirty_files": []}
    return {
        "head": head.stdout.strip(),
        "branch": branch.stdout.strip(),
        "dirty_files": [line for line in status.stdout.splitlines() if line.strip()],
    }


def _format_train_markdown(report: dict[str, Any]) -> str:
    model = report["model"]
    data = report["data"]
    test = report["test_metrics"]
    return "\n".join(
        [
            "# EthosSafeDetV2 Training Report",
            "",
            "This is a host-side Model A V2 training report. It is not a board acceptance report.",
            "",
            "## Dataset",
            "",
            f"- Manifest: `{data['manifest']}`",
            f"- Records: {data['record_count']}",
            f"- Objects: {data['object_count']}",
            f"- Splits: `{data['split_counts']}`",
            f"- Classes: `{data['class_counts']}`",
            f"- Theta-valid counts: `{data['theta_valid_counts']}`",
            "",
            "## Model",
            "",
            f"- Width: {model['width']}",
            f"- Parameters: {model['parameter_count']}",
            f"- Estimated INT8 weights: {model['estimated_int8_weight_bytes']} bytes",
            f"- Outputs: `{model['output_names']}`",
            "",
            "## Result",
            "",
            f"- Best epoch: {report['best_epoch']}",
            f"- Best metric: {report['best_metric']:.6f}",
            f"- Test recall@0.5: {_fmt(test.get('recall50'))}",
            f"- Test mean best IoU: {_fmt(test.get('best_iou_mean'))}",
            f"- Test theta MAE rad: {_fmt(test.get('theta_abs_error_rad_mean'))}",
            "",
            "## Limitations",
            "",
            "- TFLite, host MERA, RUHMI dispatch, and board static golden remain future gates.",
            "",
        ]
    )


def _write_history_csv(path: Path, history: list[dict[str, Any]]) -> None:
    fieldnames = sorted({key for row in history for key, value in row.items() if not isinstance(value, (dict, list))})
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        for row in history:
            writer.writerow({key: _csv_value(row.get(key)) for key in fieldnames})


def _csv_value(value: Any) -> Any:
    if isinstance(value, float):
        return f"{value:.8f}"
    return "" if value is None else value


def _fmt(value: Any) -> str:
    return "n/a" if value is None else f"{float(value):.6f}"


def _seed_everything(torch, seed: int) -> None:  # type: ignore[no-untyped-def]
    random.seed(int(seed))
    np.random.seed(int(seed))
    torch.manual_seed(int(seed))
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(int(seed))


def _autocast_context(torch, enabled: bool):  # type: ignore[no-untyped-def]
    if enabled and torch.cuda.is_available():
        return torch.amp.autocast("cuda")
    return torch.autocast("cpu", enabled=False)


def _make_grad_scaler(torch, enabled: bool):  # type: ignore[no-untyped-def]
    if enabled and torch.cuda.is_available():
        return torch.amp.GradScaler("cuda")
    return None


def _clone_state_dict(state_dict: dict[str, Any]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in state_dict.items():
        result[key] = value.detach().cpu().clone() if hasattr(value, "detach") else value
    return result


def _sha256_file(path: str | Path) -> str:
    h = hashlib.sha256()
    with Path(path).open("rb") as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def _torch_modules():
    try:
        import torch
        from torch import nn
    except ImportError as exc:  # pragma: no cover - depends on local ML env
        raise RuntimeError("EthosSafeDetV2 training requires PyTorch") from exc
    return torch, nn


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Train Model A V2 / EthosSafeDetV2.")
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--input-size", type=int, default=320)
    parser.add_argument("--width", type=int, default=40)
    parser.add_argument("--epochs", type=int, default=120)
    parser.add_argument("--batch", type=int, default=32)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--eval-score-threshold", type=float, default=0.25)
    parser.add_argument("--nms-iou", type=float, default=0.5)
    parser.add_argument("--eval-every", type=int, default=1)
    parser.add_argument("--eval-limit", type=int, default=None)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--amp", action="store_true")
    parser.add_argument("--no-amp", action="store_true")
    parser.add_argument("--no-cache-images", action="store_true")
    parser.add_argument("--min-int8-weight-bytes", type=int, default=100 * 1024)
    parser.add_argument("--max-int8-weight-bytes", type=int, default=300 * 1024)
    parser.add_argument("--init-checkpoint", default=None)
    args = parser.parse_args(argv)
    try:
        report = train_ethossafedet_v2(
            args.manifest,
            args.out,
            input_size=args.input_size,
            width=args.width,
            epochs=args.epochs,
            batch_size=args.batch,
            lr=args.lr,
            weight_decay=args.weight_decay,
            device=args.device,
            seed=args.seed,
            eval_score_threshold=args.eval_score_threshold,
            nms_iou_threshold=args.nms_iou,
            eval_every=args.eval_every,
            eval_limit=args.eval_limit,
            num_workers=args.num_workers,
            amp=(False if args.no_amp else True if args.amp else True),
            cache_images=not args.no_cache_images,
            min_int8_weight_bytes=args.min_int8_weight_bytes,
            max_int8_weight_bytes=args.max_int8_weight_bytes,
            init_checkpoint=args.init_checkpoint,
        )
    except Exception as exc:
        print(str(exc), file=sys.stderr)
        return 1
    print(json.dumps({"ok": True, "report": report["report_paths"]["json"], "checkpoint": report["checkpoint"]}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
