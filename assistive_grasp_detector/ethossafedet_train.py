"""Training helpers for EthosSafeDet-A v1."""

from __future__ import annotations

import csv
import hashlib
import json
import platform
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image, ImageEnhance

from assistive_grasp_detector.coords import bbox_vga_to_model, letterbox_rgb_image, make_letterbox_transform
from assistive_grasp_detector.ethossafedet_manifest import load_manifest_records, resolve_record_image
from assistive_grasp_detector.ethossafedet_model import EthosSafeDetConfig, load_checkpoint_config, load_checkpoint_state, make_ethossafedet_a
from assistive_grasp_detector.ethossafedet_postprocess import bbox_iou, decode_ltrb_outputs
from assistive_grasp_detector.schema import ETHOSSAFEDET_CLASS_NAMES, ETHOSSAFEDET_NUM_CLASSES, ETHOSSAFEDET_STRIDE


def assign_targets(
    objects: list[dict[str, Any]],
    input_size: int = 320,
    stride: int = ETHOSSAFEDET_STRIDE,
    num_classes: int = ETHOSSAFEDET_NUM_CLASSES,
    source_size: tuple[int, int] = (640, 480),
    center_radius: float = 1.5,
) -> dict[str, np.ndarray]:
    grid = input_size // stride
    cls_target = np.zeros((num_classes, grid, grid), dtype=np.float32)
    box_target = np.zeros((4, grid, grid), dtype=np.float32)
    positive = np.zeros((grid, grid), dtype=bool)
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
                left = max(0.0, center_x - x1)
                top = max(0.0, center_y - y1)
                right = max(0.0, x2 - center_x)
                bottom = max(0.0, y2 - center_y)
                cls_target[:, gy, gx] = 0.0
                cls_target[class_id, gy, gx] = 1.0
                box_target[:, gy, gx] = np.asarray(
                    [
                        left,
                        top,
                        right,
                        bottom,
                    ],
                    dtype=np.float32,
                )
                positive[gy, gx] = True
                owner_area[gy, gx] = area
                owner_center_distance[gy, gx] = distance
    return {"cls": cls_target, "box": box_target, "positive": positive}

class EthosSafeDetDataset:
    def __init__(self, manifest_path: str | Path, split: str, input_size: int = 320, augment: bool = False, cache_images: bool = True) -> None:
        torch, _ = _torch_modules()
        self.torch = torch
        self.records = [record for record in load_manifest_records(manifest_path) if record.get("split") == split]
        self.input_size = int(input_size)
        self.augment = bool(augment)
        self.cache_images = bool(cache_images)
        if not self.records:
            raise ValueError(f"manifest has no records for split {split!r}")
        self.cache = self._build_cache() if self.cache_images else None

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, index: int):  # type: ignore[no-untyped-def]
        record = self.records[index]
        if self.cache is not None:
            entry = self.cache[index]
            arr_u8 = np.asarray(entry["image"], dtype=np.uint8)
            targets = entry["targets"]
            if self.augment and np.random.random() < 0.5:
                arr_u8 = np.ascontiguousarray(arr_u8[:, ::-1, :])
                targets = entry["targets_flip"]
            if self.augment:
                arr_u8 = _augment_cached_image(arr_u8)
            arr = arr_u8.astype(np.float32) / 255.0
        else:
            image_path = resolve_record_image(record)
            with Image.open(image_path) as image:
                image = image.convert("RGB")
                objects = [dict(obj) for obj in record.get("objects", [])]
                if self.augment:
                    image, objects = _augment_image_and_objects(image, objects)
                model_image = letterbox_rgb_image(image, self.input_size, self.input_size)
            arr = np.asarray(model_image, dtype=np.float32) / 255.0
            targets = assign_targets(objects, input_size=self.input_size)
        chw = np.transpose(arr, (2, 0, 1))
        return {
            "image": self.torch.from_numpy(chw),
            "cls": self.torch.from_numpy(targets["cls"]),
            "box": self.torch.from_numpy(targets["box"]),
            "positive": self.torch.from_numpy(targets["positive"]),
            "record_index": self.torch.tensor(index, dtype=self.torch.long),
        }

    def _build_cache(self) -> list[dict[str, Any]]:
        cache: list[dict[str, Any]] = []
        for record in self.records:
            image_path = resolve_record_image(record)
            with Image.open(image_path) as image:
                image = image.convert("RGB")
                model_image = letterbox_rgb_image(image, self.input_size, self.input_size)
            objects = [dict(obj) for obj in record.get("objects", [])]
            flipped_objects = _flip_objects(objects, width=int(record.get("width", 640)))
            cache.append(
                {
                    "image": np.asarray(model_image, dtype=np.uint8),
                    "targets": assign_targets(objects, input_size=self.input_size),
                    "targets_flip": assign_targets(flipped_objects, input_size=self.input_size),
                }
            )
        return cache


def train_ethossafedet_a(
    manifest_path: str | Path,
    output_dir: str | Path,
    input_size: int = 320,
    epochs: int = 1,
    batch_size: int = 8,
    lr: float = 1e-3,
    device: str = "cpu",
    seed: int = 0,
    eval_score_threshold: float = 0.25,
    nms_iou_threshold: float = 0.5,
    num_workers: int = 0,
    eval_every: int = 1,
    eval_limit: int | None = None,
    amp: bool = False,
    cache_images: bool = True,
    resume_checkpoint: str | Path | None = None,
) -> dict[str, Any]:
    torch, nn = _torch_modules()
    from torch.utils.data import DataLoader

    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    manifest = Path(manifest_path)
    records = load_manifest_records(manifest)
    _seed_everything(torch, seed)

    train_ds = EthosSafeDetDataset(manifest_path, "train", input_size=input_size, augment=True, cache_images=cache_images)
    try:
        val_ds: EthosSafeDetDataset | None = EthosSafeDetDataset(manifest_path, "val", input_size=input_size, cache_images=cache_images)
    except ValueError:
        val_ds = None
    generator = torch.Generator()
    generator.manual_seed(int(seed))
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=num_workers, generator=generator)
    val_loader = (
        DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=num_workers)
        if val_ds is not None
        else None
    )
    if resume_checkpoint is not None:
        resume_config = load_checkpoint_config(str(resume_checkpoint))
        config = EthosSafeDetConfig(input_size=input_size, num_classes=resume_config.num_classes, width=resume_config.width)
    else:
        config = EthosSafeDetConfig(input_size=input_size)
    model = make_ethossafedet_a(config).to(device)
    if resume_checkpoint is not None:
        model.load_state_dict(load_checkpoint_state(str(resume_checkpoint)))
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)
    use_amp = bool(amp and str(device).startswith("cuda") and torch.cuda.is_available())
    scaler = _make_grad_scaler(torch, use_amp)
    cls_loss_fn = nn.BCEWithLogitsLoss(reduction="none")
    box_loss_fn = nn.SmoothL1Loss(reduction="none")
    history: list[dict[str, Any]] = []
    best_state: dict[str, Any] | None = None
    best_metric = -float("inf")
    best_metric_name = "none"
    best_epoch = 0

    for epoch in range(int(epochs)):
        train_metrics = _run_epoch(
            model,
            train_loader,
            optimizer,
            cls_loss_fn,
            box_loss_fn,
            device=device,
            train=True,
            use_amp=use_amp,
            scaler=scaler,
        )
        val_metrics = (
            _run_epoch(
                model,
                val_loader,
                None,
                cls_loss_fn,
                box_loss_fn,
                device=device,
                train=False,
                use_amp=use_amp,
                scaler=None,
            )
            if val_loader is not None
            else _empty_epoch_metrics()
        )
        should_eval_detections = val_ds is not None and (int(eval_every) <= 1 or (epoch + 1) % int(eval_every) == 0 or epoch + 1 == int(epochs))
        detection_metrics = (
            _evaluate_val_detections(
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
            if should_eval_detections
            else {"val_top1_class_acc": None, "val_top1_iou_mean": None, "val_eval_count": 0}
        )
        row = {
            "epoch": epoch + 1,
            "train_loss": train_metrics["loss"],
            "train_cls_loss": train_metrics["cls_loss"],
            "train_box_loss": train_metrics["box_loss"],
            "val_loss": val_metrics["loss"],
            "val_cls_loss": val_metrics["cls_loss"],
            "val_box_loss": val_metrics["box_loss"],
            "positive_cells": train_metrics["positive_cells"],
            **detection_metrics,
        }
        history.append(row)
        metric, metric_name = _checkpoint_metric(row)
        if metric >= best_metric:
            best_metric = metric
            best_metric_name = metric_name
            best_epoch = epoch + 1
            best_state = _clone_state_dict(model.state_dict())

    last_state = _clone_state_dict(model.state_dict())
    if best_state is None:
        best_state = last_state
        best_epoch = int(epochs)

    checkpoint = out / "ethossafedet_a.pt"
    last_checkpoint = out / "ethossafedet_a_last.pt"
    torch.save(
        {
            "model_id": "EthosSafeDet-A",
            "input_size": input_size,
            "num_classes": ETHOSSAFEDET_NUM_CLASSES,
            "width": config.width,
            "model_state": best_state,
            "history": history,
            "best_epoch": best_epoch,
            "best_metric": best_metric,
            "best_metric_name": best_metric_name,
        },
        checkpoint,
    )
    torch.save(
        {
            "model_id": "EthosSafeDet-A",
            "input_size": input_size,
            "num_classes": ETHOSSAFEDET_NUM_CLASSES,
            "width": config.width,
            "model_state": last_state,
            "history": history,
            "best_epoch": best_epoch,
            "best_metric": best_metric,
            "best_metric_name": best_metric_name,
        },
        last_checkpoint,
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
        epochs=int(epochs),
        batch_size=int(batch_size),
        lr=float(lr),
        device=device,
        seed=int(seed),
        eval_score_threshold=float(eval_score_threshold),
        nms_iou_threshold=float(nms_iou_threshold),
        num_workers=int(num_workers),
        eval_every=int(eval_every),
        eval_limit=eval_limit,
        amp=use_amp,
        cache_images=bool(cache_images),
        resume_checkpoint=Path(resume_checkpoint) if resume_checkpoint is not None else None,
        history=history,
        best_epoch=best_epoch,
        best_metric=best_metric,
        best_metric_name=best_metric_name,
        checkpoint=checkpoint,
        last_checkpoint=last_checkpoint,
        model=model,
    )
    report["report_paths"] = report_paths
    _write_history_csv(Path(report_paths["csv"]), history)
    Path(report_paths["json"]).write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    Path(report_paths["markdown"]).write_text(_format_markdown_report(report), encoding="utf-8")
    return report


def _focal_bce_loss(loss, logits, targets, gamma: float = 2.0, alpha: float = 0.25):  # type: ignore[no-untyped-def]
    torch, _ = _torch_modules()
    probs = torch.sigmoid(logits)
    pt = targets * probs + (1.0 - targets) * (1.0 - probs)
    alpha_t = targets * alpha + (1.0 - targets) * (1.0 - alpha)
    return loss * alpha_t * torch.pow(1.0 - pt, gamma)


def _run_epoch(
    model,  # type: ignore[no-untyped-def]
    loader,  # type: ignore[no-untyped-def]
    optimizer,  # type: ignore[no-untyped-def]
    cls_loss_fn,  # type: ignore[no-untyped-def]
    box_loss_fn,  # type: ignore[no-untyped-def]
    device: str,
    train: bool,
    use_amp: bool,
    scaler,  # type: ignore[no-untyped-def]
) -> dict[str, Any]:
    torch, _ = _torch_modules()
    model.train(mode=train)
    totals = {"loss": 0.0, "cls_loss": 0.0, "box_loss": 0.0, "samples": 0, "positive_cells": 0}
    context = torch.enable_grad() if train else torch.no_grad()
    with context:
        for batch in loader:
            images = batch["image"].to(device)
            cls_target = batch["cls"].to(device)
            box_target = batch["box"].to(device)
            positive = batch["positive"].to(device)
            with _autocast_context(torch, use_amp):
                cls_logits, box_ltrb = model(images)
                loss, cls_loss, box_loss = _batch_losses(cls_logits, box_ltrb, cls_target, box_target, positive, cls_loss_fn, box_loss_fn)
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
            totals["cls_loss"] += float(cls_loss.detach().cpu()) * batch_size
            totals["box_loss"] += float(box_loss.detach().cpu()) * batch_size
            totals["samples"] += batch_size
            totals["positive_cells"] += int(positive.detach().cpu().sum())
    samples = max(1, int(totals["samples"]))
    return {
        "loss": totals["loss"] / samples,
        "cls_loss": totals["cls_loss"] / samples,
        "box_loss": totals["box_loss"] / samples,
        "positive_cells": int(totals["positive_cells"]),
    }


def _batch_losses(cls_logits, box_ltrb, cls_target, box_target, positive, cls_loss_fn, box_loss_fn):  # type: ignore[no-untyped-def]
    cls_loss = _hard_negative_focal_loss(cls_loss_fn(cls_logits, cls_target), cls_logits, cls_target)
    if bool(positive.any()):
        pos = positive.unsqueeze(1).expand_as(box_ltrb)
        smooth_l1 = box_loss_fn(box_ltrb[pos], box_target[pos]).mean()
        iou_loss = _positive_ltrb_iou_loss(box_ltrb, box_target, positive)
        box_loss = smooth_l1 + 10.0 * iou_loss
    else:
        box_loss = box_ltrb.sum() * 0.0
    return cls_loss + box_loss, cls_loss, box_loss


def _hard_negative_focal_loss(loss, logits, targets, hard_negative_ratio: int = 20):  # type: ignore[no-untyped-def]
    torch, _ = _torch_modules()
    focal = _focal_bce_loss(loss, logits, targets)
    positive = targets > 0.0
    positive_losses = focal[positive]
    negative_losses = focal[~positive].flatten()
    positive_count = int(positive_losses.numel())
    if positive_count > 0:
        negative_count = min(int(negative_losses.numel()), positive_count * int(hard_negative_ratio))
    else:
        negative_count = min(int(negative_losses.numel()), 256)
    if negative_count > 0:
        negative_losses = torch.topk(negative_losses, k=negative_count).values
        total = positive_losses.sum() + negative_losses.sum()
    else:
        total = positive_losses.sum()
    denom = max(1, positive_count + negative_count)
    return total / float(denom)


def _positive_ltrb_iou_loss(pred_ltrb, target_ltrb, positive, stride: int = ETHOSSAFEDET_STRIDE):  # type: ignore[no-untyped-def]
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
        [
            center_x[pos] - pred[:, 0][pos],
            center_y[pos] - pred[:, 1][pos],
            center_x[pos] + pred[:, 2][pos],
            center_y[pos] + pred[:, 3][pos],
        ],
        dim=1,
    )
    target_boxes = torch.stack(
        [
            center_x[pos] - target[:, 0][pos],
            center_y[pos] - target[:, 1][pos],
            center_x[pos] + target[:, 2][pos],
            center_y[pos] + target[:, 3][pos],
        ],
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


def _empty_epoch_metrics() -> dict[str, Any]:
    return {"loss": None, "cls_loss": None, "box_loss": None, "positive_cells": 0}


def _checkpoint_metric(row: dict[str, Any]) -> tuple[float, str]:
    class_acc = row.get("val_top1_class_acc")
    iou_mean = row.get("val_top1_iou_mean")
    if class_acc is not None and iou_mean is not None:
        return float(class_acc) + float(iou_mean), "val_top1_class_acc_plus_iou_mean"
    val_loss = row.get("val_loss")
    if val_loss is not None:
        return -float(val_loss), "negative_val_loss"
    return -float(row["train_loss"]), "negative_train_loss"


def _evaluate_val_detections(
    model,  # type: ignore[no-untyped-def]
    dataset: EthosSafeDetDataset,
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

    class_matches = 0
    ious: list[float] = []
    count = 0
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    model.eval()
    with torch.no_grad():
        for batch in loader:
            images = batch["image"].to(device)
            with _autocast_context(torch, use_amp):
                cls_logits, box_ltrb = model(images)
            cls_np = cls_logits.detach().cpu().numpy()
            box_np = box_ltrb.detach().cpu().numpy()
            record_indices = batch["record_index"].detach().cpu().numpy().tolist()
            for batch_index, record_index in enumerate(record_indices):
                if limit is not None and int(limit) > 0 and count >= int(limit):
                    break
                record = dataset.records[int(record_index)]
                gt = _primary_object(record.get("objects", []))
                if gt is None:
                    continue
                detections = decode_ltrb_outputs(
                    cls_np[batch_index : batch_index + 1],
                    box_np[batch_index : batch_index + 1],
                    input_size=input_size,
                    score_threshold=score_threshold,
                    nms_iou_threshold=nms_iou_threshold,
                )
                count += 1
                if not detections:
                    ious.append(0.0)
                    continue
                top = detections[0]
                if int(top.class_id) == int(gt["class_id"]):
                    class_matches += 1
                ious.append(bbox_iou(top.bbox_xyxy_vga, gt["bbox_xyxy_vga"]))
            if limit is not None and int(limit) > 0 and count >= int(limit):
                break
    return {
        "val_top1_class_acc": (class_matches / count) if count else None,
        "val_top1_iou_mean": (float(np.mean(ious)) if ious else None),
        "val_eval_count": count,
    }


def _primary_object(objects: list[dict[str, Any]]) -> dict[str, Any] | None:
    if not objects:
        return None
    return max(objects, key=lambda obj: _bbox_area(obj.get("bbox_xyxy_vga", [0, 0, 0, 0])))


def _augment_image_and_objects(image: Image.Image, objects: list[dict[str, Any]]) -> tuple[Image.Image, list[dict[str, Any]]]:
    width, _ = image.size
    if np.random.random() < 0.5:
        image = image.transpose(Image.Transpose.FLIP_LEFT_RIGHT)
        objects = _flip_objects(objects, width)
    brightness = 0.85 + np.random.random() * 0.3
    contrast = 0.85 + np.random.random() * 0.3
    image = ImageEnhance.Brightness(image).enhance(float(brightness))
    image = ImageEnhance.Contrast(image).enhance(float(contrast))
    return image, objects


def _flip_objects(objects: list[dict[str, Any]], width: int) -> list[dict[str, Any]]:
    flipped: list[dict[str, Any]] = []
    for obj in objects:
        item = dict(obj)
        x1, y1, x2, y2 = [float(v) for v in item["bbox_xyxy_vga"]]
        item["bbox_xyxy_vga"] = [float(width) - x2, y1, float(width) - x1, y2]
        flipped.append(item)
    return flipped


def _augment_cached_image(image: np.ndarray) -> np.ndarray:
    arr = image.astype(np.float32)
    brightness = 0.85 + np.random.random() * 0.3
    contrast = 0.85 + np.random.random() * 0.3
    arr = (arr - 127.5) * float(contrast) + 127.5
    arr = arr * float(brightness)
    return np.clip(arr, 0.0, 255.0).astype(np.uint8)


def _bbox_area(bbox: Any) -> float:
    try:
        x1, y1, x2, y2 = [float(v) for v in bbox]
    except Exception:
        return 0.0
    return max(0.0, x2 - x1) * max(0.0, y2 - y1)


def _seed_everything(torch, seed: int) -> None:  # type: ignore[no-untyped-def]
    np.random.seed(int(seed))
    torch.manual_seed(int(seed))
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(int(seed))


def _autocast_context(torch, enabled: bool):  # type: ignore[no-untyped-def]
    return torch.amp.autocast("cuda", enabled=enabled)


def _make_grad_scaler(torch, enabled: bool):  # type: ignore[no-untyped-def]
    if not enabled:
        return None
    try:
        return torch.amp.GradScaler("cuda", enabled=True)
    except TypeError:
        return torch.cuda.amp.GradScaler(enabled=True)


def _clone_state_dict(state_dict: dict[str, Any]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in state_dict.items():
        if hasattr(value, "detach"):
            result[key] = value.detach().cpu().clone()
        else:
            result[key] = value
    return result


def _build_train_report(
    manifest_path: Path,
    records: list[dict[str, Any]],
    output_dir: Path,
    config: EthosSafeDetConfig,
    epochs: int,
    batch_size: int,
    lr: float,
    device: str,
    seed: int,
    eval_score_threshold: float,
    nms_iou_threshold: float,
    num_workers: int,
    eval_every: int,
    eval_limit: int | None,
    amp: bool,
    cache_images: bool,
    resume_checkpoint: Path | None,
    history: list[dict[str, Any]],
    best_epoch: int,
    best_metric: float,
    best_metric_name: str,
    checkpoint: Path,
    last_checkpoint: Path,
    model,  # type: ignore[no-untyped-def]
) -> dict[str, Any]:
    checkpoint_sha = _sha256_file(checkpoint)
    last_checkpoint_sha = _sha256_file(last_checkpoint)
    return {
        "schema_version": "ethossafedet_train_report_v1",
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "model_id": "EthosSafeDet-A",
        "output_dir": output_dir.resolve().as_posix(),
        "resume_checkpoint": resume_checkpoint.resolve().as_posix() if resume_checkpoint is not None else "",
        "resume_checkpoint_sha256": _sha256_file(resume_checkpoint) if resume_checkpoint is not None else "",
        "environment": _environment(device),
        "git": _git_summary(),
        "data": _manifest_summary(manifest_path, records),
        "model": {
            "input_size": config.input_size,
            "num_classes": config.num_classes,
            "class_names": list(ETHOSSAFEDET_CLASS_NAMES),
            "width": config.width,
            "stride": ETHOSSAFEDET_STRIDE,
            "parameter_count": _parameter_count(model),
        },
        "hyperparameters": {
            "epochs": epochs,
            "batch_size": batch_size,
            "lr": lr,
            "device": device,
            "seed": seed,
            "eval_score_threshold": eval_score_threshold,
            "nms_iou_threshold": nms_iou_threshold,
            "num_workers": num_workers,
            "eval_every": eval_every,
            "eval_limit": eval_limit,
            "amp": amp,
            "cache_images": cache_images,
            "classification_loss": "focal_bce_hard_negative_ratio_20",
            "box_loss": "smooth_l1_plus_10x_iou_positive_cells",
        },
        "history": history,
        "best_epoch": best_epoch,
        "best_metric": best_metric,
        "best_metric_name": best_metric_name,
        "checkpoint": checkpoint.resolve().as_posix(),
        "checkpoint_sha256": checkpoint_sha,
        "last_checkpoint": last_checkpoint.resolve().as_posix(),
        "last_checkpoint_sha256": last_checkpoint_sha,
    }


def _manifest_summary(path: Path, records: list[dict[str, Any]]) -> dict[str, Any]:
    split_counts: dict[str, int] = {}
    class_counts: dict[str, int] = {str(i): 0 for i in range(ETHOSSAFEDET_NUM_CLASSES)}
    negative_count = 0
    object_count = 0
    for record in records:
        split = str(record.get("split", ""))
        split_counts[split] = split_counts.get(split, 0) + 1
        if bool(record.get("negative", False)):
            negative_count += 1
        for obj in record.get("objects", []):
            class_id = str(int(obj["class_id"]))
            class_counts[class_id] = class_counts.get(class_id, 0) + 1
            object_count += 1
    return {
        "manifest": path.resolve().as_posix(),
        "manifest_sha256": _sha256_file(path),
        "record_count": len(records),
        "object_count": object_count,
        "negative_count": negative_count,
        "split_counts": split_counts,
        "class_counts": class_counts,
    }


def _environment(device: str) -> dict[str, Any]:
    torch, _ = _torch_modules()
    return {
        "python": sys.version.split()[0],
        "platform": platform.platform(),
        "torch": torch.__version__,
        "cuda_available": bool(torch.cuda.is_available()),
        "cuda_device_count": int(torch.cuda.device_count()),
        "cuda_device_name": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "",
        "requested_device": device,
    }


def _git_summary() -> dict[str, Any]:
    try:
        head = subprocess.run(["git", "rev-parse", "HEAD"], text=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=False)
        status = subprocess.run(["git", "status", "--short"], text=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=False)
    except Exception as exc:
        return {"available": False, "error": str(exc)}
    lines = [line for line in status.stdout.splitlines() if line.strip()]
    return {
        "available": head.returncode == 0 and status.returncode == 0,
        "head": head.stdout.strip() if head.returncode == 0 else "",
        "dirty": bool(lines),
        "status_line_count": len(lines),
        "status_short": lines[:200],
    }


def _parameter_count(model) -> int:  # type: ignore[no-untyped-def]
    return int(sum(parameter.numel() for parameter in model.parameters()))


def _sha256_file(path: str | Path) -> str:
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def _write_history_csv(path: Path, history: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    columns = [
        "epoch",
        "train_loss",
        "train_cls_loss",
        "train_box_loss",
        "val_loss",
        "val_cls_loss",
        "val_box_loss",
        "positive_cells",
        "val_top1_class_acc",
        "val_top1_iou_mean",
    ]
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=columns)
        writer.writeheader()
        for row in history:
            writer.writerow({column: _csv_value(row.get(column)) for column in columns})


def _csv_value(value: Any) -> Any:
    if value is None:
        return ""
    if isinstance(value, float):
        return f"{value:.8f}"
    return value


def _format_markdown_report(report: dict[str, Any]) -> str:
    data = report["data"]
    hparams = report["hyperparameters"]
    model = report["model"]
    env = report["environment"]
    git = report["git"]
    history = report["history"]
    best_epoch = int(report["best_epoch"])
    best_row = next((row for row in history if int(row["epoch"]) == best_epoch), history[-1])
    last_row = history[-1]
    first_row = history[0]
    measured_rows = [row for row in history if row.get("val_top1_class_acc") is not None]
    last_measured = measured_rows[-1] if measured_rows else last_row
    report_paths = report.get("report_paths", {})
    lines = [
        "# EthosSafeDet-A Training Report",
        "",
        "## Abstract",
        "",
        (
            "This report documents an EthosSafeDet-A training run as a reproducible "
            "experiment rather than an informal loss log. The current firmware acceptance "
            "contract is Model A V2 / EthosSafeDetV2: six-class detection with bbox and "
            "orientation outputs designed backwards from RA8P1 Ethos-U55 and RUHMI "
            "deployment constraints. The exported inference graph is required to keep "
            "classification, box, and orientation tensors separate and to leave sigmoid, "
            "decode, atan2, NMS, TopK, and board-specific post-processing outside the neural graph."
        ),
        "",
        "## 1. Experiment Identity",
        "",
        "| Field | Value |",
        "|---|---|",
        f"| Generated at | `{report['generated_at']}` |",
        f"| Model id | `{report['model_id']}` |",
        f"| Input tensor | `1x3x{model['input_size']}x{model['input_size']}` PC reference; static batch 1 deployment target |",
        f"| Best checkpoint | `{report['checkpoint']}` |",
        f"| Best checkpoint SHA256 | `{report['checkpoint_sha256']}` |",
        f"| Last checkpoint | `{report['last_checkpoint']}` |",
        f"| Last checkpoint SHA256 | `{report['last_checkpoint_sha256']}` |",
        f"| Resume checkpoint | `{report.get('resume_checkpoint') or 'none'}` |",
        f"| Best selection metric | `{report['best_metric_name']}` = {_markdown_metric(report['best_metric'])} |",
        "",
        "## 2. Research Objective And Deployment Envelope",
        "",
        "Model A V2 / EthosSafeDetV2 is the active system contract for six desktop object categories in "
        "VGA camera frames. A run generated by the bbox-only training path is a legacy/pre-V2 baseline: "
        "it can support migration analysis, but it is not a firmware-accepted Model A V2 candidate until "
        "the orientation head, theta_valid masking, two-scale output, and V2 gates are present.",
        "",
        "The deployment envelope is intentionally narrower than a generic detector: the final graph must be "
        "static-shape, batch-one, quantization-friendly, and expressible with Ethos-U/RUHMI-friendly operators. "
        "The model therefore uses a single stride-8 detection grid and avoids FPN-style concatenation, DFL bins, "
        "graph-internal decode, graph-internal NMS, and graph-internal sigmoid/exp.",
        "",
        "## 3. Dataset And Annotation Contract",
        "",
        "| Item | Value |",
        "|---|---:|",
        f"| Manifest records | {data['record_count']} |",
        f"| Annotated objects | {data['object_count']} |",
        f"| Negative images | {data['negative_count']} |",
        "",
        f"- Manifest: `{data['manifest']}`",
        f"- Manifest SHA256: `{data['manifest_sha256']}`",
        "- Coordinate source: VGA `640x480` camera image space.",
        "- Training transform: deterministic letterbox into the model input size; bbox targets are mapped from `bbox_xyxy_vga`.",
        "- Primary training format: `ethossafedet_manifest.jsonl`; YOLO dataset files are not a training dependency.",
        "",
        "### 3.1 Split Distribution",
        "",
        "| Split | Images | Share |",
        "|---|---:|---:|",
        *_split_rows(data),
        "",
        "### 3.2 Class Distribution",
        "",
        "| Class id | Class name | Objects | Share |",
        "|---:|---|---:|---:|",
        *_class_rows(data),
        "",
        "The class distribution is not perfectly balanced. `phone_A`, `earbud_A`, and `bottle_A` dominate the current set, "
        "while `phial_A`, `remote_A`, and `tissue_A` have fewer examples. This imbalance should be considered when "
        "interpreting the top-1 validation metric and when choosing future data collection targets.",
        "",
        "## 4. Model Design",
        "",
        "| Property | Value |",
        "|---|---:|",
        f"| Width multiplier base | {model['width']} |",
        f"| Parameters | {model['parameter_count']} |",
        f"| Classes | {model['num_classes']} |",
        f"| Detection stride | {model['stride']} |",
        f"| Grid size | {model['input_size'] // model['stride']} x {model['input_size'] // model['stride']} |",
        "",
        "The architecture uses MobileNet-style depthwise-separable blocks with ReLU-family activations. Detection uses "
        "separate heads for class logits and LTRB distances. The exported inference graph returns exactly two tensors: "
        "`cls_logits` and `box_ltrb`. The box branch terminates with ReLU to keep distances non-negative without using "
        "exp or grid decode in the graph.",
        "",
        "The architecture deliberately excludes SiLU/Swish/Mish/GELU, attention, SPPF, C2f-style large concat blocks, "
        "DFL/Softmax bins, reg-max distributions, and any graph-internal NMS/TopK/ArgMax/Gather/Range/Meshgrid/dynamic "
        "reshape logic. CM85 C post-processing owns sigmoid, scoring, grid decode, candidate ranking, and NMS.",
        "",
        "## 5. Training Methodology",
        "",
        "| Hyperparameter | Value |",
        "|---|---:|",
        f"| Epochs in this invocation | {hparams['epochs']} |",
        f"| Batch size | {hparams['batch_size']} |",
        f"| Learning rate | {hparams['lr']} |",
        f"| Optimizer | AdamW |",
        f"| Requested device | `{hparams['device']}` |",
        f"| Seed | {hparams['seed']} |",
        f"| AMP enabled | {hparams['amp']} |",
        f"| Image cache enabled | {hparams['cache_images']} |",
        f"| Data loader workers | {hparams['num_workers']} |",
        f"| Eval score threshold | {hparams['eval_score_threshold']} |",
        f"| NMS IoU threshold | {hparams['nms_iou_threshold']} |",
        f"| Detection eval interval | every {hparams['eval_every']} epoch(s) |",
        "",
        "Positive cells are assigned with an FCOS-like center-region rule on the stride-8 grid. If multiple objects compete "
        "for the same cell, the assignment favors the smaller object, then the object whose center is closer to the cell. "
        f"Classification uses `{hparams['classification_loss']}` and box regression uses `{hparams['box_loss']}`. "
        "The training path includes light image augmentation and hard-negative sampling while keeping the inference graph "
        "free of loss-only and post-processing operators.",
        "",
        "## 6. Validation Results",
        "",
        "| Metric | First epoch | Best checkpoint epoch | Last epoch | Last measured epoch |",
        "|---|---:|---:|---:|---:|",
        _metric_compare_row("train_loss", first_row, best_row, last_row, last_measured),
        _metric_compare_row("val_loss", first_row, best_row, last_row, last_measured),
        _metric_compare_row("val_top1_class_acc", first_row, best_row, last_row, last_measured),
        _metric_compare_row("val_top1_iou_mean", first_row, best_row, last_row, last_measured),
        _metric_compare_row("positive_cells", first_row, best_row, last_row, last_measured),
        "",
        f"The best checkpoint is epoch {best_epoch}. Its validation top-1 class accuracy is "
        f"{_markdown_metric(best_row.get('val_top1_class_acc'))}, and its mean top-1 IoU is "
        f"{_markdown_metric(best_row.get('val_top1_iou_mean'))}. The final epoch is reported separately because "
        "the last model is not automatically the deployment candidate; the deployment candidate is the best checkpoint.",
        "",
        "The validation metric is a coarse detector-level proxy: it compares the top decoded prediction against the primary "
        "ground-truth object for each validation image after CPU-side sigmoid, decode, candidate filtering, and NMS. It is "
        "useful for candidate selection, but it is not a full mAP protocol and should not be cited as final detector mAP.",
        "",
        "## 7. Epoch History",
        "",
        "| Epoch | Train loss | Train cls | Train box | Val loss | Val cls | Val box | Positive cells | Val top-1 class acc | Val top-1 IoU mean |",
        "|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in history:
        lines.append(
            "| {epoch} | {train_loss} | {train_cls} | {train_box} | {val_loss} | {val_cls} | {val_box} | {positive} | {acc} | {iou} |".format(
                epoch=row["epoch"],
                train_loss=_markdown_metric(row.get("train_loss")),
                train_cls=_markdown_metric(row.get("train_cls_loss")),
                train_box=_markdown_metric(row.get("train_box_loss")),
                val_loss=_markdown_metric(row.get("val_loss")),
                val_cls=_markdown_metric(row.get("val_cls_loss")),
                val_box=_markdown_metric(row.get("val_box_loss")),
                positive=_markdown_metric(row.get("positive_cells")),
                acc=_markdown_metric(row.get("val_top1_class_acc")),
                iou=_markdown_metric(row.get("val_top1_iou_mean")),
            )
        )
    lines.extend(
        [
            "",
            "## 8. Reproducibility And Provenance",
            "",
            "| Item | Value |",
            "|---|---|",
            f"| Python | `{env['python']}` |",
            f"| Platform | `{env['platform']}` |",
            f"| PyTorch | `{env['torch']}` |",
            f"| CUDA available | `{env['cuda_available']}` |",
            f"| CUDA device count | `{env['cuda_device_count']}` |",
            f"| CUDA device | `{env.get('cuda_device_name', '')}` |",
            f"| Git HEAD | `{git.get('head', '')}` |",
            f"| Dirty worktree | `{git.get('dirty', '')}` |",
            f"| Dirty status line count | `{git.get('status_line_count', '')}` |",
            f"| JSON report | `{report_paths.get('json', '')}` |",
            f"| CSV history | `{report_paths.get('csv', '')}` |",
            f"| Markdown report | `{report_paths.get('markdown', '')}` |",
            "",
            "A dirty worktree is recorded rather than hidden because this project is under active migration from the old "
            "YOLO route to EthosSafeDet-A/Model A V2. The artifact hashes above are the reproducibility anchors for the trained "
            "candidate and its last-epoch counterpart.",
            "",
            "## 9. Limitations And Risk Register",
            "",
            "- This report covers training only. Export, quantization, host MERA, RUHMI dispatch, memory budget, and board static golden must be documented in the formal chain report.",
            "- The detector-level validation metric is top-1/primary-object oriented, not full mAP.",
            "- Class imbalance remains visible and should guide the next data collection batch.",
            "- Board latency, arena allocation, RUHMI region placement, and CM85 post-processing cost cannot be inferred from this training report.",
            "",
            "## 10. Conclusion",
            "",
            "This training run produced a best-checkpoint candidate with separated class and LTRB outputs. "
            "If the candidate lacks the V2 orientation head and theta_valid-aware gates, it remains a pre-V2 "
            "migration artifact and is not eligible for board flashing or firmware acceptance. A V2 candidate also "
            "requires the full formal gate chain, including host MERA alignment and RUHMI dispatch inspection.",
            "",
        ]
    )
    return "\n".join(lines)


def _split_rows(data: dict[str, Any]) -> list[str]:
    total = max(1, int(data.get("record_count", 0)))
    rows: list[str] = []
    for split, count in sorted(data.get("split_counts", {}).items()):
        rows.append(f"| `{split}` | {int(count)} | {_markdown_percent(int(count), total)} |")
    return rows


def _class_rows(data: dict[str, Any]) -> list[str]:
    total = max(1, int(data.get("object_count", 0)))
    counts = data.get("class_counts", {})
    rows: list[str] = []
    for class_id, name in enumerate(ETHOSSAFEDET_CLASS_NAMES):
        count = int(counts.get(str(class_id), 0))
        rows.append(f"| {class_id} | `{name}` | {count} | {_markdown_percent(count, total)} |")
    return rows


def _metric_compare_row(
    metric: str,
    first_row: dict[str, Any],
    best_row: dict[str, Any],
    last_row: dict[str, Any],
    last_measured: dict[str, Any],
) -> str:
    return (
        f"| `{metric}` | {_markdown_metric(first_row.get(metric))} | "
        f"{_markdown_metric(best_row.get(metric))} | {_markdown_metric(last_row.get(metric))} | "
        f"{_markdown_metric(last_measured.get(metric))} |"
    )


def _markdown_percent(value: int, total: int) -> str:
    return f"{(100.0 * float(value) / float(max(1, total))):.2f}%"


def _markdown_metric(value: Any) -> str:
    if value is None:
        return "n/a"
    if isinstance(value, float):
        return f"{value:.6f}"
    return str(value)


def _torch_modules():
    try:
        import torch
        from torch import nn
    except ImportError as exc:  # pragma: no cover - depends on local ML env
        raise RuntimeError("EthosSafeDet training requires PyTorch") from exc
    return torch, nn
