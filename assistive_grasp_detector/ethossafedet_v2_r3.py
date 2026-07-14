"""Versioned real-scene R3 data policy, evaluation, and report extension."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
from collections import Counter
from pathlib import Path
from typing import Any

from PIL import Image, ImageDraw

from assistive_grasp_detector.annotator_dataset import annotations_root, images_root, read_annotation
from assistive_grasp_detector.ethossafedet_postprocess import bbox_iou
from assistive_grasp_detector.ethossafedet_v2_manifest import (
    ETHOSSAFEDET_V2_SCHEMA_VERSION,
    load_v2_manifest_records,
    prepare_ethossafedet_v2_manifest,
    resolve_v2_record_image,
)
from assistive_grasp_detector.ethossafedet_v2_model import EthosSafeDetV2Config, make_ethossafedet_v2
from assistive_grasp_detector.ethossafedet_v2_report import (
    _annotate_validation_example,
    _predict_record,
    _validation_panel,
    make_v2_formal_report,
)
from assistive_grasp_detector.ethossafedet_v2_train import (
    EthosSafeDetV2Dataset,
    evaluate_v2_model,
    orientation_abs_error,
)
from assistive_grasp_detector.schema import ETHOSSAFEDET_CLASS_NAMES

R3_SCHEMA_VERSION = "ethossafedet_v2_r3_real_scene_v1"
R3_NEW_CAMERA = "camera_1"
R3_NEW_IMAGE_MIN = 3870
R3_NEW_IMAGE_MAX = 4545
R3_EMPTY_SPLITS = {
    "003347": "train",
    "003348": "train",
    "003349": "train",
    "003350": "train",
    "003351": "train",
    "003352": "train",
    "003353": "val",
    "003354": "empty_table_holdout",
    "003355": "empty_table_holdout",
}
R3_EXCLUDED_IMAGE_IDS = ("004056", "004057", "004138", "004170")
R3_EXCLUDED_IMAGE_REASONS = {
    image_id: "user-specified R3 exclusion; not eligible for training, selection, or terminal evaluation"
    for image_id in R3_EXCLUDED_IMAGE_IDS
}
R3_WEAK_VIEW_CLASSES = {"phial", "bottle", "phone"}


def stable_r3_split_for_key(key: str) -> str:
    """Assign only new positive records without perturbing frozen R2 membership."""
    bucket = int(hashlib.sha1(f"{R3_SCHEMA_VERSION}:{key}".encode("utf-8")).hexdigest()[:8], 16) % 100
    if bucket < 80:
        return "train"
    if bucket < 90:
        return "val"
    return "real_scene_holdout"


def prepare_v2_r3_manifest(
    dataset_root: str | Path,
    r2_manifest_path: str | Path,
    output_path: str | Path,
    *,
    policy_path: str | Path | None = None,
    snapshot_path: str | Path | None = None,
    write_empty_annotations: bool = False,
) -> dict[str, Any]:
    """Build a policy-controlled R3 manifest while preserving all R2 split assignments."""
    root = Path(dataset_root).resolve()
    out = Path(output_path)
    r2_manifest = Path(r2_manifest_path)
    if not r2_manifest.is_file():
        raise FileNotFoundError(f"R2 manifest does not exist: {r2_manifest}")

    _ensure_r3_empty_annotations(root, write=write_empty_annotations)
    _validate_excluded_images(root)

    source_manifest = out.with_name("ethossafedet_v2_r3_source_manifest.jsonl")
    source_result = prepare_ethossafedet_v2_manifest(root, source_manifest)
    if not source_result.ok:
        raise ValueError(f"source dataset validation failed: {source_result.to_dict()}")
    source_records = load_v2_manifest_records(source_manifest)
    legacy_splits = {_record_key(record, root): str(record["split"]) for record in load_v2_manifest_records(r2_manifest)}

    records: list[dict[str, Any]] = []
    unclassified: list[str] = []
    excluded_seen: set[str] = set()
    for source_record in source_records:
        record = dict(source_record)
        key = _record_key(record, root)
        image_id = str(record.get("image_id", ""))
        if image_id in R3_EXCLUDED_IMAGE_IDS:
            excluded_seen.add(image_id)
            continue
        if key in legacy_splits:
            split = legacy_splits[key]
            tags = ["legacy_r2"]
            multiplier = 1.0
        elif image_id in R3_EMPTY_SPLITS:
            split = R3_EMPTY_SPLITS[image_id]
            tags = ["empty_table", "real_scene"]
            multiplier = 2.0 if split == "train" else 1.0
        elif _is_r3_new_key(key):
            split = stable_r3_split_for_key(key)
            class_names = {str(obj.get("class_name", "")) for obj in record.get("objects", [])}
            tags = ["real_scene"]
            if "tissue" in class_names:
                tags.append("tissue_variant")
            multiplier = 1.0
            if class_names & R3_WEAK_VIEW_CLASSES:
                tags.append("weak_view")
                multiplier = 1.5
        else:
            unclassified.append(key)
            continue
        record["split"] = split
        record["r3_tags"] = tags
        record["sampler_multiplier"] = multiplier
        records.append(record)

    if unclassified:
        raise ValueError(f"records absent from R2 and outside the approved R3 policy: {sorted(unclassified)}")
    if excluded_seen:
        raise ValueError(f"excluded images unexpectedly have annotations: {sorted(excluded_seen)}")
    _validate_r3_records(records, legacy_splits)

    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text("\n".join(json.dumps(record, ensure_ascii=False, sort_keys=True) for record in records) + "\n", encoding="utf-8")
    policy_out = Path(policy_path) if policy_path is not None else out.with_name("r3_dataset_policy.json")
    snapshot_out = Path(snapshot_path) if snapshot_path is not None else out.with_name("r3_dataset_snapshot.json")
    policy = _build_policy(root, r2_manifest, out, records, source_manifest)
    policy_out.write_text(json.dumps(policy, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    snapshot = _build_snapshot(root, out, policy_out, records)
    snapshot_out.write_text(json.dumps(snapshot, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    return {
        "ok": True,
        "manifest": out.resolve().as_posix(),
        "source_manifest": source_manifest.resolve().as_posix(),
        "policy": policy_out.resolve().as_posix(),
        "snapshot": snapshot_out.resolve().as_posix(),
        "record_count": len(records),
        "split_counts": dict(Counter(str(record["split"]) for record in records)),
        "excluded_image_ids": list(R3_EXCLUDED_IMAGE_IDS),
    }


def evaluate_v2_r3_run(run_dir: str | Path, r2_run_dir: str | Path) -> dict[str, Any]:
    """Evaluate terminal R3 holdouts without feeding either result into checkpoint selection."""
    run = Path(run_dir)
    train_report = _read_json(run / "train_report.json")
    r2_report = _read_json(Path(r2_run_dir) / "train_report.json")
    manifest = Path(train_report["data"]["manifest"])
    checkpoint_path = Path(train_report["checkpoint"])
    input_size = int(train_report["model"]["input_size"])
    width = int(train_report["model"]["width"])
    score_threshold = float(train_report["hyperparameters"]["eval_score_threshold"])
    nms_iou_threshold = float(train_report["hyperparameters"]["nms_iou_threshold"])
    torch = __import__("torch")
    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    model = make_ethossafedet_v2(EthosSafeDetV2Config(input_size=input_size, width=width))
    model.load_state_dict(checkpoint.get("model_state", checkpoint))
    model.eval()

    evaluations: dict[str, dict[str, Any]] = {}
    for split in ("real_scene_holdout", "empty_table_holdout"):
        dataset = EthosSafeDetV2Dataset(manifest, split, input_size=input_size, augment=False, cache_images=True)
        evaluations[split] = evaluate_v2_model(
            model,
            dataset,
            device="cpu",
            input_size=input_size,
            score_threshold=score_threshold,
            nms_iou_threshold=nms_iou_threshold,
            batch_size=24,
            num_workers=0,
            limit=None,
            use_amp=False,
        )
    payload = {
        "schema_version": R3_SCHEMA_VERSION,
        "selection_boundary": "best checkpoint selected from validation only; holdouts evaluated after selection",
        "run": run.resolve().as_posix(),
        "checkpoint": checkpoint_path.resolve().as_posix(),
        "checkpoint_sha256": _sha256_file(checkpoint_path),
        "manifest": manifest.resolve().as_posix(),
        "manifest_sha256": _sha256_file(manifest),
        "score_threshold": score_threshold,
        "nms_iou_threshold": nms_iou_threshold,
        "r2_legacy_test": r2_report.get("test_metrics", {}),
        "r3_legacy_test": train_report.get("test_metrics", {}),
        "holdouts": evaluations,
    }
    out = run / "r3_evaluation.json"
    out.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    _write_csv(run / "r3_evaluation_summary.csv", _evaluation_summary_rows(payload))
    return {"ok": True, "evaluation": out.resolve().as_posix(), "holdouts": evaluations}


def make_v2_r3_formal_report(run_dir: str | Path) -> dict[str, Any]:
    """Extend the base formal report with immutable R3 data and holdout evidence."""
    run = Path(run_dir)
    base = make_v2_formal_report(run)
    evaluation = _read_json(run / "r3_evaluation.json")
    train_report = _read_json(run / "train_report.json")
    manifest = Path(train_report["data"]["manifest"])
    policy_path = manifest.with_name("r3_dataset_policy.json")
    snapshot_path = manifest.with_name("r3_dataset_snapshot.json")
    assets_dir = run / "formal_report_assets"
    input_size = int(train_report["model"]["input_size"])
    width = int(train_report["model"]["width"])
    score_threshold = float(train_report["hyperparameters"]["eval_score_threshold"])
    nms_iou_threshold = float(train_report["hyperparameters"]["nms_iou_threshold"])
    torch = __import__("torch")
    import numpy as np

    checkpoint = torch.load(Path(train_report["checkpoint"]), map_location="cpu")
    model = make_ethossafedet_v2(EthosSafeDetV2Config(input_size=input_size, width=width))
    model.load_state_dict(checkpoint.get("model_state", checkpoint))
    model.eval()
    records = load_v2_manifest_records(manifest)
    real_records = [record for record in records if record.get("split") == "real_scene_holdout"]
    empty_records = [record for record in records if record.get("split") == "empty_table_holdout"]
    real_examples = _collect_object_examples(model, torch, np, real_records, input_size, score_threshold, nms_iou_threshold)
    empty_examples = _collect_empty_examples(model, torch, np, empty_records, input_size, score_threshold, nms_iou_threshold)

    figures: dict[str, str] = {}
    real_figure = assets_dir / "fig13_real_scene_holdout_examples.png"
    selected_real = _select_real_examples(real_examples, limit=4)
    if _validation_panel(real_figure, selected_real, "R3 real-scene: tissue, rear views, and robot-arm-edge captures"):
        figures["real_scene_holdout_examples"] = real_figure.relative_to(run).as_posix()
    empty_figure = assets_dir / "fig14_empty_table_holdout_examples.png"
    if _empty_panel(empty_figure, empty_examples):
        figures["empty_table_holdout_examples"] = empty_figure.relative_to(run).as_posix()

    tables = {
        "r3_evaluation_summary": _write_table(assets_dir / "r3_evaluation_summary.csv", _evaluation_summary_rows(evaluation), run),
        "r3_real_scene_per_class": _write_table(
            assets_dir / "r3_real_scene_per_class.csv",
            _per_class_rows(evaluation["holdouts"]["real_scene_holdout"]),
            run,
        ),
        "r3_real_scene_examples": _write_table(assets_dir / "r3_real_scene_examples.csv", _example_rows(selected_real), run),
        "r3_empty_table_predictions": _write_table(assets_dir / "r3_empty_table_predictions.csv", _empty_rows(empty_examples), run),
    }
    report_path = run / "formal_report.md"
    report_path.write_text(report_path.read_text(encoding="utf-8") + _r3_markdown(evaluation, policy_path, snapshot_path, figures, tables), encoding="utf-8")
    return {
        **base,
        "figure_count": int(base["figure_count"]) + len(figures),
        "table_count": int(base["table_count"]) + len(tables),
        "r3_figures": figures,
        "r3_tables": tables,
    }


def _ensure_r3_empty_annotations(root: Path, *, write: bool) -> None:
    for image_id in R3_EMPTY_SPLITS:
        image_path = images_root(root) / R3_NEW_CAMERA / f"{image_id}.png"
        annotation_path = annotations_root(root) / R3_NEW_CAMERA / f"{image_id}.json"
        if not image_path.is_file():
            raise FileNotFoundError(f"R3 empty-table image is missing: {image_path}")
        if annotation_path.is_file():
            annotation = read_annotation(annotation_path)
            if annotation.get("objects"):
                raise ValueError(f"R3 empty-table annotation contains objects: {annotation_path}")
            continue
        if not write:
            raise FileNotFoundError(f"R3 empty-table annotation is missing: {annotation_path}")
        with Image.open(image_path) as image:
            width, height = image.size
        annotation_path.parent.mkdir(parents=True, exist_ok=True)
        annotation_path.write_text(
            json.dumps(
                {
                    "image_id": image_id,
                    "image_path": str(image_path),
                    "width": width,
                    "height": height,
                    "camera": R3_NEW_CAMERA,
                    "source": "model_a_v2_r3_empty_table",
                    "split": R3_EMPTY_SPLITS[image_id],
                    "objects": [],
                },
                indent=2,
                ensure_ascii=False,
            )
            + "\n",
            encoding="utf-8",
        )


def _validate_excluded_images(root: Path) -> None:
    for image_id in R3_EXCLUDED_IMAGE_IDS:
        image_path = images_root(root) / R3_NEW_CAMERA / f"{image_id}.png"
        if not image_path.is_file():
            raise FileNotFoundError(f"R3 excluded image is missing: {image_path}")


def _record_key(record: dict[str, Any], root: Path) -> str:
    annotation = Path(str(record["annotation_path"])).resolve()
    try:
        return annotation.relative_to(annotations_root(root).resolve()).with_suffix("").as_posix()
    except ValueError as exc:
        raise ValueError(f"annotation path is outside the dataset root: {annotation}") from exc


def _is_r3_new_key(key: str) -> bool:
    parts = Path(key).parts
    if len(parts) != 2 or parts[0] != R3_NEW_CAMERA:
        return False
    try:
        image_id = int(parts[1])
    except ValueError:
        return False
    return R3_NEW_IMAGE_MIN <= image_id <= R3_NEW_IMAGE_MAX


def _validate_r3_records(records: list[dict[str, Any]], legacy_splits: dict[str, str]) -> None:
    by_key = {_record_key(record, Path(record["dataset_root"])): record for record in records}
    for key, legacy_split in legacy_splits.items():
        record = by_key.get(key)
        if record is None:
            raise ValueError(f"R2 record is missing from R3 manifest: {key}")
        if record.get("split") != legacy_split:
            raise ValueError(f"R2 split changed for {key}: {record.get('split')} != {legacy_split}")
    for record in records:
        split = str(record.get("split"))
        if split in {"real_scene_holdout", "empty_table_holdout"} and "legacy_r2" in record.get("r3_tags", []):
            raise ValueError(f"legacy R2 record leaked into R3 holdout: {_record_key(record, Path(record['dataset_root']))}")
    for split in ("train", "val", "test"):
        if not any(record.get("split") == split for record in records):
            raise ValueError(f"R3 manifest has no {split!r} records")
    if not any(record.get("split") == "real_scene_holdout" for record in records):
        raise ValueError("R3 manifest has no real-scene holdout records")
    if sum(1 for record in records if record.get("split") == "empty_table_holdout") != 2:
        raise ValueError("R3 manifest must contain exactly two empty-table holdout records")
    legacy_class_ids = {
        int(obj["class_id"])
        for record in records
        if "legacy_r2" in record.get("r3_tags", [])
        for obj in record.get("objects", [])
    }
    required_class_ids = set(range(len(ETHOSSAFEDET_CLASS_NAMES)))
    if legacy_class_ids != required_class_ids:
        missing = sorted(required_class_ids - legacy_class_ids)
        raise ValueError(f"frozen R2 records do not retain all seven classes: missing={missing}")
    for record in records:
        image_path = resolve_v2_record_image(record)
        with Image.open(image_path) as image:
            image_size = image.size
        if image_size != (640, 480):
            raise ValueError(f"R3 requires VGA 640x480 images: {image_path} is {image_size}")
        if int(record.get("width", 0)) != 640 or int(record.get("height", 0)) != 480:
            raise ValueError(f"R3 manifest image dimensions are not VGA: {image_path}")


def _build_policy(root: Path, r2_manifest: Path, manifest: Path, records: list[dict[str, Any]], source_manifest: Path) -> dict[str, Any]:
    return {
        "schema_version": R3_SCHEMA_VERSION,
        "dataset_root": root.as_posix(),
        "r2_manifest": r2_manifest.resolve().as_posix(),
        "r2_manifest_sha256": _sha256_file(r2_manifest),
        "source_manifest": source_manifest.resolve().as_posix(),
        "manifest": manifest.resolve().as_posix(),
        "rules": {
            "legacy": "preserve every R2 annotation-relative key and split",
            "new_range": {"camera": R3_NEW_CAMERA, "first_image_id": R3_NEW_IMAGE_MIN, "last_image_id": R3_NEW_IMAGE_MAX},
            "new_split": "sha1(schema_version + annotation-relative-key) % 100: train<80, val<90, real_scene_holdout>=90",
            "empty_table_splits": R3_EMPTY_SPLITS,
            "excluded_image_ids": list(R3_EXCLUDED_IMAGE_IDS),
            "excluded_image_reasons": R3_EXCLUDED_IMAGE_REASONS,
            "sampler_multiplier": {
                "empty_table_train": 2.0,
                "new_real_scene_with_phial_bottle_or_phone": 1.5,
                "other_records": 1.0,
                "combined_cap": 3.0,
            },
        },
        "record_counts": dict(Counter(str(record["split"]) for record in records)),
    }


def _build_snapshot(root: Path, manifest: Path, policy_path: Path, records: list[dict[str, Any]]) -> dict[str, Any]:
    rows = []
    for record in records:
        annotation = Path(str(record["annotation_path"]))
        image = resolve_v2_record_image(record)
        rows.append(
            {
                "annotation_key": _record_key(record, root),
                "image_id": str(record["image_id"]),
                "split": str(record["split"]),
                "r3_tags": list(record.get("r3_tags", [])),
                "annotation_sha256": _sha256_file(annotation),
                "image_sha256": _sha256_file(image),
            }
        )
    return {
        "schema_version": R3_SCHEMA_VERSION,
        "manifest": manifest.resolve().as_posix(),
        "manifest_sha256": _sha256_file(manifest),
        "policy": policy_path.resolve().as_posix(),
        "policy_sha256": _sha256_file(policy_path),
        "records": rows,
    }


def _collect_object_examples(model, torch, np, records: list[dict[str, Any]], input_size: int, score_threshold: float, nms_iou_threshold: float) -> list[dict[str, Any]]:  # type: ignore[no-untyped-def]
    examples: list[dict[str, Any]] = []
    with torch.no_grad():
        for record in records:
            detections = _predict_record(model, torch, np, record, input_size, score_threshold, nms_iou_threshold)
            objects = [obj for obj in record.get("objects", []) if not record.get("negative")]
            for target in objects:
                class_id = int(target["class_id"])
                same_class = [det for det in detections if int(det["class_id"]) == class_id]
                match = max(same_class, key=lambda det: bbox_iou(det["bbox_xyxy_vga"], target["bbox_xyxy_vga"]), default=None)
                best_iou = bbox_iou(match["bbox_xyxy_vga"], target["bbox_xyxy_vga"]) if match is not None else 0.0
                theta_error = None
                if match is not None and target.get("theta_valid") and match.get("orientation_rad") is not None:
                    theta_error = orientation_abs_error(float(match["orientation_rad"]), float(target["orientation_rad"]))
                examples.append(
                    {
                        "record": record,
                        "objects": objects,
                        "detections": detections,
                        "target": target,
                        "match": match,
                        "class_id": class_id,
                        "class_name": ETHOSSAFEDET_CLASS_NAMES[class_id],
                        "image_id": str(record["image_id"]),
                        "best_iou": float(best_iou),
                        "score": float(match.get("score", 0.0)) if match is not None else 0.0,
                        "theta_error_rad": theta_error,
                    }
                )
    return examples


def _collect_empty_examples(model, torch, np, records: list[dict[str, Any]], input_size: int, score_threshold: float, nms_iou_threshold: float) -> list[dict[str, Any]]:  # type: ignore[no-untyped-def]
    examples: list[dict[str, Any]] = []
    with torch.no_grad():
        for record in records:
            detections = _predict_record(model, torch, np, record, input_size, score_threshold, nms_iou_threshold)
            examples.append({"record": record, "detections": detections, "max_score": max((float(det["score"]) for det in detections), default=0.0)})
    return examples


def _select_real_examples(examples: list[dict[str, Any]], limit: int) -> list[dict[str, Any]]:
    selected: list[dict[str, Any]] = []
    used: set[tuple[str, int]] = set()
    for class_name in ("tissue", "phial", "bottle", "phone"):
        candidates = [item for item in examples if item["class_name"] == class_name]
        candidates.sort(key=lambda item: (float(item["best_iou"]), float(item["score"])), reverse=True)
        if candidates:
            selected.append(candidates[0])
            used.add((str(candidates[0]["image_id"]), int(candidates[0]["class_id"])))
    for item in sorted(examples, key=lambda item: (float(item["best_iou"]), -float(item["score"]))):
        key = (str(item["image_id"]), int(item["class_id"]))
        if key in used:
            continue
        selected.append(item)
        used.add(key)
        if len(selected) >= limit:
            break
    return selected[:limit]


def _empty_panel(path: Path, examples: list[dict[str, Any]]) -> bool:
    if not examples:
        return False
    import matplotlib

    matplotlib.use("Agg", force=True)
    import matplotlib.pyplot as plt

    count = len(examples)
    fig, axes = plt.subplots(1, count, figsize=(5.2 * count, 4.0))
    axes_list = [axes] if count == 1 else list(axes)
    for ax, example in zip(axes_list, examples):
        image = Image.open(resolve_v2_record_image(example["record"])).convert("RGB")
        draw = ImageDraw.Draw(image)
        for detection in example["detections"]:
            class_name = ETHOSSAFEDET_CLASS_NAMES[int(detection["class_id"])]
            x1, y1, x2, y2 = [float(value) for value in detection["bbox_xyxy_vga"]]
            draw.rectangle((x1, y1, x2, y2), outline="#d62728", width=3)
            draw.text((x1 + 2.0, max(0.0, y1 - 14.0)), f"P {class_name} {float(detection['score']):.2f}", fill="#d62728")
        ax.imshow(image)
        ax.set_title(f"{example['record']['image_id']} | predictions {len(example['detections'])} | max {float(example['max_score']):.2f}", fontsize=9)
        ax.axis("off")
    fig.suptitle("R3 empty-table holdout at score threshold 0.25", fontsize=13)
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    fig.savefig(path, dpi=180)
    plt.close(fig)
    return True


def _evaluation_summary_rows(evaluation: dict[str, Any]) -> list[dict[str, Any]]:
    rows = []
    metrics_by_name = {
        "r2_legacy_test": evaluation.get("r2_legacy_test", {}),
        "r3_legacy_test": evaluation.get("r3_legacy_test", {}),
        **dict(evaluation.get("holdouts", {})),
    }
    for name, metrics in metrics_by_name.items():
        rows.append(
            {
                "evaluation_set": name,
                "eval_count": metrics.get("eval_count", 0),
                "gt_count": metrics.get("gt_count", 0),
                "recall50": metrics.get("recall50"),
                "best_iou_mean": metrics.get("best_iou_mean"),
                "primary_class_acc": metrics.get("primary_class_acc"),
                "theta_abs_error_rad_mean": metrics.get("theta_abs_error_rad_mean"),
                "negative_image_count": metrics.get("negative_image_count", 0),
                "negative_image_false_positive_count": metrics.get("negative_image_false_positive_count", 0),
                "negative_image_false_positive_rate": metrics.get("negative_image_false_positive_rate"),
                "negative_detection_count": metrics.get("negative_detection_count", 0),
            }
        )
    return rows


def _per_class_rows(metrics: dict[str, Any]) -> list[dict[str, Any]]:
    rows = []
    for class_name in ETHOSSAFEDET_CLASS_NAMES:
        per_class = dict(metrics.get("per_class", {}).get(class_name, {}))
        if int(per_class.get("gt", 0)) <= 0:
            continue
        rows.append(
            {
                "class_name": class_name,
                "gt_count": int(per_class.get("gt", 0)),
                "recall50": per_class.get("recall50"),
                "best_iou_mean": per_class.get("best_iou_mean"),
                "theta_abs_error_rad_mean": per_class.get("theta_abs_error_rad_mean"),
                "theta_eval_count": int(per_class.get("theta_count", 0)),
            }
        )
    return rows


def _example_rows(examples: list[dict[str, Any]]) -> list[dict[str, Any]]:
    rows = []
    for example in examples:
        match = example.get("match") or {}
        target = example["target"]
        rows.append(
            {
                "image_id": example["image_id"],
                "class_name": example["class_name"],
                "target_bbox_xyxy_vga": json.dumps(target.get("bbox_xyxy_vga", [])),
                "prediction_bbox_xyxy_vga": json.dumps(match.get("bbox_xyxy_vga", [])),
                "score": example["score"],
                "best_iou": example["best_iou"],
                "theta_error_rad": example["theta_error_rad"],
            }
        )
    return rows


def _empty_rows(examples: list[dict[str, Any]]) -> list[dict[str, Any]]:
    rows = []
    for example in examples:
        record = example["record"]
        if not example["detections"]:
            rows.append({"image_id": record["image_id"], "class_name": "", "score": None, "bbox_xyxy_vga": "", "prediction_count": 0})
        for detection in example["detections"]:
            rows.append(
                {
                    "image_id": record["image_id"],
                    "class_name": ETHOSSAFEDET_CLASS_NAMES[int(detection["class_id"])],
                    "score": float(detection["score"]),
                    "bbox_xyxy_vga": json.dumps(detection["bbox_xyxy_vga"]),
                    "prediction_count": len(example["detections"]),
                }
            )
    return rows


def _write_table(path: Path, rows: list[dict[str, Any]], run: Path) -> str:
    _write_csv(path, rows)
    return path.relative_to(run).as_posix()


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    fields = sorted({field for row in rows for field in row}) or ["empty"]
    with path.open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def _r3_markdown(evaluation: dict[str, Any], policy_path: Path, snapshot_path: Path, figures: dict[str, str], tables: dict[str, str]) -> str:
    real = evaluation["holdouts"]["real_scene_holdout"]
    empty = evaluation["holdouts"]["empty_table_holdout"]
    r2 = evaluation.get("r2_legacy_test", {})
    r3 = evaluation.get("r3_legacy_test", {})
    lines = [
        "",
        "## R3 Real-Scene Data Policy",
        "",
        f"- Policy: `{policy_path.as_posix()}`",
        f"- Immutable dataset snapshot: `{snapshot_path.as_posix()}`",
        "- R2 legacy membership remains frozen. The new real-scene and empty-table holdouts were evaluated only after the validation-selected checkpoint was fixed.",
        "",
        "## R3 Dual-Track Results",
        "",
        "| Evaluation set | primary class acc | recall@0.5 | mean best IoU | theta MAE rad | negative FP rate |",
        "|---|---:|---:|---:|---:|---:|",
        _metric_row("R2 frozen legacy test", r2),
        _metric_row("R3 frozen legacy test", r3),
        _metric_row("R3 real-scene holdout", real),
        _metric_row("R3 empty-table holdout", empty),
        "",
        "The real-scene holdout covers the newly introduced capture distribution, including the tissue variant and the targeted phial/bottle/phone views. The fixed qualitative panel retains the robot-arm edge at left in every capture. It is terminal evidence rather than a model-selection signal.",
        "Per-class denominators remain binding: the current remote holdout has one object and it is missed, so this holdout supports the targeted phial/bottle/phone/tissue evidence but is not a broad all-class acceptance claim.",
        "No new real-scene holdout record has a valid orientation label, so direction MAE is explicitly not evaluable for this split rather than a zero-error result.",
        "",
        "### R3 Real-Scene Holdout Per-Class",
        "",
        "| Class | GT | recall@0.5 | mean best IoU | theta MAE rad | theta labels |",
        "|---|---:|---:|---:|---:|---:|",
        *_per_class_markdown_rows(real),
        "",
    ]
    if "real_scene_holdout_examples" in figures:
        lines.extend([f"![Figure 13. R3 real-scene holdout examples.]({figures['real_scene_holdout_examples']})", "", "*Figure 13. Tissue variant and phial/bottle/phone rear-view detections in real captures with the robot-arm edge retained at left.*", ""])
    lines.extend(["## R3 Empty-Table False-Positive Check", "", f"- Threshold: `{float(evaluation['score_threshold']):.2f}`", f"- Empty images: `{empty.get('negative_image_count', 0)}`", f"- Images with any prediction: `{empty.get('negative_image_false_positive_count', 0)}`", f"- False-positive rate: `{_fmt(empty.get('negative_image_false_positive_rate'))}`", "- This is a two-image regression check, not a sufficient empty-table field-rate estimate.", ""])
    if "empty_table_holdout_examples" in figures:
        lines.extend([f"![Figure 14. R3 empty-table holdout predictions.]({figures['empty_table_holdout_examples']})", "", "*Figure 14. Empty-table predictions after class-wise NMS at the fixed evaluation threshold.*", ""])
    lines.extend(["R3 derived tables:", "", *[f"- `{name}`: `{path}`" for name, path in sorted(tables.items())], ""])
    return "\n".join(lines)


def _metric_row(name: str, metrics: dict[str, Any]) -> str:
    return "| {} | {} | {} | {} | {} | {} |".format(
        name,
        _fmt(metrics.get("primary_class_acc")),
        _fmt(metrics.get("recall50")),
        _fmt(metrics.get("best_iou_mean")),
        _fmt(metrics.get("theta_abs_error_rad_mean")),
        _fmt(metrics.get("negative_image_false_positive_rate")),
    )


def _per_class_markdown_rows(metrics: dict[str, Any]) -> list[str]:
    rows = []
    for row in _per_class_rows(metrics):
        rows.append(
            "| {class_name} | {gt_count} | {recall50} | {best_iou_mean} | {theta_abs_error_rad_mean} | {theta_eval_count} |".format(
                class_name=row["class_name"],
                gt_count=row["gt_count"],
                recall50=_fmt(row["recall50"]),
                best_iou_mean=_fmt(row["best_iou_mean"]),
                theta_abs_error_rad_mean=_fmt(row["theta_abs_error_rad_mean"]),
                theta_eval_count=row["theta_eval_count"],
            )
        )
    return rows


def _fmt(value: Any) -> str:
    return "n/a" if value is None else f"{float(value):.6f}"


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Prepare and report the Model A V2 R3 real-scene run.")
    subparsers = parser.add_subparsers(dest="command", required=True)
    prepare = subparsers.add_parser("prepare")
    prepare.add_argument("--dataset", required=True)
    prepare.add_argument("--r2-manifest", required=True)
    prepare.add_argument("--out", required=True)
    prepare.add_argument("--policy-out", default="")
    prepare.add_argument("--snapshot-out", default="")
    prepare.add_argument("--write-empty-annotations", action="store_true")
    evaluate = subparsers.add_parser("evaluate")
    evaluate.add_argument("--run", required=True)
    evaluate.add_argument("--r2-run", required=True)
    report = subparsers.add_parser("report")
    report.add_argument("--run", required=True)
    args = parser.parse_args(argv)
    if args.command == "prepare":
        result = prepare_v2_r3_manifest(
            args.dataset,
            args.r2_manifest,
            args.out,
            policy_path=args.policy_out or None,
            snapshot_path=args.snapshot_out or None,
            write_empty_annotations=args.write_empty_annotations,
        )
    elif args.command == "evaluate":
        result = evaluate_v2_r3_run(args.run, args.r2_run)
    else:
        result = make_v2_r3_formal_report(args.run)
    print(json.dumps(result, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
