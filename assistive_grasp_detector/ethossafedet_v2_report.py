"""Paper-style report generator for EthosSafeDetV2 training runs."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
from pathlib import Path
from typing import Any

from PIL import Image, ImageDraw

from assistive_grasp_detector.coords import letterbox_rgb_image
from assistive_grasp_detector.ethossafedet_postprocess import bbox_iou
from assistive_grasp_detector.ethossafedet_v2_manifest import load_v2_manifest_records, resolve_v2_record_image
from assistive_grasp_detector.ethossafedet_v2_model import EthosSafeDetV2Config, make_ethossafedet_v2
from assistive_grasp_detector.ethossafedet_v2_train import decode_v2_outputs, orientation_abs_error
from assistive_grasp_detector.schema import ETHOSSAFEDET_CLASS_NAMES


def make_v2_formal_report(run_dir: str | Path, output_path: str | Path | None = None) -> dict[str, Any]:
    run = Path(run_dir)
    train_report = _read_json(run / "train_report.json")
    gates = _read_gates(run / "gates")
    assets_dir = run / "formal_report_assets"
    assets_dir.mkdir(parents=True, exist_ok=True)
    _clear_generated_assets(assets_dir)
    qualitative = _build_validation_qualitative_examples(train_report)
    tables = _write_tables(assets_dir, train_report, gates, qualitative)
    figures = _write_figures(assets_dir, train_report, gates, qualitative)
    out = Path(output_path) if output_path is not None else run / "formal_report.md"
    markdown = _format_report(run, train_report, gates, figures, tables, qualitative)
    out.write_text(markdown, encoding="utf-8")
    return {
        "ok": True,
        "report": out.resolve().as_posix(),
        "figure_count": len(figures),
        "table_count": len(tables),
        "figures": figures,
        "tables": tables,
    }


def _format_report(
    run: Path,
    report: dict[str, Any],
    gates: dict[str, Any],
    figures: dict[str, str],
    tables: dict[str, str],
    qualitative: dict[str, Any],
) -> str:
    data = report["data"]
    model = report["model"]
    test = report.get("test_metrics", {})
    onnx_gate = gates.get("check_v2_onnx_ops", {})
    budget_gate = gates.get("check_v2_weight_budget", {})
    lines = [
        "# EthosSafeDetV2 Model A Formal Training Report",
        "",
        "## Abstract",
        "",
        (
            "This report documents the host-side formal training run for Model A V2 / EthosSafeDetV2. "
            "The candidate uses a 320x320 RGB input, seven canonical object classes, two detection scales, "
            "and separated class, LTRB box, and orientation heads. The evidence here covers dataset, training, "
            "validation, ONNX static gates, and artifact provenance; it does not claim board acceptance."
        ),
        "",
        "## Dataset And Annotation Contract",
        "",
        f"- Manifest: `{data['manifest']}`",
        f"- Manifest SHA-256: `{data['manifest_sha256']}`",
        f"- Images: `{data['record_count']}`",
        f"- Objects: `{data['object_count']}`",
        f"- Split counts: `{data['split_counts']}`",
        f"- Class counts: `{data['class_counts']}`",
        f"- Theta-valid counts: `{data.get('theta_valid_counts', {})}`",
        "",
        *_figure_block(figures, "split_distribution", "Figure 1. Stable train/validation/test split distribution."),
        *_figure_block(figures, "class_distribution", "Figure 2. Object count by class."),
        *_figure_block(figures, "theta_valid_coverage", "Figure 3. Orientation-label coverage by class."),
        *_figure_block(figures, "class_weight_sampling", "Figure 4. Class weighting used for imbalance control."),
        "",
        "## Model Architecture",
        "",
        f"- Width multiplier base: `{model['width']}`",
        f"- Parameter count: `{model['parameter_count']}`",
        f"- Estimated INT8 weights: `{model['estimated_int8_weight_bytes']}` bytes",
        f"- Estimated FP32 weights: `{model['estimated_fp32_weight_bytes']}` bytes",
        f"- Output tensors: `{model['output_names']}`",
        "",
        "The graph intentionally leaves sigmoid, decode, atan2, candidate filtering, and NMS outside the model graph.",
        "",
        "## Training Protocol",
        "",
        f"- Hyperparameters: `{report['hyperparameters']}`",
        f"- Class balance strategy: `{report['class_balance']['strategy']}`",
        f"- Best epoch: `{report['best_epoch']}`",
        f"- Best metric: `{report['best_metric']:.6f}`",
        f"- Best checkpoint SHA-256: `{report['checkpoint_sha256']}`",
        "",
        *_figure_block(figures, "loss_curves", "Figure 5. Training and validation loss curves."),
        *_figure_block(figures, "validation_metrics", "Figure 6. Validation metrics over epochs."),
        "",
        "## Validation Qualitative Results",
        "",
        (
            "Validation examples are rendered from the best checkpoint. Green boxes are ground truth, "
            "orange boxes are the class-matched prediction used for the displayed IoU, and blue boxes "
            "are other high-scoring predictions after class-wise NMS. Thin green/orange axes mark "
            "ground-truth and predicted orientation where available."
        ),
        "",
        *_figure_block(figures, "validation_success_examples", "Figure 7. Representative validation detections across classes."),
        *_figure_block(figures, "validation_hard_examples", "Figure 8. Hard validation examples and residual failure modes."),
        *_qualitative_note(qualitative),
        "",
        "## Results",
        "",
        f"- Test recall@0.5: `{_fmt(test.get('recall50'))}`",
        f"- Test mean best IoU: `{_fmt(test.get('best_iou_mean'))}`",
        f"- Test primary class accuracy: `{_fmt(test.get('primary_class_acc'))}`",
        f"- Test theta MAE rad: `{_fmt(test.get('theta_abs_error_rad_mean'))}` over `{test.get('theta_eval_count', 0)}` matched theta-valid objects",
        "",
        *_figure_block(figures, "per_class_val_metrics", "Figure 9. Per-class validation metrics."),
        *_figure_block(figures, "test_metrics", "Figure 10. Per-class test metrics."),
        "",
        "## Deployment Gate Status",
        "",
        f"- ONNX static gate: `{_status(onnx_gate)}`",
        f"- Weight budget gate: `{_status(budget_gate)}`",
        "- TFLite full-int8: `BLOCKED/not run` because TensorFlow is not installed in the current ma2 training environment.",
        "- Host MERA, RUHMI dispatch, and board static golden: `BLOCKED/not run`; these remain post-training acceptance gates.",
        "",
        *_figure_block(figures, "artifact_sizes", "Figure 11. Artifact and estimated model sizes."),
        *_figure_block(figures, "gate_status", "Figure 12. Gate status matrix."),
        "",
        "## Reproducibility Artifacts",
        "",
        "| Artifact | Path | SHA-256 | Bytes |",
        "|---|---:|---:|---:|",
        *[_artifact_row(item) for item in _artifact_rows(run, report)],
        "",
        "## Derived Tables",
        "",
        "| Table | Path |",
        "|---|---|",
        *[f"| `{name}` | `{path}` |" for name, path in sorted(tables.items())],
        "",
        "## Limitations And Next Work",
        "",
        "- This report is a host-side formal training report, not a board acceptance report.",
        "- TFLite, host MERA, RUHMI dispatch, and board static golden remain required before firmware can consume this candidate.",
        "- The detector metric is a pragmatic per-object recall/IoU/theta proxy, not a full COCO mAP benchmark.",
        "",
    ]
    return "\n".join(lines)


def _write_tables(assets_dir: Path, report: dict[str, Any], gates: dict[str, Any], qualitative: dict[str, Any]) -> dict[str, str]:
    tables = {
        "split_distribution": _split_rows(report["data"]),
        "class_distribution": _class_rows(report["data"]),
        "theta_coverage": _theta_rows(report["data"]),
        "class_weights": _class_weight_rows(report),
        "epoch_history": _history_rows(report),
        "per_class_validation": _per_class_rows(_last_measured(report["history"])),
        "per_class_test": _per_class_rows(report.get("test_metrics", {})),
        "qualitative_validation_examples": _qualitative_rows(qualitative),
        "gate_matrix": _gate_rows(gates),
        "artifact_provenance": _artifact_rows(assets_dir.parent, report),
    }
    result: dict[str, str] = {}
    for name, rows in tables.items():
        if not rows:
            continue
        path = assets_dir / f"{name}.csv"
        _write_csv(path, rows)
        result[name] = path.relative_to(assets_dir.parent).as_posix()
    return result


def _write_figures(assets_dir: Path, report: dict[str, Any], gates: dict[str, Any], qualitative: dict[str, Any]) -> dict[str, str]:
    jobs = [
        ("split_distribution", "fig01_split_distribution.png", lambda p: _bar(p, _split_rows(report["data"]), "split", "image_count", "Split distribution", "Images")),
        ("class_distribution", "fig02_class_distribution.png", lambda p: _bar(p, _class_rows(report["data"]), "class_name", "object_count", "Class distribution", "Objects", rotate=25)),
        ("theta_valid_coverage", "fig03_theta_valid_coverage.png", lambda p: _bar(p, _theta_rows(report["data"]), "class_name", "theta_valid", "Theta-valid coverage", "Objects", rotate=25)),
        ("class_weight_sampling", "fig04_class_weight_sampling.png", lambda p: _bar(p, _class_weight_rows(report), "class_name", "class_weight", "Class weights", "Weight", rotate=25)),
        ("loss_curves", "fig05_loss_curves.png", lambda p: _plot_history(p, report, ["train_loss", "val_loss", "train_cls_loss", "train_box_loss", "train_ori_loss"], "Loss curves")),
        ("validation_metrics", "fig06_validation_metrics.png", lambda p: _plot_history(p, report, ["recall50", "best_iou_mean", "primary_class_acc"], "Validation metrics")),
        ("validation_success_examples", "fig07_validation_success_examples.png", lambda p: _validation_panel(p, qualitative.get("representative", []), "Representative validation detections")),
        ("validation_hard_examples", "fig08_validation_hard_examples.png", lambda p: _validation_panel(p, qualitative.get("hard_cases", []), "Hard validation examples")),
        ("per_class_val_metrics", "fig09_per_class_val_metrics.png", lambda p: _bar(p, _per_class_rows(_last_measured(report["history"])), "class_name", "recall50", "Per-class validation recall@0.5", "Recall", rotate=25)),
        ("test_metrics", "fig10_test_metrics.png", lambda p: _bar(p, _per_class_rows(report.get("test_metrics", {})), "class_name", "recall50", "Per-class test recall@0.5", "Recall", rotate=25)),
        ("artifact_sizes", "fig11_artifact_sizes.png", lambda p: _bar(p, _artifact_rows(assets_dir.parent, report), "artifact", "bytes", "Artifact sizes", "Bytes", rotate=25)),
        ("gate_status", "fig12_gate_status.png", lambda p: _bar(p, _gate_rows(gates), "gate", "score", "Gate status", "1=PASS", rotate=25)),
    ]
    result: dict[str, str] = {}
    for key, filename, writer in jobs:
        path = assets_dir / filename
        if writer(path):
            result[key] = path.relative_to(assets_dir.parent).as_posix()
    return result


def _split_rows(data: dict[str, Any]) -> list[dict[str, Any]]:
    total = max(1, int(data.get("record_count", 0)))
    return [{"split": split, "image_count": int(count), "share": float(count) / total} for split, count in sorted(data.get("split_counts", {}).items())]


def _class_rows(data: dict[str, Any]) -> list[dict[str, Any]]:
    total = max(1, int(data.get("object_count", 0)))
    return [
        {"class_name": name, "object_count": int(data.get("class_counts", {}).get(name, 0)), "share": int(data.get("class_counts", {}).get(name, 0)) / total}
        for name in ETHOSSAFEDET_CLASS_NAMES
    ]


def _theta_rows(data: dict[str, Any]) -> list[dict[str, Any]]:
    rows = []
    theta_counts = data.get("theta_valid_counts", {})
    class_counts = data.get("class_counts", {})
    for name in ETHOSSAFEDET_CLASS_NAMES:
        total = int(class_counts.get(name, 0))
        theta = int(theta_counts.get(name, 0))
        rows.append({"class_name": name, "theta_valid": theta, "object_count": total, "coverage": theta / max(1, total)})
    return rows


def _class_weight_rows(report: dict[str, Any]) -> list[dict[str, Any]]:
    weights = report.get("class_balance", {}).get("class_weights", {})
    return [{"class_name": name, "class_weight": float(weights.get(name, 0.0))} for name in ETHOSSAFEDET_CLASS_NAMES]


def _history_rows(report: dict[str, Any]) -> list[dict[str, Any]]:
    rows = []
    for row in report.get("history", []):
        rows.append({key: value for key, value in row.items() if not isinstance(value, (dict, list))})
    return rows


def _per_class_rows(metrics: dict[str, Any]) -> list[dict[str, Any]]:
    per_class = metrics.get("per_class", {}) if isinstance(metrics, dict) else {}
    return [
        {
            "class_name": name,
            "gt": per_class.get(name, {}).get("gt", 0),
            "recall50": _num(per_class.get(name, {}).get("recall50")),
            "best_iou_mean": _num(per_class.get(name, {}).get("best_iou_mean")),
            "theta_abs_error_rad_mean": _num(per_class.get(name, {}).get("theta_abs_error_rad_mean")),
            "theta_count": per_class.get(name, {}).get("theta_count", 0),
        }
        for name in ETHOSSAFEDET_CLASS_NAMES
    ]


def _gate_rows(gates: dict[str, Any]) -> list[dict[str, Any]]:
    rows = []
    for name in ("check_v2_onnx_ops", "check_v2_weight_budget"):
        gate = gates.get(name, {})
        rows.append({"gate": name, "status": _status(gate), "score": 1.0 if gate.get("ok") else 0.0, "reason": gate.get("reason", "")})
    rows.extend(
        [
            {"gate": "tflite_full_int8", "status": "BLOCKED/not_run", "score": 0.0, "reason": "tensorflow not installed"},
            {"gate": "host_mera", "status": "BLOCKED/not_run", "score": 0.0, "reason": "post-training acceptance gate"},
            {"gate": "ruhmi_dispatch", "status": "BLOCKED/not_run", "score": 0.0, "reason": "post-training acceptance gate"},
            {"gate": "board_static_golden", "status": "BLOCKED/not_run", "score": 0.0, "reason": "post-training acceptance gate"},
        ]
    )
    return rows


def _artifact_rows(run: Path, report: dict[str, Any]) -> list[dict[str, Any]]:
    items = [
        ("best_checkpoint", Path(report["checkpoint"])),
        ("last_checkpoint", Path(report["last_checkpoint"])),
        ("train_report_json", run / "train_report.json"),
        ("train_report_md", run / "train_report.md"),
        ("train_history_csv", run / "train_history.csv"),
    ]
    export_dir = run / "export"
    if export_dir.is_dir():
        for path in sorted(export_dir.glob("*")):
            if path.is_file():
                items.append((path.stem, path))
    rows = []
    for name, path in items:
        if path.is_file():
            rows.append({"artifact": name, "path": path.as_posix(), "sha256": _sha256_file(path), "bytes": path.stat().st_size})
    return rows


def _build_validation_qualitative_examples(report: dict[str, Any], max_representative: int = 8, max_hard: int = 4) -> dict[str, Any]:
    manifest = Path(report.get("data", {}).get("manifest", ""))
    checkpoint_path = Path(report.get("checkpoint", ""))
    if not manifest.is_file() or not checkpoint_path.is_file():
        return {"representative": [], "hard_cases": [], "rows": [], "note": "manifest or checkpoint missing"}
    try:
        torch = __import__("torch")
        import numpy as local_np
    except Exception as exc:  # pragma: no cover - dependency availability is environment-specific.
        return {"representative": [], "hard_cases": [], "rows": [], "note": f"qualitative rendering skipped: {exc}"}

    input_size = int(report.get("model", {}).get("input_size", 320))
    width = int(report.get("model", {}).get("width", 40))
    score_threshold = float(report.get("hyperparameters", {}).get("eval_score_threshold", 0.25))
    nms_iou_threshold = float(report.get("hyperparameters", {}).get("nms_iou_threshold", 0.5))

    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    state = checkpoint.get("model_state", checkpoint)
    model = make_ethossafedet_v2(EthosSafeDetV2Config(input_size=input_size, num_classes=len(ETHOSSAFEDET_CLASS_NAMES), width=width))
    model.load_state_dict(state)
    model.eval()

    candidates: list[dict[str, Any]] = []
    val_records = [record for record in load_v2_manifest_records(manifest) if record.get("split") == "val"]
    with torch.no_grad():
        for record in val_records:
            detections = _predict_record(model, torch, local_np, record, input_size, score_threshold, nms_iou_threshold)
            objects = [obj for obj in record.get("objects", []) if not record.get("negative")]
            for obj in objects:
                class_id = int(obj["class_id"])
                same_class = [det for det in detections if int(det["class_id"]) == class_id]
                best = max(same_class, key=lambda det: bbox_iou(det["bbox_xyxy_vga"], obj["bbox_xyxy_vga"]), default=None)
                best_iou = bbox_iou(best["bbox_xyxy_vga"], obj["bbox_xyxy_vga"]) if best is not None else 0.0
                theta_error = None
                if best is not None and obj.get("theta_valid") and best.get("orientation_rad") is not None:
                    theta_error = orientation_abs_error(float(best["orientation_rad"]), float(obj["orientation_rad"]))
                candidates.append(
                    {
                        "panel": "",
                        "record": record,
                        "objects": objects,
                        "detections": detections,
                        "target": obj,
                        "match": best,
                        "class_id": class_id,
                        "class_name": ETHOSSAFEDET_CLASS_NAMES[class_id],
                        "image_id": str(record.get("image_id", "")),
                        "best_iou": float(best_iou),
                        "score": float(best.get("score", 0.0)) if best is not None else 0.0,
                        "theta_error_rad": theta_error,
                    }
                )

    representative = _select_representative_examples(candidates, max_representative)
    hard_cases = _select_hard_examples(candidates, representative, max_hard)
    for item in representative:
        item["panel"] = "representative"
    for item in hard_cases:
        item["panel"] = "hard_case"
    rows = [_qualitative_row(item) for item in [*representative, *hard_cases]]
    return {"representative": representative, "hard_cases": hard_cases, "rows": rows, "note": ""}


def _predict_record(
    model,  # type: ignore[no-untyped-def]
    torch,  # type: ignore[no-untyped-def]
    local_np,  # type: ignore[no-untyped-def]
    record: dict[str, Any],
    input_size: int,
    score_threshold: float,
    nms_iou_threshold: float,
) -> list[dict[str, Any]]:
    with Image.open(resolve_v2_record_image(record)) as image:
        model_image = letterbox_rgb_image(image.convert("RGB"), input_size, input_size)
    arr = local_np.asarray(model_image, dtype=local_np.uint8)
    chw = local_np.ascontiguousarray(local_np.transpose(arr, (2, 0, 1)))
    tensor = torch.from_numpy(chw).to(dtype=torch.float32).div(255.0).unsqueeze(0)
    outputs = model(tensor)
    outputs_np = [output.detach().float().cpu().numpy() for output in outputs]
    return decode_v2_outputs(outputs_np, input_size=input_size, score_threshold=score_threshold, nms_iou_threshold=nms_iou_threshold)


def _select_representative_examples(candidates: list[dict[str, Any]], limit: int) -> list[dict[str, Any]]:
    selected: list[dict[str, Any]] = []
    used_keys: set[tuple[str, int]] = set()
    for class_id, class_name in enumerate(ETHOSSAFEDET_CLASS_NAMES):
        class_candidates = [item for item in candidates if int(item["class_id"]) == class_id]
        class_candidates.sort(key=lambda item: (float(item["best_iou"]), float(item["score"])), reverse=True)
        for item in class_candidates:
            key = (str(item["image_id"]), int(item["class_id"]))
            if key not in used_keys:
                selected.append(item)
                used_keys.add(key)
                break
        if len(selected) >= limit:
            break
    if len(selected) < limit:
        remaining = sorted(candidates, key=lambda item: (float(item["best_iou"]), float(item["score"])), reverse=True)
        for item in remaining:
            key = (str(item["image_id"]), int(item["class_id"]))
            if key in used_keys:
                continue
            selected.append(item)
            used_keys.add(key)
            if len(selected) >= limit:
                break
    return selected


def _select_hard_examples(candidates: list[dict[str, Any]], representative: list[dict[str, Any]], limit: int) -> list[dict[str, Any]]:
    representative_keys = {(str(item["image_id"]), int(item["class_id"])) for item in representative}
    hard_pool = [
        item
        for item in candidates
        if (str(item["image_id"]), int(item["class_id"])) not in representative_keys
        and (float(item["best_iou"]) < 0.5 or float(item["score"]) < 0.5 or item.get("match") is None)
    ]
    if len(hard_pool) < limit:
        hard_pool = [item for item in candidates if (str(item["image_id"]), int(item["class_id"])) not in representative_keys]
    hard_pool.sort(key=lambda item: (float(item["best_iou"]), -float(item["score"])))
    selected: list[dict[str, Any]] = []
    used_images: set[str] = set()
    for item in hard_pool:
        image_id = str(item["image_id"])
        if image_id in used_images and len(used_images) < limit:
            continue
        selected.append(item)
        used_images.add(image_id)
        if len(selected) >= limit:
            break
    return selected


def _qualitative_rows(qualitative: dict[str, Any]) -> list[dict[str, Any]]:
    return list(qualitative.get("rows", []))


def _qualitative_row(item: dict[str, Any]) -> dict[str, Any]:
    match = item.get("match") or {}
    target = item.get("target") or {}
    return {
        "panel": item.get("panel", ""),
        "image_id": item.get("image_id", ""),
        "class_name": item.get("class_name", ""),
        "target_bbox_xyxy_vga": json.dumps(target.get("bbox_xyxy_vga", [])),
        "prediction_bbox_xyxy_vga": json.dumps(match.get("bbox_xyxy_vga", [])),
        "score": float(item.get("score", 0.0)),
        "best_iou": float(item.get("best_iou", 0.0)),
        "theta_error_rad": _num(item.get("theta_error_rad")),
    }


def _last_measured(history: list[dict[str, Any]]) -> dict[str, Any]:
    for row in reversed(history):
        if row.get("per_class"):
            return row
    return history[-1] if history else {}


def _bar(path: Path, rows: list[dict[str, Any]], label_key: str, value_key: str, title: str, ylabel: str, rotate: int = 0) -> bool:
    if not rows:
        return False
    plt = _pyplot()
    labels = [str(row[label_key]) for row in rows]
    values = [float(row.get(value_key) or 0.0) for row in rows]
    fig, ax = plt.subplots(figsize=(8.8, 4.8))
    ax.bar(labels, values, color="#3f6fa6")
    ax.set_title(title)
    ax.set_ylabel(ylabel)
    if rotate:
        ax.tick_params(axis="x", rotation=rotate)
    ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)
    return True


def _plot_history(path: Path, report: dict[str, Any], keys: list[str], title: str) -> bool:
    history = report.get("history", [])
    if not history:
        return False
    plt = _pyplot()
    fig, ax = plt.subplots(figsize=(8.8, 4.8))
    epochs = [int(row["epoch"]) for row in history]
    for key in keys:
        values = [row.get(key) for row in history]
        if all(value is None for value in values):
            continue
        ax.plot(epochs, [float(value) if value is not None else np.nan for value in values], marker="o", markersize=2.5, linewidth=1.5, label=key)
    ax.set_title(title)
    ax.set_xlabel("Epoch")
    ax.grid(alpha=0.25)
    ax.legend()
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)
    return True


def _validation_panel(path: Path, examples: list[dict[str, Any]], title: str) -> bool:
    if not examples:
        return False
    plt = _pyplot()
    count = len(examples)
    cols = 2 if count <= 4 else 4
    rows = int(math.ceil(count / cols))
    fig, axes = plt.subplots(rows, cols, figsize=(4.4 * cols, 3.7 * rows))
    axes_list = np.asarray(axes).reshape(-1).tolist()
    for ax, example in zip(axes_list, examples):
        annotated = _annotate_validation_example(example)
        ax.imshow(annotated)
        theta = example.get("theta_error_rad")
        theta_text = "theta n/a" if theta is None else f"theta {float(theta):.2f} rad"
        ax.set_title(
            f"{example.get('class_name', '')} | IoU {float(example.get('best_iou', 0.0)):.2f} | "
            f"score {float(example.get('score', 0.0)):.2f} | {theta_text}",
            fontsize=9,
        )
        ax.axis("off")
    for ax in axes_list[count:]:
        ax.axis("off")
    fig.suptitle(title, fontsize=13)
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    fig.savefig(path, dpi=180)
    plt.close(fig)
    return True


def _annotate_validation_example(example: dict[str, Any]) -> Image.Image:
    record = example["record"]
    image = Image.open(resolve_v2_record_image(record)).convert("RGB")
    draw = ImageDraw.Draw(image)
    target = example.get("target") or {}
    match = example.get("match") or {}
    target_id = int(target.get("instance_id", -1))

    for obj in example.get("objects", []):
        class_name = ETHOSSAFEDET_CLASS_NAMES[int(obj["class_id"])]
        is_target = int(obj.get("instance_id", -2)) == target_id and int(obj["class_id"]) == int(target.get("class_id", -1))
        color = "#00aa44" if is_target else "#77cc88"
        width = 4 if is_target else 2
        _draw_box(draw, obj["bbox_xyxy_vga"], color, width, f"GT {class_name}")
        if obj.get("theta_valid") and obj.get("orientation_rad") is not None:
            _draw_axis(draw, obj["bbox_xyxy_vga"], float(obj["orientation_rad"]), color, width=width)

    for index, det in enumerate((example.get("detections") or [])[:6]):
        is_match = bool(match) and det is match
        color = "#ff8c00" if is_match else "#2f70d0"
        width = 4 if is_match else 2
        class_name = ETHOSSAFEDET_CLASS_NAMES[int(det["class_id"])]
        label = f"P {class_name} {float(det.get('score', 0.0)):.2f}"
        _draw_box(draw, det["bbox_xyxy_vga"], color, width, label)
        if det.get("orientation_rad") is not None:
            _draw_axis(draw, det["bbox_xyxy_vga"], float(det["orientation_rad"]), color, width=width)

    return image


def _draw_box(draw: ImageDraw.ImageDraw, bbox: list[float], color: str, width: int, label: str) -> None:
    x1, y1, x2, y2 = [float(v) for v in bbox]
    draw.rectangle((x1, y1, x2, y2), outline=color, width=width)
    text_pos = (x1 + 3, max(0.0, y1 - 14.0))
    text_bbox = draw.textbbox(text_pos, label)
    draw.rectangle(text_bbox, fill="black")
    draw.text(text_pos, label, fill=color)


def _draw_axis(draw: ImageDraw.ImageDraw, bbox: list[float], theta: float, color: str, width: int = 2) -> None:
    x1, y1, x2, y2 = [float(v) for v in bbox]
    cx = (x1 + x2) * 0.5
    cy = (y1 + y2) * 0.5
    length = max(12.0, min(abs(x2 - x1), abs(y2 - y1)) * 0.45)
    dx = math.cos(theta) * length
    dy = math.sin(theta) * length
    draw.line((cx - dx, cy - dy, cx + dx, cy + dy), fill=color, width=max(1, width))


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    fieldnames = sorted({key for row in rows for key in row.keys()})
    with path.open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def _figure_block(figures: dict[str, str], key: str, caption: str) -> list[str]:
    path = figures.get(key)
    if not path:
        return []
    return [f"![{caption}]({path})", "", f"*{caption}*", ""]


def _qualitative_note(qualitative: dict[str, Any]) -> list[str]:
    note = str(qualitative.get("note", "") or "")
    if not note:
        return []
    return [f"> Qualitative validation rendering note: {note}", ""]


def _artifact_row(item: dict[str, Any]) -> str:
    return f"| `{item['artifact']}` | `{item['path']}` | `{item['sha256']}` | {item['bytes']} |"


def _read_gates(path: Path) -> dict[str, Any]:
    if not path.is_dir():
        return {}
    return {item.stem: _read_json(item) for item in sorted(path.glob("*.json"))}


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _clear_generated_assets(assets_dir: Path) -> None:
    for pattern in ("fig*.png", "*.csv"):
        for path in assets_dir.glob(pattern):
            if path.is_file():
                path.unlink()


def _sha256_file(path: str | Path) -> str:
    h = hashlib.sha256()
    with Path(path).open("rb") as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def _status(gate: dict[str, Any]) -> str:
    if not gate:
        return "MISSING"
    return "PASS" if gate.get("ok") else "FAIL"


def _num(value: Any) -> float | None:
    return None if value is None else float(value)


def _fmt(value: Any) -> str:
    return "n/a" if value is None else f"{float(value):.6f}"


def _pyplot():
    import matplotlib

    matplotlib.use("Agg", force=True)
    import matplotlib.pyplot as plt
    import numpy as np

    globals()["np"] = np
    return plt


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Generate a paper-style EthosSafeDetV2 training report.")
    parser.add_argument("--run", required=True)
    parser.add_argument("--out", default="")
    args = parser.parse_args(argv)
    result = make_v2_formal_report(args.run, args.out or None)
    print(json.dumps(result, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
