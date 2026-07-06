"""Paper-style report generator for EthosSafeDetV2 training runs."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path
from typing import Any

from assistive_grasp_detector.schema import ETHOSSAFEDET_CLASS_NAMES


def make_v2_formal_report(run_dir: str | Path, output_path: str | Path | None = None) -> dict[str, Any]:
    run = Path(run_dir)
    train_report = _read_json(run / "train_report.json")
    gates = _read_gates(run / "gates")
    assets_dir = run / "formal_report_assets"
    assets_dir.mkdir(parents=True, exist_ok=True)
    tables = _write_tables(assets_dir, train_report, gates)
    figures = _write_figures(assets_dir, train_report, gates)
    out = Path(output_path) if output_path is not None else run / "formal_report.md"
    markdown = _format_report(run, train_report, gates, figures, tables)
    out.write_text(markdown, encoding="utf-8")
    return {
        "ok": True,
        "report": out.resolve().as_posix(),
        "figure_count": len(figures),
        "table_count": len(tables),
        "figures": figures,
        "tables": tables,
    }


def _format_report(run: Path, report: dict[str, Any], gates: dict[str, Any], figures: dict[str, str], tables: dict[str, str]) -> str:
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
        "## Results",
        "",
        f"- Test recall@0.5: `{_fmt(test.get('recall50'))}`",
        f"- Test mean best IoU: `{_fmt(test.get('best_iou_mean'))}`",
        f"- Test primary class accuracy: `{_fmt(test.get('primary_class_acc'))}`",
        f"- Test theta MAE rad: `{_fmt(test.get('theta_abs_error_rad_mean'))}` over `{test.get('theta_eval_count', 0)}` matched theta-valid objects",
        "",
        *_figure_block(figures, "per_class_val_metrics", "Figure 7. Per-class validation metrics."),
        *_figure_block(figures, "test_metrics", "Figure 8. Per-class test metrics."),
        "",
        "## Deployment Gate Status",
        "",
        f"- ONNX static gate: `{_status(onnx_gate)}`",
        f"- Weight budget gate: `{_status(budget_gate)}`",
        "- TFLite full-int8: `BLOCKED/not run` because TensorFlow is not installed in the current ma2 training environment.",
        "- Host MERA, RUHMI dispatch, and board static golden: `BLOCKED/not run`; these remain post-training acceptance gates.",
        "",
        *_figure_block(figures, "artifact_sizes", "Figure 9. Artifact and estimated model sizes."),
        *_figure_block(figures, "gate_status", "Figure 10. Gate status matrix."),
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


def _write_tables(assets_dir: Path, report: dict[str, Any], gates: dict[str, Any]) -> dict[str, str]:
    tables = {
        "split_distribution": _split_rows(report["data"]),
        "class_distribution": _class_rows(report["data"]),
        "theta_coverage": _theta_rows(report["data"]),
        "class_weights": _class_weight_rows(report),
        "epoch_history": _history_rows(report),
        "per_class_validation": _per_class_rows(_last_measured(report["history"])),
        "per_class_test": _per_class_rows(report.get("test_metrics", {})),
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


def _write_figures(assets_dir: Path, report: dict[str, Any], gates: dict[str, Any]) -> dict[str, str]:
    jobs = [
        ("split_distribution", "fig01_split_distribution.png", lambda p: _bar(p, _split_rows(report["data"]), "split", "image_count", "Split distribution", "Images")),
        ("class_distribution", "fig02_class_distribution.png", lambda p: _bar(p, _class_rows(report["data"]), "class_name", "object_count", "Class distribution", "Objects", rotate=25)),
        ("theta_valid_coverage", "fig03_theta_valid_coverage.png", lambda p: _bar(p, _theta_rows(report["data"]), "class_name", "theta_valid", "Theta-valid coverage", "Objects", rotate=25)),
        ("class_weight_sampling", "fig04_class_weight_sampling.png", lambda p: _bar(p, _class_weight_rows(report), "class_name", "class_weight", "Class weights", "Weight", rotate=25)),
        ("loss_curves", "fig05_loss_curves.png", lambda p: _plot_history(p, report, ["train_loss", "val_loss", "train_cls_loss", "train_box_loss", "train_ori_loss"], "Loss curves")),
        ("validation_metrics", "fig06_validation_metrics.png", lambda p: _plot_history(p, report, ["recall50", "best_iou_mean", "primary_class_acc"], "Validation metrics")),
        ("per_class_val_metrics", "fig07_per_class_val_metrics.png", lambda p: _bar(p, _per_class_rows(_last_measured(report["history"])), "class_name", "recall50", "Per-class validation recall@0.5", "Recall", rotate=25)),
        ("test_metrics", "fig08_test_metrics.png", lambda p: _bar(p, _per_class_rows(report.get("test_metrics", {})), "class_name", "recall50", "Per-class test recall@0.5", "Recall", rotate=25)),
        ("artifact_sizes", "fig09_artifact_sizes.png", lambda p: _bar(p, _artifact_rows(assets_dir.parent, report), "artifact", "bytes", "Artifact sizes", "Bytes", rotate=25)),
        ("gate_status", "fig10_gate_status.png", lambda p: _bar(p, _gate_rows(gates), "gate", "score", "Gate status", "1=PASS", rotate=25)),
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


def _artifact_row(item: dict[str, Any]) -> str:
    return f"| `{item['artifact']}` | `{item['path']}` | `{item['sha256']}` | {item['bytes']} |"


def _read_gates(path: Path) -> dict[str, Any]:
    if not path.is_dir():
        return {}
    return {item.stem: _read_json(item) for item in sorted(path.glob("*.json"))}


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


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
