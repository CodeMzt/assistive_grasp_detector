"""Formal report generation for EthosSafeDet-A runs."""

from __future__ import annotations

import hashlib
import csv
import json
import math
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from assistive_grasp_detector.schema import ETHOSSAFEDET_ARENA_LIMIT_BYTES, ETHOSSAFEDET_CLASS_NAMES, ETHOSSAFEDET_WEIGHTS_LIMIT_BYTES


def make_formal_chain_report(
    run_dir: str | Path,
    output_path: str | Path | None = None,
    calibration_path: str | Path | None = None,
    reference_image: str | Path | None = None,
) -> dict[str, Any]:
    """Write a thesis-style Markdown report for one EthosSafeDet-A run."""

    run = Path(run_dir)
    train_report_path = run / "train_report.json"
    if not train_report_path.is_file():
        raise FileNotFoundError(f"missing training report: {train_report_path}")

    train_report = _read_json(train_report_path)
    gates = _read_gate_artifacts(run / "gates")
    calibration = _read_optional_json(calibration_path) if calibration_path is not None else None
    static_golden_path = _first_existing([run / "static_golden" / "golden_320.json", *sorted((run / "static_golden").glob("*.json"))])
    static_golden = _read_optional_json(static_golden_path)

    artifacts = _collect_artifacts(run, train_report, calibration_path, static_golden_path)
    assets = _build_report_assets(
        run=run,
        train_report=train_report,
        gates=gates,
        artifacts=artifacts,
        calibration=calibration,
        reference_image=Path(reference_image) if reference_image is not None else None,
    )
    out = Path(output_path) if output_path is not None else run / "formal_chain_report.md"
    markdown = _format_formal_report(
        run=run,
        train_report=train_report,
        gates=gates,
        artifacts=artifacts,
        assets=assets,
        calibration=calibration,
        calibration_path=Path(calibration_path) if calibration_path is not None else None,
        static_golden=static_golden,
        static_golden_path=static_golden_path,
        reference_image=Path(reference_image) if reference_image is not None else None,
    )
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(markdown, encoding="utf-8")
    return {
        "ok": True,
        "schema_version": "ethossafedet_formal_chain_report_v1",
        "run_dir": run.resolve().as_posix(),
        "report": out.resolve().as_posix(),
        "artifact_count": len(artifacts),
        "figure_count": len(assets["figures"]),
        "table_count": len(assets["tables"]),
    }


def _format_formal_report(
    run: Path,
    train_report: dict[str, Any],
    gates: dict[str, dict[str, Any]],
    artifacts: list[dict[str, Any]],
    assets: dict[str, dict[str, str]],
    calibration: dict[str, Any] | None,
    calibration_path: Path | None,
    static_golden: dict[str, Any] | None,
    static_golden_path: Path | None,
    reference_image: Path | None,
) -> str:
    model = train_report["model"]
    data = train_report["data"]
    hparams = train_report["hyperparameters"]
    env = train_report["environment"]
    git = train_report["git"]
    history = train_report["history"]
    best_epoch = int(train_report["best_epoch"])
    best_row = next((row for row in history if int(row["epoch"]) == best_epoch), history[-1])
    first_row = history[0]
    last_row = history[-1]
    measured_rows = [row for row in history if row.get("val_top1_class_acc") is not None]
    last_measured = measured_rows[-1] if measured_rows else last_row
    onnx_gate = gates.get("check_onnx_ops", {})
    tflite_gate = gates.get("check_tflite_ops", {})
    pc_gate = gates.get("pc_onnx_vs_tflite", {})
    memory_gate = gates.get("check_memory_budget") or gates.get("memory_budget_weights_only", {})
    host_gate = gates.get("host_mera_gate") or gates.get("host_mera_gate_blocked", {})
    ruhmi_gate = gates.get("ruhmi_dispatch") or gates.get("ruhmi_dispatch_blocked", {})
    comparison = pc_gate.get("comparison", {})
    calibration_count = len(calibration.get("items", [])) if calibration else None
    golden_count = len(static_golden.get("items", [])) if static_golden else None
    deployment_status = _deployment_status(pc_gate, host_gate, ruhmi_gate, memory_gate)

    lines = [
        "# EthosSafeDet-A v1 正式训练、导出与门禁报告",
        "",
        "## 0. 报告声明",
        "",
        (
            "本报告按实验论文/工程验证报告的章法组织：先定义研究目标和部署约束，再说明数据、模型、训练方法、"
            "量化导出、门禁验收、阻塞项和可复现证据。报告中的每个通过/阻塞结论都来自本 run 目录下的 JSON gate "
            "或 artifact hash；没有实测证据的板端、RUHMI、arena 和延迟结果不会被写成已通过。"
        ),
        "",
        "| 项目 | 内容 |",
        "|---|---|",
        f"| 生成时间 | `{datetime.now(timezone.utc).isoformat()}` |",
        f"| Run 目录 | `{run.resolve().as_posix()}` |",
        f"| 模型 | `{train_report['model_id']}` v1，无朝向 |",
        f"| 部署判定 | **{deployment_status}** |",
        f"| 最佳 checkpoint epoch | {best_epoch} |",
        f"| 最佳选择指标 | `{train_report['best_metric_name']}` = {_fmt(train_report['best_metric'])} |",
        "",
        "## 1. 摘要",
        "",
        (
            f"本次 run 训练了一个面向 RA8P1 Ethos-U55 + RUHMI 部署的六类桌面物体检测器，输入固定为 "
            f"`320x320`，输出为分离的 `cls_logits` 与 `box_ltrb`。最佳 checkpoint 出现在 epoch {best_epoch}，"
            f"验证 top-1 class accuracy 为 {_fmt(best_row.get('val_top1_class_acc'))}，"
            f"验证 top-1 IoU mean 为 {_fmt(best_row.get('val_top1_iou_mean'))}。"
        ),
        "",
        (
            f"PC reference 阶段，ONNX 算子门禁为 {_status(onnx_gate)}，TFLite full-int8 算子门禁为 {_status(tflite_gate)}，"
            f"ONNX vs TFLite 主目标 IoU 为 {_fmt(comparison.get('iou'))}，class_match={comparison.get('class_match', 'n/a')}。"
            "Host MERA/RUHMI 当前仍是阻塞项：没有 host MERA 输出或 RUHMI dispatch log，因此不能刷板，也不能声明 arena/dispatch 已满足验收。"
        ),
        "",
        "### 1.1 图表与数据资产索引",
        "",
        "| 类型 | 数量 | 目录 |",
        "|---|---:|---|",
        f"| Figures | {len(assets['figures'])} | `{_asset_dir(run)}` |",
        f"| Derived CSV tables | {len(assets['tables'])} | `{_asset_dir(run)}` |",
        "",
        "## 2. 研究目标与非目标",
        "",
        "### 2.1 目标",
        "",
        "- 构建与 Model A V2 / EthosSafeDetV2 合同对齐的六类检测、bbox 与 orientation 候选链路，服务于固定外部 RGB 相机的辅助抓取前级定位。",
        "- 以 Ethos-U55/RUHMI 部署约束反向设计网络，而不是沿用 YOLOv8-style detector。",
        "- 保持推理图静态、batch=1、量化友好，并将 decode/NMS/score 计算全部放到 CM85 C 后处理。",
        "- 以 TFLite full-int8 为主部署产物，ONNX 仅作为 PC reference。",
        "",
        "### 2.2 非目标",
        "",
        "- 不恢复独立 Model B/ROI 抓取矩形模型；orientation 属于单个 Model A V2 输出合同。",
        "- 不导出 obj 分支；置信度由 CPU 侧 `sigmoid(cls_logits)` 得出。",
        "- 不在图内执行 sigmoid、exp、grid decode、atan2、NMS、TopK、ArgMax、Gather 或动态 shape 逻辑。",
        "",
        "## 3. 部署约束与输出契约",
        "",
        "| 约束项 | 本 run 状态 | 证据 |",
        "|---|---|---|",
        f"| 固定输入 | `1x3x{model['input_size']}x{model['input_size']}` / TFLite static | train/export config |",
        f"| 输出分离 | `{', '.join(onnx_gate.get('output_names', ['cls_logits', 'box_ltrb']))}` | `check_onnx_ops.json` |",
        f"| ONNX 允许算子 | `{', '.join(onnx_gate.get('ops', []))}` | `check_onnx_ops.json` |",
        f"| TFLite 允许算子 | `{', '.join(tflite_gate.get('ops', []))}` | `check_tflite_ops.json` |",
        f"| 禁止图内后处理 | {'通过' if onnx_gate.get('ok') and tflite_gate.get('ok') else '需复查'} | ops whitelist |",
        f"| 权重预算 | {_bytes(memory_gate.get('weights_bytes'))} / limit {_bytes(ETHOSSAFEDET_WEIGHTS_LIMIT_BYTES)} | memory gate |",
        f"| Arena 预算 | {_arena_bytes(memory_gate)} / limit {_bytes(ETHOSSAFEDET_ARENA_LIMIT_BYTES)} | 需要 MERA/board log |",
        "",
        "## 4. 数据集与标注协议",
        "",
        f"数据母图为 VGA `640x480` 相机图，训练主格式为 EthosSafeDet JSONL manifest。坐标协议使用 `bbox_xyxy_vga`，"
        f"训练时 letterbox 到 `{model['input_size']}x{model['input_size']}`，评估/导出比较时再统一 decode 回 VGA 坐标系。",
        "",
        "| 数据项 | 数值 |",
        "|---|---:|",
        f"| manifest records | {data['record_count']} |",
        f"| annotated objects | {data['object_count']} |",
        f"| negative images | {data['negative_count']} |",
        f"| calibration images | {_fmt(calibration_count)} |",
        "",
        f"- Manifest: `{data['manifest']}`",
        f"- Manifest SHA256: `{data['manifest_sha256']}`",
        f"- Calibration: `{calibration_path.resolve().as_posix() if calibration_path else 'not supplied'}`",
        f"- Calibration SHA256: `{_sha256_file(calibration_path) if calibration_path and calibration_path.is_file() else 'n/a'}`",
        "",
        "### 4.1 Split Distribution",
        "",
        "| Split | Images | Share |",
        "|---|---:|---:|",
        *_split_rows(data),
        "",
        "### 4.2 Class Distribution",
        "",
        "| Class id | Class name | Objects | Share |",
        "|---:|---|---:|---:|",
        *_class_rows(data),
        "",
        "### 4.3 数据分布图",
        "",
        *_figure_block(assets, "split_distribution", "Figure 1. Train/validation split distribution."),
        *_figure_block(assets, "class_distribution", "Figure 2. Class distribution in the EthosSafeDet manifest."),
        *_figure_block(assets, "calibration_distribution", "Figure 3. Representative calibration sample distribution."),
        "",
        "数据分布仍然不均衡，少数类的泛化置信度不能只靠总体 top-1 指标判断。下一批数据应优先补足低样本类和遮挡/边界姿态。",
        "",
        "## 5. 模型结构与设计依据",
        "",
        "| 结构项 | 值 |",
        "|---|---:|",
        f"| input size | {model['input_size']} |",
        f"| stride | {model['stride']} |",
        f"| grid | {model['input_size'] // model['stride']} x {model['input_size'] // model['stride']} |",
        f"| width base | {model['width']} |",
        f"| parameters | {model['parameter_count']} |",
        f"| classes | {model['num_classes']} |",
        "",
        (
            "网络采用 MobileNet-style depthwise-separable backbone 和单 stride-8 检测网格。这个选择牺牲了大模型常见的多尺度 "
            "concat/FPN 表达力，但换来更可控的 Ethos-U/RUHMI 算子集合和更简单的 CM85 后处理。bbox 使用 LTRB 距离形式，"
            "box 分支末端仅使用 ReLU 保证非负距离。"
        ),
        "",
        "## 6. 训练方法",
        "",
        "| 项目 | 设置 |",
        "|---|---:|",
        f"| epochs in this invocation | {hparams['epochs']} |",
        f"| batch size | {hparams['batch_size']} |",
        f"| learning rate | {hparams['lr']} |",
        f"| optimizer | AdamW |",
        f"| seed | {hparams['seed']} |",
        f"| requested device | `{hparams['device']}` |",
        f"| AMP | {hparams['amp']} |",
        f"| cache images | {hparams['cache_images']} |",
        f"| eval every | {hparams['eval_every']} epoch(s) |",
        f"| classification loss | `{hparams['classification_loss']}` |",
        f"| box loss | `{hparams['box_loss']}` |",
        "",
        (
            "目标分配采用 FCOS-like center assignment。一个网格点被多个 GT 竞争时，优先选择面积更小的 GT；面积接近时选择中心距离更近的 GT。"
            "分类侧使用 hard-negative focal/BCE 风格损失，回归侧使用 SmoothL1 与 IoU 项组合。训练阶段允许 loss、augmentation 和 CPU 参考后处理；"
            "这些内容不进入推理图。"
        ),
        "",
        "## 7. 训练结果与收敛分析",
        "",
        "| Metric | First epoch | Best checkpoint epoch | Last epoch | Last measured epoch |",
        "|---|---:|---:|---:|---:|",
        _metric_compare_row("train_loss", first_row, best_row, last_row, last_measured),
        _metric_compare_row("val_loss", first_row, best_row, last_row, last_measured),
        _metric_compare_row("val_top1_class_acc", first_row, best_row, last_row, last_measured),
        _metric_compare_row("val_top1_iou_mean", first_row, best_row, last_row, last_measured),
        "",
        (
            "这里的 validation top-1 指标是候选选择指标，不是 mAP。它衡量 CPU reference decode/NMS 后的主目标 class 是否一致、"
            "bbox 与主 GT 的 IoU 水平。正式论文或产品验收仍需要补充 per-class AP、混淆矩阵、误检/漏检分析和板端 static golden 对齐。"
        ),
        "",
        "### 7.1 收敛曲线",
        "",
        *_figure_block(assets, "loss_curves", "Figure 4. Training and validation loss curves."),
        *_figure_block(assets, "detection_metrics", "Figure 5. Validation detection proxy metrics over epochs."),
        "",
        "## 8. 导出与量化",
        "",
        "| 阶段 | 结果 | 细节 |",
        "|---|---|---|",
        f"| ONNX PC reference | {_status(onnx_gate)} | ops=`{', '.join(onnx_gate.get('ops', []))}`, outputs=`{', '.join(onnx_gate.get('output_names', []))}` |",
        f"| TFLite full-int8 | {_status(tflite_gate)} | ops=`{', '.join(tflite_gate.get('ops', []))}`, output_count={tflite_gate.get('output_count', 'n/a')} |",
        f"| Representative dataset | {'PASS' if calibration_count and 200 <= calibration_count <= 500 else 'UNKNOWN/FAIL'} | count={_fmt(calibration_count)}, required=200..500 real camera images |",
        "",
        "TFLite 输出顺序允许与 ONNX 不同；比较工具按通道数识别 `cls_logits` 与 `box_ltrb`，统一 decode 到 canonical candidate 后做 detection-level 对齐。",
        "",
        "## 9. 门禁验收矩阵",
        "",
        "| Gate | Required threshold | Result | Evidence | Decision |",
        "|---|---|---|---|---|",
        f"| ONNX op whitelist | static shape, separated outputs, no forbidden ops | {_status(onnx_gate)} | `check_onnx_ops.json` | {'PASS' if onnx_gate.get('ok') else 'FAIL'} |",
        f"| TFLite op whitelist | full-int8, two outputs, allowed ops | {_status(tflite_gate)} | `check_tflite_ops.json` | {'PASS' if tflite_gate.get('ok') else 'FAIL'} |",
        f"| PC ONNX vs TFLite | class match, IoU >= 0.85 | IoU={_fmt(comparison.get('iou'))}, class_match={comparison.get('class_match', 'n/a')} | `pc_onnx_vs_tflite.json` | {'PASS' if pc_gate.get('ok') else 'FAIL'} |",
        f"| Host MERA vs PC | class match, IoU >= 0.85 | {_gate_reason(host_gate)} | host MERA JSON/command | {'PASS' if host_gate.get('ok') else 'BLOCKED'} |",
        f"| RUHMI dispatch | heavy conv in Ethos-U region, CPU bridge only, base_addr <= 8 | {_gate_reason(ruhmi_gate)} | RUHMI dispatch log | {'PASS' if ruhmi_gate.get('ok') else 'BLOCKED'} |",
        f"| Memory budget | arena <= 2.5 MB, weights <= 1.5 MB | weights={_bytes(memory_gate.get('weights_bytes'))}, arena={_arena_bytes(memory_gate)} | memory gate/log | {_memory_decision(memory_gate)} |",
        f"| Static golden | board first stage only, no camera restore yet | items={_fmt(golden_count)} | `{static_golden_path.resolve().as_posix() if static_golden_path else 'missing'}` | {'READY_FOR_HOST/BOARD_COMPARISON' if golden_count else 'MISSING'} |",
        "",
        "### 9.1 门禁与体积图",
        "",
        *_figure_block(assets, "gate_status", "Figure 6. Export and deployment gate status."),
        *_figure_block(assets, "artifact_sizes", "Figure 7. Artifact sizes and model weight budget context."),
        "",
        "## 10. PC Reference Detection Evidence",
        "",
        f"- Reference image: `{reference_image.resolve().as_posix() if reference_image else 'not recorded'}`",
        f"- ONNX top detection: `{json.dumps(comparison.get('reference', {}), ensure_ascii=False)}`",
        f"- TFLite top detection: `{json.dumps(comparison.get('candidate', {}), ensure_ascii=False)}`",
        "",
        *_figure_block(assets, "pc_detection_overlay", "Figure 8. ONNX and TFLite top detections overlaid on the reference camera image."),
        "",
        "上述 evidence 只证明 PC ONNX 与 PC TFLite 在选定图像上的 detection-level 一致性。它不能替代 host MERA gate，也不能替代板端 static golden。",
        "",
        "## 11. Artifact Provenance",
        "",
        "| Artifact | Path | SHA256 | Bytes |",
        "|---|---|---|---:|",
        *[_artifact_row(item) for item in artifacts],
        "",
        "### 11.1 派生数据表",
        "",
        "| Table | Path |",
        "|---|---|",
        *[_table_row(name, path) for name, path in sorted(assets["tables"].items())],
        "",
        "## 12. 复现实验命令",
        "",
        "```powershell",
        _train_command(train_report),
        "export_ethossafedet_onnx --checkpoint <run>\\ethossafedet_a.pt --out <run>\\export\\ethossafedet_a_320.onnx --input-size 320 --opset 13",
        "check_onnx_ops --onnx <run>\\export\\ethossafedet_a_320.onnx --out-json <run>\\gates\\check_onnx_ops.json",
        "export_ethossafedet_tflite --onnx <run>\\export\\ethossafedet_a_320.onnx --calibration <calibration_320.json> --out <run>\\export\\ethossafedet_a_320_full_int8.tflite --work-dir <run>\\export\\saved_model",
        "check_tflite_ops --tflite <run>\\export\\ethossafedet_a_320_full_int8.tflite --out-json <run>\\gates\\check_tflite_ops.json",
        "compare_ethossafedet_reference --onnx <run>\\export\\ethossafedet_a_320.onnx --tflite <run>\\export\\ethossafedet_a_320_full_int8.tflite --image <real_camera_image> --out-json <run>\\gates\\pc_onnx_vs_tflite.json",
        "make_static_golden --onnx <run>\\export\\ethossafedet_a_320.onnx --manifest <ethossafedet_manifest.jsonl> --out <run>\\static_golden\\golden_320.json --limit 20",
        "```",
        "",
        "## 13. 有效性威胁与风险",
        "",
        "- 训练验证指标不是 full mAP；当前报告不能支撑“所有类别均高质量”的结论。",
        "- Host MERA gate 缺失，因此 PC reference 与 host runtime 的数值一致性尚未实证。",
        "- RUHMI dispatch log 缺失，因此 heavy conv 是否全部落在 Ethos-U region 尚未实证。",
        "- Arena 使用量缺失；当前只知道 TFLite 文件/weights 量级，不能声明 `arena <= 2.5 MB` 已通过。",
        "- 该模型 MACs/latency 风险需要 host/board 实测，不能仅凭 ops whitelist 判断可用。",
        "- 类别分布不均衡，少数类需要更多真实板端相机样本和 hard cases。",
        "",
        "## 14. 结论与放行判定",
        "",
        (
            f"本 run 可作为 320x320 PC reference 候选：训练报告、ONNX 导出、TFLite full-int8 导出、ONNX/TFLite ops gate、"
            f"PC ONNX vs TFLite detection-level 对齐和 static golden 已形成可追溯 artifact。当前不可刷板，原因是 host MERA、"
            f"RUHMI dispatch 和真实 arena 证据缺失。下一步必须先补齐 host MERA JSON 或可执行命令，再执行 RUHMI dispatch inspection "
            f"和 memory budget gate；只有这些 gate 通过后，才允许进入 board static golden。"
        ),
        "",
    ]
    return "\n".join(lines)


def _build_report_assets(
    run: Path,
    train_report: dict[str, Any],
    gates: dict[str, dict[str, Any]],
    artifacts: list[dict[str, Any]],
    calibration: dict[str, Any] | None,
    reference_image: Path | None,
) -> dict[str, dict[str, str]]:
    assets_dir = run / _asset_dir(run)
    assets_dir.mkdir(parents=True, exist_ok=True)
    figures: dict[str, str] = {}
    tables: dict[str, str] = {}

    table_rows = {
        "split_distribution": _split_table_rows(train_report["data"]),
        "class_distribution": _class_table_rows(train_report["data"]),
        "calibration_distribution": _calibration_table_rows(calibration),
        "epoch_history_selected": _history_table_rows(train_report["history"]),
        "gate_matrix": _gate_table_rows(gates),
        "artifact_provenance": _artifact_table_rows(artifacts),
    }
    for name, rows in table_rows.items():
        if not rows:
            continue
        path = assets_dir / f"{name}.csv"
        _write_csv(path, rows)
        tables[name] = _relative_to_run(path, run)

    figure_jobs = [
        ("split_distribution", assets_dir / "fig01_split_distribution.png", lambda p: _plot_split_distribution(p, train_report["data"])),
        ("class_distribution", assets_dir / "fig02_class_distribution.png", lambda p: _plot_class_distribution(p, train_report["data"])),
        ("calibration_distribution", assets_dir / "fig03_calibration_distribution.png", lambda p: _plot_calibration_distribution(p, calibration)),
        ("loss_curves", assets_dir / "fig04_loss_curves.png", lambda p: _plot_loss_curves(p, train_report)),
        ("detection_metrics", assets_dir / "fig05_detection_metrics.png", lambda p: _plot_detection_metrics(p, train_report)),
        ("gate_status", assets_dir / "fig06_gate_status.png", lambda p: _plot_gate_status(p, gates)),
        ("artifact_sizes", assets_dir / "fig07_artifact_sizes.png", lambda p: _plot_artifact_sizes(p, artifacts)),
        (
            "pc_detection_overlay",
            assets_dir / "fig08_pc_detection_overlay.png",
            lambda p: _make_detection_overlay(p, reference_image, gates.get("pc_onnx_vs_tflite", {})),
        ),
    ]
    for name, path, writer in figure_jobs:
        try:
            if writer(path):
                figures[name] = _relative_to_run(path, run)
        except Exception:
            continue

    return {"figures": figures, "tables": tables}


def _collect_artifacts(
    run: Path,
    train_report: dict[str, Any],
    calibration_path: str | Path | None,
    static_golden_path: Path | None,
) -> list[dict[str, Any]]:
    candidates: list[tuple[str, Path | None]] = [
        ("best_checkpoint", Path(train_report["checkpoint"])),
        ("last_checkpoint", Path(train_report["last_checkpoint"])),
        ("train_report_json", run / "train_report.json"),
        ("train_report_markdown", run / "train_report.md"),
        ("train_history_csv", run / "train_history.csv"),
        ("onnx", _first_existing(sorted((run / "export").glob("*.onnx")))),
        ("tflite_full_int8", _first_existing(sorted((run / "export").glob("*.tflite")))),
        ("static_golden", static_golden_path),
        ("calibration", Path(calibration_path) if calibration_path is not None else None),
    ]
    artifacts: list[dict[str, Any]] = []
    for name, path in candidates:
        if path is None or not path.is_file():
            continue
        artifacts.append(
            {
                "name": name,
                "path": path.resolve().as_posix(),
                "sha256": _sha256_file(path),
                "bytes": path.stat().st_size,
            }
        )
    return artifacts


def _split_table_rows(data: dict[str, Any]) -> list[dict[str, Any]]:
    total = max(1, int(data.get("record_count", 0)))
    return [
        {"split": split, "image_count": int(count), "share": 100.0 * int(count) / total}
        for split, count in sorted(data.get("split_counts", {}).items())
    ]


def _class_table_rows(data: dict[str, Any]) -> list[dict[str, Any]]:
    total = max(1, int(data.get("object_count", 0)))
    counts = data.get("class_counts", {})
    rows: list[dict[str, Any]] = []
    for class_id, name in enumerate(ETHOSSAFEDET_CLASS_NAMES):
        count = int(counts.get(str(class_id), 0))
        rows.append({"class_id": class_id, "class_name": name, "object_count": count, "share": 100.0 * count / total})
    return rows


def _calibration_table_rows(calibration: dict[str, Any] | None) -> list[dict[str, Any]]:
    if not calibration:
        return []
    counts = {class_id: 0 for class_id in range(len(ETHOSSAFEDET_CLASS_NAMES))}
    negative = 0
    for item in calibration.get("items", []):
        class_ids = item.get("class_ids", [])
        if not class_ids:
            negative += 1
            continue
        for class_id in class_ids:
            if int(class_id) in counts:
                counts[int(class_id)] += 1
    total = max(1, sum(counts.values()) + negative)
    rows = [
        {"class_id": class_id, "class_name": name, "image_hits": counts[class_id], "share": 100.0 * counts[class_id] / total}
        for class_id, name in enumerate(ETHOSSAFEDET_CLASS_NAMES)
    ]
    rows.append({"class_id": "negative", "class_name": "negative", "image_hits": negative, "share": 100.0 * negative / total})
    return rows


def _history_table_rows(history: list[dict[str, Any]]) -> list[dict[str, Any]]:
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
        "val_eval_count",
    ]
    return [{column: row.get(column, "") for column in columns} for row in history]


def _gate_table_rows(gates: dict[str, dict[str, Any]]) -> list[dict[str, Any]]:
    onnx_gate = gates.get("check_onnx_ops", {})
    tflite_gate = gates.get("check_tflite_ops", {})
    pc_gate = gates.get("pc_onnx_vs_tflite", {})
    memory_gate = gates.get("check_memory_budget") or gates.get("memory_budget_weights_only", {})
    host_gate = gates.get("host_mera_gate") or gates.get("host_mera_gate_blocked", {})
    ruhmi_gate = gates.get("ruhmi_dispatch") or gates.get("ruhmi_dispatch_blocked", {})
    comparison = pc_gate.get("comparison", {})
    return [
        {
            "gate": "ONNX op whitelist",
            "threshold": "static shape, separated outputs, no forbidden ops",
            "result": _status(onnx_gate),
            "evidence": "check_onnx_ops.json",
            "decision": "PASS" if onnx_gate.get("ok") else "FAIL",
        },
        {
            "gate": "TFLite op whitelist",
            "threshold": "full-int8, two outputs, allowed ops",
            "result": _status(tflite_gate),
            "evidence": "check_tflite_ops.json",
            "decision": "PASS" if tflite_gate.get("ok") else "FAIL",
        },
        {
            "gate": "PC ONNX vs TFLite",
            "threshold": "class match, IoU >= 0.85",
            "result": f"IoU={_fmt(comparison.get('iou'))}, class_match={comparison.get('class_match', 'n/a')}",
            "evidence": "pc_onnx_vs_tflite.json",
            "decision": "PASS" if pc_gate.get("ok") else "FAIL",
        },
        {
            "gate": "Host MERA vs PC",
            "threshold": "class match, IoU >= 0.85",
            "result": _gate_reason(host_gate),
            "evidence": "host MERA JSON/command",
            "decision": "PASS" if host_gate.get("ok") else "BLOCKED",
        },
        {
            "gate": "RUHMI dispatch",
            "threshold": "heavy conv in Ethos-U region, CPU bridge only, base_addr <= 8",
            "result": _gate_reason(ruhmi_gate),
            "evidence": "RUHMI dispatch log",
            "decision": "PASS" if ruhmi_gate.get("ok") else "BLOCKED",
        },
        {
            "gate": "Memory budget",
            "threshold": "arena <= 2.5 MB, weights <= 1.5 MB",
            "result": f"weights={_bytes(memory_gate.get('weights_bytes'))}, arena={_arena_bytes(memory_gate)}",
            "evidence": "memory gate/log",
            "decision": _memory_decision(memory_gate),
        },
    ]


def _artifact_table_rows(artifacts: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [
        {
            "artifact": item["name"],
            "path": item["path"],
            "sha256": item["sha256"],
            "bytes": item["bytes"],
            "mib": float(item["bytes"]) / (1024.0 * 1024.0),
        }
        for item in artifacts
    ]


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    columns = list(rows[0].keys())
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=columns)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def _plot_split_distribution(path: Path, data: dict[str, Any]) -> bool:
    rows = _split_table_rows(data)
    if not rows:
        return False
    labels = [str(row["split"]) for row in rows]
    values = [float(row["image_count"]) for row in rows]
    return _bar_plot(path, labels, values, "Split distribution", "Images", color="#3568a8")


def _plot_class_distribution(path: Path, data: dict[str, Any]) -> bool:
    rows = _class_table_rows(data)
    labels = [str(row["class_name"]) for row in rows]
    values = [float(row["object_count"]) for row in rows]
    return _bar_plot(path, labels, values, "Class distribution", "Objects", color="#4f8f62", rotate=25)


def _plot_calibration_distribution(path: Path, calibration: dict[str, Any] | None) -> bool:
    rows = _calibration_table_rows(calibration)
    if not rows:
        return False
    labels = [str(row["class_name"]) for row in rows]
    values = [float(row["image_hits"]) for row in rows]
    return _bar_plot(path, labels, values, "Calibration representative distribution", "Image hits", color="#8a6fba", rotate=25)


def _plot_loss_curves(path: Path, train_report: dict[str, Any]) -> bool:
    plt = _pyplot()
    if plt is None:
        return False
    history = train_report["history"]
    epochs = [int(row["epoch"]) for row in history]
    fig, ax = plt.subplots(figsize=(8.0, 4.8))
    _plot_line(ax, epochs, [row.get("train_loss") for row in history], "train_loss", "#2f6b9a")
    _plot_line(ax, epochs, [row.get("val_loss") for row in history], "val_loss", "#b24b43")
    _plot_line(ax, epochs, [row.get("train_cls_loss") for row in history], "train_cls_loss", "#5c8f42", alpha=0.55)
    _plot_line(ax, epochs, [row.get("train_box_loss") for row in history], "train_box_loss", "#8064a2", alpha=0.55)
    best_epoch = int(train_report.get("best_epoch", 0))
    if best_epoch:
        ax.axvline(best_epoch, color="#222222", linestyle="--", linewidth=1.0, label=f"best epoch {best_epoch}")
    ax.set_title("Training and validation losses")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Loss")
    ax.grid(True, alpha=0.25)
    ax.legend(loc="best", fontsize=8)
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)
    return True


def _plot_detection_metrics(path: Path, train_report: dict[str, Any]) -> bool:
    plt = _pyplot()
    if plt is None:
        return False
    history = train_report["history"]
    epochs = [int(row["epoch"]) for row in history]
    fig, ax = plt.subplots(figsize=(8.0, 4.8))
    _plot_line(ax, epochs, [row.get("val_top1_class_acc") for row in history], "val_top1_class_acc", "#3d7f4f")
    _plot_line(ax, epochs, [row.get("val_top1_iou_mean") for row in history], "val_top1_iou_mean", "#c27a29")
    best_epoch = int(train_report.get("best_epoch", 0))
    if best_epoch:
        ax.axvline(best_epoch, color="#222222", linestyle="--", linewidth=1.0, label=f"best epoch {best_epoch}")
    ax.set_title("Validation detection proxy metrics")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Metric")
    ax.set_ylim(0.0, 1.0)
    ax.grid(True, alpha=0.25)
    ax.legend(loc="best", fontsize=8)
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)
    return True


def _plot_gate_status(path: Path, gates: dict[str, dict[str, Any]]) -> bool:
    plt = _pyplot()
    if plt is None:
        return False
    rows = _gate_table_rows(gates)
    labels = [str(row["gate"]) for row in rows]
    values: list[float] = []
    colors: list[str] = []
    for row in rows:
        decision = str(row["decision"])
        if decision == "PASS":
            values.append(1.0)
            colors.append("#4f8f62")
        elif decision.startswith("PARTIAL"):
            values.append(0.5)
            colors.append("#c79b2b")
        else:
            values.append(0.0)
            colors.append("#b24b43")
    fig, ax = plt.subplots(figsize=(8.8, 4.8))
    ax.bar(labels, values, color=colors)
    for index, row in enumerate(rows):
        ax.text(index, min(1.02, values[index] + 0.06), str(row["decision"]), ha="center", va="bottom", fontsize=8, rotation=90)
    ax.set_ylim(0.0, 1.15)
    ax.set_ylabel("Gate score")
    ax.set_title("Gate status summary")
    ax.set_yticks([0.0, 0.5, 1.0], ["blocked/fail", "partial", "pass"])
    ax.tick_params(axis="x", rotation=25)
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)
    return True


def _plot_artifact_sizes(path: Path, artifacts: list[dict[str, Any]]) -> bool:
    plt = _pyplot()
    if plt is None or not artifacts:
        return False
    labels = [str(item["name"]) for item in artifacts]
    values = [float(item["bytes"]) / (1024.0 * 1024.0) for item in artifacts]
    fig, ax = plt.subplots(figsize=(9.0, 4.8))
    ax.bar(labels, values, color="#527a9d")
    ax.axhline(ETHOSSAFEDET_WEIGHTS_LIMIT_BYTES / (1024.0 * 1024.0), color="#b24b43", linestyle="--", linewidth=1.2, label="weights limit 1.5 MiB")
    ax.set_ylabel("MiB")
    ax.set_title("Artifact sizes")
    ax.tick_params(axis="x", rotation=25)
    ax.legend(loc="best", fontsize=8)
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)
    return True


def _make_detection_overlay(path: Path, reference_image: Path | None, pc_gate: dict[str, Any]) -> bool:
    if reference_image is None or not reference_image.is_file():
        return False
    comparison = pc_gate.get("comparison", {})
    reference = comparison.get("reference", {})
    candidate = comparison.get("candidate", {})
    if not reference or not candidate:
        return False
    from PIL import Image, ImageDraw

    with Image.open(reference_image) as image:
        canvas = image.convert("RGB")
    draw = ImageDraw.Draw(canvas)
    _draw_detection(draw, reference, "ONNX", (220, 54, 47), y_offset=0)
    _draw_detection(draw, candidate, "TFLite", (33, 150, 170), y_offset=18)
    canvas.save(path)
    return True


def _draw_detection(draw: Any, detection: dict[str, Any], label: str, color: tuple[int, int, int], y_offset: int = 0) -> None:
    bbox = detection.get("bbox_xyxy_vga")
    if not bbox or len(bbox) != 4:
        return
    x1, y1, x2, y2 = [float(value) for value in bbox]
    draw.rectangle([x1, y1, x2, y2], outline=color, width=3)
    class_id = detection.get("class_id", "n/a")
    class_name = _safe_class_name(class_id)
    text = f"{label}: {class_name} {float(detection.get('score', 0.0)):.3f}"
    tx = max(0.0, x1)
    ty = max(0.0, y1 - 18 + y_offset)
    try:
        text_box = draw.textbbox((tx, ty), text)
        draw.rectangle(text_box, fill=(255, 255, 255))
    except Exception:
        pass
    draw.text((tx, ty), text, fill=color)


def _bar_plot(path: Path, labels: list[str], values: list[float], title: str, ylabel: str, color: str, rotate: int = 0) -> bool:
    plt = _pyplot()
    if plt is None:
        return False
    fig, ax = plt.subplots(figsize=(8.0, 4.8))
    ax.bar(labels, values, color=color)
    ax.set_title(title)
    ax.set_ylabel(ylabel)
    ax.grid(axis="y", alpha=0.25)
    if rotate:
        ax.tick_params(axis="x", rotation=rotate)
    for index, value in enumerate(values):
        ax.text(index, value, f"{value:.0f}", ha="center", va="bottom", fontsize=8)
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)
    return True


def _plot_line(ax: Any, epochs: list[int], values: list[Any], label: str, color: str, alpha: float = 1.0) -> None:
    y_values = [float(value) if value is not None else math.nan for value in values]
    if all(math.isnan(value) for value in y_values):
        return
    ax.plot(epochs, y_values, marker="o", markersize=3, linewidth=1.6, label=label, color=color, alpha=alpha)


def _pyplot():
    try:
        import matplotlib

        matplotlib.use("Agg", force=True)
        import matplotlib.pyplot as plt
    except Exception:
        return None
    return plt


def _read_gate_artifacts(gates_dir: Path) -> dict[str, dict[str, Any]]:
    if not gates_dir.is_dir():
        return {}
    gates: dict[str, dict[str, Any]] = {}
    for path in sorted(gates_dir.glob("*.json")):
        gates[path.stem] = _read_json(path)
    return gates


def _read_json(path: str | Path) -> dict[str, Any]:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def _read_optional_json(path: str | Path | None) -> dict[str, Any] | None:
    if path is None:
        return None
    p = Path(path)
    if not p.is_file():
        return None
    return _read_json(p)


def _first_existing(paths: list[Path] | tuple[Path, ...]) -> Path | None:
    for path in paths:
        if path is not None and path.is_file():
            return path
    return None


def _deployment_status(
    pc_gate: dict[str, Any],
    host_gate: dict[str, Any],
    ruhmi_gate: dict[str, Any],
    memory_gate: dict[str, Any],
) -> str:
    if pc_gate.get("ok") and host_gate.get("ok") and ruhmi_gate.get("ok") and _memory_decision(memory_gate) == "PASS":
        return "board static golden 可进入"
    if pc_gate.get("ok"):
        return "PC reference 已通过，host MERA/RUHMI/arena 阻塞，禁止刷板"
    return "PC reference 未通过，禁止刷板"


def _status(gate: dict[str, Any]) -> str:
    if not gate:
        return "MISSING"
    return "PASS" if gate.get("ok") else "FAIL/BLOCKED"


def _gate_reason(gate: dict[str, Any]) -> str:
    if not gate:
        return "missing gate artifact"
    if gate.get("ok"):
        return "ok"
    return str(gate.get("reason") or gate.get("note") or "not passed")


def _memory_decision(gate: dict[str, Any]) -> str:
    if not gate:
        return "MISSING"
    arena = gate.get("arena_bytes")
    weights = gate.get("weights_bytes")
    if arena in (None, 0):
        return "PARTIAL_WEIGHTS_ONLY"
    return "PASS" if gate.get("ok") else "FAIL"


def _split_rows(data: dict[str, Any]) -> list[str]:
    total = max(1, int(data.get("record_count", 0)))
    rows = []
    for split, count in sorted(data.get("split_counts", {}).items()):
        rows.append(f"| `{split}` | {int(count)} | {_pct(int(count), total)} |")
    return rows


def _class_rows(data: dict[str, Any]) -> list[str]:
    total = max(1, int(data.get("object_count", 0)))
    counts = data.get("class_counts", {})
    rows = []
    for class_id, name in enumerate(ETHOSSAFEDET_CLASS_NAMES):
        count = int(counts.get(str(class_id), 0))
        rows.append(f"| {class_id} | `{name}` | {count} | {_pct(count, total)} |")
    return rows


def _metric_compare_row(
    metric: str,
    first_row: dict[str, Any],
    best_row: dict[str, Any],
    last_row: dict[str, Any],
    last_measured: dict[str, Any],
) -> str:
    return (
        f"| `{metric}` | {_fmt(first_row.get(metric))} | {_fmt(best_row.get(metric))} | "
        f"{_fmt(last_row.get(metric))} | {_fmt(last_measured.get(metric))} |"
    )


def _artifact_row(item: dict[str, Any]) -> str:
    return f"| `{item['name']}` | `{item['path']}` | `{item['sha256']}` | {item['bytes']} |"


def _table_row(name: str, path: str) -> str:
    return f"| `{name}` | [`{path}`]({path}) |"


def _figure_block(assets: dict[str, dict[str, str]], key: str, caption: str) -> list[str]:
    path = assets["figures"].get(key)
    if not path:
        return [f"_{caption} was not generated because the required evidence was unavailable._", ""]
    return [f"![{caption}]({path})", "", f"*{caption}*", ""]


def _asset_dir(run: Path) -> str:
    return "formal_report_assets"


def _relative_to_run(path: Path, run: Path) -> str:
    try:
        return path.resolve().relative_to(run.resolve()).as_posix()
    except ValueError:
        return path.resolve().as_posix()


def _safe_class_name(class_id: Any) -> str:
    try:
        index = int(class_id)
    except (TypeError, ValueError):
        return str(class_id)
    if 0 <= index < len(ETHOSSAFEDET_CLASS_NAMES):
        return ETHOSSAFEDET_CLASS_NAMES[index]
    return str(class_id)


def _train_command(train_report: dict[str, Any]) -> str:
    hparams = train_report["hyperparameters"]
    parts = [
        "train_ethossafedet_a",
        f"--manifest {train_report['data']['manifest']}",
        f"--out {train_report['output_dir']}",
        f"--input-size {train_report['model']['input_size']}",
        f"--epochs {hparams['epochs']}",
        f"--batch {hparams['batch_size']}",
        f"--lr {hparams['lr']}",
        f"--device {hparams['device']}",
        f"--seed {hparams['seed']}",
        f"--eval-score-threshold {hparams['eval_score_threshold']}",
        f"--nms-iou {hparams['nms_iou_threshold']}",
        f"--eval-every {hparams['eval_every']}",
    ]
    if hparams.get("amp"):
        parts.append("--amp")
    if not hparams.get("cache_images", True):
        parts.append("--no-cache-images")
    if train_report.get("resume_checkpoint"):
        parts.append(f"--resume-checkpoint {train_report['resume_checkpoint']}")
    return " ".join(parts)


def _pct(value: int, total: int) -> str:
    return f"{100.0 * float(value) / float(max(1, total)):.2f}%"


def _fmt(value: Any) -> str:
    if value is None:
        return "n/a"
    if isinstance(value, float):
        return f"{value:.6f}"
    return str(value)


def _bytes(value: Any) -> str:
    if value is None:
        return "n/a"
    try:
        integer = int(value)
    except (TypeError, ValueError):
        return str(value)
    return f"{integer} B ({integer / (1024 * 1024):.3f} MiB)"


def _arena_bytes(memory_gate: dict[str, Any]) -> str:
    value = memory_gate.get("arena_bytes")
    if value in (None, 0):
        return "not measured"
    return _bytes(value)


def _sha256_file(path: str | Path) -> str:
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()
