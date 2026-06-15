from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from assistive_grasp_detector.ethossafedet_gates import (
    check_memory_budget,
    check_onnx_ops,
    inspect_ruhmi_dispatch,
)
from assistive_grasp_detector.ethossafedet_postprocess import bbox_iou, decode_ltrb_outputs
from assistive_grasp_detector.ethossafedet_report import make_formal_chain_report


def test_decode_ltrb_outputs_and_nms() -> None:
    cls = np.full((1, 6, 2, 2), -8.0, dtype=np.float32)
    box = np.zeros((1, 4, 2, 2), dtype=np.float32)
    cls[0, 3, 0, 0] = 8.0
    box[0, :, 0, 0] = [20.0, 20.0, 40.0, 40.0]

    detections = decode_ltrb_outputs(cls, box, input_size=16, score_threshold=0.5, pre_nms_top_k=10)

    assert len(detections) == 1
    assert detections[0].class_id == 3
    assert detections[0].score > 0.99
    assert detections[0].bbox_xyxy_vga[2] > detections[0].bbox_xyxy_vga[0]


def test_bbox_iou() -> None:
    assert bbox_iou([0, 0, 10, 10], [0, 0, 10, 10]) == 1.0
    assert bbox_iou([0, 0, 10, 10], [20, 20, 30, 30]) == 0.0


def test_check_onnx_ops_accepts_static_split_outputs(tmp_path: Path) -> None:
    onnx = pytest_import_onnx()
    graph = onnx.helper.make_graph(
        nodes=[
            onnx.helper.make_node("Conv", ["input_image", "w_cls", "b_cls"], ["cls_logits"]),
            onnx.helper.make_node("Conv", ["input_image", "w_box", "b_box"], ["box_ltrb"]),
        ],
        name="ethossafedet_test",
        inputs=[onnx.helper.make_tensor_value_info("input_image", onnx.TensorProto.FLOAT, [1, 3, 8, 8])],
        outputs=[
            onnx.helper.make_tensor_value_info("cls_logits", onnx.TensorProto.FLOAT, [1, 6, 8, 8]),
            onnx.helper.make_tensor_value_info("box_ltrb", onnx.TensorProto.FLOAT, [1, 4, 8, 8]),
        ],
        initializer=[
            onnx.helper.make_tensor("w_cls", onnx.TensorProto.FLOAT, [6, 3, 1, 1], [0.0] * 18),
            onnx.helper.make_tensor("b_cls", onnx.TensorProto.FLOAT, [6], [0.0] * 6),
            onnx.helper.make_tensor("w_box", onnx.TensorProto.FLOAT, [4, 3, 1, 1], [0.0] * 12),
            onnx.helper.make_tensor("b_box", onnx.TensorProto.FLOAT, [4], [0.0] * 4),
        ],
    )
    model = onnx.helper.make_model(graph)
    path = tmp_path / "model.onnx"
    onnx.save(model, path)

    result = check_onnx_ops(path)

    assert result["ok"]
    assert result["output_names"] == ["cls_logits", "box_ltrb"]


def test_check_onnx_ops_rejects_forbidden_sigmoid(tmp_path: Path) -> None:
    onnx = pytest_import_onnx()
    graph = onnx.helper.make_graph(
        nodes=[onnx.helper.make_node("Sigmoid", ["input_image"], ["cls_logits"])],
        name="bad",
        inputs=[onnx.helper.make_tensor_value_info("input_image", onnx.TensorProto.FLOAT, [1, 6, 8, 8])],
        outputs=[
            onnx.helper.make_tensor_value_info("cls_logits", onnx.TensorProto.FLOAT, [1, 6, 8, 8]),
            onnx.helper.make_tensor_value_info("box_ltrb", onnx.TensorProto.FLOAT, [1, 4, 8, 8]),
        ],
    )
    path = tmp_path / "bad.onnx"
    onnx.save(onnx.helper.make_model(graph), path)

    result = check_onnx_ops(path)

    assert not result["ok"]
    assert "Sigmoid" in result["forbidden_ops"]


def test_inspect_ruhmi_dispatch_accepts_ethos_conv_and_bridge_cpu(tmp_path: Path) -> None:
    log = tmp_path / "dispatch.log"
    log.write_text(
        "\n".join(
            [
                "num_base_addr = 7",
                "Conv2D_0 -> Ethos-U region",
                "DepthwiseConv2D_1 -> Ethos-U region",
                "CPU RESHAPE bridge",
            ]
        ),
        encoding="utf-8",
    )

    result = inspect_ruhmi_dispatch(log)

    assert result["ok"]
    assert result["num_base_addr"] == 7


def test_inspect_ruhmi_dispatch_rejects_cpu_conv(tmp_path: Path) -> None:
    log = tmp_path / "dispatch.log"
    log.write_text("num_base_addr: 7\nCPU Conv2D_0\n", encoding="utf-8")

    result = inspect_ruhmi_dispatch(log)

    assert not result["ok"]
    assert result["cpu_violations"]


def test_check_memory_budget_from_log(tmp_path: Path) -> None:
    log = tmp_path / "memory.log"
    log.write_text("tensor arena: 2.0 MiB\nmodel weights: 1.0 MiB\n", encoding="utf-8")

    result = check_memory_budget(log_path=log)

    assert result["ok"]


def test_reference_json_shape_for_host_mera_gate(tmp_path: Path) -> None:
    sample = {
        "detections": [
            {"class_id": 1, "score": 0.9, "bbox_xyxy_vga": [10, 20, 100, 120]},
        ]
    }
    path = tmp_path / "detections.json"
    path.write_text(json.dumps(sample), encoding="utf-8")
    assert json.loads(path.read_text(encoding="utf-8"))["detections"][0]["class_id"] == 1


def test_formal_chain_report_has_paper_structure_and_blocked_gate(tmp_path: Path) -> None:
    run = tmp_path / "run"
    export_dir = run / "export"
    gates_dir = run / "gates"
    golden_dir = run / "static_golden"
    export_dir.mkdir(parents=True)
    gates_dir.mkdir()
    golden_dir.mkdir()
    checkpoint = run / "ethossafedet_a.pt"
    last_checkpoint = run / "ethossafedet_a_last.pt"
    onnx = export_dir / "ethossafedet_a_320.onnx"
    tflite = export_dir / "ethossafedet_a_320_full_int8.tflite"
    for path in (checkpoint, last_checkpoint, onnx, tflite):
        path.write_bytes(b"artifact")
    (run / "train_history.csv").write_text("epoch,train_loss\n1,1.0\n", encoding="utf-8")
    (run / "train_report.md").write_text("# placeholder\n", encoding="utf-8")

    train_report = {
        "schema_version": "ethossafedet_train_report_v1",
        "generated_at": "2026-06-13T00:00:00+00:00",
        "model_id": "EthosSafeDet-A",
        "output_dir": run.as_posix(),
        "resume_checkpoint": "",
        "environment": {
            "python": "3.10",
            "platform": "Windows",
            "torch": "2.x",
            "cuda_available": False,
            "cuda_device_count": 0,
            "cuda_device_name": "",
        },
        "git": {"head": "abc", "dirty": True, "status_line_count": 1},
        "data": {
            "manifest": "manifest.jsonl",
            "manifest_sha256": "mhash",
            "record_count": 4,
            "object_count": 4,
            "negative_count": 0,
            "split_counts": {"train": 2, "val": 2},
            "class_counts": {"0": 1, "1": 1, "2": 1, "3": 1, "4": 0, "5": 0},
        },
        "model": {"input_size": 320, "num_classes": 7, "class_names": [], "width": 32, "stride": 8, "parameter_count": 1234},
        "hyperparameters": {
            "epochs": 1,
            "batch_size": 2,
            "lr": 0.001,
            "device": "cpu",
            "seed": 0,
            "eval_score_threshold": 0.25,
            "nms_iou_threshold": 0.5,
            "num_workers": 0,
            "eval_every": 1,
            "eval_limit": None,
            "amp": False,
            "cache_images": True,
            "classification_loss": "focal",
            "box_loss": "smooth_l1_iou",
        },
        "history": [
            {
                "epoch": 1,
                "train_loss": 1.0,
                "val_loss": 1.2,
                "val_top1_class_acc": 0.5,
                "val_top1_iou_mean": 0.6,
            }
        ],
        "best_epoch": 1,
        "best_metric": 1.1,
        "best_metric_name": "val_top1_class_acc_plus_iou_mean",
        "checkpoint": checkpoint.as_posix(),
        "checkpoint_sha256": "besthash",
        "last_checkpoint": last_checkpoint.as_posix(),
        "last_checkpoint_sha256": "lasthash",
    }
    (run / "train_report.json").write_text(json.dumps(train_report), encoding="utf-8")
    (gates_dir / "check_onnx_ops.json").write_text(
        json.dumps({"ok": True, "ops": ["Conv", "Relu"], "output_names": ["cls_logits", "box_ltrb"]}),
        encoding="utf-8",
    )
    (gates_dir / "check_tflite_ops.json").write_text(
        json.dumps({"ok": True, "ops": ["CONV_2D"], "output_count": 2}),
        encoding="utf-8",
    )
    (gates_dir / "pc_onnx_vs_tflite.json").write_text(
        json.dumps({"ok": True, "comparison": {"iou": 0.9, "class_match": True}}),
        encoding="utf-8",
    )
    (gates_dir / "memory_budget_weights_only.json").write_text(
        json.dumps({"ok": True, "arena_bytes": 0, "weights_bytes": 100}),
        encoding="utf-8",
    )
    (gates_dir / "host_mera_gate_blocked.json").write_text(
        json.dumps({"ok": False, "reason": "tool_or_host_json_missing"}),
        encoding="utf-8",
    )
    (gates_dir / "ruhmi_dispatch_blocked.json").write_text(
        json.dumps({"ok": False, "reason": "tool_or_host_json_missing"}),
        encoding="utf-8",
    )
    (golden_dir / "golden_320.json").write_text(json.dumps({"items": [{"detections": []}]}), encoding="utf-8")
    calibration = tmp_path / "calibration_320.json"
    calibration.write_text(json.dumps({"items": [{"image": "a.png"}] * 200}), encoding="utf-8")

    result = make_formal_chain_report(run, calibration_path=calibration)

    assert result["ok"]
    assert result["table_count"] >= 6
    assert (run / "formal_report_assets" / "class_distribution.csv").is_file()
    assert (run / "formal_report_assets" / "gate_matrix.csv").is_file()
    if result["figure_count"]:
        assert (run / "formal_report_assets" / "fig02_class_distribution.png").is_file()
        assert (run / "formal_report_assets" / "fig06_gate_status.png").is_file()
    text = (run / "formal_chain_report.md").read_text(encoding="utf-8")
    assert "## 1. 摘要" in text
    assert "### 1.1 图表与数据资产索引" in text
    assert "## 9. 门禁验收矩阵" in text
    assert "Figure 2. Class distribution" in text
    assert "派生数据表" in text
    assert "## 13. 有效性威胁与风险" in text
    assert "Host MERA" in text
    assert "BLOCKED" in text


def pytest_import_onnx():
    import pytest

    return pytest.importorskip("onnx")
