# Assistive Grasp Detector

本仓库当前系统合同与固件仓库对齐为 **Model A V2 / EthosSafeDetV2**：面向 RA8P1 Ethos-U55 + RUHMI 的单模型 7 类桌面物体检测、定位和朝向估计。历史 bbox-only EthosSafeDet-A v1 产物和旧 ROI 抓取矩形方案已退役，不能作为当前固件接收合同；新的轮廓训练在平行仓库 `assistive_grasp_contour_model` 中进行。

## Current Contract

- 相机与坐标母图：OV5640 VGA 640x480，固定外部 RGB 相机，eye-to-hand。
- 业务类别：`earbud`、`phial`、`bottle`、`phone`、`remote`、`tissue`、`apple`。
- 模型输入：静态 `batch=1`，主线 `320x320`，失败兜底 `256x256`。
- 模型输出：两尺度 stride 8 与 stride 16；每个尺度分离输出 `cls[7]`、`box[4]`、`orientation[2]`，orientation 为 `sin(2theta), cos(2theta)`。
- 朝向有效性：只有 `theta_valid=true` 的正样本参与朝向监督和验收；方向不明确或不需要朝向的目标必须显式记录为无效朝向。
- 图内非目标：不导出 `obj`，不做图内 sigmoid/exp/grid decode/NMS/TopK/ArgMax/Gather/dynamic shape。
- 部署产物：TFLite full-int8 优先；ONNX 只作为 PC reference。
- 校准数据：必须来自 200-500 张真实板端相机图，不能用随机 npy 或单张测试图。
- 交付边界：Git 保存 manifest、hash、接口说明、gate 摘要和脚本；权重、训练输出、RUHMI/板端大产物不直接进入 Git。

## Current Implementation State

- 现有部分 CLI、schema 和训练报告仍带 `EthosSafeDet-A/v1` 历史命名。它们可以用于迁移验证，但输出若只有 class+bbox 且没有 orientation head，不满足 Model A V2 固件接收条件。
- 任何准备交给固件的候选模型都必须附带 V2 tensor contract、operator gate、静态 golden、MERA/RUHMI/memory 摘要和 SHA-256 manifest。

## Main Commands

```powershell
# 从现有导出布局生成 EthosSafeDet manifest
prepare_ethossafedet_manifest `
  --dataset data\raw\model_a\first_batch_20260604 `
  --out data\generated\ethossafedet_a\first_batch_20260604\ethossafedet_manifest.jsonl `
  --negative-image-id 000139 --negative-image-id 000739 --negative-image-id 000837 --negative-image-id 000838 --negative-image-id 001151

# 生成 full-int8 representative calibration manifest
build_ethossafedet_calibration `
  --manifest data\generated\ethossafedet_a\first_batch_20260604\ethossafedet_manifest.jsonl `
  --out data\generated\ethossafedet_a\first_batch_20260604\calibration_320.json `
  --target-count 320 --seed 0

# 训练与导出；历史命令名保留，产物必须在报告中声明是否满足 Model A V2
train_ethossafedet_a --manifest <manifest.jsonl> --out runs\ethossafedet_a_v2_candidate --input-size 320 --epochs 1 --device cuda
export_ethossafedet_onnx --checkpoint runs\ethossafedet_a_v2_candidate\ethossafedet_a.pt --out runs\ethossafedet_a_v2_candidate\ethossafedet_a_320.onnx
export_ethossafedet_tflite --onnx <model.onnx> --calibration <calibration.json> --out <model_full_int8.tflite>
```

## Required Gates

每次导出都按顺序运行：

```powershell
check_onnx_ops --onnx <model.onnx>
check_tflite_ops --tflite <model_full_int8.tflite>
compare_ethossafedet_reference --onnx <model.onnx> --tflite <model_full_int8.tflite> --image <board_image.png>
run_host_mera_gate --reference-json <pc_reference.json> --host-json <host_mera.json>
inspect_ruhmi_dispatch --log <ruhmi_dispatch.log>
check_memory_budget --log <mera_or_board_memory.log>
make_static_golden --onnx <model.onnx> --manifest <manifest.jsonl> --out <static_golden.json>
```

验收阈值：PC ONNX vs host MERA 主目标 class 一致且 bbox IoU >= 0.85；board static vs host MERA 主目标 class 一致且 bbox IoU >= 0.85；RUHMI heavy conv 必须在 Ethos-U region，CPU 只允许 quant/dequant/reshape bridge，`num_base_addr <= 8`；arena <= 2.5 MiB，weights <= 1.5 MiB。

V2 静态 golden 还必须覆盖 orientation 输出存在性、`theta_valid` mask 语义和 CM85 后处理的 `atan2` 解码。具体角度误差阈值需要以 V2 gate 记录冻结，未冻结前不得把候选模型标成固件可接收。

## Important Files

- `assistive_grasp_detector/ethossafedet_manifest.py`: JSONL manifest 与 calibration manifest。
- `assistive_grasp_detector/ethossafedet_model.py`: EthosSafeDet-A 模型图；bbox-only 输出为 pre-V2/legacy，不等同于固件 V2 合同。
- `assistive_grasp_detector/ethossafedet_train.py`: 最小训练链路。
- `assistive_grasp_detector/ethossafedet_export.py`: ONNX reference 与 TFLite full-int8 导出。
- `assistive_grasp_detector/ethossafedet_gates.py`: ONNX/TFLite/MERA/RUHMI/memory/static golden gates。
- `docs/PROJECT_FACTS.md`: 当前冻结事实与未冻结事项。

真实数据、generated 数据、runs 与模型权重保持在 `.gitignore` 覆盖路径中；实验结论必须以可复现记录和 gate 输出为准。

## 2026-06-14 object_vocab_v1 alignment

Detector training and annotation configs now consume `configs/classes/object_vocab_v1.json` and the canonical seven-class order `0 earbud`, `1 phial`, `2 bottle`, `3 phone`, `4 remote`, `5 tissue`, `6 apple`. Existing six-class board exports remain legacy until a retrained seven-class model is exported and accepted by main firmware static golden checks.

