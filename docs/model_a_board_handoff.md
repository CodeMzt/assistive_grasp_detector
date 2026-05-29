# Model A Board-Side Handoff

本文件用于把本机训练得到的模型 A 交给另一台板侧开发电脑。训练侧只提供 `.pt` 和 `.onnx`，不在本机做最终量化或部署判断。

## Handoff Artifacts

优先交付：

```text
runs/detect/runs/model_a_yolov8n_pc_export_run_001/train/weights/best.pt
runs/detect/runs/model_a_yolov8n_pc_export_run_001/train/weights/best_raw.onnx
```

可选对照：

```text
runs/detect/runs/model_a_yolov8n_pc_export_run_001/train/weights/best_nms.onnx
```

建议板侧优先尝试 `best_raw.onnx`，因为它保留 YOLOv8n 原始检测输出，后处理策略更容易由我们按项目接口控制。`best_nms.onnx` 只作为 e2 studio 导入、PC 对照或后处理诊断参考。

## Tensor Contract

当前 raw ONNX：

```text
input:  images  [1, 3, 416, 416]
output: output0 [1, 84, 3549]
```

当前 NMS ONNX：

```text
input:  images  [1, 3, 416, 416]
output: output0 [1, 300, 6]
```

导入 e2 studio / RA 工具链时需要确认工具是否自动处理 NCHW/NHWC 转换。如果工具要求 NHWC，板侧工程必须记录转换发生在工具链内部还是需要我们在预处理代码里处理。

## Board-Side Checklist

1. 记录 e2 studio、FSP、RA 工具链、模型转换工具和板卡 BSP 版本。
2. 导入 `best_raw.onnx`，确认输入输出 tensor shape 和数据类型。
3. 执行工具链量化，校准数据优先使用本项目 OV5640 VGA 桌面图片经同一套 `416x416` letterbox 预处理后的样本。
4. 编译生成板侧模型 artifact，记录模型大小、Tensor Arena、是否使用 NPU/CPU、编译 warning/error。
5. 在板端接入预处理：
   `OV5640 UYVY -> RGB -> letterbox/resize -> 416x416 tensor`。
6. 在板端接入后处理：
   raw output -> score threshold -> NMS/top-k -> bbox 反变换回 VGA -> `semantic_det_raw_t`。
7. 用固定测试画面先验证输出稳定，再接语音目标选择和模型 B。

## What To Send Back

板侧每次尝试后，把这些信息贴回实验记录：

```text
model artifact used:
conversion tool/version:
quantization type:
calibration data:
input tensor:
output tensor:
compile result:
flash/run result:
latency:
memory/Tensor Arena:
first detection output:
problem/next step:
```
