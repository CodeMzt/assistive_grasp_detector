# Model A YOLOv8n Bring-Up

目标：先用本项目模型 A 的训练基础模型 `yolov8n.pt` 跑通本机 PC 侧检测链路，训练侧只交付 `.pt` 和/或 `.onnx`；量化、转换和板端集成由另一台板侧开发电脑的 e2 studio / RA 工具链完成。

当前状态：PC 侧 smoke training/export 待开始实测。

## Bring-Up Principle

这里的“预训练模型”指 Ultralytics `yolov8n.pt` COCO-pretrained detect model，不指 Renesas 官方 Vision AI demo 的分类模型。

第一次不追求本项目类别表，也不追求抓取输出。第一次只验证模型 A 检测链路：

```text
yolov8n.pt
-> 416x416 smoke training
-> predict on one image
-> export ONNX
-> inspect raw output tensor and Python post-processing
-> map bbox back to VGA 640x480 coordinates
-> record artifacts, hashes, output shape and problems
```

RA8P1/e2 studio 转换和上板是下一阶段。Renesas 官方示例只作为后续工程集成参考，不作为模型 A 的基础模型。

## What Counts As Success

一次有效的 PC 侧 YOLOv8n 基线实验至少满足：

1. 能明确记录 Python、Ultralytics、Torch、ONNX/ONNX Runtime 相关版本。
2. 能成功加载 `yolov8n.pt`。
3. 能以 `imgsz=416` 完成 1 epoch 或等价 smoke training。
4. 能对一张测试图执行 predict 并输出检测框。
5. 能成功导出 ONNX。
6. 能记录原始输出 tensor shape、NMS/阈值策略和输出样例。
7. 能把检测框统一整理为 `semantic_det_raw_t` 风格结果。
8. 能验证 `416x416` letterbox 坐标反变换回 VGA `640x480` 坐标。
9. 能明确给出板侧 handoff artifact：优先 `.onnx`，保留 `.pt` 作为训练基线备份。

## Current Project Input Domain

本项目已验证输入域：

```text
OV5640
DVP/CEU
VGA 640x480
YUV422 UYVY
```

PC 侧首跑可以使用普通图片文件作为输入，但所有坐标检查必须回到 VGA `640x480` 母图语义。若测试图片不是 640x480，实验记录必须标明它不是项目输入域验证。

## Recommended Step Order

1. 建立 Python 虚拟环境。
2. 安装 `ultralytics`、`onnx`、`onnxruntime`。
3. 加载 `yolov8n.pt`。
4. 用 `coco8.yaml` 或等价最小数据集做 `imgsz=416` smoke training。
5. 对一张测试图做 predict，保存预测图和 JSON 摘要。
6. 导出 ONNX，记录路径和 hash。
7. 可选导出 NMS ONNX 作为 PC 侧输出对照；板侧优先评估 raw ONNX。
8. 记录 raw tensor shape、score/NMS 策略。
9. 用 `scripts/model_a_letterbox_demo.py` 验证 bbox 反变换规则。
10. 把 `.pt` / `.onnx` 交给板侧电脑，在 e2 studio / RA 工具链中执行量化、转换和集成。

## Information To Capture

| Item | Value |
|---|---|
| Python version | TBD |
| Ultralytics version | TBD |
| Torch version | TBD |
| Base weights | `yolov8n.pt` |
| Training dataset | `coco8.yaml` / TBD |
| Input size | 416x416 RGB |
| Test image | 640x480 preferred |
| PT artifact | TBD |
| raw ONNX artifact | TBD |
| NMS ONNX artifact | optional PC reference |
| Raw output tensor shape | TBD |
| NMS/conf/IoU parameters | TBD |
| bbox reverse transform check | TBD |

## Expected Adaptation After PC Export

PC 侧 YOLOv8n 链路跑通后，本项目模型 A 需要逐步推进为：

```text
VGA 640x480 UYVY
-> UYVY to RGB
-> workspace crop / letterbox / resize
-> 416x416 RGB
-> e2 studio / RA 工具链完成量化和转换
-> board-side detector inference
-> detector post-processing
-> bbox reverse transform to VGA coordinates
```

此阶段仍不引入模型 B。模型 A 的第一阶段目标是可靠输出：

```text
class_id
confidence
bbox_xyxy in VGA coordinates
```

## Open Questions

1. 板侧 e2 studio / RA 工具链更适合接受 raw ONNX 还是带 NMS 的 ONNX。
2. YOLOv8n head 的后处理是否适合在 CM85 上实现。
3. Tensor Arena、SDRAM、OSPI 模型存储和推理延迟是否满足触发式运行。
4. 自采桌面数据的最终类别表和样本量。
5. 板端是否直接运行 detector NMS，还是输出 top-k raw candidates 后由 CPU 后处理。
