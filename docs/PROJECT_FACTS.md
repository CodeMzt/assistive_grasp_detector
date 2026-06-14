# Project Facts

Last updated: 2026-06-14

本文件只记录当前已经明确或需要实测确认的项目事实。未实测内容不得写成已完成事实。

## Frozen Facts

1. 视觉硬件路线采用固定外部 RGB 相机，即 eye-to-hand。
2. 当前相机输入基准为 OV5640 VGA 640x480 YUV422，按 UYVY 解释。
3. VGA 640x480 原图是视觉系统唯一坐标母图。
4. 当前系统检测合同为 **Model A V2 / EthosSafeDetV2**，目标平台为 RA8P1 Ethos-U55 + RUHMI。
5. Model A V2 是触发式 6 类桌面物体检测、定位和朝向估计单模型；不再设独立 Model B/ROI 抓取矩形模型。
6. V2 类别固定为 `earbud_A`、`phial_A`、`bottle_A`、`phone_A`、`remote_A`、`tissue_A`。
7. V2 主输入为静态 `batch=1, 320x320`；如果 full-int8/MERA/内存 gate 失败，才评估同架构 `256x256` fallback。
8. V2 模型图只允许 Conv2D、DepthwiseConv2D、PointwiseConv2D、Add、ReLU/ReLU6、静态 Reshape/Pad/Pool 类算子。
9. V2 禁止 SiLU/Swish/Mish/GELU/Attention/SPPF、大量 concat、DFL、Softmax bins、图内 NMS/TopK/ArgMax/Gather/Shape/Range/Meshgrid/dynamic Slice/dynamic Reshape。
10. V2 输出 tensor 按 stride 8 和 stride 16 两尺度分离为 `cls_logits`、`box_ltrb`、`orientation`；不输出 `obj`，不把 bbox/class/orientation concat 到同一个 tensor。
11. V2 模型最后不加 sigmoid/exp/grid decode/atan2/NMS；这些全部在 CM85 C 后处理或 PC reference 后处理中完成。
12. calibration 必须使用 200-500 张真实板端相机图，不能使用随机 npy 或单张测试图。
13. ONNX 只作为 PC reference，opset 12/13，static shape，无 dynamic axes。
14. 主部署产物优先 TFLite full-int8；如果必须走 ONNX -> RUHMI，也必须先过 ONNX op whitelist 和 host MERA gate。
15. 只有 host MERA detection-level 与 PC reference 对齐后，才允许刷板。
16. 板端第一阶段只跑 static golden，不恢复 camera/annotated-vga。
17. 桌面平面标定负责从 VGA 像素坐标到机械臂桌面坐标映射。
18. 轨迹规划、RL、力触觉闭环负责接触、夹持、抬升、滑移补偿与安全递送。

## Interface Draft

```c
typedef struct
{
    uint8_t class_id;
    uint8_t theta_valid;
    float confidence;
    float bbox_x1_vga;
    float bbox_y1_vga;
    float bbox_x2_vga;
    float bbox_y2_vga;
    float orientation_sin2theta;
    float orientation_cos2theta;
    float orientation_rad;
} ethossafedet_det_t;
```

模型 raw outputs:

```text
stride 8:
  cls_logits: int8/float tensor, [1, 40, 40, 6]
  box_ltrb:   int8/float tensor, [1, 40, 40, 4]
  orientation:int8/float tensor, [1, 40, 40, 2]  # sin(2theta), cos(2theta)

stride 16:
  cls_logits: int8/float tensor, [1, 20, 20, 6]
  box_ltrb:   int8/float tensor, [1, 20, 20, 4]
  orientation:int8/float tensor, [1, 20, 20, 2]  # sin(2theta), cos(2theta)
```

## Acceptance Gates

1. PC ONNX vs host MERA：主目标 class 一致，bbox IoU >= 0.85，目标 >= 0.90，并记录 orientation 解码差异。
2. board static vs host MERA：主目标 class 一致，bbox IoU >= 0.85，top-k 候选点一致或可解释，并覆盖 `theta_valid=true` 样本的 orientation 输出。
3. RUHMI dispatch：heavy conv 全在 Ethos-U region；CPU 只允许 quant/dequant/reshape bridge；`num_base_addr <= 8`。
4. Memory：arena <= 2.5 MiB，weights <= 1.5 MiB。

## Not Frozen Yet

1. 检测器仓库中的 V2 两尺度 orientation head、训练目标、导出和 gate 实现仍在迁移；bbox-only 候选不满足固件 V2 合同。
2. `theta_valid=true` 样本的 orientation 误差阈值、mask 统计和失败报告格式。
3. `320x320` 是否满足 arena/weights/dispatch；若不满足，切换 `256x256` fallback。
4. PTQ 是否足够；如掉点明显，再评估 QAT。
5. CM85 C 后处理阈值、候选 cap、NMS 参数、`atan2` 解码与耗时。
6. 板侧 RUHMI dispatch、Tensor Arena、延迟和内存均以实验记录为准。
