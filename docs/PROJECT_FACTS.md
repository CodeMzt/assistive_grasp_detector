# Project Facts

Last updated: 2026-05-29

本文件只记录当前已经明确或需要实测确认的项目事实。未实测内容不得写成已完成事实。

## Frozen Facts

1. 视觉硬件路线采用固定外部 RGB 相机，即 eye-to-hand。
2. 当前相机输入基准为 OV5640 VGA 640x480 YUV422，按 UYVY 解释。
3. VGA 640x480 原图是视觉系统唯一坐标母图。
4. 模型 A 和模型 B 都是触发式运行，不设计为两个模型同时实时常跑。
5. 模型 A 是语义检测/定位模型，不是纯分类模型。
6. 模型 A 训练验证阶段输入暂定为 416x416 RGB。
7. 模型 A 输出 class、confidence、bbox，bbox 必须反变换回 VGA 坐标。
8. 预设身份由 class 表达，例如 `phone_A`、`phone_other`，不单独输出身份字段。
9. 模型 A v0 基础训练模型定为 Ultralytics `yolov8n.pt` COCO-pretrained detect model。
10. 模型 A v0 首轮实验只要求 PC 侧跑通加载、smoke training、predict 和 PT/ONNX 导出。
11. YOLO 原始输出 tensor、score 计算、NMS 和阈值属于模型 A 内部实现细节；对外接口统一收敛到 `semantic_det_raw_t`。
12. 模型 B 是 ROI 级 RGB 抓取矩形模型。
13. 模型 B 的 ROI 必须从 VGA 原图裁剪，而不是从模型 A 的 416x416 输入图裁剪。
14. 模型 B 输出抓取中心、角度、宽度、质量分数等候选，不直接控制机械臂。
15. 桌面平面标定负责从 VGA 像素坐标到机械臂桌面坐标映射。
16. 轨迹规划、RL、力触觉闭环负责接触、夹持、抬升、滑移补偿与安全递送。
17. 数据集采用场景级主标注，再生成模型 A 和模型 B 的训练数据。

## Not Frozen Yet

1. 模型 A 最终上板版本是否仍使用 YOLOv8n。
2. YOLOv8n 经板侧 e2 studio / RA 工具链转换、量化和后处理验证后是否满足 RA8P1 资源约束。
3. 模型 B 的具体网络结构。
4. 模型 B 的输入尺寸。
5. 模型 B 是否加入 mask、坐标通道、class embedding。
6. COCO、Jacquard、VMRD 等公开数据的具体混合比例。
7. `grasp_quality` 阈值。
8. ROI 外扩比例。
9. 工作区裁剪范围。
10. 最终类别表数量。
11. 板侧转换结果、Tensor Arena 占用、推理延迟、后处理成本。

## Coordinate Contract

```text
VGA 640x480
-> 模型 A 预处理到 416x416
-> 模型 A 输出 bbox_416
-> bbox_416 反变换回 bbox_vga
-> 从 VGA 原图裁剪 ROI
-> 模型 B 输出 grasp_center_roi / angle_roi / width_roi
-> 反变换回 grasp_center_vga / grasp_axis_vga
-> homography 映射到机械臂桌面坐标
-> 得到 x_mm / y_mm / grasp_yaw / grasp_width
```

## Model A Interface Draft

```c
typedef struct
{
    uint8_t class_id;
    float confidence;

    // Already transformed back to VGA 640x480 coordinates.
    float bbox_x1;
    float bbox_y1;
    float bbox_x2;
    float bbox_y2;
} semantic_det_raw_t;
```

```c
typedef struct
{
    uint8_t class_id;
    float confidence;

    float bbox_x1;
    float bbox_y1;
    float bbox_x2;
    float bbox_y2;

    uint8_t instance_index;
    uint8_t graspable;
    uint8_t need_grasp_model;
    uint8_t grasp_template_id;
} semantic_object_t;
```

## Model B Interface Draft

```c
typedef struct
{
    float quality;

    float center_u_roi;
    float center_v_roi;
    float angle_img_rad;
    float width_px_roi;

    float center_u_vga;
    float center_v_vga;
    float width_px_vga;

    uint8_t valid;
} grasp_candidate_img_t;
```

```c
typedef struct
{
    float x_mm;
    float y_mm;
    float grasp_yaw_rad;
    float grasp_width_mm;

    float quality;
    uint8_t valid;
} grasp_candidate_robot_t;
```
