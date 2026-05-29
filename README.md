# Assistive Grasp Detector

辅助抓取项目的视觉与模型落地仓库。

当前阶段目标：

1. 建立可追踪的项目事实与实验记录。
2. 先在 PC 侧跑通模型 A 的基础检测链路：`yolov8n.pt` COCO-pretrained -> smoke training -> predict -> PT/ONNX 导出。
3. 在 PC 侧确认 YOLO 检测输出、NMS、阈值和 `416x416` 到 VGA `640x480` 的坐标回映射。
4. 再把 PT/ONNX 交给板侧电脑，在 e2 studio / RA 工具链中进行转换、量化、部署验证。
5. 后续进入自采桌面数据 fine-tune 和模型 B ROI 抓取矩形部分。

当前视觉路线一句话版：

> 固定 RGB 相机触发式双模型架构：模型 A 在 VGA 桌面图像中检测定位用户目标，模型 B 在选中 ROI 中输出抓取候选，所有结果回到 640x480 VGA 坐标，再经桌面平面标定进入机械臂控制链路。

## Important Files

- `AGENTS.md`: 仓库协作规则与 CodeGraph 使用约定。
- `docs/PROJECT_FACTS.md`: 当前已冻结/未冻结的项目事实。
- `docs/model_a_board_bringup.md`: 模型 A YOLOv8n 基线从 PC 训练/导出到后续上板的路线。
- `docs/model_a_board_handoff.md`: 模型 A PT/ONNX 交付给板侧 e2 studio 工具链的检查清单。
- `experiments/model_a_yolov8n_pc_export_run_001.md`: 第一次 YOLOv8n PC smoke training/export 实验记录。
- `scripts/model_a_yolov8n_smoke.py`: PC 侧 YOLOv8n 加载、短训练、预测和导出脚本。
- `scripts/model_a_letterbox_demo.py`: `416x416` letterbox 检测框反变换回 VGA 坐标的最小验证脚本。
