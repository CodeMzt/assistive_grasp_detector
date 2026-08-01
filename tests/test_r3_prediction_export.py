from __future__ import annotations

from assistive_grasp_detector.ethossafedet_v2_r3_predictions import match_record_detections, residual_from_boxes


def test_one_to_one_same_class_matching_and_cascade_availability():
    detections = [
        {"class_id": 5, "score": 0.9, "bbox_xyxy_vga": [10, 10, 30, 30]},
        {"class_id": 5, "score": 0.8, "bbox_xyxy_vga": [50, 50, 70, 70]},
    ]
    targets = [
        {"instance_id": 1, "class_id": 5, "class_name": "tissue", "bbox_xyxy_vga": [11, 11, 29, 29]},
        {"instance_id": 2, "class_id": 5, "class_name": "tissue", "bbox_xyxy_vga": [100, 100, 120, 120]},
    ]
    rows, used = match_record_detections(detections, targets)
    assert rows[0]["roi_input_available"] is True
    assert rows[0]["match_iou50"] is True
    assert rows[1]["roi_input_available"] is False
    assert used == {0}


def test_bbox_residual_is_relative_to_ground_truth_geometry():
    residual = residual_from_boxes([15, 20, 35, 60], [10, 20, 30, 60])
    assert residual["dx_over_gt_w"] == 0.25
    assert residual["dy_over_gt_h"] == 0.0
    assert residual["log_w_over_gt_w"] == 0.0
