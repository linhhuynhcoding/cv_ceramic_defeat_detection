from __future__ import annotations

import numpy as np


def xywh_to_xyxy(boxes: np.ndarray) -> np.ndarray:
    boxes = np.asarray(boxes, dtype=np.float32)
    if boxes.size == 0:
        return np.zeros((0, 4), dtype=np.float32)
    converted = boxes.copy()
    converted[:, 2] = boxes[:, 0] + boxes[:, 2]
    converted[:, 3] = boxes[:, 1] + boxes[:, 3]
    return converted


def box_iou_xywh(box: tuple[int, int, int, int] | np.ndarray, boxes: np.ndarray) -> np.ndarray:
    if boxes.size == 0:
        return np.zeros((0,), dtype=np.float32)
    box_arr = np.asarray(box, dtype=np.float32).reshape(1, 4)
    a = xywh_to_xyxy(box_arr)[0]
    b = xywh_to_xyxy(boxes)
    x1 = np.maximum(a[0], b[:, 0])
    y1 = np.maximum(a[1], b[:, 1])
    x2 = np.minimum(a[2], b[:, 2])
    y2 = np.minimum(a[3], b[:, 3])
    inter = np.maximum(0, x2 - x1) * np.maximum(0, y2 - y1)
    area_a = max(0.0, (a[2] - a[0]) * (a[3] - a[1]))
    area_b = np.maximum(0, b[:, 2] - b[:, 0]) * np.maximum(0, b[:, 3] - b[:, 1])
    union = area_a + area_b - inter
    return inter / np.maximum(union, 1e-6)


def nms_xywh(boxes: list[tuple[int, int, int, int]], scores: list[float], threshold: float) -> list[int]:
    if not boxes:
        return []
    boxes_arr = np.asarray(boxes, dtype=np.float32)
    scores_arr = np.asarray(scores, dtype=np.float32)
    order = scores_arr.argsort()[::-1]
    keep: list[int] = []
    while order.size > 0:
        current = int(order[0])
        keep.append(current)
        if order.size == 1:
            break
        ious = box_iou_xywh(boxes_arr[current], boxes_arr[order[1:]])
        order = order[1:][ious <= threshold]
    return keep
