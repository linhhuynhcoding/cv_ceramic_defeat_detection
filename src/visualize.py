from __future__ import annotations

import cv2
import numpy as np

COLORS = {
    "crack": (0, 0, 255),
    "spalling": (0, 220, 255),
    "exfoliation": (255, 80, 0),
    "background": (150, 150, 150),
}


def draw_predictions(image_bgr: np.ndarray, predictions) -> np.ndarray:
    output = image_bgr.copy()
    for prediction in predictions:
        x, y, w, h = map(int, prediction.box_xywh)
        color = COLORS.get(prediction.label, (180, 180, 180))
        cv2.rectangle(output, (x, y), (x + w, y + h), color, 2)
        text = f"{prediction.label} {prediction.score:.2f}"
        cv2.putText(output, text, (x, max(15, y - 5)), cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 1, cv2.LINE_AA)
    return output


def draw_ground_truth(image_bgr: np.ndarray, boxes, labels) -> np.ndarray:
    output = image_bgr.copy()
    for box, label in zip(boxes, labels):
        x, y, w, h = map(int, box)
        color = COLORS.get(label, (180, 180, 180))
        cv2.rectangle(output, (x, y), (x + w, y + h), color, 2)
        cv2.putText(output, label, (x, max(15, y - 5)), cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 1, cv2.LINE_AA)
    return output


def draw_candidates(image_bgr: np.ndarray, candidates) -> np.ndarray:
    output = image_bgr.copy()
    for candidate in candidates:
        x, y, w, h = candidate.box_xywh
        cv2.rectangle(output, (x, y), (x + w, y + h), (160, 160, 160), 1)
    return output


def mask_to_bgr(mask: np.ndarray) -> np.ndarray:
    return cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)


def bgr_to_rgb(image_bgr: np.ndarray) -> np.ndarray:
    return cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
