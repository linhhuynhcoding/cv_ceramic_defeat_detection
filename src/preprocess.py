from __future__ import annotations

import cv2
import numpy as np


def _odd(value: int) -> int:
    value = max(1, int(value))
    return value if value % 2 == 1 else value + 1


def preprocess_image(image_bgr: np.ndarray, params: dict | None = None) -> dict:
    params = params or {}
    gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
    lab = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2LAB)

    clahe_clip_limit = float(params.get("clahe_clip_limit", 2.0))
    clahe_tile_size = max(2, int(params.get("clahe_tile_size", 8)))
    bilateral_d = _odd(int(params.get("bilateral_d", 7)))
    bilateral_sigma_color = float(params.get("bilateral_sigma_color", 50))
    bilateral_sigma_space = float(params.get("bilateral_sigma_space", 50))
    valid_threshold = int(params.get("valid_threshold", 20))
    valid_morph_kernel = _odd(int(params.get("valid_morph_kernel", 7)))

    clahe = cv2.createCLAHE(clipLimit=clahe_clip_limit, tileGridSize=(clahe_tile_size, clahe_tile_size))
    enhanced_l = clahe.apply(lab[:, :, 0])
    enhanced_gray = cv2.bilateralFilter(enhanced_l, bilateral_d, bilateral_sigma_color, bilateral_sigma_space)

    valid_tile_mask = (gray > valid_threshold).astype(np.uint8) * 255
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (valid_morph_kernel, valid_morph_kernel))
    valid_tile_mask = cv2.morphologyEx(valid_tile_mask, cv2.MORPH_CLOSE, kernel)
    valid_tile_mask = cv2.morphologyEx(valid_tile_mask, cv2.MORPH_OPEN, kernel)

    return {
        "original": image_bgr,
        "gray": gray,
        "lab": lab,
        "enhanced_gray": enhanced_gray,
        "valid_tile_mask": valid_tile_mask,
    }
