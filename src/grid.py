from __future__ import annotations

import cv2
import numpy as np


def _odd(value: int) -> int:
    value = max(1, int(value))
    return value if value % 2 == 1 else value + 1


def detect_grid_debug(preprocessed: dict, params: dict | None = None) -> dict:
    params = params or {}
    gray = preprocessed["enhanced_gray"]
    valid = preprocessed["valid_tile_mask"]
    h, w = gray.shape

    dark_threshold = int(params.get("grid_dark_threshold", 95))
    horizontal_scale = max(4, int(params.get("grid_horizontal_scale", 18)))
    vertical_scale = max(4, int(params.get("grid_vertical_scale", 12)))
    hough_threshold = int(params.get("hough_threshold", 45))
    hough_min_line_divisor = max(2, int(params.get("hough_min_line_divisor", 6)))
    hough_max_gap = int(params.get("hough_max_gap", 12))
    hough_min_length_ratio = float(params.get("hough_min_length_ratio", 0.16))
    line_thickness = int(params.get("grid_line_thickness", 5))
    dilation_kernel = _odd(int(params.get("grid_dilation_kernel", 5)))
    grid_canny_low = int(params.get("grid_canny_low", 40))
    grid_canny_high = int(params.get("grid_canny_high", 120))

    dark = cv2.inRange(gray, 0, dark_threshold)
    dark = cv2.bitwise_and(dark, valid)

    horizontal_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (max(15, w // horizontal_scale), 3))
    vertical_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, max(15, h // vertical_scale)))
    horizontal = cv2.morphologyEx(dark, cv2.MORPH_OPEN, horizontal_kernel)
    vertical = cv2.morphologyEx(dark, cv2.MORPH_OPEN, vertical_kernel)
    mask = cv2.bitwise_or(horizontal, vertical)
    mask_before_hough = mask.copy()

    edges = cv2.Canny(gray, grid_canny_low, grid_canny_high)
    lines = cv2.HoughLinesP(
        edges,
        rho=1,
        theta=np.pi / 180,
        threshold=hough_threshold,
        minLineLength=max(35, min(h, w) // hough_min_line_divisor),
        maxLineGap=hough_max_gap,
    )
    hough_mask = np.zeros_like(mask)
    if lines is not None:
        for line in np.asarray(lines).reshape(-1, 4):
            x1, y1, x2, y2 = map(int, line)
            length = np.hypot(x2 - x1, y2 - y1)
            if length >= min(h, w) * hough_min_length_ratio:
                cv2.line(mask, (x1, y1), (x2, y2), 255, line_thickness)
                cv2.line(hough_mask, (x1, y1), (x2, y2), 255, line_thickness)
    
    mask_after_hough = mask.copy()

    dilate_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (dilation_kernel, dilation_kernel))
    mask = cv2.dilate(mask, dilate_kernel, iterations=1)
    mask = cv2.bitwise_and(mask, valid)
    
    return {
        "mask": mask,
        "dark": dark,
        "horizontal": horizontal,
        "vertical": vertical,
        "morph_combined": mask_before_hough,
        "edges": edges,
        "hough_lines": hough_mask,
        "mask_after_hough": mask_after_hough,
    }

def detect_grid_mask(preprocessed: dict, params: dict | None = None) -> np.ndarray:
    return detect_grid_debug(preprocessed, params)["mask"]
