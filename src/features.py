from __future__ import annotations

import cv2
import numpy as np
from skimage.feature import local_binary_pattern

from src.candidates import Candidate

SOURCE_NAMES = ("crack_branch", "spalling_branch", "exfoliation_branch", "gt_seed")


def _safe_stats(values: np.ndarray) -> tuple[float, float]:
    if values.size == 0:
        return 0.0, 0.0
    return float(values.mean()), float(values.std())


def extract_features(image_bgr: np.ndarray, candidate: Candidate, grid_mask: np.ndarray) -> dict[str, float]:
    x, y, w, h = candidate.box_xywh
    image_h, image_w = image_bgr.shape[:2]
    roi = image_bgr[y : y + h, x : x + w]
    roi_mask = candidate.mask[y : y + h, x : x + w]
    gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
    lab = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2LAB)
    roi_gray = gray[y : y + h, x : x + w]
    roi_lab = lab[y : y + h, x : x + w]

    mask_pixels = roi_mask > 0
    area = float(mask_pixels.sum())
    bbox_area = float(max(w * h, 1))
    contours, _ = cv2.findContours(roi_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    perimeter = float(sum(cv2.arcLength(contour, True) for contour in contours))
    contour_area = float(sum(cv2.contourArea(contour) for contour in contours))
    hull_area = 0.0
    for contour in contours:
        if len(contour) >= 3:
            hull_area += cv2.contourArea(cv2.convexHull(contour))

    circularity = 0.0 if perimeter <= 0 else float(4 * np.pi * contour_area / (perimeter * perimeter))
    solidity = 0.0 if hull_area <= 0 else float(contour_area / hull_area)
    extent = float(area / bbox_area)
    aspect_ratio = float(w / max(h, 1))
    area_ratio = float(area / max(image_h * image_w, 1))

    gray_values = roi_gray[mask_pixels]
    gray_mean, gray_std = _safe_stats(gray_values)
    dark_ratio = float((gray_values < 100).mean()) if gray_values.size else 0.0

    features: dict[str, float] = {
        "area": area,
        "width": float(w),
        "height": float(h),
        "aspect_ratio": aspect_ratio,
        "max_side": float(max(w, h)),
        "min_side": float(min(w, h)),
        "extent": extent,
        "solidity": solidity,
        "perimeter": perimeter,
        "circularity": circularity,
        "area_ratio": area_ratio,
        "gray_mean": gray_mean,
        "gray_std": gray_std,
        "dark_ratio": dark_ratio,
    }

    for idx, channel in enumerate(("l", "a", "b")):
        values = roi_lab[:, :, idx][mask_pixels]
        mean, std = _safe_stats(values)
        features[f"{channel}_mean"] = mean
        features[f"{channel}_std"] = std

    pad = 8
    sx0 = max(0, x - pad)
    sy0 = max(0, y - pad)
    sx1 = min(image_w, x + w + pad)
    sy1 = min(image_h, y + h + pad)
    surround = gray[sy0:sy1, sx0:sx1]
    surround_mean = float(surround.mean()) if surround.size else gray_mean
    features["surround_contrast"] = abs(gray_mean - surround_mean)

    edges = cv2.Canny(roi_gray, 40, 120)
    features["edge_density"] = float((edges > 0).mean()) if edges.size else 0.0
    features["local_variance"] = float(roi_gray.var()) if roi_gray.size else 0.0

    if roi_gray.size and min(roi_gray.shape) >= 3:
        lbp = local_binary_pattern(roi_gray, P=8, R=1, method="uniform")
        hist, _ = np.histogram(lbp[mask_pixels] if mask_pixels.any() else lbp.ravel(), bins=10, range=(0, 10), density=True)
    else:
        hist = np.zeros(10, dtype=np.float32)
    for idx, value in enumerate(hist):
        features[f"lbp_{idx}"] = float(value)

    grid_roi = grid_mask[y : y + h, x : x + w]
    features["grid_overlap_ratio"] = float(((grid_roi > 0) & mask_pixels).sum() / max(area, 1.0))

    for source in SOURCE_NAMES:
        features[f"source_{source}"] = 1.0 if candidate.source == source else 0.0

    return features


def features_to_matrix(feature_rows: list[dict[str, float]], schema: list[str] | None = None) -> tuple[np.ndarray, list[str]]:
    if not feature_rows:
        if schema is None:
            return np.zeros((0, 0), dtype=np.float32), []
        return np.zeros((0, len(schema)), dtype=np.float32), schema
    keys = schema or sorted(feature_rows[0].keys())
    matrix = np.array([[row.get(key, 0.0) for key in keys] for row in feature_rows], dtype=np.float32)
    return matrix, keys
