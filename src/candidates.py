from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np

from src.grid import detect_grid_mask
from src.preprocess import preprocess_image


@dataclass(frozen=True)
class Candidate:
    box_xywh: tuple[int, int, int, int]
    mask: np.ndarray
    source: str


def _clip_box(x: int, y: int, w: int, h: int, image_shape: tuple[int, int]) -> tuple[int, int, int, int]:
    ih, iw = image_shape
    x = max(0, min(x, iw - 1))
    y = max(0, min(y, ih - 1))
    w = max(1, min(w, iw - x))
    h = max(1, min(h, ih - y))
    return x, y, w, h


def _component_candidates(binary: np.ndarray, source: str, min_area: int, max_area_ratio: float) -> list[Candidate]:
    count, labels, stats, _ = cv2.connectedComponentsWithStats(binary, connectivity=8)
    image_area = binary.shape[0] * binary.shape[1]
    candidates: list[Candidate] = []
    for idx in range(1, count):
        x, y, w, h, area = stats[idx]
        if area < min_area or area > image_area * max_area_ratio:
            continue
        if w < 2 or h < 2:
            continue
        component_mask = np.zeros_like(binary)
        component_mask[labels == idx] = 255
        candidates.append(Candidate(_clip_box(int(x), int(y), int(w), int(h), binary.shape), component_mask, source))
    return candidates


def _dedupe_candidates(candidates: list[Candidate]) -> list[Candidate]:
    seen: set[tuple[int, int, int, int, str]] = set()
    deduped: list[Candidate] = []
    for candidate in candidates:
        key = (*candidate.box_xywh, candidate.source)
        if key in seen:
            continue
        seen.add(key)
        deduped.append(candidate)
    return deduped


def _odd(value: int) -> int:
    value = max(1, int(value))
    return value if value % 2 == 1 else value + 1


def generate_candidate_debug(image_bgr: np.ndarray, params: dict | None = None) -> dict:
    params = params or {}
    pre = preprocess_image(image_bgr, params)
    gray = pre["enhanced_gray"]
    valid = pre["valid_tile_mask"]
    grid = detect_grid_mask(pre, params)
    non_grid = cv2.bitwise_and(valid, cv2.bitwise_not(grid))
    h, w = gray.shape

    candidates: list[Candidate] = []

    crack_blackhat_kernel = _odd(int(params.get("crack_blackhat_kernel", 13)))
    crack_close_kernel = _odd(int(params.get("crack_close_kernel", 3)))
    crack_canny_low = int(params.get("crack_canny_low", 45))
    crack_canny_high = int(params.get("crack_canny_high", 135))
    crack_min_area = int(params.get("crack_min_area", 8))
    crack_max_area_ratio = float(params.get("crack_max_area_ratio", 0.08))
    crack_min_aspect = float(params.get("crack_min_aspect", 1.6))
    crack_min_side = int(params.get("crack_min_side", 10))

    crack_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (crack_blackhat_kernel, crack_blackhat_kernel))
    blackhat = cv2.morphologyEx(gray, cv2.MORPH_BLACKHAT, crack_kernel)
    _, crack_thresh = cv2.threshold(blackhat, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    edges = cv2.Canny(gray, crack_canny_low, crack_canny_high)
    crack_mask = cv2.bitwise_or(crack_thresh, edges)
    crack_mask = cv2.bitwise_and(crack_mask, non_grid)
    crack_mask = cv2.morphologyEx(
        crack_mask,
        cv2.MORPH_CLOSE,
        cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (crack_close_kernel, crack_close_kernel)),
        iterations=1,
    )
    for candidate in _component_candidates(
        crack_mask,
        "crack_branch",
        min_area=crack_min_area,
        max_area_ratio=crack_max_area_ratio,
    ):
        x, y, cw, ch = candidate.box_xywh
        aspect = max(cw / max(ch, 1), ch / max(cw, 1))
        if aspect >= crack_min_aspect or max(cw, ch) >= crack_min_side:
            candidates.append(candidate)

    lab = pre["lab"]
    l_channel = lab[:, :, 0]
    spalling_blur_sigma = float(params.get("spalling_blur_sigma", 9))
    spalling_dark_threshold = int(params.get("spalling_dark_threshold", 110))
    spalling_open_kernel = _odd(int(params.get("spalling_open_kernel", 3)))
    spalling_min_area = int(params.get("spalling_min_area", 12))
    spalling_max_area_ratio = float(params.get("spalling_max_area_ratio", 0.12))

    blur = cv2.GaussianBlur(l_channel, (0, 0), spalling_blur_sigma)
    local_diff = cv2.absdiff(l_channel, blur)
    _, spot_mask = cv2.threshold(local_diff, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    dark = cv2.inRange(gray, 0, spalling_dark_threshold)
    spalling_mask = cv2.bitwise_or(spot_mask, dark)
    spalling_mask = cv2.bitwise_and(spalling_mask, non_grid)
    spalling_mask = cv2.morphologyEx(
        spalling_mask,
        cv2.MORPH_OPEN,
        cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (spalling_open_kernel, spalling_open_kernel)),
        iterations=1,
    )
    candidates.extend(
        _component_candidates(
            spalling_mask,
            "spalling_branch",
            min_area=spalling_min_area,
            max_area_ratio=spalling_max_area_ratio,
        )
    )

    ab = lab[:, :, 1:].astype(np.float32)
    valid_pixels = ab[valid > 0]
    if valid_pixels.size:
        mean_ab = valid_pixels.mean(axis=0)
        color_dist = np.linalg.norm(ab - mean_ab, axis=2).astype(np.uint8)
        _, exf_mask = cv2.threshold(color_dist, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    else:
        exf_mask = np.zeros((h, w), dtype=np.uint8)
    variance = cv2.Laplacian(gray, cv2.CV_64F)
    variance = np.uint8(np.clip(np.abs(variance), 0, 255))
    _, var_mask = cv2.threshold(variance, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    exfoliation_close_kernel = _odd(int(params.get("exfoliation_close_kernel", 9)))
    exfoliation_min_area = int(params.get("exfoliation_min_area", 80))
    exfoliation_max_area_ratio = float(params.get("exfoliation_max_area_ratio", 0.35))
    exfoliation_mask = cv2.bitwise_or(exf_mask, var_mask)
    exfoliation_mask = cv2.bitwise_and(exfoliation_mask, non_grid)
    exfoliation_mask = cv2.morphologyEx(
        exfoliation_mask,
        cv2.MORPH_CLOSE,
        cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (exfoliation_close_kernel, exfoliation_close_kernel)),
        iterations=1,
    )
    candidates.extend(
        _component_candidates(
            exfoliation_mask,
            "exfoliation_branch",
            min_area=exfoliation_min_area,
            max_area_ratio=exfoliation_max_area_ratio,
        )
    )

    candidates = _dedupe_candidates(candidates)
    return {
        "preprocessed": pre,
        "grid_mask": grid,
        "non_grid_mask": non_grid,
        "blackhat": blackhat,
        "crack_threshold": crack_thresh,
        "crack_edges": edges,
        "crack_mask": crack_mask,
        "spalling_local_diff": local_diff,
        "spalling_spot_mask": spot_mask,
        "spalling_dark_mask": dark,
        "spalling_mask": spalling_mask,
        "exfoliation_color_mask": exf_mask,
        "exfoliation_variance_mask": var_mask,
        "exfoliation_mask": exfoliation_mask,
        "candidates": candidates,
    }


def generate_candidates(image_bgr: np.ndarray, params: dict | None = None) -> list[Candidate]:
    return generate_candidate_debug(image_bgr, params)["candidates"]
