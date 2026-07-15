from __future__ import annotations

from io import BytesIO
from dataclasses import dataclass
from pathlib import Path
from typing import BinaryIO

import cv2
import joblib
import numpy as np
import streamlit as st
from PIL import Image
from skimage.filters import frangi
from skimage.morphology import skeletonize

from src.predict import draw_predictions, predict_image


TARGET_IMAGE = Path("IMG_1260_000045_jpg.rf.e832a11a8a69ad957cf26409c80008c9.jpg")
CLEAN_IMAGE = Path("IMG_1260_000045_jpg.rf.e832a11a8a69ad957cf26409c80008c9_1.jpg")


@dataclass(frozen=True)
class Params:
    ignore_annotation_overlay: bool
    annotation_min_saturation: int
    annotation_dilation: int
    background_blur_kernel: int
    clahe_clip_limit: float
    clahe_tile_grid_size: int
    grout_dark_threshold: int
    vertical_kernel_height: int
    horizontal_kernel_width: int
    grout_dilation: int
    blackhat_kernel_size: int
    enhancement_blur: int
    use_frangi: bool
    crack_response_threshold: int
    min_area: int
    max_area: int
    min_aspect_ratio: float
    max_extent: float
    max_grout_overlap_ratio: float
    closing_kernel_size: int
    closing_iterations: int
    use_skeleton: bool
    show_grout_mask: bool
    show_rejected_candidates: bool
    show_component_boxes: bool


@dataclass(frozen=True)
class ExfoliationParams:
    texture_window_size: int
    edge_window_size: int
    dark_gray_threshold: int
    rough_std_threshold: int
    saturated_dark_threshold: int
    edge_density_threshold: float
    open_kernel_size: int
    close_kernel_size: int
    close_iterations: int
    min_area: int
    max_area: int
    min_width: int
    min_height: int
    min_extent: float
    max_aspect_ratio: float
    max_mean_gray: int
    min_mean_saturation: int
    max_grout_overlap_ratio: float
    show_rejected: bool
    show_boxes: bool


def odd(value: int) -> int:
    value = max(1, int(value))
    return value if value % 2 == 1 else value + 1


def load_image(source: str | BinaryIO) -> np.ndarray:
    if isinstance(source, str):
        bgr = cv2.imread(source, cv2.IMREAD_COLOR)
        if bgr is None:
            raise FileNotFoundError(f"Could not read image: {source}")
        return cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)

    image = Image.open(source).convert("RGB")
    return np.array(image)


def create_annotation_mask(rgb: np.ndarray, params: Params) -> np.ndarray:
    if not params.ignore_annotation_overlay:
        return np.zeros(rgb.shape[:2], dtype=np.uint8)

    hsv = cv2.cvtColor(rgb, cv2.COLOR_RGB2HSV)
    hue = hsv[:, :, 0]
    saturation = hsv[:, :, 1]
    value = hsv[:, :, 2]

    pink_hue = ((hue >= 155) & (hue <= 179)) | ((hue >= 0) & (hue <= 8))
    mask = pink_hue & (saturation >= params.annotation_min_saturation) & (value >= 80)
    mask = mask.astype(np.uint8) * 255

    dilation = max(0, params.annotation_dilation)
    if dilation:
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (dilation, dilation))
        mask = cv2.dilate(mask, kernel, iterations=1)

    return mask


def correct_lighting(gray: np.ndarray, params: Params) -> np.ndarray:
    blur_kernel = odd(params.background_blur_kernel)
    background = cv2.GaussianBlur(gray, (blur_kernel, blur_kernel), 0)
    corrected = cv2.addWeighted(gray, 1.45, background, -0.45, 80)
    corrected = cv2.normalize(corrected, None, 0, 255, cv2.NORM_MINMAX)

    tile_grid = max(2, int(params.clahe_tile_grid_size))
    clahe = cv2.createCLAHE(
        clipLimit=float(params.clahe_clip_limit),
        tileGridSize=(tile_grid, tile_grid),
    )
    return clahe.apply(corrected.astype(np.uint8))


def detect_grout_mask(gray: np.ndarray, params: Params) -> np.ndarray:
    dark = cv2.inRange(gray, 0, params.grout_dark_threshold)
    dark = cv2.morphologyEx(dark, cv2.MORPH_CLOSE, np.ones((5, 5), np.uint8), iterations=1)

    lines = cv2.HoughLinesP(
        dark,
        rho=1,
        theta=np.pi / 180,
        threshold=80,
        minLineLength=70,
        maxLineGap=20,
    )
    line_mask = np.zeros_like(gray)
    if lines is not None:
        for x1, y1, x2, y2 in lines.reshape(-1, 4):
            angle = abs(np.degrees(np.arctan2(int(y2) - int(y1), int(x2) - int(x1))))
            angle = min(angle, 180 - angle)
            if angle <= 20 or angle >= 70:
                cv2.line(
                    line_mask,
                    (int(x1), int(y1)),
                    (int(x2), int(y2)),
                    255,
                    max(3, params.grout_dilation * 2 - 1),
                )

    if np.count_nonzero(line_mask) > 0:
        grout = cv2.bitwise_and(line_mask, dark)
        grout = cv2.morphologyEx(grout, cv2.MORPH_CLOSE, np.ones((7, 7), np.uint8), iterations=1)
        return cv2.dilate(grout, np.ones((3, 3), np.uint8), iterations=1)

    vertical_kernel = cv2.getStructuringElement(
        cv2.MORPH_RECT,
        (3, max(3, params.vertical_kernel_height)),
    )
    horizontal_kernel = cv2.getStructuringElement(
        cv2.MORPH_RECT,
        (max(3, params.horizontal_kernel_width), 3),
    )

    vertical = cv2.morphologyEx(dark, cv2.MORPH_OPEN, vertical_kernel)
    horizontal = cv2.morphologyEx(dark, cv2.MORPH_OPEN, horizontal_kernel)
    grout = cv2.bitwise_or(vertical, horizontal)
    grout = cv2.morphologyEx(grout, cv2.MORPH_CLOSE, np.ones((5, 5), np.uint8))

    dilation = max(0, params.grout_dilation)
    if dilation:
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (dilation, dilation))
        grout = cv2.dilate(grout, kernel, iterations=1)

    return grout


def enhance_cracks(corrected: np.ndarray, ignore_mask: np.ndarray, params: Params) -> np.ndarray:
    kernel_size = odd(params.blackhat_kernel_size)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (kernel_size, kernel_size))
    enhanced = cv2.morphologyEx(corrected, cv2.MORPH_BLACKHAT, kernel)

    if params.use_frangi:
        normalized = corrected.astype(np.float32) / 255.0
        vesselness = frangi(1.0 - normalized, sigmas=range(1, 4), black_ridges=False)
        vesselness = np.nan_to_num(vesselness)
        vesselness = cv2.normalize(vesselness, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
        enhanced = cv2.addWeighted(enhanced, 0.65, vesselness, 0.35, 0)

    blur = max(0, params.enhancement_blur)
    if blur:
        enhanced = cv2.GaussianBlur(enhanced, (odd(blur * 2 + 1), odd(blur * 2 + 1)), 0)

    enhanced = enhanced.copy()
    enhanced[ignore_mask > 0] = 0
    return enhanced


def threshold_cracks(enhanced: np.ndarray, ignore_mask: np.ndarray, params: Params) -> np.ndarray:
    _, mask = cv2.threshold(
        enhanced,
        params.crack_response_threshold,
        255,
        cv2.THRESH_BINARY,
    )
    mask[ignore_mask > 0] = 0
    return mask


def remove_small_components(mask: np.ndarray, min_area: int) -> np.ndarray:
    count, labels, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
    cleaned = np.zeros_like(mask)

    for label in range(1, count):
        area = stats[label, cv2.CC_STAT_AREA]
        if area >= min_area:
            cleaned[labels == label] = 255

    return cleaned


def clean_crack_mask(mask: np.ndarray, params: Params) -> np.ndarray:
    cleaned = remove_small_components(mask, params.min_area)

    if params.closing_iterations > 0 and params.closing_kernel_size > 1:
        kernel_size = odd(params.closing_kernel_size)
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_size, kernel_size))
        cleaned = cv2.morphologyEx(
            cleaned,
            cv2.MORPH_CLOSE,
            kernel,
            iterations=params.closing_iterations,
        )
        cleaned = remove_small_components(cleaned, params.min_area)

    if params.use_skeleton:
        cleaned = skeletonize(cleaned > 0).astype(np.uint8) * 255

    return cleaned


def filter_candidates(
    mask: np.ndarray,
    grout_mask: np.ndarray,
    annotation_mask: np.ndarray,
    params: Params,
) -> tuple[np.ndarray, np.ndarray, list[dict]]:
    count, labels, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
    accepted = np.zeros_like(mask)
    rejected = np.zeros_like(mask)
    components: list[dict] = []

    for label in range(1, count):
        x = int(stats[label, cv2.CC_STAT_LEFT])
        y = int(stats[label, cv2.CC_STAT_TOP])
        width = int(stats[label, cv2.CC_STAT_WIDTH])
        height = int(stats[label, cv2.CC_STAT_HEIGHT])
        area = int(stats[label, cv2.CC_STAT_AREA])

        component_mask = labels == label
        bbox_area = max(1, width * height)
        aspect_ratio = max(width, height) / max(1, min(width, height))
        extent = area / bbox_area
        grout_overlap = int(np.count_nonzero(component_mask & (grout_mask > 0))) / max(1, area)
        annotation_overlap = int(np.count_nonzero(component_mask & (annotation_mask > 0))) / max(1, area)

        is_long = aspect_ratio >= params.min_aspect_ratio
        is_curved = area >= params.min_area * 2 and extent <= params.max_extent * 0.40
        keep = (
            params.min_area <= area <= params.max_area
            and (is_long or is_curved)
            and extent <= params.max_extent
            and grout_overlap <= params.max_grout_overlap_ratio
            and annotation_overlap == 0
        )

        target = accepted if keep else rejected
        target[component_mask] = 255
        components.append(
            {
                "accepted": keep,
                "x": x,
                "y": y,
                "width": width,
                "height": height,
                "area": area,
                "aspect_ratio": aspect_ratio,
                "extent": extent,
                "grout_overlap": grout_overlap,
            }
        )

    return accepted, rejected, components


def make_overlay(
    rgb: np.ndarray,
    final_mask: np.ndarray,
    grout_mask: np.ndarray,
    rejected_mask: np.ndarray,
    params: Params,
    components: list[dict] | None = None,
) -> np.ndarray:
    overlay = rgb.copy()

    if params.show_grout_mask:
        overlay[grout_mask > 0] = (0.55 * overlay[grout_mask > 0] + np.array([20, 95, 230]) * 0.45).astype(np.uint8)

    if params.show_rejected_candidates:
        overlay[rejected_mask > 0] = (0.55 * overlay[rejected_mask > 0] + np.array([230, 45, 45]) * 0.45).astype(np.uint8)

    overlay[final_mask > 0] = (0.35 * overlay[final_mask > 0] + np.array([255, 230, 0]) * 0.65).astype(np.uint8)

    if params.show_component_boxes and components:
        for component in components:
            if not component["accepted"]:
                continue
            x = component["x"]
            y = component["y"]
            w = component["width"]
            h = component["height"]
            cv2.rectangle(overlay, (x, y), (x + w, y + h), (0, 255, 255), 1)

    return overlay


def make_crack_annotation(rgb: np.ndarray, components: list[dict]) -> np.ndarray:
    annotated = rgb.copy()
    for component in components:
        if not component["accepted"]:
            continue
        x = component["x"]
        y = component["y"]
        width = component["width"]
        height = component["height"]
        pad = 5
        x1 = max(0, x - pad)
        y1 = max(0, y - pad)
        x2 = min(annotated.shape[1] - 1, x + width + pad)
        y2 = min(annotated.shape[0] - 1, y + height + pad)
        cv2.rectangle(annotated, (x1, y1), (x2, y2), (255, 110, 130), 2)
        label_y = max(0, y1 - 28)
        cv2.rectangle(annotated, (x1, label_y), (min(x1 + 66, annotated.shape[1] - 1), y1), (255, 110, 130), -1)
        cv2.putText(
            annotated,
            "crack",
            (x1 + 8, max(17, y1 - 9)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (255, 255, 255),
            1,
            cv2.LINE_AA,
        )
    return annotated


def enhance_exfoliation(
    rgb: np.ndarray,
    gray: np.ndarray,
    grout_mask: np.ndarray,
    annotation_mask: np.ndarray,
    params: ExfoliationParams,
) -> tuple[np.ndarray, dict[str, np.ndarray]]:
    hsv = cv2.cvtColor(rgb, cv2.COLOR_RGB2HSV)
    saturation = hsv[:, :, 1]

    texture_kernel = odd(params.texture_window_size)
    gray_float = gray.astype(np.float32)
    local_mean = cv2.blur(gray_float, (texture_kernel, texture_kernel))
    local_mean_sq = cv2.blur(gray_float * gray_float, (texture_kernel, texture_kernel))
    local_std = np.sqrt(np.maximum(0, local_mean_sq - local_mean * local_mean))

    edges = cv2.Canny(gray, 60, 160)
    edge_kernel = odd(params.edge_window_size)
    edge_density = cv2.blur((edges > 0).astype(np.float32), (edge_kernel, edge_kernel))

    dark_rough = (gray < params.dark_gray_threshold) & (local_std > params.rough_std_threshold)
    saturated_dark = (gray < params.saturated_dark_threshold) & (saturation > 35)
    textured_dark = (
        (gray < params.dark_gray_threshold + 15)
        & (edge_density > params.edge_density_threshold)
        & (local_std > max(10, params.rough_std_threshold - 5))
    )

    response = np.zeros_like(gray, dtype=np.uint8)
    response[dark_rough | saturated_dark | textured_dark] = 255
    response[(grout_mask > 0) | (annotation_mask > 0)] = 0

    debug_maps = {
        "local_std": cv2.normalize(local_std, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8),
        "edge_density": cv2.normalize(edge_density, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8),
    }
    return response, debug_maps


def clean_exfoliation_mask(mask: np.ndarray, params: ExfoliationParams) -> np.ndarray:
    cleaned = mask.copy()
    if params.open_kernel_size > 1:
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (odd(params.open_kernel_size), odd(params.open_kernel_size)))
        cleaned = cv2.morphologyEx(cleaned, cv2.MORPH_OPEN, kernel, iterations=1)

    if params.close_kernel_size > 1 and params.close_iterations > 0:
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (odd(params.close_kernel_size), odd(params.close_kernel_size)))
        cleaned = cv2.morphologyEx(cleaned, cv2.MORPH_CLOSE, kernel, iterations=params.close_iterations)

    return cleaned


def filter_exfoliation_candidates(
    mask: np.ndarray,
    grout_mask: np.ndarray,
    rgb: np.ndarray,
    gray: np.ndarray,
    params: ExfoliationParams,
) -> tuple[np.ndarray, np.ndarray, list[dict]]:
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    accepted = np.zeros_like(mask)
    rejected = np.zeros_like(mask)
    components: list[dict] = []
    saturation = cv2.cvtColor(rgb, cv2.COLOR_RGB2HSV)[:, :, 1]

    for contour in contours:
        x, y, width, height = cv2.boundingRect(contour)
        contour_area = float(cv2.contourArea(contour))
        bbox_area = max(1, width * height)
        extent = contour_area / bbox_area
        aspect_ratio = max(width, height) / max(1, min(width, height))

        component_mask = np.zeros_like(mask)
        cv2.drawContours(component_mask, [contour], -1, 255, -1)
        component_pixels = component_mask > 0
        component_area = max(1, int(np.count_nonzero(component_pixels)))
        grout_overlap = int(np.count_nonzero(component_pixels & (grout_mask > 0))) / component_area
        mean_gray = float(np.mean(gray[component_pixels]))
        mean_saturation = float(np.mean(saturation[component_pixels]))

        keep = (
            params.min_area <= contour_area <= params.max_area
            and width >= params.min_width
            and height >= params.min_height
            and extent >= params.min_extent
            and aspect_ratio <= params.max_aspect_ratio
            and mean_gray <= params.max_mean_gray
            and mean_saturation >= params.min_mean_saturation
            and grout_overlap <= params.max_grout_overlap_ratio
        )

        target = accepted if keep else rejected
        cv2.drawContours(target, [contour], -1, 255, -1)
        components.append(
            {
                "accepted": keep,
                "x": int(x),
                "y": int(y),
                "width": int(width),
                "height": int(height),
                "area": int(round(contour_area)),
                "aspect_ratio": aspect_ratio,
                "extent": extent,
                "mean_gray": mean_gray,
                "mean_saturation": mean_saturation,
                "grout_overlap": grout_overlap,
            }
        )

    return accepted, rejected, components


def make_exfoliation_overlay(
    rgb: np.ndarray,
    final_mask: np.ndarray,
    rejected_mask: np.ndarray,
    components: list[dict],
    params: ExfoliationParams,
) -> np.ndarray:
    overlay = rgb.copy()
    if params.show_rejected:
        overlay[rejected_mask > 0] = (0.55 * overlay[rejected_mask > 0] + np.array([230, 45, 45]) * 0.45).astype(np.uint8)

    overlay[final_mask > 0] = (0.35 * overlay[final_mask > 0] + np.array([255, 125, 0]) * 0.65).astype(np.uint8)

    if params.show_boxes:
        for component in components:
            if not component["accepted"]:
                continue
            x = component["x"]
            y = component["y"]
            w = component["width"]
            h = component["height"]
            cv2.rectangle(overlay, (x, y), (x + w, y + h), (255, 150, 0), 2)

    return overlay


def make_exfoliation_annotation(rgb: np.ndarray, components: list[dict]) -> np.ndarray:
    annotated = rgb.copy()
    for component in components:
        if not component["accepted"]:
            continue
        x = component["x"]
        y = component["y"]
        width = component["width"]
        height = component["height"]
        pad = 8
        x1 = max(0, x - pad)
        y1 = max(0, y - pad)
        x2 = min(annotated.shape[1] - 1, x + width + pad)
        y2 = min(annotated.shape[0] - 1, y + height + pad)
        cv2.rectangle(annotated, (x1, y1), (x2, y2), (255, 110, 130), 2)
        label_y = max(0, y1 - 28)
        cv2.rectangle(
            annotated,
            (x1, label_y),
            (min(x1 + 116, annotated.shape[1] - 1), y1),
            (255, 110, 130),
            -1,
        )
        cv2.putText(
            annotated,
            "exfoliation",
            (x1 + 7, max(17, y1 - 9)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.48,
            (255, 255, 255),
            1,
            cv2.LINE_AA,
        )
    return annotated


def image_to_jpeg_bytes(rgb: np.ndarray) -> bytes:
    buffer = BytesIO()
    Image.fromarray(rgb).save(buffer, format="JPEG", quality=95)
    return buffer.getvalue()


def sidebar_params() -> Params:
    st.sidebar.header("Parameters")

    st.sidebar.subheader("Annotation mask")
    ignore_annotation_overlay = st.sidebar.checkbox("Ignore pink annotation overlay", value=True)
    annotation_min_saturation = st.sidebar.slider("Annotation min saturation", 0, 255, 80)
    annotation_dilation = st.sidebar.slider("Annotation dilation", 0, 15, 5)

    st.sidebar.subheader("Image correction")
    background_blur_kernel = st.sidebar.slider("Background blur kernel", 15, 101, 41, step=2)
    clahe_clip_limit = st.sidebar.slider("CLAHE clip limit", 1.0, 5.0, 2.0, step=0.1)
    clahe_tile_grid_size = st.sidebar.slider("CLAHE tile grid size", 2, 16, 8)

    st.sidebar.subheader("Grout detection")
    grout_dark_threshold = st.sidebar.slider("Grout dark threshold", 20, 120, 70)
    vertical_kernel_height = st.sidebar.slider("Vertical kernel height", 10, 80, 35)
    horizontal_kernel_width = st.sidebar.slider("Horizontal kernel width", 10, 80, 35)
    grout_dilation = st.sidebar.slider("Grout dilation", 1, 15, 5)

    st.sidebar.subheader("Crack enhancement")
    blackhat_kernel_size = st.sidebar.slider("Black-hat kernel size", 3, 31, 9, step=2)
    enhancement_blur = st.sidebar.slider("Enhancement blur", 0, 5, 1)
    use_frangi = st.sidebar.checkbox("Blend Frangi thin-line response", value=False)
    crack_response_threshold = st.sidebar.slider("Crack response threshold", 0, 120, 35)

    st.sidebar.subheader("Cleanup and shape filter")
    min_area = st.sidebar.slider("Minimum area", 1, 500, 120)
    max_area = st.sidebar.slider("Maximum area", 50, 6000, 3000)
    min_aspect_ratio = st.sidebar.slider("Minimum aspect ratio", 1.0, 10.0, 2.5, step=0.1)
    max_extent = st.sidebar.slider("Maximum extent", 0.05, 1.0, 0.80, step=0.05)
    max_grout_overlap_ratio = st.sidebar.slider("Maximum grout overlap", 0.0, 1.0, 0.10, step=0.01)
    closing_kernel_size = st.sidebar.slider("Closing kernel size", 1, 15, 7, step=2)
    closing_iterations = st.sidebar.slider("Closing iterations", 0, 3, 1)
    use_skeleton = st.sidebar.checkbox("Skeletonize cleaned mask", value=False)

    st.sidebar.subheader("Overlay")
    show_grout_mask = st.sidebar.checkbox("Show grout mask", value=True)
    show_rejected_candidates = st.sidebar.checkbox("Show rejected candidates", value=False)
    show_component_boxes = st.sidebar.checkbox("Show component boxes", value=True)

    return Params(
        ignore_annotation_overlay=ignore_annotation_overlay,
        annotation_min_saturation=annotation_min_saturation,
        annotation_dilation=annotation_dilation,
        background_blur_kernel=background_blur_kernel,
        clahe_clip_limit=clahe_clip_limit,
        clahe_tile_grid_size=clahe_tile_grid_size,
        grout_dark_threshold=grout_dark_threshold,
        vertical_kernel_height=vertical_kernel_height,
        horizontal_kernel_width=horizontal_kernel_width,
        grout_dilation=grout_dilation,
        blackhat_kernel_size=blackhat_kernel_size,
        enhancement_blur=enhancement_blur,
        use_frangi=use_frangi,
        crack_response_threshold=crack_response_threshold,
        min_area=min_area,
        max_area=max_area,
        min_aspect_ratio=min_aspect_ratio,
        max_extent=max_extent,
        max_grout_overlap_ratio=max_grout_overlap_ratio,
        closing_kernel_size=closing_kernel_size,
        closing_iterations=closing_iterations,
        use_skeleton=use_skeleton,
        show_grout_mask=show_grout_mask,
        show_rejected_candidates=show_rejected_candidates,
        show_component_boxes=show_component_boxes,
    )


def sidebar_exfoliation_params() -> ExfoliationParams:
    st.sidebar.header("Exfoliation")
    texture_window_size = st.sidebar.slider("Texture window size", 9, 51, 31, step=2)
    edge_window_size = st.sidebar.slider("Edge density window", 7, 41, 21, step=2)
    dark_gray_threshold = st.sidebar.slider("Dark rough gray threshold", 40, 180, 110)
    rough_std_threshold = st.sidebar.slider("Texture std threshold", 10, 80, 35)
    saturated_dark_threshold = st.sidebar.slider("Saturated dark threshold", 40, 180, 95)
    edge_density_threshold = st.sidebar.slider("Edge density threshold", 0.01, 0.40, 0.12, step=0.01)

    st.sidebar.subheader("Exfoliation cleanup")
    open_kernel_size = st.sidebar.slider("Exfoliation open kernel", 1, 15, 3, step=2)
    close_kernel_size = st.sidebar.slider("Exfoliation close kernel", 3, 31, 17, step=2)
    close_iterations = st.sidebar.slider("Exfoliation close iterations", 0, 4, 2)

    st.sidebar.subheader("Exfoliation shape filter")
    min_area = st.sidebar.slider("Minimum exfoliation area", 100, 10000, 700, step=100)
    max_area = st.sidebar.slider("Maximum exfoliation area", 1000, 120000, 60000, step=1000)
    min_width = st.sidebar.slider("Minimum exfoliation width", 5, 120, 25)
    min_height = st.sidebar.slider("Minimum exfoliation height", 5, 120, 25)
    min_extent = st.sidebar.slider("Minimum exfoliation extent", 0.01, 1.0, 0.12, step=0.01)
    max_aspect_ratio = st.sidebar.slider("Maximum exfoliation aspect ratio", 1.0, 10.0, 4.5, step=0.1)
    max_mean_gray = st.sidebar.slider("Maximum exfoliation mean gray", 20, 220, 130)
    min_mean_saturation = st.sidebar.slider("Minimum exfoliation mean saturation", 0, 180, 35)
    max_grout_overlap_ratio = st.sidebar.slider("Maximum exfoliation grout overlap", 0.0, 1.0, 0.15, step=0.01)

    st.sidebar.subheader("Exfoliation overlay")
    show_rejected = st.sidebar.checkbox("Show rejected exfoliation", value=False)
    show_boxes = st.sidebar.checkbox("Show exfoliation boxes", value=True)

    return ExfoliationParams(
        texture_window_size=texture_window_size,
        edge_window_size=edge_window_size,
        dark_gray_threshold=dark_gray_threshold,
        rough_std_threshold=rough_std_threshold,
        saturated_dark_threshold=saturated_dark_threshold,
        edge_density_threshold=edge_density_threshold,
        open_kernel_size=open_kernel_size,
        close_kernel_size=close_kernel_size,
        close_iterations=close_iterations,
        min_area=min_area,
        max_area=max_area,
        min_width=min_width,
        min_height=min_height,
        min_extent=min_extent,
        max_aspect_ratio=max_aspect_ratio,
        max_mean_gray=max_mean_gray,
        min_mean_saturation=min_mean_saturation,
        max_grout_overlap_ratio=max_grout_overlap_ratio,
        show_rejected=show_rejected,
        show_boxes=show_boxes,
    )


def sidebar_model_controls() -> tuple[str, Path, float]:
    st.sidebar.header("Model")
    mode = st.sidebar.selectbox(
        "Model mode",
        ["Classical pipeline only", "SVM detector", "Compare both"],
        index=0,
    )
    model_path = Path(st.sidebar.text_input("SVM model path", "models/crack_svm.joblib"))
    score_threshold = st.sidebar.slider("SVM score threshold", -2.0, 4.0, 0.0, step=0.1)
    return mode, model_path, score_threshold


def select_image_pair() -> tuple[np.ndarray, np.ndarray, str]:
    options = {
        "Clean image": CLEAN_IMAGE,
        "Annotated source truth": TARGET_IMAGE,
    }
    selected = st.selectbox("Image", list(options), index=0)
    upload = st.file_uploader("Upload another tile image", type=["jpg", "jpeg", "png"])

    truth_rgb = load_image(str(TARGET_IMAGE))
    if upload is not None:
        st.caption("Processing image: uploaded image")
        st.caption(f"Source truth reference: `{TARGET_IMAGE.name}`")
        return load_image(upload), truth_rgb, "Uploaded image used for processing"

    selected_path = options[selected]
    st.caption(f"Processing image: `{selected_path.name}`")
    st.caption(f"Source truth reference: `{TARGET_IMAGE.name}`")
    return load_image(str(selected_path)), truth_rgb, f"{selected} used for processing"


def show_metric_row(components: list[dict], final_mask: np.ndarray, grout_mask: np.ndarray) -> None:
    accepted = sum(1 for component in components if component["accepted"])
    rejected = len(components) - accepted
    crack_pixels = int(np.count_nonzero(final_mask))
    grout_pixels = int(np.count_nonzero(grout_mask))
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Accepted", accepted)
    col2.metric("Rejected", rejected)
    col3.metric("Crack pixels", crack_pixels)
    col4.metric("Grout pixels", grout_pixels)


def show_exfoliation_metric_row(components: list[dict], final_mask: np.ndarray) -> None:
    accepted = sum(1 for component in components if component["accepted"])
    rejected = len(components) - accepted
    exfoliation_pixels = int(np.count_nonzero(final_mask))
    col1, col2, col3 = st.columns(3)
    col1.metric("Exfoliation accepted", accepted)
    col2.metric("Exfoliation rejected", rejected)
    col3.metric("Exfoliation pixels", exfoliation_pixels)


def run_pipeline(rgb: np.ndarray, params: Params) -> dict[str, np.ndarray | list[dict]]:
    gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
    annotation_mask = create_annotation_mask(rgb, params)
    corrected = correct_lighting(gray, params)
    grout_mask = detect_grout_mask(gray, params)
    ignore_mask = cv2.bitwise_or(grout_mask, annotation_mask)
    enhanced = enhance_cracks(corrected, ignore_mask, params)
    raw_mask = threshold_cracks(enhanced, ignore_mask, params)
    cleaned_mask = clean_crack_mask(raw_mask, params)
    final_mask, rejected_mask, components = filter_candidates(
        cleaned_mask,
        grout_mask,
        annotation_mask,
        params,
    )
    overlay = make_overlay(rgb, final_mask, grout_mask, rejected_mask, params, components)
    annotation = make_crack_annotation(rgb, components)

    return {
        "gray": gray,
        "annotation_mask": annotation_mask,
        "corrected": corrected,
        "grout_mask": grout_mask,
        "ignore_mask": ignore_mask,
        "enhanced": enhanced,
        "raw_mask": raw_mask,
        "cleaned_mask": cleaned_mask,
        "final_mask": final_mask,
        "rejected_mask": rejected_mask,
        "components": components,
        "overlay": overlay,
        "annotation": annotation,
    }


def run_exfoliation_pipeline(
    rgb: np.ndarray,
    gray: np.ndarray,
    grout_mask: np.ndarray,
    annotation_mask: np.ndarray,
    params: ExfoliationParams,
) -> dict[str, np.ndarray | list[dict]]:
    raw_mask, debug_maps = enhance_exfoliation(rgb, gray, grout_mask, annotation_mask, params)
    cleaned_mask = clean_exfoliation_mask(raw_mask, params)
    final_mask, rejected_mask, components = filter_exfoliation_candidates(
        cleaned_mask,
        grout_mask,
        rgb,
        gray,
        params,
    )
    overlay = make_exfoliation_overlay(rgb, final_mask, rejected_mask, components, params)
    annotation = make_exfoliation_annotation(rgb, components)

    return {
        "raw_mask": raw_mask,
        "local_std": debug_maps["local_std"],
        "edge_density": debug_maps["edge_density"],
        "cleaned_mask": cleaned_mask,
        "final_mask": final_mask,
        "rejected_mask": rejected_mask,
        "components": components,
        "overlay": overlay,
        "annotation": annotation,
    }


def main() -> None:
    st.set_page_config(page_title="Ceramic Tile Defect Detection", layout="wide")
    st.title("Ceramic Tile Defect Detection")

    params = sidebar_params()
    exfoliation_params = sidebar_exfoliation_params()
    model_mode, model_path, svm_score_threshold = sidebar_model_controls()
    rgb, truth_rgb, processing_caption = select_image_pair()
    results = run_pipeline(rgb, params)
    exfoliation_results = run_exfoliation_pipeline(
        rgb,
        results["gray"],
        results["grout_mask"],
        results["annotation_mask"],
        exfoliation_params,
    )
    components = results["components"]
    exfoliation_components = exfoliation_results["components"]
    svm_predictions = None
    svm_overlay = None
    svm_error = None

    if model_mode != "Classical pipeline only":
        if model_path.exists():
            try:
                model_bundle = joblib.load(model_path)
                svm_predictions = predict_image(
                    model_bundle,
                    rgb,
                    score_threshold=svm_score_threshold,
                    nms_threshold=0.3,
                )
                svm_overlay = cv2.cvtColor(draw_predictions(rgb, svm_predictions), cv2.COLOR_BGR2RGB)
            except Exception as exc:  # pragma: no cover - shown in Streamlit UI
                svm_error = str(exc)
        else:
            svm_error = f"Model not found: {model_path}"

    st.subheader("Crack detection")
    show_metric_row(components, results["final_mask"], results["grout_mask"])
    st.subheader("Exfoliation detection")
    show_exfoliation_metric_row(exfoliation_components, exfoliation_results["final_mask"])

    tabs = st.tabs(
        [
            "Source truth",
            "Original",
            "Corrected grayscale",
            "Grout mask",
            "Crack enhancement",
            "Raw crack mask",
            "Cleaned crack mask",
            "Final crack overlay",
            "Crack annotation",
            "Exfoliation response",
            "Raw exfoliation mask",
            "Cleaned exfoliation mask",
            "Final exfoliation overlay",
            "Exfoliation annotation",
            "SVM detector",
        ]
    )

    with tabs[0]:
        left, right = st.columns(2)
        left.image(truth_rgb, caption="Source truth reference", width="stretch")
        right.image(rgb, caption=processing_caption, width="stretch")

    with tabs[1]:
        left, right = st.columns(2)
        left.image(rgb, caption="Original", width="stretch")
        right.image(results["annotation_mask"], caption="Annotation exclusion mask", width="stretch")

    with tabs[2]:
        left, right = st.columns(2)
        left.image(results["gray"], caption="Grayscale", clamp=True, width="stretch")
        right.image(results["corrected"], caption="Corrected grayscale", clamp=True, width="stretch")

    with tabs[3]:
        left, right = st.columns(2)
        left.image(results["grout_mask"], caption="Detected grout mask", clamp=True, width="stretch")
        right.image(results["ignore_mask"], caption="Combined ignore mask", clamp=True, width="stretch")

    with tabs[4]:
        st.image(results["enhanced"], caption="Thin dark line enhancement", clamp=True, width="stretch")

    with tabs[5]:
        st.image(results["raw_mask"], caption="Raw crack candidates", clamp=True, width="stretch")

    with tabs[6]:
        left, right = st.columns(2)
        left.image(results["cleaned_mask"], caption="Cleaned candidates", clamp=True, width="stretch")
        right.image(results["rejected_mask"], caption="Rejected candidates", clamp=True, width="stretch")

    with tabs[7]:
        st.image(results["overlay"], caption="Final overlay", width="stretch")
        st.dataframe(
            components,
            width="stretch",
            hide_index=True,
            column_config={
                "accepted": st.column_config.CheckboxColumn("Accepted"),
                "aspect_ratio": st.column_config.NumberColumn("Aspect ratio", format="%.2f"),
                "extent": st.column_config.NumberColumn("Extent", format="%.2f"),
                "grout_overlap": st.column_config.NumberColumn("Grout overlap", format="%.2f"),
            },
        )

    with tabs[8]:
        st.image(results["annotation"], caption="Crack-only annotation output", width="stretch")
        st.download_button(
            "Download crack-only JPG",
            data=image_to_jpeg_bytes(results["annotation"]),
            file_name=TARGET_IMAGE.name,
            mime="image/jpeg",
        )

    with tabs[9]:
        left, right = st.columns(2)
        left.image(exfoliation_results["local_std"], caption="Local texture standard deviation", clamp=True, width="stretch")
        right.image(exfoliation_results["edge_density"], caption="Local edge density", clamp=True, width="stretch")

    with tabs[10]:
        st.image(exfoliation_results["raw_mask"], caption="Raw exfoliation candidates", clamp=True, width="stretch")

    with tabs[11]:
        left, right = st.columns(2)
        left.image(exfoliation_results["cleaned_mask"], caption="Cleaned exfoliation candidates", clamp=True, width="stretch")
        right.image(exfoliation_results["rejected_mask"], caption="Rejected exfoliation candidates", clamp=True, width="stretch")

    with tabs[12]:
        st.image(exfoliation_results["overlay"], caption="Final exfoliation overlay", width="stretch")
        st.dataframe(
            exfoliation_components,
            width="stretch",
            hide_index=True,
            column_config={
                "accepted": st.column_config.CheckboxColumn("Accepted"),
                "aspect_ratio": st.column_config.NumberColumn("Aspect ratio", format="%.2f"),
                "extent": st.column_config.NumberColumn("Extent", format="%.2f"),
                "grout_overlap": st.column_config.NumberColumn("Grout overlap", format="%.2f"),
            },
        )

    with tabs[13]:
        st.image(exfoliation_results["annotation"], caption="Exfoliation-only annotation output", width="stretch")
        st.download_button(
            "Download exfoliation-only JPG",
            data=image_to_jpeg_bytes(exfoliation_results["annotation"]),
            file_name=f"exfoliation_{TARGET_IMAGE.name}",
            mime="image/jpeg",
        )

    with tabs[14]:
        if model_mode == "Classical pipeline only":
            st.info("Select SVM detector or Compare both in the sidebar to run the trained model.")
        elif svm_error:
            st.error(svm_error)
        else:
            st.image(svm_overlay, caption="SVM crack predictions", width="stretch")
            st.json(svm_predictions)


if __name__ == "__main__":
    main()
