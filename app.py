from __future__ import annotations

import json
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import streamlit as st
from PIL import Image

from src.candidates import generate_candidate_debug, generate_candidates
from src.coco import count_annotations, list_splits, load_coco_records
from src.config import DEFAULT_DATASET_ROOT, DEFAULT_MODEL_PATH, DEFAULT_SCHEMA_PATH, REPORTS_DIR
from src.evaluate import evaluate
from src.grid import detect_grid_mask
from src.predict import predict_image
from src.preprocess import preprocess_image
from src.train_model import train
from src.visualize import bgr_to_rgb, draw_candidates, draw_ground_truth, draw_predictions, mask_to_bgr


st.set_page_config(page_title="Ceramic Tile Defect Detection", layout="wide")


def _uploaded_to_bgr(uploaded_file) -> np.ndarray:
    image = Image.open(uploaded_file).convert("RGB")
    rgb = np.array(image)
    return cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)


def _show_bgr(label: str, image_bgr: np.ndarray) -> None:
    st.image(bgr_to_rgb(image_bgr), caption=label, width="stretch")


def _processing_parameter_controls(prefix: str = "") -> dict:
    params: dict[str, int | float] = {}
    with st.expander("Image processing parameters", expanded=False):
        st.markdown("Preprocessing")
        c1, c2, c3 = st.columns(3)
        with c1:
            params["valid_threshold"] = st.slider(f"{prefix}Black/background threshold", 0, 80, 20, 1)
            params["clahe_clip_limit"] = st.slider(f"{prefix}CLAHE clip limit", 0.5, 6.0, 2.0, 0.1)
        with c2:
            params["clahe_tile_size"] = st.slider(f"{prefix}CLAHE tile size", 2, 16, 8, 1)
            params["bilateral_d"] = st.slider(f"{prefix}Bilateral diameter", 1, 21, 7, 2)
        with c3:
            params["bilateral_sigma_color"] = st.slider(f"{prefix}Bilateral sigma color", 5, 120, 50, 5)
            params["valid_morph_kernel"] = st.slider(f"{prefix}Valid-mask morph kernel", 1, 31, 7, 2)

        st.markdown("Grout / grid suppression")
        c1, c2, c3 = st.columns(3)
        with c1:
            params["grid_dark_threshold"] = st.slider(f"{prefix}Grid dark threshold", 20, 180, 95, 1)
            params["grid_horizontal_scale"] = st.slider(f"{prefix}Horizontal kernel divisor", 4, 40, 18, 1)
            params["grid_canny_low"] = st.slider(f"{prefix}Grid Canny low", 0, 200, 40, 1)
        with c2:
            params["grid_vertical_scale"] = st.slider(f"{prefix}Vertical kernel divisor", 4, 40, 12, 1)
            params["grid_dilation_kernel"] = st.slider(f"{prefix}Grid dilation kernel", 1, 21, 5, 2)
            params["grid_canny_high"] = st.slider(f"{prefix}Grid Canny high", 20, 300, 120, 1)
        with c3:
            params["hough_threshold"] = st.slider(f"{prefix}Hough threshold", 5, 120, 45, 1)
            params["hough_max_gap"] = st.slider(f"{prefix}Hough max line gap", 0, 40, 12, 1)

        st.markdown("Crack candidate branch")
        c1, c2, c3 = st.columns(3)
        with c1:
            params["crack_blackhat_kernel"] = st.slider(f"{prefix}Black-hat kernel", 3, 41, 13, 2)
            params["crack_close_kernel"] = st.slider(f"{prefix}Crack close kernel", 1, 15, 3, 2)
        with c2:
            params["crack_canny_low"] = st.slider(f"{prefix}Canny low", 0, 200, 45, 1)
            params["crack_canny_high"] = st.slider(f"{prefix}Canny high", 20, 300, 135, 1)
        with c3:
            params["crack_min_area"] = st.slider(f"{prefix}Crack min area", 1, 200, 8, 1)
            params["crack_min_aspect"] = st.slider(f"{prefix}Crack min aspect", 1.0, 8.0, 1.6, 0.1)

        st.markdown("Spalling and exfoliation branches")
        c1, c2, c3 = st.columns(3)
        with c1:
            params["spalling_blur_sigma"] = st.slider(f"{prefix}Spalling blur sigma", 1.0, 25.0, 9.0, 0.5)
            params["spalling_dark_threshold"] = st.slider(f"{prefix}Spalling dark threshold", 20, 200, 110, 1)
        with c2:
            params["spalling_min_area"] = st.slider(f"{prefix}Spalling min area", 1, 500, 12, 1)
            params["exfoliation_min_area"] = st.slider(f"{prefix}Exfoliation min area", 10, 3000, 80, 10)
        with c3:
            params["spalling_open_kernel"] = st.slider(f"{prefix}Spalling open kernel", 1, 15, 3, 2)
            params["exfoliation_close_kernel"] = st.slider(f"{prefix}Exfoliation close kernel", 1, 31, 9, 2)
    return params


def page_predict() -> None:
    st.header("Predict")
    model_path = Path(st.text_input("Model path", str(DEFAULT_MODEL_PATH)))
    schema_path = Path(st.text_input("Feature schema path", str(DEFAULT_SCHEMA_PATH)))
    confidence = st.slider("Confidence threshold", 0.05, 0.95, 0.40, 0.05)
    nms_iou = st.slider("NMS IoU threshold", 0.05, 0.90, 0.30, 0.05)
    processing_params = _processing_parameter_controls("Predict ")
    uploaded = st.file_uploader("Upload tile image", type=["jpg", "jpeg", "png"])

    if not model_path.exists():
        st.warning("No trained model found. Go to Train Model first.")
        return
    if uploaded is None:
        return

    image_bgr = _uploaded_to_bgr(uploaded)
    if st.button("Run detection", type="primary"):
        predictions, debug = predict_image(
            image_bgr,
            model_path,
            schema_path,
            confidence,
            nms_iou,
            processing_params,
        )
        left, right = st.columns(2)
        with left:
            _show_bgr("Original", image_bgr)
        with right:
            _show_bgr("Prediction overlay", draw_predictions(image_bgr, predictions))

        st.subheader("Predictions")
        st.dataframe(
            pd.DataFrame(
                [
                    {
                        "label": p.label,
                        "score": round(p.score, 3),
                        "x": p.box_xywh[0],
                        "y": p.box_xywh[1],
                        "width": p.box_xywh[2],
                        "height": p.box_xywh[3],
                        "source": p.source,
                    }
                    for p in predictions
                ]
            ),
            width="stretch",
        )

        with st.expander("Debug images", expanded=False):
            d1, d2, d3 = st.columns(3)
            with d1:
                _show_bgr("Enhanced grayscale", mask_to_bgr(debug["preprocessed"]["enhanced_gray"]))
            with d2:
                _show_bgr("Valid tile mask", mask_to_bgr(debug["preprocessed"]["valid_tile_mask"]))
            with d3:
                _show_bgr("Grid mask", mask_to_bgr(debug["grid_mask"]))
            _show_bgr("Candidate overlay", draw_candidates(image_bgr, debug["candidates"]))


def page_processing_lab() -> None:
    st.header("Image Processing Lab")
    st.caption("Tune the classical CV stages before model classification. These controls affect masks and candidate boxes.")

    source = st.radio("Image source", ["Upload image", "Dataset image"], horizontal=True)
    image_bgr = None
    record = None

    if source == "Upload image":
        uploaded = st.file_uploader("Upload tile image", type=["jpg", "jpeg", "png"], key="lab_upload")
        if uploaded is not None:
            image_bgr = _uploaded_to_bgr(uploaded)
    else:
        dataset_root = Path(st.text_input("Dataset root", str(DEFAULT_DATASET_ROOT), key="lab_dataset_root"))
        if dataset_root.exists():
            split = st.selectbox("Split", list_splits(dataset_root), key="lab_split")
            records = load_coco_records(dataset_root, split)
            selected_name = st.selectbox("Image", [r.image_path.name for r in records], key="lab_image")
            record = next(r for r in records if r.image_path.name == selected_name)
            image_bgr = cv2.imread(str(record.image_path))
        else:
            st.warning("Dataset root does not exist.")

    processing_params = _processing_parameter_controls("Lab ")
    if image_bgr is None:
        return

    debug = generate_candidate_debug(image_bgr, processing_params)
    candidates = debug["candidates"]
    by_source = pd.Series([candidate.source for candidate in candidates], name="source").value_counts().reset_index()
    by_source.columns = ["source", "count"]

    st.subheader("Candidate Summary")
    c1, c2, c3 = st.columns(3)
    c1.metric("Total candidates", len(candidates))
    c2.metric("Grid mask pixels", int((debug["grid_mask"] > 0).sum()))
    c3.metric("Valid tile pixels", int((debug["preprocessed"]["valid_tile_mask"] > 0).sum()))
    st.dataframe(by_source, width="stretch")

    overview_tab, preprocess_tab, grid_tab, crack_tab, spalling_tab, exfoliation_tab = st.tabs(
        ["Overview", "Preprocess", "Grid", "Crack", "Spalling", "Exfoliation"]
    )

    with overview_tab:
        left, right = st.columns(2)
        with left:
            _show_bgr("Original", image_bgr)
        with right:
            _show_bgr("Candidate overlay", draw_candidates(image_bgr, candidates))
        if record is not None:
            _show_bgr("Ground truth boxes", draw_ground_truth(image_bgr, record.boxes, record.labels))

    with preprocess_tab:
        c1, c2, c3 = st.columns(3)
        with c1:
            _show_bgr("Grayscale", mask_to_bgr(debug["preprocessed"]["gray"]))
        with c2:
            _show_bgr("CLAHE + bilateral", mask_to_bgr(debug["preprocessed"]["enhanced_gray"]))
        with c3:
            _show_bgr("Valid tile mask", mask_to_bgr(debug["preprocessed"]["valid_tile_mask"]))

    with grid_tab:
        c1, c2, c3, c4 = st.columns(4)
        with c1:
            _show_bgr("1. Dark threshold", mask_to_bgr(debug["grid_dark"]))
            _show_bgr("5. Canny edges", mask_to_bgr(debug["grid_edges"]))
        with c2:
            _show_bgr("2. Horizontal morph", mask_to_bgr(debug["grid_horizontal"]))
            _show_bgr("6. Hough lines", mask_to_bgr(debug["grid_hough_lines"]))
        with c3:
            _show_bgr("3. Vertical morph", mask_to_bgr(debug["grid_vertical"]))
            _show_bgr("7. Final Grid Mask", mask_to_bgr(debug["grid_mask"]))
        with c4:
            _show_bgr("4. Combined morph", mask_to_bgr(debug["grid_morph_combined"]))
            _show_bgr("8. Non-grid valid", mask_to_bgr(debug["non_grid_mask"]))

    with crack_tab:
        c1, c2, c3 = st.columns(3)
        with c1:
            _show_bgr("Black-hat response", mask_to_bgr(debug["blackhat"]))
        with c2:
            _show_bgr("Canny edges", mask_to_bgr(debug["crack_edges"]))
        with c3:
            _show_bgr("Final crack mask", mask_to_bgr(debug["crack_mask"]))

    with spalling_tab:
        c1, c2, c3 = st.columns(3)
        with c1:
            _show_bgr("Local contrast", mask_to_bgr(debug["spalling_local_diff"]))
        with c2:
            _show_bgr("Dark patch mask", mask_to_bgr(debug["spalling_dark_mask"]))
        with c3:
            _show_bgr("Final spalling mask", mask_to_bgr(debug["spalling_mask"]))

    with exfoliation_tab:
        c1, c2, c3 = st.columns(3)
        with c1:
            _show_bgr("LAB color anomaly mask", mask_to_bgr(debug["exfoliation_color_mask"]))
        with c2:
            _show_bgr("Laplacian variance mask", mask_to_bgr(debug["exfoliation_variance_mask"]))
        with c3:
            _show_bgr("Final exfoliation mask", mask_to_bgr(debug["exfoliation_mask"]))

    with st.expander("Candidate boxes table", expanded=False):
        st.dataframe(
            pd.DataFrame(
                [
                    {
                        "source": candidate.source,
                        "x": candidate.box_xywh[0],
                        "y": candidate.box_xywh[1],
                        "width": candidate.box_xywh[2],
                        "height": candidate.box_xywh[3],
                    }
                    for candidate in candidates
                ]
            ),
            width="stretch",
        )


def page_train() -> None:
    st.header("Train Model")
    dataset_root = Path(st.text_input("Dataset root", str(DEFAULT_DATASET_ROOT)))
    model_kind = st.selectbox("Model type", ["extra_trees", "random_forest"], index=0)
    max_images = st.number_input("Max training images, 0 means all", min_value=0, max_value=10000, value=800, step=100)
    st.caption("Training all images is slower. Use 800-1500 images for fast iteration, then train all for final metrics.")

    if dataset_root.exists():
        counts = count_annotations(dataset_root, "train")
        st.dataframe(pd.DataFrame([{"class": k, "annotations": v} for k, v in counts.items()]), width="stretch")
    else:
        st.warning("Dataset root does not exist.")

    if st.button("Train model", type="primary"):
        with st.spinner("Training classical feature classifier..."):
            report = train(dataset_root=dataset_root, model_kind=model_kind, limit=max_images or None)
        st.success(f"Saved model to {DEFAULT_MODEL_PATH}")
        st.json(report)


def page_evaluate() -> None:
    st.header("Evaluate")
    dataset_root = Path(st.text_input("Dataset root", str(DEFAULT_DATASET_ROOT)))
    model_path = Path(st.text_input("Model path", str(DEFAULT_MODEL_PATH)))
    max_images = st.number_input("Max validation images, 0 means all", min_value=0, max_value=5000, value=100, step=50)

    if not model_path.exists():
        st.warning("No trained model found. Go to Train Model first.")
        return

    if st.button("Run evaluation", type="primary"):
        with st.spinner("Evaluating validation split..."):
            report = evaluate(dataset_root, model_path, DEFAULT_SCHEMA_PATH, limit=max_images or None)
        st.json(report)
        cm_path = Path(report["confusion_matrix_path"])
        if cm_path.exists():
            st.image(str(cm_path), caption="Confusion matrix", width="content")


def page_dataset_explorer() -> None:
    st.header("Dataset Explorer")
    dataset_root = Path(st.text_input("Dataset root", str(DEFAULT_DATASET_ROOT)))
    if not dataset_root.exists():
        st.warning("Dataset root does not exist.")
        return
    splits = list_splits(dataset_root)
    if not splits:
        st.warning("No COCO train/valid splits found.")
        return
    split = st.selectbox("Split", splits)
    records = load_coco_records(dataset_root, split)
    counts = count_annotations(dataset_root, split)
    st.dataframe(pd.DataFrame([{"class": k, "annotations": v} for k, v in counts.items()]), width="stretch")

    options = [record.image_path.name for record in records]
    selected_name = st.selectbox("Image", options)
    record = next(r for r in records if r.image_path.name == selected_name)
    image = cv2.imread(str(record.image_path))
    if image is None:
        st.error("Could not load selected image.")
        return
    left, right = st.columns(2)
    with left:
        _show_bgr("Original", image)
    with right:
        _show_bgr("Ground truth", draw_ground_truth(image, record.boxes, record.labels))
    st.dataframe(
        pd.DataFrame(
            [
                {"label": label, "x": box[0], "y": box[1], "width": box[2], "height": box[3]}
                for box, label in zip(record.boxes, record.labels)
            ]
        ),
        width="stretch",
    )


def page_about() -> None:
    st.header("About Pipeline")
    st.markdown(
        """
This app uses classical computer vision and classical machine learning only.

The detector first enhances the tile image, masks the black background, estimates grout/grid lines,
generates defect candidates using morphology, thresholding, edges, color anomalies, and connected
components, then classifies each candidate with handcrafted features.

The model is not a neural network. The default classifier is `ExtraTreesClassifier`.

The dataset contains bounding boxes, not segmentation masks, so final boundaries are approximate
candidate contours while bounding boxes are the main detection output.

Use the Image Processing Lab page to inspect and tune the pipeline stages:

1. Preprocessing creates grayscale, contrast-enhanced grayscale, and valid-tile mask.
2. Grid suppression detects long dark grout lines and removes them from candidate search.
3. Crack branch uses black-hat morphology, Canny edges, closing, and connected components.
4. Spalling branch uses local contrast, dark-patch thresholding, opening, and components.
5. Exfoliation branch uses LAB color anomaly, Laplacian texture response, closing, and components.
        """.strip()
    )
    report_path = Path("models/training_report.json")
    if report_path.exists():
        st.subheader("Latest Training Report")
        st.json(json.loads(report_path.read_text(encoding="utf-8")))


def main() -> None:
    page = st.sidebar.radio(
        "Page",
        ["Predict", "Image Processing Lab", "Train Model", "Evaluate", "Dataset Explorer", "About Pipeline"],
    )
    if page == "Predict":
        page_predict()
    elif page == "Image Processing Lab":
        page_processing_lab()
    elif page == "Train Model":
        page_train()
    elif page == "Evaluate":
        page_evaluate()
    elif page == "Dataset Explorer":
        page_dataset_explorer()
    else:
        page_about()


if __name__ == "__main__":
    main()
