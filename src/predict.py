from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np

from src.candidates import generate_candidates
from src.config import DEFAULT_MODEL_PATH, DEFAULT_SCHEMA_PATH, MIN_CONFIDENCE, NMS_IOU_THRESHOLD
from src.features import extract_features, features_to_matrix
from src.grid import detect_grid_mask
from src.matching import nms_xywh
from src.model import load_model
from src.preprocess import preprocess_image
from src.visualize import draw_predictions


@dataclass(frozen=True)
class Prediction:
    box_xywh: tuple[int, int, int, int]
    label: str
    score: float
    source: str


def predict_image(
    image_bgr: np.ndarray,
    model_path: Path = DEFAULT_MODEL_PATH,
    schema_path: Path = DEFAULT_SCHEMA_PATH,
    min_confidence: float = MIN_CONFIDENCE,
    nms_iou_threshold: float = NMS_IOU_THRESHOLD,
    processing_params: dict | None = None,
) -> tuple[list[Prediction], dict]:
    model, schema = load_model(model_path, schema_path)
    pre = preprocess_image(image_bgr, processing_params)
    grid_mask = detect_grid_mask(pre, processing_params)
    candidates = generate_candidates(image_bgr, processing_params)
    rows = [extract_features(image_bgr, candidate, grid_mask) for candidate in candidates]
    matrix, schema = features_to_matrix(rows, schema)

    predictions: list[Prediction] = []
    if matrix.size:
        probabilities = model.predict_proba(matrix)
        classes = list(model.classes_)
        for candidate, probs in zip(candidates, probabilities):
            best_idx = int(np.argmax(probs))
            label = classes[best_idx]
            score = float(probs[best_idx])
            if label == "background" or score < min_confidence:
                continue
            predictions.append(Prediction(candidate.box_xywh, label, score, candidate.source))

    filtered: list[Prediction] = []
    for label in sorted({prediction.label for prediction in predictions}):
        group = [prediction for prediction in predictions if prediction.label == label]
        keep = nms_xywh([p.box_xywh for p in group], [p.score for p in group], nms_iou_threshold)
        filtered.extend(group[idx] for idx in keep)

    debug = {
        "preprocessed": pre,
        "grid_mask": grid_mask,
        "candidates": candidates,
    }
    return sorted(filtered, key=lambda p: p.score, reverse=True), debug


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--image", required=True)
    parser.add_argument("--model", default=str(DEFAULT_MODEL_PATH))
    parser.add_argument("--schema", default=str(DEFAULT_SCHEMA_PATH))
    parser.add_argument("--output", default="outputs/predictions/prediction.jpg")
    parser.add_argument("--confidence", type=float, default=MIN_CONFIDENCE)
    args = parser.parse_args()

    image = cv2.imread(args.image)
    if image is None:
        raise SystemExit(f"Could not read image: {args.image}")
    predictions, _ = predict_image(image, Path(args.model), Path(args.schema), args.confidence)
    output = draw_predictions(image, predictions)
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(output_path), output)
    print(f"Saved prediction overlay to {output_path}")


if __name__ == "__main__":
    main()
