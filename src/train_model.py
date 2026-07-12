from __future__ import annotations

import argparse
import json
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

import cv2
import joblib
import numpy as np

from src.candidates import Candidate
from src.candidates import generate_candidates
from src.coco import count_annotations, load_coco_records
from src.config import (
    BACKGROUND_IOU_THRESHOLD,
    DEFAULT_DATASET_ROOT,
    DEFAULT_MODEL_PATH,
    DEFAULT_REPORT_PATH,
    DEFAULT_SCHEMA_PATH,
    POSITIVE_IOU_THRESHOLD,
)
from src.features import extract_features, features_to_matrix
from src.grid import detect_grid_mask
from src.matching import box_iou_xywh
from src.model import build_classifier, save_model
from src.preprocess import preprocess_image


def build_training_examples(dataset_root: Path, limit: int | None = None, max_background_per_image: int = 8):
    records = load_coco_records(dataset_root, "train")
    if limit:
        records = records[:limit]

    feature_rows: list[dict[str, float]] = []
    labels: list[str] = []
    rng = np.random.default_rng(42)

    for index, record in enumerate(records, start=1):
        image = cv2.imread(str(record.image_path))
        if image is None:
            continue
        pre = preprocess_image(image)
        grid_mask = detect_grid_mask(pre)
        candidates = generate_candidates(image)
        for box in record.boxes:
            x, y, w, h = [int(round(value)) for value in box]
            x = max(0, min(x, image.shape[1] - 1))
            y = max(0, min(y, image.shape[0] - 1))
            w = max(1, min(w, image.shape[1] - x))
            h = max(1, min(h, image.shape[0] - y))
            mask = np.zeros(image.shape[:2], dtype=np.uint8)
            mask[y : y + h, x : x + w] = 255
            candidates.append(Candidate((x, y, w, h), mask, "gt_seed"))
        background_added = 0
        rng.shuffle(candidates)

        for candidate in candidates:
            ious = box_iou_xywh(candidate.box_xywh, record.boxes)
            if ious.size and float(ious.max()) >= POSITIVE_IOU_THRESHOLD:
                label = record.labels[int(ious.argmax())]
            elif ious.size == 0 or float(ious.max()) < BACKGROUND_IOU_THRESHOLD:
                if background_added >= max_background_per_image:
                    continue
                label = "background"
                background_added += 1
            else:
                continue
            feature_rows.append(extract_features(image, candidate, grid_mask))
            labels.append(label)

        if index % 100 == 0:
            print(f"Processed {index}/{len(records)} training images; examples={len(labels)}")

    return feature_rows, labels


def oversample_minority(X: np.ndarray, y: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    counts = Counter(y.tolist())
    if not counts:
        return X, y
    target = min(max(counts.values()), 1200)
    rows = [X]
    labels = [y]
    rng = np.random.default_rng(42)
    for label, count in counts.items():
        if label == "background" or count == 0 or count >= target:
            continue
        indexes = np.where(y == label)[0]
        sampled = rng.choice(indexes, size=target - count, replace=True)
        rows.append(X[sampled])
        labels.append(y[sampled])
    return np.vstack(rows), np.concatenate(labels)


def train(
    dataset_root: Path = DEFAULT_DATASET_ROOT,
    model_path: Path = DEFAULT_MODEL_PATH,
    schema_path: Path = DEFAULT_SCHEMA_PATH,
    report_path: Path = DEFAULT_REPORT_PATH,
    model_kind: str = "extra_trees",
    limit: int | None = None,
) -> dict:
    feature_rows, labels = build_training_examples(dataset_root, limit=limit)
    if not labels:
        raise RuntimeError("No training examples were generated. Check dataset path and candidate thresholds.")

    X, schema = features_to_matrix(feature_rows)
    y = np.array(labels)
    X_train, y_train = oversample_minority(X, y)
    classifier = build_classifier(model_kind)
    classifier.fit(X_train, y_train)
    save_model(classifier, schema, model_path, schema_path)

    report = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "dataset_root": str(dataset_root),
        "model_type": type(classifier).__name__,
        "feature_count": len(schema),
        "feature_names": schema,
        "raw_class_counts": dict(Counter(labels)),
        "balanced_class_counts": dict(Counter(y_train.tolist())),
        "annotation_counts": dict(count_annotations(dataset_root, "train")),
    }
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    joblib.dump(schema, schema_path)
    return report


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", default=str(DEFAULT_DATASET_ROOT))
    parser.add_argument("--model", default=str(DEFAULT_MODEL_PATH))
    parser.add_argument("--schema", default=str(DEFAULT_SCHEMA_PATH))
    parser.add_argument("--report", default=str(DEFAULT_REPORT_PATH))
    parser.add_argument("--model-kind", choices=("extra_trees", "random_forest"), default="extra_trees")
    parser.add_argument("--limit", type=int, default=None, help="Optional image limit for fast smoke training.")
    args = parser.parse_args()

    report = train(
        dataset_root=Path(args.data),
        model_path=Path(args.model),
        schema_path=Path(args.schema),
        report_path=Path(args.report),
        model_kind=args.model_kind,
        limit=args.limit,
    )
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
