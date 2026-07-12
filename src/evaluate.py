from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path

import cv2
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import confusion_matrix, precision_recall_fscore_support

from src.coco import load_coco_records
from src.config import DEFAULT_DATASET_ROOT, DEFAULT_MODEL_PATH, DEFAULT_SCHEMA_PATH, REPORTS_DIR, VALID_LABELS
from src.matching import box_iou_xywh
from src.predict import predict_image
from src.visualize import draw_predictions


def _match_predictions(predictions, gt_boxes, gt_labels, iou_threshold: float):
    used_gt: set[int] = set()
    y_true: list[str] = []
    y_pred: list[str] = []
    true_positives = 0
    false_positives = 0

    for prediction in predictions:
        ious = box_iou_xywh(prediction.box_xywh, gt_boxes)
        if ious.size == 0 or float(ious.max()) < iou_threshold:
            y_true.append("background")
            y_pred.append(prediction.label)
            false_positives += 1
            continue
        gt_idx = int(ious.argmax())
        if gt_idx in used_gt:
            y_true.append("background")
            y_pred.append(prediction.label)
            false_positives += 1
            continue
        used_gt.add(gt_idx)
        y_true.append(gt_labels[gt_idx])
        y_pred.append(prediction.label)
        if prediction.label == gt_labels[gt_idx]:
            true_positives += 1
        else:
            false_positives += 1

    for idx, label in enumerate(gt_labels):
        if idx not in used_gt:
            y_true.append(label)
            y_pred.append("background")

    false_negatives = len(gt_labels) - len(used_gt)
    return y_true, y_pred, true_positives, false_positives, false_negatives


def evaluate(
    dataset_root: Path = DEFAULT_DATASET_ROOT,
    model_path: Path = DEFAULT_MODEL_PATH,
    schema_path: Path = DEFAULT_SCHEMA_PATH,
    limit: int | None = None,
) -> dict:
    records = load_coco_records(dataset_root, "valid")
    if limit:
        records = records[:limit]

    all_true_03: list[str] = []
    all_pred_03: list[str] = []
    all_true_05: list[str] = []
    all_pred_05: list[str] = []
    counts = defaultdict(Counter)
    sample_dir = REPORTS_DIR / "sample_predictions"
    sample_dir.mkdir(parents=True, exist_ok=True)

    for index, record in enumerate(records, start=1):
        image = cv2.imread(str(record.image_path))
        if image is None:
            continue
        predictions, _ = predict_image(image, model_path, schema_path)
        t03, p03, tp03, fp03, fn03 = _match_predictions(predictions, record.boxes, record.labels, 0.3)
        t05, p05, tp05, fp05, fn05 = _match_predictions(predictions, record.boxes, record.labels, 0.5)
        all_true_03.extend(t03)
        all_pred_03.extend(p03)
        all_true_05.extend(t05)
        all_pred_05.extend(p05)
        counts["iou_03"].update({"tp": tp03, "fp": fp03, "fn": fn03})
        counts["iou_05"].update({"tp": tp05, "fp": fp05, "fn": fn05})

        if index <= 12:
            overlay = draw_predictions(image, predictions)
            cv2.imwrite(str(sample_dir / record.image_path.name), overlay)
        if index % 50 == 0:
            print(f"Evaluated {index}/{len(records)} validation images")

    labels = list(VALID_LABELS) + ["background"]
    precision, recall, f1, support = precision_recall_fscore_support(
        all_true_03,
        all_pred_03,
        labels=labels,
        zero_division=0,
    )
    per_class = {
        label: {
            "precision": float(precision[idx]),
            "recall": float(recall[idx]),
            "f1": float(f1[idx]),
            "support": int(support[idx]),
        }
        for idx, label in enumerate(labels)
    }

    cm = confusion_matrix(all_true_03, all_pred_03, labels=labels)
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(7, 6))
    ax.imshow(cm, cmap="Blues")
    ax.set_xticks(range(len(labels)), labels=labels, rotation=45, ha="right")
    ax.set_yticks(range(len(labels)), labels=labels)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    for y in range(cm.shape[0]):
        for x in range(cm.shape[1]):
            ax.text(x, y, str(cm[y, x]), ha="center", va="center", color="black")
    fig.tight_layout()
    cm_path = REPORTS_DIR / "confusion_matrix.png"
    fig.savefig(cm_path)
    plt.close(fig)

    def score(counter):
        tp = counter["tp"]
        fp = counter["fp"]
        fn = counter["fn"]
        return {
            "tp": int(tp),
            "fp": int(fp),
            "fn": int(fn),
            "precision": float(tp / max(tp + fp, 1)),
            "recall": float(tp / max(tp + fn, 1)),
            "f1": float(2 * tp / max(2 * tp + fp + fn, 1)),
        }

    report = {
        "images_evaluated": len(records),
        "per_class_iou_0_3": per_class,
        "detection_iou_0_3": score(counts["iou_03"]),
        "detection_iou_0_5": score(counts["iou_05"]),
        "confusion_matrix_labels": labels,
        "confusion_matrix": cm.tolist(),
        "confusion_matrix_path": str(cm_path),
        "sample_predictions_dir": str(sample_dir),
    }
    (REPORTS_DIR / "evaluation.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    return report


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", default=str(DEFAULT_DATASET_ROOT))
    parser.add_argument("--model", default=str(DEFAULT_MODEL_PATH))
    parser.add_argument("--schema", default=str(DEFAULT_SCHEMA_PATH))
    parser.add_argument("--limit", type=int, default=None)
    args = parser.parse_args()
    report = evaluate(Path(args.data), Path(args.model), Path(args.schema), args.limit)
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
