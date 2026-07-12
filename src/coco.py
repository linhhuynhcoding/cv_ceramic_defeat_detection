from __future__ import annotations

import json
from collections import Counter
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from src.config import CATEGORY_ID_TO_NAME


@dataclass(frozen=True)
class CocoRecord:
    image_path: Path
    image_id: int
    width: int
    height: int
    boxes: np.ndarray
    labels: list[str]


def annotation_path(dataset_root: Path, split: str) -> Path:
    return dataset_root / split / "_annotations.coco.json"


def load_coco_json(dataset_root: Path, split: str) -> dict:
    path = annotation_path(dataset_root, split)
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def load_coco_records(dataset_root: Path, split: str) -> list[CocoRecord]:
    data = load_coco_json(dataset_root, split)
    split_dir = dataset_root / split
    annotations_by_image: dict[int, list[dict]] = {}
    for ann in data.get("annotations", []):
        label = CATEGORY_ID_TO_NAME.get(ann.get("category_id"))
        if label is None:
            continue
        annotations_by_image.setdefault(ann["image_id"], []).append(ann)

    records: list[CocoRecord] = []
    for image in data.get("images", []):
        anns = annotations_by_image.get(image["id"], [])
        boxes = np.array([ann["bbox"] for ann in anns], dtype=np.float32)
        if boxes.size == 0:
            boxes = np.zeros((0, 4), dtype=np.float32)
        labels = [CATEGORY_ID_TO_NAME[ann["category_id"]] for ann in anns]
        records.append(
            CocoRecord(
                image_path=split_dir / image["file_name"],
                image_id=int(image["id"]),
                width=int(image["width"]),
                height=int(image["height"]),
                boxes=boxes,
                labels=labels,
            )
        )
    return records


def count_annotations(dataset_root: Path, split: str) -> Counter[str]:
    counts: Counter[str] = Counter()
    data = load_coco_json(dataset_root, split)
    for ann in data.get("annotations", []):
        label = CATEGORY_ID_TO_NAME.get(ann.get("category_id"))
        if label:
            counts[label] += 1
    return counts


def list_splits(dataset_root: Path) -> list[str]:
    return [
        split
        for split in ("train", "valid")
        if annotation_path(dataset_root, split).exists()
    ]
