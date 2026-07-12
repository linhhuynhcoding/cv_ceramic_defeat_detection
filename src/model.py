from __future__ import annotations

from pathlib import Path

import joblib
from sklearn.ensemble import ExtraTreesClassifier, RandomForestClassifier

from src.config import RANDOM_STATE


def build_classifier(kind: str = "extra_trees"):
    if kind == "random_forest":
        return RandomForestClassifier(
            n_estimators=250,
            class_weight="balanced",
            random_state=RANDOM_STATE,
            n_jobs=-1,
        )
    return ExtraTreesClassifier(
        n_estimators=300,
        class_weight="balanced",
        random_state=RANDOM_STATE,
        n_jobs=-1,
    )


def save_model(model, feature_schema: list[str], model_path: Path, schema_path: Path) -> None:
    model_path.parent.mkdir(parents=True, exist_ok=True)
    schema_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, model_path)
    joblib.dump(feature_schema, schema_path)


def load_model(model_path: Path, schema_path: Path | None = None):
    model = joblib.load(model_path)
    schema = None
    if schema_path and schema_path.exists():
        schema = joblib.load(schema_path)
    return model, schema
