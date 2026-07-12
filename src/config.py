from pathlib import Path

CATEGORY_ID_TO_NAME = {
    1: "crack",
    2: "exfoliation",
    3: "spalling",
}

VALID_LABELS = tuple(CATEGORY_ID_TO_NAME.values())
MODEL_LABELS = ("background", "crack", "exfoliation", "spalling")

DEFAULT_DATASET_ROOT = Path("2024_ground_wall_tile_dataset_latest.v2i.coco")
DEFAULT_MODEL_PATH = Path("models/defect_classifier.joblib")
DEFAULT_SCHEMA_PATH = Path("models/feature_schema.joblib")
DEFAULT_REPORT_PATH = Path("models/training_report.json")

OUTPUTS_DIR = Path("outputs")
PREDICTIONS_DIR = OUTPUTS_DIR / "predictions"
DEBUG_DIR = OUTPUTS_DIR / "debug"
REPORTS_DIR = OUTPUTS_DIR / "reports"

MIN_CONFIDENCE = 0.4
NMS_IOU_THRESHOLD = 0.3
POSITIVE_IOU_THRESHOLD = 0.3
BACKGROUND_IOU_THRESHOLD = 0.1

RANDOM_STATE = 42
