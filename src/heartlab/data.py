"""Acquire and validate the original UCI processed Cleveland records."""

from pathlib import Path
from urllib.request import urlopen

import pandas as pd

SOURCE_URL = (
    "https://archive.ics.uci.edu/ml/machine-learning-databases/"
    "heart-disease/processed.cleveland.data"
)
FIELDS = [
    "age", "sex", "cp", "trestbps", "chol", "fbs", "restecg", "thalach",
    "exang", "oldpeak", "slope", "ca", "thal", "num",
]
FEATURES = FIELDS[:-1]
NUMERIC = ["age", "trestbps", "chol", "thalach", "oldpeak"]
CATEGORICAL = [name for name in FEATURES if name not in NUMERIC]
DATA_PATH = Path("data/cleveland_clean.csv")


def clean_data(frame: pd.DataFrame) -> pd.DataFrame:
    """Convert UCI's 0-4 outcome to a binary label, retaining missing features."""
    if list(frame.columns) != FIELDS:
        raise ValueError(f"Expected UCI columns {FIELDS}")
    result = frame.copy()
    for name in FIELDS:
        result[name] = pd.to_numeric(result[name], errors="coerce")
    if result["num"].isna().any() or not result["num"].isin(range(5)).all():
        raise ValueError("Target must contain integer values 0 to 4")
    if result[FEATURES].isna().all(axis=1).any():
        raise ValueError("A row has no usable features")
    result = result.rename(columns={"num": "target"})
    result["target"] = (result["target"] > 0).astype(int)
    return result


def download_data(destination: Path = DATA_PATH) -> Path:
    """Download the public source once; no patient data is sent to another service."""
    destination.parent.mkdir(parents=True, exist_ok=True)
    with urlopen(SOURCE_URL, timeout=30) as response:
        raw = response.read()
    if not raw:
        raise ValueError("Empty download from UCI")
    raw_path = destination.parent / "processed.cleveland.data"
    raw_path.write_bytes(raw)
    frame = pd.read_csv(raw_path, names=FIELDS, na_values="?")
    if len(frame) != 303:
        raise ValueError(f"Expected 303 Cleveland rows, received {len(frame)}")
    cleaned = clean_data(frame)
    cleaned.to_csv(destination, index=False)
    return destination


def load_data(path: Path = DATA_PATH) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Missing {path}. Run python -m heartlab.data first.")
    frame = pd.read_csv(path)
    if list(frame.columns) != FEATURES + ["target"]:
        raise ValueError("Unexpected cleaned dataset schema")
    return frame


if __name__ == "__main__":
    print(download_data())
