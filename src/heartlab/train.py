"""Compare two pipelines with stratified CV and one untouched test split."""

import json
import os
from pathlib import Path

import joblib
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import mlflow
import numpy as np
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    RocCurveDisplay,
    accuracy_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import GridSearchCV, StratifiedKFold, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from heartlab.data import CATEGORICAL, FEATURES, NUMERIC, load_data
from heartlab.eda import make_plots

SEED = 42
ARTIFACT_DIR = Path("artifacts")


def build_pipeline(classifier) -> Pipeline:
    numeric = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
    ])
    categorical = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("onehot", OneHotEncoder(handle_unknown="ignore")),
    ])
    preprocess = ColumnTransformer([
        ("numeric", numeric, NUMERIC),
        ("categorical", categorical, CATEGORICAL),
    ])
    return Pipeline([("preprocess", preprocess), ("classifier", classifier)])


def search_spaces() -> dict:
    return {
        "logistic_regression": (
            LogisticRegression(max_iter=2000, random_state=SEED),
            {"classifier__C": [0.1, 1.0, 10.0]},
        ),
        "random_forest": (
            RandomForestClassifier(random_state=SEED, n_jobs=1),
            {"classifier__n_estimators": [100, 200],
             "classifier__max_depth": [3, None],
             "classifier__min_samples_leaf": [1, 3]},
        ),
    }


def calculate_metrics(y_true, probability: np.ndarray) -> dict[str, float]:
    predicted = (probability >= 0.5).astype(int)
    return {
        "accuracy": float(accuracy_score(y_true, predicted)),
        "precision": float(precision_score(y_true, predicted, zero_division=0)),
        "recall": float(recall_score(y_true, predicted, zero_division=0)),
        "roc_auc": float(roc_auc_score(y_true, probability)),
    }


def train() -> dict:
    frame = load_data()
    if len(frame) < 100 or frame["target"].nunique() != 2:
        raise ValueError("Training needs at least 100 rows and both classes")
    ARTIFACT_DIR.mkdir(exist_ok=True)
    plots = make_plots()
    x_train, x_test, y_train, y_test = train_test_split(
        frame[FEATURES], frame["target"], test_size=0.2,
        stratify=frame["target"], random_state=SEED,
    )
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)
    mlflow.set_tracking_uri(os.getenv("MLFLOW_TRACKING_URI", "sqlite:///mlflow.db"))
    mlflow.set_experiment("cleveland-heart-disease")
    comparisons = {}
    winners = {}
    for name, (classifier, grid) in search_spaces().items():
        with mlflow.start_run(run_name=name) as run:
            search = GridSearchCV(
                build_pipeline(classifier), grid, scoring={
                    "roc_auc": "roc_auc", "accuracy": "accuracy",
                    "precision": "precision", "recall": "recall",
                }, refit="roc_auc", cv=cv, n_jobs=1, error_score="raise",
                return_train_score=False,
            )
            search.fit(x_train, y_train)
            index = search.best_index_
            cv_metrics = {
                metric: float(search.cv_results_[f"mean_test_{metric}"][index])
                for metric in ["roc_auc", "accuracy", "precision", "recall"]
            }
            mlflow.log_params({key.replace("classifier__", ""): value
                               for key, value in search.best_params_.items()})
            mlflow.log_param("seed", SEED)
            mlflow.log_param("train_rows", len(x_train))
            mlflow.log_metrics({f"cv_{key}": value for key, value in cv_metrics.items()})
            mlflow.log_artifacts(str(ARTIFACT_DIR / "eda"), artifact_path="eda")
            comparisons[name] = {
                "cv": cv_metrics, "best_params": search.best_params_, "run_id": run.info.run_id,
            }
            winners[name] = search.best_estimator_

    # Select with training CV alone. The test split is evaluated once below.
    selected_name = max(comparisons, key=lambda name: comparisons[name]["cv"]["roc_auc"])
    model = winners[selected_name]
    probability = model.predict_proba(x_test)[:, 1]
    metrics = calculate_metrics(y_test, probability)
    model_path = ARTIFACT_DIR / "model.joblib"
    joblib.dump(model, model_path)
    metadata = {
        "dataset": "UCI processed Cleveland Heart Disease",
        "source_url": "https://archive.ics.uci.edu/dataset/45/heart+disease",
        "seed": SEED,
        "split": {"train": len(x_train), "test": len(x_test), "test_fraction": 0.2},
        "selection_rule": "highest mean 5-fold training ROC-AUC",
        "selected_model": selected_name,
        "candidate_results": comparisons,
        "test_metrics": metrics,
        "features": FEATURES,
        "threshold": 0.5,
        "note": "Educational classifier only; not a clinical decision tool.",
    }
    metadata_path = ARTIFACT_DIR / "metadata.json"
    metadata_path.write_text(json.dumps(metadata, indent=2) + "\n")

    fig, axis = plt.subplots(figsize=(5, 4))
    ConfusionMatrixDisplay.from_predictions(y_test, probability >= 0.5, ax=axis)
    fig.tight_layout()
    confusion_path = ARTIFACT_DIR / "confusion_matrix.png"
    fig.savefig(confusion_path, dpi=160)
    plt.close(fig)
    fig, axis = plt.subplots(figsize=(5, 4))
    RocCurveDisplay.from_predictions(y_test, probability, ax=axis)
    fig.tight_layout()
    roc_path = ARTIFACT_DIR / "roc_curve.png"
    fig.savefig(roc_path, dpi=160)
    plt.close(fig)
    with mlflow.start_run(run_name="selected-model"):
        mlflow.log_param("selected_model", selected_name)
        mlflow.log_param("selection_rule", metadata["selection_rule"])
        mlflow.log_metrics({f"test_{key}": value for key, value in metrics.items()})
        for path in [model_path, metadata_path, confusion_path, roc_path, *plots]:
            mlflow.log_artifact(str(path), artifact_path="final")
    return metadata


if __name__ == "__main__":
    print(json.dumps(train(), indent=2))
