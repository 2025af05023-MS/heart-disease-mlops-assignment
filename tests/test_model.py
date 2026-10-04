import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression

from heartlab.data import FEATURES
from heartlab.train import build_pipeline, calculate_metrics


def test_pipeline_handles_missing_values_without_pre_split_fitting():
    data = pd.DataFrame([
        [55, 1, 4, 140, 250, 0, 2, 130, 1, 2.1, 2, None, 7],
        [40, 0, 2, 120, 210, 0, 0, 170, 0, 0.0, 1, 0, 3],
        [62, 1, 4, 155, 280, 1, 2, 120, 1, 3.0, 3, 2, 7],
        [45, 0, 3, 125, 220, 0, 0, 160, 0, 0.2, 1, 0, 3],
    ], columns=FEATURES)
    model = build_pipeline(LogisticRegression(max_iter=1000))
    model.fit(data, [1, 0, 1, 0])
    probabilities = model.predict_proba(data)[:, 1]
    assert len(probabilities) == 4
    assert np.isfinite(probabilities).all()


def test_metrics_threshold_and_auc():
    result = calculate_metrics([0, 1, 0, 1], np.array([0.1, 0.8, 0.2, 0.9]))
    assert result == {"accuracy": 1.0, "precision": 1.0, "recall": 1.0, "roc_auc": 1.0}
