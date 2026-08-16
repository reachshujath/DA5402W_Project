from __future__ import annotations

from pathlib import Path

import joblib
import pandas as pd

from fraud_mlops.config import FEATURE_COLUMNS, MODEL_PATH
from fraud_mlops.validation import validate_prediction_payload


class FraudModelService:
    def __init__(self, model_path: Path = MODEL_PATH):
        self.model_path = model_path
        self.artifact = None

    def load(self) -> None:
        if not self.model_path.exists():
            raise FileNotFoundError(f"Model artifact not found: {self.model_path}")
        self.artifact = joblib.load(self.model_path)

    @property
    def is_loaded(self) -> bool:
        return self.artifact is not None

    def model_info(self) -> dict:
        self._ensure_loaded()
        return {
            "model_name": self.artifact["model_name"],
            "model_version": self.artifact["model_version"],
            "model_type": self.artifact.get("model_type", "logistic_regression"),
            "threshold": self.artifact["threshold"],
            "feature_count": len(self.artifact["feature_columns"]),
            "mlflow_run_id": self.artifact.get("mlflow_run_id"),
            "dataset_fingerprint": self.artifact.get("dataset_fingerprint"),
            "trained_at": self.artifact.get("trained_at"),
        }

    def predict(self, payload: dict) -> dict:
        self._ensure_loaded()
        record = validate_prediction_payload(payload)
        frame = pd.DataFrame([record], columns=FEATURE_COLUMNS)
        probability = float(self.artifact["pipeline"].predict_proba(frame)[0, 1])
        threshold = float(self.artifact["threshold"])
        prediction = int(probability >= threshold)
        return {
            "fraud_probability": probability,
            "predicted_class": prediction,
            "threshold": threshold,
            "model_name": self.artifact["model_name"],
            "model_type": self.artifact.get("model_type", "logistic_regression"),
            "model_version": self.artifact["model_version"],
        }

    def _ensure_loaded(self) -> None:
        if self.artifact is None:
            self.load()
