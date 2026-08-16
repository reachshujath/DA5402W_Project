from __future__ import annotations

import argparse
import hashlib
import json
import logging
import os
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
import yaml
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    average_precision_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split

from fraud_mlops.config import (
    DATA_PATH,
    FEATURE_COLUMNS,
    METRICS_PATH,
    MODEL_DIR,
    MODEL_NAME,
    MODEL_PATH,
    PARAMS_PATH,
    PROCESSED_DIR,
    REFERENCE_STATS_PATH,
    REPORT_DIR,
    TARGET_COLUMN,
)
from fraud_mlops.preprocessing import build_model_pipeline
from fraud_mlops.validation import validate_dataframe

LOGGER = logging.getLogger(__name__)


def load_training_params(path: Path = PARAMS_PATH) -> dict[str, Any]:
    with path.open(encoding="utf-8") as stream:
        document = yaml.safe_load(stream) or {}
    params = document.get("train")
    if not isinstance(params, dict):
        raise ValueError("params.yaml must contain a 'train' mapping.")
    return params


def dataset_fingerprint(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def build_candidates(params: dict[str, Any]) -> dict[str, Any]:
    seed = int(params["random_state"])
    logistic = params["logistic_regression"]
    forest = params["random_forest"]
    return {
        "logistic_regression": LogisticRegression(
            class_weight=logistic["class_weight"],
            max_iter=int(logistic["max_iter"]),
            solver="lbfgs",
            random_state=seed,
        ),
        "random_forest": RandomForestClassifier(
            class_weight=forest["class_weight"],
            n_estimators=int(forest["n_estimators"]),
            max_depth=int(forest["max_depth"]),
            random_state=seed,
            n_jobs=-1,
        ),
    }


def choose_threshold(
    y_true: np.ndarray,
    probabilities: np.ndarray,
    minimum: float = 0.05,
    maximum: float = 0.95,
    steps: int = 181,
    min_recall: float = 0.0,
    min_precision: float = 0.0,
) -> tuple[float, dict[str, float]]:
    best_threshold = 0.5
    best_metrics = {"precision": 0.0, "recall": 0.0, "f1": -1.0}
    for threshold in np.linspace(minimum, maximum, steps):
        predictions = (probabilities >= threshold).astype(int)
        metrics = {
            "precision": float(precision_score(y_true, predictions, zero_division=0)),
            "recall": float(recall_score(y_true, predictions, zero_division=0)),
            "f1": float(f1_score(y_true, predictions, zero_division=0)),
        }
        eligible = metrics["recall"] >= min_recall and metrics["precision"] >= min_precision
        current_eligible = best_metrics["recall"] >= min_recall and best_metrics["precision"] >= min_precision
        ranking = (eligible, metrics["f1"], metrics["recall"], metrics["precision"])
        current = (current_eligible, best_metrics["f1"], best_metrics["recall"], best_metrics["precision"])
        if ranking > current:
            best_threshold = float(threshold)
            best_metrics = metrics
    return best_threshold, best_metrics


def evaluate(y_true: np.ndarray, probabilities: np.ndarray, threshold: float) -> dict[str, Any]:
    predictions = (probabilities >= threshold).astype(int)
    matrix = confusion_matrix(y_true, predictions, labels=[0, 1])
    return {
        "threshold": float(threshold),
        "precision": float(precision_score(y_true, predictions, zero_division=0)),
        "recall": float(recall_score(y_true, predictions, zero_division=0)),
        "f1": float(f1_score(y_true, predictions, zero_division=0)),
        "roc_auc": float(roc_auc_score(y_true, probabilities)),
        "pr_auc": float(average_precision_score(y_true, probabilities)),
        "confusion_matrix": {
            "tn": int(matrix[0, 0]),
            "fp": int(matrix[0, 1]),
            "fn": int(matrix[1, 0]),
            "tp": int(matrix[1, 1]),
        },
    }


def passes_gates(metrics: dict[str, Any], params: dict[str, Any]) -> bool:
    gates = params["promotion"]
    return metrics["recall"] >= float(gates["min_recall"]) and metrics["precision"] >= float(
        gates["min_precision"]
    )


def select_candidate(results: dict[str, dict[str, Any]], params: dict[str, Any]) -> tuple[str, str]:
    eligible = [name for name, result in results.items() if passes_gates(result["validation"], params)]
    pool = eligible or list(results)
    winner = max(
        pool,
        key=lambda name: (
            results[name]["validation"]["pr_auc"],
            results[name]["validation"]["f1"],
            results[name]["validation"]["recall"],
        ),
    )
    reason = "highest validation PR-AUC among candidates passing promotion gates"
    if not eligible:
        reason = "highest validation PR-AUC; no candidate passed all promotion gates"
    return winner, reason


def log_candidate_with_mlflow(name: str, result: dict[str, Any], model: Any) -> str | None:
    try:
        import mlflow
        import mlflow.sklearn
        from mlflow.models import infer_signature

        if os.getenv("MLFLOW_TRACKING_URI"):
            mlflow.set_tracking_uri(os.environ["MLFLOW_TRACKING_URI"])
        mlflow.set_experiment("credit-card-fraud-detection")
        with mlflow.start_run(run_name=name) as run:
            parameters = model.named_steps["model"].get_params()
            mlflow.log_params({f"model_{key}": value for key, value in parameters.items() if len(str(value)) < 250})
            mlflow.log_param("model_type", name)
            for split in ("validation", "test"):
                for metric in ("precision", "recall", "f1", "roc_auc", "pr_auc"):
                    mlflow.log_metric(f"{split}_{metric}", result[split][metric])
            mlflow.log_metric("threshold", result["threshold"])
            mlflow.log_dict(result, "evaluation.json")
            input_example = pd.DataFrame([{feature: 0.0 for feature in FEATURE_COLUMNS}])
            signature = infer_signature(input_example, model.predict_proba(input_example))
            mlflow.sklearn.log_model(
                model, artifact_path="model", input_example=input_example, signature=signature
            )
            return run.info.run_id
    except Exception as exc:  # MLflow must not make local training unusable.
        LOGGER.warning("MLflow logging failed for %s: %s", name, exc)
        return None


def register_winner(run_id: str | None, result: dict[str, Any], should_promote: bool) -> dict[str, Any]:
    registration: dict[str, Any] = {"run_id": run_id, "version": None, "promoted": False, "error": None}
    if not run_id:
        return registration
    try:
        import mlflow

        registered = mlflow.register_model(f"runs:/{run_id}/model", MODEL_NAME)
        registration["version"] = str(registered.version)
        client = mlflow.MlflowClient()
        client.set_model_version_tag(
            MODEL_NAME, registered.version, "validation_pr_auc", str(result["validation"]["pr_auc"])
        )
        if should_promote:
            current_score = -1.0
            try:
                champion = client.get_model_version_by_alias(MODEL_NAME, "champion")
                current_score = float(champion.tags.get("validation_pr_auc", -1.0))
            except Exception:
                pass
            if result["validation"]["pr_auc"] >= current_score:
                client.set_registered_model_alias(MODEL_NAME, "champion", registered.version)
                registration["promoted"] = True
        return registration
    except Exception as exc:
        LOGGER.warning("MLflow registration failed: %s", exc)
        registration["error"] = str(exc)
        return registration


def train(data_path: Path = DATA_PATH, sample_rows: int | None = None, enable_mlflow: bool = True) -> dict[str, Any]:
    params = load_training_params()
    seed = int(params["random_state"])
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(data_path, nrows=sample_rows)
    validate_dataframe(df, require_target=True)
    x = df[FEATURE_COLUMNS]
    y = df[TARGET_COLUMN].astype(int)
    x_train_validation, x_test, y_train_validation, y_test = train_test_split(
        x, y, test_size=float(params["test_size"]), random_state=seed, stratify=y
    )
    x_train, x_validation, y_train, y_validation = train_test_split(
        x_train_validation,
        y_train_validation,
        test_size=float(params["validation_size"]),
        random_state=seed,
        stratify=y_train_validation,
    )

    candidates: dict[str, Any] = {}
    results: dict[str, dict[str, Any]] = {}
    for name, estimator in build_candidates(params).items():
        pipeline = build_model_pipeline(estimator)
        pipeline.fit(x_train, y_train)
        validation_probabilities = pipeline.predict_proba(x_validation)[:, 1]
        threshold, _ = choose_threshold(
            y_validation.to_numpy(),
            validation_probabilities,
            float(params["threshold_min"]),
            float(params["threshold_max"]),
            int(params["threshold_steps"]),
            float(params["promotion"]["min_recall"]),
            float(params["promotion"]["min_precision"]),
        )
        result = {
            "threshold": threshold,
            "validation": evaluate(y_validation.to_numpy(), validation_probabilities, threshold),
            "test": evaluate(y_test.to_numpy(), pipeline.predict_proba(x_test)[:, 1], threshold),
            "mlflow_run_id": None,
        }
        if enable_mlflow:
            result["mlflow_run_id"] = log_candidate_with_mlflow(name, result, pipeline)
        candidates[name] = pipeline
        results[name] = result

    winner, selection_reason = select_candidate(results, params)
    winner_result = results[winner]
    promotion_eligible = passes_gates(winner_result["validation"], params)
    registration = (
        register_winner(winner_result["mlflow_run_id"], winner_result, promotion_eligible)
        if enable_mlflow
        else {"run_id": None, "version": None, "promoted": False, "error": None}
    )
    trained_at = datetime.now(UTC).isoformat()
    fingerprint = dataset_fingerprint(data_path)
    version = registration["version"] or f"local-{trained_at}"
    joblib.dump(
        {
            "pipeline": candidates[winner],
            "threshold": winner_result["threshold"],
            "feature_columns": FEATURE_COLUMNS,
            "model_name": MODEL_NAME,
            "model_type": winner,
            "model_version": str(version),
            "mlflow_run_id": winner_result["mlflow_run_id"],
            "dataset_fingerprint": fingerprint,
            "trained_at": trained_at,
        },
        MODEL_PATH,
    )

    for name, frame in (("x_train", x_train), ("x_validation", x_validation), ("x_test", x_test)):
        frame.to_csv(PROCESSED_DIR / f"{name}.csv", index=False)
    for name, series in (("y_train", y_train), ("y_validation", y_validation), ("y_test", y_test)):
        series.to_csv(PROCESSED_DIR / f"{name}.csv", index=False)
    reference_stats = x_train[["Amount", "Time", "V1", "V2", "V3", "V4"]].agg(["mean", "std"]).to_dict()
    REFERENCE_STATS_PATH.write_text(json.dumps(reference_stats, indent=2), encoding="utf-8")
    summary = {
        "selected_model": winner,
        "selection_reason": selection_reason,
        "promotion_eligible": promotion_eligible,
        "registration": registration,
        "dataset_fingerprint": fingerprint,
        "trained_at": trained_at,
        "candidates": results,
    }
    METRICS_PATH.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description="Train and compare fraud detection models.")
    parser.add_argument("--data-path", type=Path, default=DATA_PATH)
    parser.add_argument("--sample-rows", type=int, default=None)
    parser.add_argument("--no-mlflow", action="store_true")
    args = parser.parse_args()
    print(json.dumps(train(args.data_path, args.sample_rows, enable_mlflow=not args.no_mlflow), indent=2))


if __name__ == "__main__":
    main()
