from __future__ import annotations

import json
import logging
import time
import uuid
from contextlib import asynccontextmanager
from typing import Any

from fastapi import FastAPI, HTTPException
from prometheus_client import CONTENT_TYPE_LATEST, Counter, Gauge, Histogram, generate_latest
from starlette.responses import Response

from fraud_mlops.drift import DriftMonitor
from fraud_mlops.config import FEATURE_COLUMNS
from fraud_mlops.inference import FraudModelService
from fraud_mlops.validation.schema import ValidationError

LOGGER = logging.getLogger("fraud_api")
logging.basicConfig(level=logging.INFO, format="%(message)s")

model_service = FraudModelService()
drift_monitor = DriftMonitor()

REQUEST_COUNT = Counter("fraud_api_requests_total", "Total prediction API requests", ["endpoint", "status"])
ERROR_COUNT = Counter("fraud_api_errors_total", "Prediction API errors", ["type"])
PREDICTION_COUNT = Counter("fraud_predictions_total", "Prediction counts by class", ["predicted_class"])
PREDICTION_LATENCY = Histogram("fraud_prediction_latency_seconds", "Prediction latency in seconds")
FRAUD_PROBABILITY = Histogram(
    "fraud_prediction_probability", "Predicted fraud probability", buckets=(0.01, 0.05, 0.1, 0.25, 0.5, 0.75, 0.9, 0.99)
)
MODEL_LOADED = Gauge("fraud_model_loaded", "Whether the fraud model is loaded")
DRIFT_SCORE = Gauge("fraud_feature_drift_score", "Mean shift in reference standard deviations", ["feature"])
DRIFT_FLAG = Gauge("fraud_feature_drift_flag", "Whether feature drift exceeds threshold", ["feature"])


def emit_log(event: str, **fields: Any) -> None:
    LOGGER.info(json.dumps({"event": event, **fields}, default=str, sort_keys=True))


@asynccontextmanager
async def lifespan(_: FastAPI):
    try:
        model_service.load()
        MODEL_LOADED.set(1)
        emit_log("model_loaded", **model_service.model_info())
    except FileNotFoundError as exc:
        MODEL_LOADED.set(0)
        emit_log("model_missing", error=str(exc))
    yield


app = FastAPI(title="Credit Card Fraud Detection API", version="0.2.0", lifespan=lifespan)


@app.get("/health")
def health() -> dict[str, Any]:
    response: dict[str, Any] = {"status": "ok" if model_service.is_loaded else "degraded", "model_loaded": model_service.is_loaded}
    if model_service.is_loaded:
        response.update({key: value for key, value in model_service.model_info().items() if key in {"model_name", "model_version"}})
    return response


@app.get("/model-info")
def model_info() -> dict[str, Any]:
    try:
        return model_service.model_info()
    except FileNotFoundError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc


@app.post("/predict")
def predict(payload: dict[str, Any]) -> dict[str, Any]:
    start = time.perf_counter()
    request_id = str(uuid.uuid4())
    try:
        result = model_service.predict(payload)
    except FileNotFoundError as exc:
        REQUEST_COUNT.labels(endpoint="/predict", status="model_missing").inc()
        ERROR_COUNT.labels(type="model_missing").inc()
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    except ValidationError as exc:
        REQUEST_COUNT.labels(endpoint="/predict", status="validation_error").inc()
        ERROR_COUNT.labels(type="validation_error").inc()
        emit_log("prediction_rejected", request_id=request_id, reason=str(exc))
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    except Exception as exc:
        REQUEST_COUNT.labels(endpoint="/predict", status="error").inc()
        ERROR_COUNT.labels(type="internal_error").inc()
        emit_log("prediction_error", request_id=request_id, error_type=type(exc).__name__)
        raise HTTPException(status_code=500, detail="Prediction failed.") from exc

    latency = time.perf_counter() - start
    PREDICTION_LATENCY.observe(latency)
    REQUEST_COUNT.labels(endpoint="/predict", status="success").inc()
    PREDICTION_COUNT.labels(predicted_class=str(result["predicted_class"])).inc()
    FRAUD_PROBABILITY.observe(result["fraud_probability"])
    drift_results = drift_monitor.observe({feature: payload[feature] for feature in FEATURE_COLUMNS})
    if drift_results:
        for feature, values in drift_results.items():
            DRIFT_SCORE.labels(feature=feature).set(values["mean_shift_std_units"])
            DRIFT_FLAG.labels(feature=feature).set(int(values["drift_flag"]))
    result["request_id"] = request_id
    emit_log(
        "prediction_completed",
        request_id=request_id,
        status="success",
        latency_seconds=latency,
        predicted_class=result["predicted_class"],
        fraud_probability=result["fraud_probability"],
        model_version=result["model_version"],
    )
    return result


@app.get("/drift")
def drift() -> dict[str, Any]:
    return drift_monitor.snapshot()


@app.get("/metrics")
def metrics() -> Response:
    return Response(generate_latest(), media_type=CONTENT_TYPE_LATEST)
