from __future__ import annotations

import json
from collections import deque
from datetime import UTC, datetime
from pathlib import Path
from threading import Lock
from typing import Any

import pandas as pd

from fraud_mlops.config import REFERENCE_STATS_PATH


def load_reference_stats(path: Path = REFERENCE_STATS_PATH) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def compute_basic_drift(current: pd.DataFrame, reference_path: Path = REFERENCE_STATS_PATH) -> dict[str, Any]:
    reference = load_reference_stats(reference_path)
    results: dict[str, Any] = {}
    for column, stats in reference.items():
        if column not in current.columns:
            continue
        ref_mean = float(stats["mean"])
        ref_std = max(float(stats["std"]), 1e-9)
        current_mean = float(current[column].mean())
        z_score = abs(current_mean - ref_mean) / ref_std
        results[column] = {
            "reference_mean": ref_mean,
            "current_mean": current_mean,
            "mean_shift_std_units": z_score,
            "drift_flag": z_score >= 3.0,
        }
    return results


class DriftMonitor:
    """Thread-safe rolling feature monitor for a single-process demo deployment."""

    def __init__(
        self,
        reference_path: Path = REFERENCE_STATS_PATH,
        window_size: int = 500,
        calculation_interval: int = 100,
    ) -> None:
        self.reference_path = reference_path
        self.window: deque[dict[str, float]] = deque(maxlen=window_size)
        self.calculation_interval = calculation_interval
        self.total_observations = 0
        self.last_calculated_at: str | None = None
        self.results: dict[str, Any] = {}
        self._lock = Lock()

    def observe(self, record: dict[str, Any]) -> dict[str, Any] | None:
        with self._lock:
            self.window.append({key: float(value) for key, value in record.items()})
            self.total_observations += 1
            if self.total_observations % self.calculation_interval != 0:
                return None
            if not self.reference_path.exists():
                return None
            self.results = compute_basic_drift(pd.DataFrame(self.window), self.reference_path)
            self.last_calculated_at = datetime.now(UTC).isoformat()
            return dict(self.results)

    def snapshot(self) -> dict[str, Any]:
        with self._lock:
            return {
                "window_size": len(self.window),
                "window_capacity": self.window.maxlen,
                "total_observations": self.total_observations,
                "last_calculated_at": self.last_calculated_at,
                "features": dict(self.results),
            }
