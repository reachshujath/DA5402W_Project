from __future__ import annotations

from collections.abc import Mapping
import math

import numpy as np
import pandas as pd

from fraud_mlops.config import FEATURE_COLUMNS, TARGET_COLUMN


class ValidationError(ValueError):
    """Raised when input data does not match the project schema."""


def validate_dataframe(df: pd.DataFrame, require_target: bool = True) -> None:
    expected = FEATURE_COLUMNS + ([TARGET_COLUMN] if require_target else [])
    missing = [column for column in expected if column not in df.columns]
    if missing:
        raise ValidationError(f"Missing required columns: {missing}")

    checked = df[expected]
    if checked.isna().any().any():
        raise ValidationError("Dataset contains missing values.")

    non_numeric = [
        column
        for column in expected
        if not pd.api.types.is_numeric_dtype(checked[column])
    ]
    if non_numeric:
        raise ValidationError(f"Columns must be numeric: {non_numeric}")

    if not np.isfinite(checked.to_numpy(dtype=float)).all():
        raise ValidationError("Dataset contains non-finite values.")

    if (df["Amount"] < 0).any():
        raise ValidationError("Amount must be non-negative.")

    if require_target:
        labels = set(df[TARGET_COLUMN].dropna().astype(int).unique())
        if not labels.issubset({0, 1}):
            raise ValidationError("Class must contain only 0 and 1.")


def validate_prediction_payload(payload: Mapping[str, float]) -> dict[str, float]:
    missing = [column for column in FEATURE_COLUMNS if column not in payload]
    if missing:
        raise ValidationError(f"Missing required feature columns: {missing}")

    record: dict[str, float] = {}
    for column in FEATURE_COLUMNS:
        try:
            record[column] = float(payload[column])
        except (TypeError, ValueError) as exc:
            raise ValidationError(f"{column} must be numeric.") from exc
        if not math.isfinite(record[column]):
            raise ValidationError(f"{column} must be finite.")

    if record["Amount"] < 0:
        raise ValidationError("Amount must be non-negative.")

    return record
