import pandas as pd
import pytest

from fraud_mlops.config import FEATURE_COLUMNS, TARGET_COLUMN
from fraud_mlops.validation.schema import ValidationError, validate_dataframe, validate_prediction_payload


def valid_frame():
    data = {column: [0.0] for column in FEATURE_COLUMNS}
    data["Amount"] = [10.0]
    data[TARGET_COLUMN] = [0]
    return pd.DataFrame(data)


def test_validate_dataframe_accepts_valid_schema():
    validate_dataframe(valid_frame())


def test_validate_dataframe_rejects_negative_amount():
    df = valid_frame()
    df.loc[0, "Amount"] = -1.0
    with pytest.raises(ValidationError):
        validate_dataframe(df)


def test_validate_prediction_payload_returns_float_record():
    payload = {column: 1 for column in FEATURE_COLUMNS}
    record = validate_prediction_payload(payload)
    assert set(record) == set(FEATURE_COLUMNS)
    assert all(isinstance(value, float) for value in record.values())


def test_validate_prediction_payload_rejects_non_finite_value():
    payload = {column: 1 for column in FEATURE_COLUMNS}
    payload["V1"] = float("inf")
    with pytest.raises(ValidationError, match="finite"):
        validate_prediction_payload(payload)
