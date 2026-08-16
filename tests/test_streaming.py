from fraud_mlops.config import FEATURE_COLUMNS
from fraud_mlops.streaming.spark_streaming import build_schema, prediction_schema


def test_spark_schemas_cover_inputs_and_prediction_metadata():
    assert [field.name for field in build_schema().fields] == FEATURE_COLUMNS
    assert [field.name for field in prediction_schema().fields] == [
        "fraud_probability",
        "predicted_class",
        "model_version",
    ]
