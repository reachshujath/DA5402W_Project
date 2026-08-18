from __future__ import annotations

import argparse
from pathlib import Path

from pyspark.sql import SparkSession
from pyspark.sql.functions import col, current_timestamp, from_json, lit, struct, to_json, udf
from pyspark.sql.types import DoubleType, IntegerType, StringType, StructField, StructType

from fraud_mlops.config import FEATURE_COLUMNS

_ARTIFACT = None


def build_schema() -> StructType:
    return StructType([StructField(column, DoubleType(), nullable=False) for column in FEATURE_COLUMNS])


def prediction_schema() -> StructType:
    return StructType(
        [
            StructField("fraud_probability", DoubleType(), nullable=False),
            StructField("predicted_class", IntegerType(), nullable=False),
            StructField("model_version", StringType(), nullable=False),
        ]
    )


def make_predictor(model_path: str):
    def predict(*values):
        global _ARTIFACT
        import joblib
        import pandas as pd

        if _ARTIFACT is None:
            _ARTIFACT = joblib.load(model_path)
        frame = pd.DataFrame([values], columns=FEATURE_COLUMNS)
        probability = float(_ARTIFACT["pipeline"].predict_proba(frame)[0, 1])
        predicted_class = int(probability >= float(_ARTIFACT["threshold"]))
        return probability, predicted_class, str(_ARTIFACT["model_version"])

    return udf(predict, prediction_schema())


def main() -> None:
    parser = argparse.ArgumentParser(description="Consume Kafka transactions and emit fraud predictions.")
    parser.add_argument("--bootstrap-servers", default="localhost:9092")
    parser.add_argument("--topic", default="credit-card-transactions")
    parser.add_argument("--output-topic", default="fraud-predictions")
    parser.add_argument("--invalid-topic", default="fraud-invalid-transactions")
    parser.add_argument("--model-path", default="/app/models/fraud_model.joblib")
    parser.add_argument("--checkpoint", default="/tmp/fraud-stream-checkpoint")
    args = parser.parse_args()

    if not Path(args.model_path).exists():
        raise FileNotFoundError(f"Model artifact not found: {args.model_path}")
    spark = SparkSession.builder.appName("credit-card-fraud-stream").config("spark.sql.shuffle.partitions", "2").getOrCreate()
    spark.sparkContext.setLogLevel("WARN")
    raw = (
        spark.readStream.format("kafka")
        .option("kafka.bootstrap.servers", args.bootstrap_servers)
        .option("subscribe", args.topic)
        .option("startingOffsets", "latest")
        .load()
        .select(col("value").cast("string").alias("raw_value"))
    )
    parsed = raw.withColumn("transaction", from_json(col("raw_value"), build_schema()))
    valid_condition = col("transaction").isNotNull()
    for feature in FEATURE_COLUMNS:
        valid_condition = valid_condition & col(f"transaction.{feature}").isNotNull()

    valid = parsed.filter(valid_condition).select("transaction.*")
    predictor = make_predictor(args.model_path)
    predicted = valid.withColumn("prediction", predictor(*[col(feature) for feature in FEATURE_COLUMNS])).select(
        *FEATURE_COLUMNS,
        "prediction.fraud_probability",
        "prediction.predicted_class",
        "prediction.model_version",
        current_timestamp().alias("processed_at"),
    )
    prediction_values = predicted.select(to_json(struct(*predicted.columns)).alias("value"))
    invalid_values = parsed.filter(~valid_condition).select(
        to_json(
            struct(
                col("raw_value").alias("source_payload"),
                lit("schema_validation_failed").alias("reason"),
                current_timestamp().alias("processed_at"),
            )
        ).alias("value")
    )

    queries = [
        prediction_values.writeStream.format("kafka")
        .option("kafka.bootstrap.servers", args.bootstrap_servers)
        .option("topic", args.output_topic)
        .option("checkpointLocation", f"{args.checkpoint}/predictions")
        .outputMode("append")
        .start(),
        invalid_values.writeStream.format("kafka")
        .option("kafka.bootstrap.servers", args.bootstrap_servers)
        .option("topic", args.invalid_topic)
        .option("checkpointLocation", f"{args.checkpoint}/invalid")
        .outputMode("append")
        .start(),
        predicted.selectExpr(
            "CAST(Time AS BIGINT) AS Time",
            "ROUND(Amount, 2) AS Amount",
            "ROUND(fraud_probability, 6) AS fraud_probability",
            "predicted_class",
            "SUBSTRING(model_version, 1, 16) AS model_version",
            "DATE_FORMAT(processed_at, 'yyyy-MM-dd HH:mm:ss') AS processed_at",
        )
        .writeStream.format("console")
        .option("truncate", "false")
        .option("checkpointLocation", f"{args.checkpoint}/console")
        .outputMode("append")
        .start(),
    ]
    spark.streams.awaitAnyTermination()
    for query in queries:
        query.stop()


if __name__ == "__main__":
    main()
