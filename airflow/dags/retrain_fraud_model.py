from __future__ import annotations

from datetime import UTC, datetime

from airflow.decorators import dag, task


@dag(
    dag_id="credit_card_fraud_retraining",
    description="Validate data, compare candidates, register the winner, and refresh drift baselines.",
    start_date=datetime(2026, 1, 1, tzinfo=UTC),
    schedule=None,
    catchup=False,
    tags=["mlops", "fraud"],
)
def fraud_retraining_dag():
    @task
    def validate_dataset() -> dict:
        import pandas as pd

        from fraud_mlops.config import DATA_PATH
        from fraud_mlops.validation import validate_dataframe

        df = pd.read_csv(DATA_PATH)
        validate_dataframe(df, require_target=True)
        return {"rows": len(df), "fraud_rows": int(df["Class"].sum())}

    @task
    def train_and_compare(_: dict) -> dict:
        from fraud_mlops.training.train import train

        summary = train(enable_mlflow=True)
        return {
            "selected_model": summary["selected_model"],
            "selection_reason": summary["selection_reason"],
            "promotion_eligible": summary["promotion_eligible"],
            "registration": summary["registration"],
        }

    @task
    def evaluate_selection(summary: dict) -> dict:
        if not summary["selected_model"]:
            raise ValueError("Training did not select a model.")
        return summary

    @task
    def record_promotion(summary: dict) -> str:
        registration = summary["registration"]
        if registration.get("promoted"):
            return f"Promoted {summary['selected_model']} as champion version {registration['version']}."
        return f"Registered {summary['selected_model']} without champion promotion."

    @task
    def verify_drift_baseline(_: str) -> str:
        from fraud_mlops.config import REFERENCE_STATS_PATH

        if not REFERENCE_STATS_PATH.exists():
            raise FileNotFoundError("Training did not generate the drift reference statistics.")
        return str(REFERENCE_STATS_PATH)

    validation = validate_dataset()
    trained = train_and_compare(validation)
    evaluated = evaluate_selection(trained)
    promotion = record_promotion(evaluated)
    verify_drift_baseline(promotion)


fraud_retraining = fraud_retraining_dag()
