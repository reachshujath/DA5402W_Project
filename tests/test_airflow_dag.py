import importlib.util
from pathlib import Path


def test_retraining_dag_exposes_complete_lifecycle():
    path = Path(__file__).parents[1] / "airflow" / "dags" / "retrain_fraud_model.py"
    spec = importlib.util.spec_from_file_location("fraud_retraining_test_dag", path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)

    assert module.fraud_retraining.dag_id == "credit_card_fraud_retraining"
    assert set(module.fraud_retraining.task_ids) == {
        "validate_dataset",
        "train_and_compare",
        "evaluate_selection",
        "record_promotion",
        "verify_drift_baseline",
    }
