import json

from fraud_mlops.drift import DriftMonitor


def test_drift_monitor_calculates_on_configured_interval(tmp_path):
    reference = {"Amount": {"mean": 10.0, "std": 2.0}}
    reference_path = tmp_path / "reference.json"
    reference_path.write_text(json.dumps(reference), encoding="utf-8")
    monitor = DriftMonitor(reference_path=reference_path, window_size=4, calculation_interval=2)

    assert monitor.observe({"Amount": 20.0}) is None
    result = monitor.observe({"Amount": 20.0})

    assert result["Amount"]["drift_flag"] is True
    assert monitor.snapshot()["window_size"] == 2
