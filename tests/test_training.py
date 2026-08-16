import numpy as np

from fraud_mlops.training.train import choose_threshold, select_candidate


def test_choose_threshold_finds_a_useful_cutoff():
    threshold, metrics = choose_threshold(
        np.array([0, 0, 1, 1]), np.array([0.1, 0.2, 0.7, 0.9]), minimum=0.1, maximum=0.9, steps=9
    )
    assert 0.2 < threshold <= 0.7
    assert metrics["f1"] == 1.0


def test_select_candidate_applies_gates_before_pr_auc():
    params = {"promotion": {"min_recall": 0.8, "min_precision": 0.2}}
    results = {
        "high_auc_but_ineligible": {"validation": {"pr_auc": 0.9, "f1": 0.7, "recall": 0.7, "precision": 0.8}},
        "eligible": {"validation": {"pr_auc": 0.8, "f1": 0.6, "recall": 0.9, "precision": 0.3}},
    }
    winner, reason = select_candidate(results, params)
    assert winner == "eligible"
    assert "passing promotion gates" in reason
