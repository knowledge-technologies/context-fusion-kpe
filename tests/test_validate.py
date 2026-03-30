from keyword_pecker.trainer.validate import (
    compute_keyphrase_metrics,
    compute_metrics_at_k,
)


GOLD_TRUTH = [{"car", "bike"}, {"apple", "banana", "cherry"}]
FAIL_PREDICTIONS = [set(), {"dog", "elephant"}]
PERFECT_PREDICTIONS = [{"car", "bike"}, {"apple", "banana", "cherry"}]
PARTIAL_PREDICTIONS = [{"car"}, {"apple"}]


def test_compute_keyphrase_metrics():
    metrics = compute_keyphrase_metrics(
        gold_keywords=GOLD_TRUTH, pred_keywords=FAIL_PREDICTIONS
    )
    assert metrics["precision"] == 0.0
    assert metrics["recall"] == 0.0
    assert metrics["f1"] == 0.0

    metrics = compute_keyphrase_metrics(
        gold_keywords=GOLD_TRUTH, pred_keywords=PERFECT_PREDICTIONS
    )
    assert metrics["precision"] == 1.0
    assert metrics["recall"] == 1.0
    assert metrics["f1"] == 1.0

    metrics = compute_keyphrase_metrics(
        gold_keywords=GOLD_TRUTH, pred_keywords=PARTIAL_PREDICTIONS
    )
    assert metrics["precision"] == 1.0
    assert metrics["recall"] == 0.4
    assert abs(metrics["f1"] - 4 / 7) < 1e-6


def test_compute_metrics_at_k():
    gold = [{"car", "bike"}, {"apple", "banana", "cherry"}]
    pred_ranked = [
        [("car", 0.9), ("bike", 0.8), ("dog", 0.7)],
        [("apple", 0.95), ("banana", 0.85), ("elephant", 0.6)],
    ]

    metrics = compute_metrics_at_k(gold, pred_ranked, k_values=[1, 2, 3])

    assert metrics["precision@1"] == 1.0
    assert metrics["recall@1"] == 0.4

    assert metrics["precision@2"] == 1.0
    assert metrics["recall@2"] == 0.8

    assert abs(metrics["precision@3"] - 4 / 6) < 1e-6
    assert abs(metrics["recall@3"] - 4 / 5) < 1e-6


def test_compute_metrics_at_k_empty_predictions():
    gold = [{"car", "bike"}, {"apple"}]
    pred_ranked = [[], []]

    metrics = compute_metrics_at_k(gold, pred_ranked, k_values=[5])

    assert metrics["precision@5"] == 0.0
    assert metrics["recall@5"] == 0.0
    assert metrics["f1@5"] == 0.0


def test_compute_metrics_at_k_perfect():
    gold = [{"car", "bike"}]
    pred_ranked = [[("car", 0.9), ("bike", 0.8)]]

    metrics = compute_metrics_at_k(gold, pred_ranked, k_values=[1, 2, 5])

    assert metrics["precision@1"] == 1.0
    assert metrics["recall@1"] == 0.5

    assert metrics["precision@2"] == 1.0
    assert metrics["recall@2"] == 1.0

    assert metrics["precision@5"] == 1.0
    assert metrics["recall@5"] == 1.0


def test_compute_metrics_at_n():
    gold = [{"car", "bike"}, {"apple"}]
    pred_ranked = [
        [("car", 0.9), ("bike", 0.8), ("dog", 0.7)],
        [("apple", 0.95), ("banana", 0.6)],
    ]

    metrics = compute_metrics_at_k(gold, pred_ranked, k_values=["N"])

    assert metrics["precision@N"] == 1.0
    assert metrics["recall@N"] == 1.0
    assert metrics["f1@N"] == 1.0


def test_compute_metrics_at_k_with_n():
    gold = [{"car", "bike"}, {"apple", "banana", "cherry"}]
    pred_ranked = [
        [("car", 0.9), ("bike", 0.8), ("dog", 0.7)],
        [("apple", 0.95), ("banana", 0.85), ("elephant", 0.6)],
    ]

    metrics = compute_metrics_at_k(gold, pred_ranked, k_values=[1, "N"])

    assert metrics["precision@1"] == 1.0
    assert metrics["recall@1"] == 0.4

    assert abs(metrics["precision@N"] - 0.8) < 1e-6
    assert abs(metrics["recall@N"] - 0.8) < 1e-6
    assert abs(metrics["f1@N"] - 0.8) < 1e-6


def test_compute_metrics_at_n_partial():
    gold = [{"car", "bike"}, {"apple"}]
    pred_ranked = [
        [("dog", 0.9), ("car", 0.8), ("bike", 0.7)],
        [("banana", 0.95), ("apple", 0.6)],
    ]

    metrics = compute_metrics_at_k(gold, pred_ranked, k_values=["N"])

    assert abs(metrics["precision@N"] - 1 / 3) < 1e-6
    assert abs(metrics["recall@N"] - 1 / 3) < 1e-6
    assert abs(metrics["f1@N"] - 1 / 3) < 1e-6
