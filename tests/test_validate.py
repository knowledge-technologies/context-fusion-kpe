from keyword_pecker.trainer.validate import compute_keyphrase_metrics


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
