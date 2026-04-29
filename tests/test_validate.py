from numpy.testing import assert_approx_equal

from keyword_pecker.trainer.validate import (
    compute_keyphrase_metrics,
    compute_metrics_at_k,
    compute_metrics_at_k_martinc,
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


def test_compute_metrics_at_k_martinc_stemmed_matching():
    # Porter stemming: "networks" -> "network", "learning" -> "learn", etc.
    # After stemming, gold and pred keywords should match exactly.
    gold = [["networks", "learning"], ["apple", "bananas"]]
    pred = [["network", "learn"], ["apple", "banana", "cherry"]]

    metrics = compute_metrics_at_k_martinc(gold, pred, k_values=[5])

    # Doc 1: 2 gold, 2 pred, all match -> P=1.0, R=1.0
    # Doc 2: 2 gold, 3 pred, 2 match -> P=2/3=0.667, R=1.0
    # Macro P = (1.0 + 0.667) / 2 = 0.833
    assert_approx_equal(metrics["precision@5"], 0.833, significant=2)
    assert metrics["recall@5"] == 1.0


def test_compute_metrics_at_k_martinc_macro_averaged():
    # Macro-averaging: compute P/R per doc, then average.
    # Doc 1: gold=[a,b], pred=[a,b,x] -> P=2/3, R=2/2=1.0
    # Doc 2: gold=[c], pred=[c,y] -> P=1/2, R=1/1=1.0
    # Macro P = (2/3 + 1/2) / 2 = 7/12 ≈ 0.583
    # Macro R = (1.0 + 1.0) / 2 = 1.0
    gold = [["a", "b"], ["c"]]
    pred = [["a", "b", "x"], ["c", "y"]]

    metrics = compute_metrics_at_k_martinc(gold, pred, k_values=[5])

    # Macro P = (2/3 + 1/2) / 2 = 7/12 ≈ 0.5833
    expected_p = (2 / 3 + 1 / 2) / 2
    assert_approx_equal(metrics["precision@5"], expected_p)
    # Macro R = (1.0 + 1.0) / 2 = 1.0
    assert metrics["recall@5"] == 1.0


def test_compute_metrics_at_k_martinc_m_variant():
    # @M uses len(pred) as cutoff for each doc.
    # Doc 1: gold=[a,b], pred=[a,x] (len=2) -> P=1/2, R=1/2
    # Doc 2: gold=[c,d,e], pred=[c,d,y,z] (len=4) -> P=2/4, R=2/3
    # Macro P = (0.5 + 0.5) / 2 = 0.5
    # Macro R = (0.5 + 2/3) / 2 = 7/12 ≈ 0.583
    gold = [["a", "b"], ["c", "d", "e"]]
    pred = [["a", "x"], ["c", "d", "y", "z"]]

    metrics = compute_metrics_at_k_martinc(gold, pred, k_values=["M"])

    # Macro P = (1/2 + 2/4) / 2 = 0.5
    assert_approx_equal(metrics["precision@M"], 0.5)
    # Macro R = (1/2 + 2/3) / 2 = 7/12 ≈ 0.583
    expected_r = (1 / 2 + 2 / 3) / 2
    assert_approx_equal(metrics["recall@M"], expected_r)


def test_compute_metrics_at_k_martinc_remove_docs_with_no_kw():
    # Doc 2 has empty gold -> should be skipped when remove_docs_with_no_kw=True
    gold = [["a", "b"], [], ["c"]]
    pred = [["a", "x"], ["y", "z"], ["c"]]

    metrics = compute_metrics_at_k_martinc(
        gold, pred, k_values=[5], remove_docs_with_no_kw=True
    )

    # Only docs 1 and 3 are evaluated (doc 2 skipped)
    assert metrics["num_docs_total"] == 3
    assert metrics["num_docs_removed"] == 1
    assert metrics["num_docs_evaluated"] == 2
    # Doc 1: P=1/2, R=1/2; Doc 3: P=1/1, R=1/1
    # Macro P = (0.5 + 1.0) / 2 = 0.75
    expected_p = (1 / 2 + 1 / 1) / 2
    assert_approx_equal(metrics["precision@5"], expected_p)
    # Macro R = (0.5 + 1.0) / 2 = 0.75
    assert_approx_equal(metrics["recall@5"], expected_p)


def test_compute_metrics_at_k_martinc_keep_docs_with_no_kw():
    # When remove_docs_with_no_kw=False, doc 2 is kept (but has 0 gold -> R undefined)
    gold = [["a", "b"], [], ["c"]]
    pred = [["a", "x"], ["y", "z"], ["c"]]

    metrics = compute_metrics_at_k_martinc(
        gold, pred, k_values=[5], remove_docs_with_no_kw=False
    )

    # All 3 docs are kept
    assert metrics["num_docs_total"] == 3
    assert metrics["num_docs_removed"] == 0
    assert metrics["num_docs_evaluated"] == 3


def test_compute_metrics_at_k_martinc_key_naming():
    # Verify all expected keys are present with correct naming convention
    gold = [["a"]]
    pred = [["a", "b"]]

    metrics = compute_metrics_at_k_martinc(gold, pred, k_values=[5, 10, "N", "M"])

    # Check all @k metric keys exist
    assert "precision@5" in metrics
    assert "recall@5" in metrics
    assert "f1@5" in metrics
    assert "precision@10" in metrics
    assert "recall@10" in metrics
    assert "f1@10" in metrics
    assert "precision@N" in metrics
    assert "recall@N" in metrics
    assert "f1@N" in metrics
    assert "precision@M" in metrics
    assert "recall@M" in metrics
    assert "f1@M" in metrics
    # Check doc count metadata keys exist
    assert "num_docs_total" in metrics
    assert "num_docs_removed" in metrics
    assert "num_docs_evaluated" in metrics


def test_compute_metrics_at_k_martinc_hyphen_normalization():
    """Test that hyphenated and space-separated keyphrases match correctly."""
    gold = [["value based"]]
    pred = [["value-based"]]

    metrics = compute_metrics_at_k_martinc(gold, pred, k_values=[5])

    # After hyphen normalization, "value-based" and "value based" should match
    assert metrics["precision@5"] == 1.0
    assert metrics["recall@5"] == 1.0
