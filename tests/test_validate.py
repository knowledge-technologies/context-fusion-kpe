from numpy.testing import assert_approx_equal

from keyword_pecker.trainer.validate import (
    _build_sample,
    _compute_per_doc_metrics,
    compute_metrics_at_k,
    compute_metrics_at_k_martinc,
)


GOLD_TRUTH = [{"car", "bike"}, {"apple", "banana", "cherry"}]
FAIL_PREDICTIONS = [set(), {"dog", "elephant"}]
PERFECT_PREDICTIONS = [{"car", "bike"}, {"apple", "banana", "cherry"}]
PARTIAL_PREDICTIONS = [{"car"}, {"apple"}]


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


def test_normalize_keyphrases_hyphen_and_case():
    """Test that _normalize_keyphrases replaces hyphens with spaces and lowercases."""
    from keyword_pecker.trainer.validate import _normalize_keyphrases

    gold = [["Value Based", "NEURAL NETWORK"]]
    pred = [["value-based", "neural-network"]]

    gold_norm = _normalize_keyphrases(gold)
    pred_norm = _normalize_keyphrases(pred)

    assert gold_norm == [["value based", "neural network"]]
    assert pred_norm == [["value based", "neural network"]]


def test_normalize_keyphrases_empty():
    """Test that _normalize_keyphrases handles empty lists."""
    from keyword_pecker.trainer.validate import _normalize_keyphrases

    assert _normalize_keyphrases([[]]) == [[]]
    assert _normalize_keyphrases([]) == []


def test_compute_confusion_matrix_basic():
    """Test _compute_confusion_matrix with mixed correct/incorrect predictions."""
    from keyword_pecker.trainer.validate import _compute_confusion_matrix

    predictions = [0, 2, 1]
    targets = [0, 1, 2]

    cm = _compute_confusion_matrix(predictions, targets, num_classes=3)

    assert cm == [[1, 0, 0], [0, 0, 1], [0, 1, 0]]


def test_compute_confusion_matrix_perfect():
    """Test _compute_confusion_matrix with perfect predictions (diagonal only)."""
    from keyword_pecker.trainer.validate import _compute_confusion_matrix

    predictions = [0, 1, 2, 0, 1]
    targets = [0, 1, 2, 0, 1]

    cm = _compute_confusion_matrix(predictions, targets, num_classes=3)

    assert cm == [[2, 0, 0], [0, 2, 0], [0, 0, 1]]


def test_build_sample_returns_correct_structure():
    """Test that _build_sample returns the expected dict structure with scores."""
    doc_key = ("doc_1", 42)
    gold_norm = ["apple", "banana"]
    pred_with_scores = [("apple", 0.95), ("cherry", 0.72), ("banana", 0.61)]

    result = _build_sample(doc_key, gold_norm, pred_with_scores)

    assert result["doc_id"] == "doc_1"
    assert result["doc_row"] == 42
    assert result["gold_keywords"] == ["apple", "banana"]
    assert len(result["pred_keywords"]) == 3
    assert result["pred_keywords"][0] == {"keyword": "apple", "score": 0.95}
    assert result["pred_keywords"][1] == {"keyword": "cherry", "score": 0.72}
    assert result["pred_keywords"][2] == {"keyword": "banana", "score": 0.61}


def test_compute_per_doc_metrics_at5_at10():
    """Test that _compute_per_doc_metrics computes @5 and @10 metrics correctly."""
    doc_key = ("doc_1", 0)
    gold_norm = ["car", "bike", "train"]
    pred_with_scores = [
        ("car", 0.95),
        ("bike", 0.85),
        ("plane", 0.70),
        ("boat", 0.60),
        ("train", 0.55),
        ("bus", 0.40),
    ]

    result = _compute_per_doc_metrics(doc_key, gold_norm, pred_with_scores)

    assert result["doc_id"] == "doc_1"
    assert result["num_gold"] == 3
    assert result["num_pred"] == 6

    # @5: top5 = {car, bike, plane, boat, train}, matches = {car, bike, train}
    # tp=3, fp=2, fn=0 -> P=3/5=0.6, R=3/3=1.0, F1=2*3/(2*3+2)=0.75
    assert_approx_equal(result["precision@5"], 0.6)
    assert result["recall@5"] == 1.0
    assert_approx_equal(result["f1@5"], 0.75)

    # @10: same 6 preds, all included
    # tp=3, fp=3, fn=0 -> P=3/6=0.5, R=3/3=1.0, F1=2*3/(2*3+3)=0.667
    assert_approx_equal(result["precision@10"], 0.5)
    assert result["recall@10"] == 1.0
    assert_approx_equal(result["f1@10"], 2 / 3)


def test_compute_per_doc_metrics_matched_flags():
    """Test that matched flags are correctly set in pred_keywords."""
    doc_key = ("doc_1", 0)
    gold_norm = ["apple", "banana"]
    pred_with_scores = [("apple", 0.9), ("cherry", 0.8), ("banana", 0.7)]

    result = _compute_per_doc_metrics(doc_key, gold_norm, pred_with_scores)

    assert result["pred_keywords"][0]["matched"] is True
    assert result["pred_keywords"][1]["matched"] is False
    assert result["pred_keywords"][2]["matched"] is True
