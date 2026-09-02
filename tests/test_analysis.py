"""Tests for recently added analysis functions — numerical correctness and crash behavior."""

import json
import tempfile
from pathlib import Path

import polars as pl
import pytest
from numpy.testing import assert_approx_equal

from keyword_pecker.analysis.per_doc_analysis import (
    _compute_benchmark_stats,
    assign_accuracy_tiers,
    build_keyphrase_stats_table,
    build_performance_stats_table,
)
from keyword_pecker.analysis.tables import compute_dataset_stats


def _make_per_doc(n: int = 5) -> list[dict]:
    """Create a small synthetic per-doc list."""
    return [
        {
            "num_gold": 3,
            "num_pred": 7,
            "precision@5": 0.4,
            "recall@5": 0.5,
            "precision@10": 0.3,
            "recall@10": 0.6,
        }
        for _ in range(n)
    ]


# ---------------------------------------------------------------------------
# 1. _compute_benchmark_stats — correct numerical output
# ---------------------------------------------------------------------------


def test_compute_benchmark_stats_basic():
    per_doc = _make_per_doc(5)
    stats = _compute_benchmark_stats(
        per_doc,
        [
            "gold_mean",
            "gold_median",
            "pred_mean",
            "pred_median",
            "precision_5",
            "recall_5",
            "precision_10",
            "recall_10",
        ],
    )

    assert stats["gold_mean"] == pytest.approx(3.0)
    assert stats["gold_median"] == pytest.approx(3.0)
    assert stats["pred_mean"] == pytest.approx(7.0)
    assert stats["pred_median"] == pytest.approx(7.0)
    assert stats["precision_5"] == pytest.approx(0.4)
    assert stats["recall_5"] == pytest.approx(0.5)
    assert stats["precision_10"] == pytest.approx(0.3)
    assert stats["recall_10"] == pytest.approx(0.6)


def test_compute_benchmark_stats_with_variation():
    """Verify mean vs median differ when values vary."""
    per_doc = [
        {
            "num_gold": 1,
            "num_pred": 5,
            "precision@5": 0.2,
            "recall@5": 0.3,
            "precision@10": 0.1,
            "recall@10": 0.4,
        },
        {
            "num_gold": 5,
            "num_pred": 15,
            "precision@5": 0.6,
            "recall@5": 0.7,
            "precision@10": 0.5,
            "recall@10": 0.8,
        },
    ]
    stats = _compute_benchmark_stats(
        per_doc,
        [
            "gold_mean",
            "gold_median",
            "pred_mean",
            "pred_median",
            "precision_5",
            "recall_5",
            "precision_10",
            "recall_10",
        ],
    )

    assert_approx_equal(stats["gold_mean"], 3.0, significant=3)
    assert stats["gold_median"] == pytest.approx(3.0)
    assert_approx_equal(stats["pred_mean"], 10.0, significant=3)
    assert stats["pred_median"] == pytest.approx(10.0)
    assert_approx_equal(stats["precision_5"], 0.4, significant=3)
    assert_approx_equal(stats["recall_5"], 0.5, significant=3)


# ---------------------------------------------------------------------------
# 2. _compute_benchmark_stats — empty input crashes (expected)
# ---------------------------------------------------------------------------


def test_compute_benchmark_stats_empty_raises():
    with pytest.raises(Exception):
        _compute_benchmark_stats([], ["gold_mean"])


# ---------------------------------------------------------------------------
# 3. build_keyphrase_stats_table — correct values + None handling
# ---------------------------------------------------------------------------


def test_build_keyphrase_stats_table():
    training = {"kp20k": _make_per_doc(3)}
    validation = {
        "kp20k": _make_per_doc(4),
        "kptimes": [
            {
                "num_gold": 2,
                "num_pred": 10,
                "precision@5": 0.0,
                "recall@5": 0.0,
                "precision@10": 0.0,
                "recall@10": 0.0,
            }
            for _ in range(4)
        ],
    }

    df = build_keyphrase_stats_table(training, validation)

    assert df.shape[0] == 3  # 1 training + 2 validation benchmarks
    assert set(df.columns) == {
        "group",
        "benchmark",
        "gold_mean",
        "gold_median",
        "pred_mean",
        "pred_median",
    }

    training_row = df.filter(
        pl.col("group") == "Training Set FT", pl.col("benchmark") == "kp20k"
    )
    assert training_row["gold_mean"][0] == pytest.approx(3.0)
    assert training_row["pred_mean"][0] == pytest.approx(7.0)


def test_build_keyphrase_stats_table_none_inputs():
    """Both None should produce an empty DataFrame, not crash."""
    df = build_keyphrase_stats_table(None, None)
    assert df.shape[0] == 0


def test_build_keyphrase_stats_table_partial_none():
    """Only training provided — should have training rows only."""
    training = {"kp20k": _make_per_doc(2)}
    df = build_keyphrase_stats_table(training, None)
    assert df.shape[0] == 1
    assert df["group"][0] == "Training Set FT"


# ---------------------------------------------------------------------------
# 4. build_performance_stats_table — correct values + None handling
# ---------------------------------------------------------------------------


def test_build_performance_stats_table():
    training = {"kp20k": _make_per_doc(3)}
    validation = {"kp20k": _make_per_doc(4)}

    df = build_performance_stats_table(training, validation)

    assert df.shape[0] == 2
    assert set(df.columns) == {
        "group",
        "benchmark",
        "precision_5",
        "recall_5",
        "precision_10",
        "recall_10",
    }

    training_row = df.filter(
        pl.col("group") == "Training Set FT", pl.col("benchmark") == "kp20k"
    )
    assert_approx_equal(float(training_row["precision_5"][0]), 0.4, significant=3)
    assert_approx_equal(float(training_row["recall_5"][0]), 0.5, significant=3)


def test_build_performance_stats_table_none_inputs():
    df = build_performance_stats_table(None, None)
    assert df.shape[0] == 0


def test_build_performance_stats_table_partial_none():
    validation = {"inspec": _make_per_doc(5)}
    df = build_performance_stats_table(None, validation)
    assert df.shape[0] == 1
    assert df["group"][0] == "Validation Set FT"
    assert df["benchmark"][0] == "inspec"


# ---------------------------------------------------------------------------
# 5. assign_accuracy_tiers — correct tier distribution
# ---------------------------------------------------------------------------


def test_assign_accuracy_tiers():
    """12 docs with known F1 values — verify quartile boundaries."""
    df = pl.DataFrame(
        {
            "f1_5": [0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.45, 0.5, 0.55, 0.6, 0.7, 0.8],
        }
    )

    result = assign_accuracy_tiers(df)

    assert "accuracy_tier" in result.columns
    assert result.shape[0] == 12

    tiers = result["accuracy_tier"].to_list()
    # With 12 values: q25 ≈ 0.2375, q75 ≈ 0.5625
    # low: <= 0.2375 → 0.1, 0.15, 0.2  (3 docs)
    # high: >= 0.5625 → 0.6, 0.7, 0.8  (3 docs)
    # mid: the rest → 6 docs
    assert tiers.count("low") == 3
    assert tiers.count("high") == 3
    assert tiers.count("mid") == 6


def test_assign_accuracy_tiers_uniform_values():
    """All same F1 — all should land in the same tier (both q25==q75)."""
    df = pl.DataFrame({"f1_5": [0.5] * 10})
    result = assign_accuracy_tiers(df)
    tiers = result["accuracy_tier"].to_list()
    # q25 == q75 == 0.5, so f1 >= q75 → "high" for all
    assert all(t == "high" for t in tiers)


# ---------------------------------------------------------------------------
# 6. compute_dataset_stats — correct numerical output
# ---------------------------------------------------------------------------


def _write_jsonl(docs: list[dict]) -> Path:
    """Write a list of dicts to a temporary JSON Lines file."""
    tmp = Path(tempfile.mktemp(suffix=".json"))
    with open(tmp, "w") as f:
        for doc in docs:
            f.write(json.dumps(doc) + "\n")
    return tmp


def test_compute_dataset_stats_basic():
    """Verify all 6 metrics on a small synthetic dataset."""
    docs = [
        {
            "title": "Alpha Beta",
            "abstract": "one two three four five",
            "keywords": "kw1;kw2;kw3",
        },
        {"title": "Gamma", "abstract": "six seven eight", "keywords": "kw4;kw5"},
        {
            "title": "Delta Epsilon",
            "abstract": "nine ten eleven twelve",
            "keywords": "kw6",
        },
    ]
    path = _write_jsonl(docs)
    try:
        stats = compute_dataset_stats(path)

        assert stats["nr_documents"] == 3

        # Abstract word lengths: 5, 3, 4 → mean=4.0, max=5
        assert stats["mean_abstract_words"] == pytest.approx(4.0)
        assert stats["max_abstract_words"] == 5

        # Keyword counts: 3, 2, 1 → mean=2.0
        assert stats["mean_keywords"] == pytest.approx(2.0)

        # Vocabulary: alpha, beta, one, two, three, four, five, gamma, six, seven,
        #             eight, delta, epsilon, nine, ten, eleven, twelve → 17
        assert stats["vocab_size"] == 17

        # Words per keyword: 4.0 / 2.0 = 2.0
        assert stats["words_per_keyword"] == pytest.approx(2.0)
    finally:
        path.unlink()


def test_compute_dataset_stats_keyword_field():
    """Verify 'keyword' (singular) field is handled like 'keywords'."""
    docs = [
        {"title": "T1", "abstract": "a b c", "keyword": "x;y"},
        {"title": "T2", "abstract": "d e", "keyword": "z"},
    ]
    path = _write_jsonl(docs)
    try:
        stats = compute_dataset_stats(path)
        assert stats["mean_keywords"] == pytest.approx(1.5)
    finally:
        path.unlink()


def test_compute_dataset_stats_empty_keywords():
    """Documents with no keywords should not crash."""
    docs = [
        {"title": "T1", "abstract": "a b c", "keywords": ""},
        {"title": "T2", "abstract": "d e f", "keywords": "k1"},
    ]
    path = _write_jsonl(docs)
    try:
        stats = compute_dataset_stats(path)
        assert stats["mean_keywords"] == pytest.approx(0.5)
        # words_per_keyword should not divide by zero
        assert stats["words_per_keyword"] > 0
    finally:
        path.unlink()


def test_compute_dataset_stats_vocab_sample_limit():
    """Vocabulary should only count first vocab_sample documents."""
    words_a = ["alpha", "beta", "gamma", "delta", "epsilon"]
    words_b = [
        "zeta",
        "eta",
        "theta",
        "iota",
        "kappa",
        "lambda",
        "mu",
        "nu",
        "xi",
        "omicron",
        "pi",
        "rho",
        "sigma",
        "tau",
        "upsilon",
    ]
    all_words = words_a + words_b
    docs = [{"title": "", "abstract": w, "keywords": "k"} for w in all_words]
    path = _write_jsonl(docs)
    try:
        stats_full = compute_dataset_stats(path, vocab_sample=20)
        stats_limited = compute_dataset_stats(path, vocab_sample=5)

        assert stats_full["nr_documents"] == 20
        assert stats_limited["nr_documents"] == 20

        # Limited vocab should be smaller (5 unique words vs 20)
        assert stats_limited["vocab_size"] == 5
        assert stats_full["vocab_size"] == 20
    finally:
        path.unlink()
