"""Tests for recently added analysis functions — numerical correctness and crash behavior."""

import polars as pl
import pytest
from numpy.testing import assert_approx_equal

from keyword_pecker.analysis.per_doc_analysis import (
    _compute_benchmark_stats,
    assign_accuracy_tiers,
    build_keyphrase_stats_table,
    build_performance_stats_table,
)


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
