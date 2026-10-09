"""Tests for recently added analysis functions — numerical correctness and crash behavior."""

import json
import tempfile
from pathlib import Path

import polars as pl
import pytest
from numpy.testing import assert_approx_equal

from ctx_fusion.analysis.per_doc_analysis import (
    _compute_benchmark_stats,
    assign_accuracy_tiers,
    build_keyphrase_stats_table,
)
from ctx_fusion.analysis.tables import (
    build_f1_table_wide,
    build_precision_recall_table,
    compute_dataset_stats,
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
# 4. build_f1_table_wide — correct values + None handling
# ---------------------------------------------------------------------------


def test_build_f1_table_wide():
    training = {
        "SEKE": {"kp20k": {"f1_5": 0.30, "f1_10": 0.25}},
        "Ours": {"kp20k": {"f1_5": 0.40, "f1_10": 0.35}},
    }
    validation = {
        "SEKE": {"kp20k": {"f1_5": 0.28, "f1_10": 0.23}},
        "Ours": {"kp20k": {"f1_5": 0.38, "f1_10": 0.33}},
    }

    df = build_f1_table_wide(training, validation, benchmarks=["kp20k"])

    assert df.shape[0] == 4  # 2 models × 2 groups
    assert set(df.columns) == {"model", "group", "kp20k_f1_5", "kp20k_f1_10"}

    ours_train = df.filter(
        pl.col("model") == "Ours", pl.col("group") == "Training Set FT"
    )
    assert ours_train["kp20k_f1_5"][0] == pytest.approx(0.40)
    assert ours_train["kp20k_f1_10"][0] == pytest.approx(0.35)


def test_build_f1_table_wide_empty():
    df = build_f1_table_wide({}, {})
    assert df.shape[0] == 0


# ---------------------------------------------------------------------------
# 5. build_precision_recall_table — correct values + F1 consistency
# ---------------------------------------------------------------------------


def test_build_precision_recall_table():
    training = {
        "Ours": {
            "kp20k": {
                "precision_5": 0.34,
                "recall_5": 0.46,
                "precision_10": 0.28,
                "recall_10": 0.52,
            }
        },
    }
    validation = {
        "Ours": {
            "kp20k": {
                "precision_5": 0.26,
                "recall_5": 0.50,
                "precision_10": 0.22,
                "recall_10": 0.55,
            }
        },
    }

    df = build_precision_recall_table(training, validation, benchmarks=["kp20k"])

    assert df.shape[0] == 2  # 1 model × 2 groups
    assert set(df.columns) == {
        "group",
        "benchmark",
        "precision_5",
        "recall_5",
        "precision_10",
        "recall_10",
    }

    train_row = df.filter(
        pl.col("group") == "Training Set FT", pl.col("benchmark") == "kp20k"
    )
    assert_approx_equal(float(train_row["precision_5"][0]), 0.34, significant=3)
    assert_approx_equal(float(train_row["recall_5"][0]), 0.46, significant=3)


def test_build_precision_recall_table_empty():
    df = build_precision_recall_table({}, {})
    assert df.shape[0] == 0


def test_f1_pr_consistency():
    """Verify that F1 ≈ 2*P*R/(P+R) when using the same aggregate source."""
    # Use consistent values: F1 = 2*P*R/(P+R)
    # P=0.341, R=0.462 → F1 = 2*0.341*0.462/(0.341+0.462) ≈ 0.392
    training = {
        "Ours": {
            "kp20k": {
                "f1_5": 0.392,
                "f1_10": 0.336,
                "precision_5": 0.341,
                "recall_5": 0.462,
                "precision_10": 0.280,
                "recall_10": 0.420,
            }
        },
    }
    validation = {}

    f1_df = build_f1_table_wide(training, validation, benchmarks=["kp20k"])
    pr_df = build_precision_recall_table(training, validation, benchmarks=["kp20k"])

    # Recompute F1 from P and R (grab training row which has values)
    train_row = pr_df.filter(pl.col("group") == "Training Set FT")
    p5 = float(train_row["precision_5"][0])
    r5 = float(train_row["recall_5"][0])
    recomputed_f1_5 = 2 * p5 * r5 / (p5 + r5)

    reported_f1_5 = float(
        f1_df.filter(pl.col("group") == "Training Set FT")["kp20k_f1_5"][0]
    )

    # Should be consistent within rounding tolerance
    assert_approx_equal(recomputed_f1_5, reported_f1_5, significant=2)


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


def _write_parquet(docs: list[dict]) -> Path:
    """Write a list of document dicts to a temporary Parquet file with sentence-level rows."""
    tmp = Path(tempfile.mktemp(suffix=".parquet"))
    rows = []
    for i, doc in enumerate(docs):
        kws = doc.get("keywords", "")
        if isinstance(kws, str):
            kw_list = [k.strip() for k in kws.split(";") if k.strip()]
        else:
            kw_list = kws
        # Simulate one sentence per doc with keywords present
        rows.append(
            {
                "doc_id": f"doc_{i}",
                "doc_row": i,
                "sent_num": 0,
                "sent_txt": doc.get("abstract", ""),
                "keywords": kw_list,
                "sent_keywords": kw_list,  # all keywords present in this simplified case
                "token_ids": [1, 2, 3],
                "attention_mask": [1, 1, 1],
                "keyword_mask": [0, 0, 0],
                "word_ids": [0, 1, 2],
                "word_list": doc.get("abstract", "").split(),
            }
        )
    pl.DataFrame(rows).write_parquet(tmp)
    return tmp


def test_compute_dataset_stats_basic():
    """Verify all metrics on a small synthetic dataset."""
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
    json_path = _write_jsonl(docs)
    parquet_path = _write_parquet(docs)
    try:
        stats = compute_dataset_stats(json_path, parquet_path)

        assert stats["nr_documents"] == 3

        # Abstract word lengths: 5, 3, 4 → mean=4.0, max=5
        assert stats["mean_abstract_words"] == pytest.approx(4.0)
        assert stats["max_abstract_words"] == 5

        # Keyword counts: 3, 2, 1 → mean=2.0 (all and present same in simplified parquet)
        assert stats["mean_keywords_all"] == pytest.approx(2.0)
        assert stats["mean_keywords_present"] == pytest.approx(2.0)

        # Vocabulary: alpha, beta, one, two, three, four, five, gamma, six, seven,
        #             eight, delta, epsilon, nine, ten, eleven, twelve → 17
        assert stats["vocab_size"] == 17

        # Words per keyword: each keyword is 1 word → 1.0
        assert stats["words_per_keyword_all"] == pytest.approx(1.0)
        assert stats["words_per_keyword_present"] == pytest.approx(1.0)
    finally:
        json_path.unlink()
        parquet_path.unlink()


def test_compute_dataset_stats_keyword_field():
    """Verify parquet keywords are counted correctly."""
    docs = [
        {"title": "T1", "abstract": "a b c", "keywords": "x;y"},
        {"title": "T2", "abstract": "d e", "keywords": "z"},
    ]
    json_path = _write_jsonl(docs)
    parquet_path = _write_parquet(docs)
    try:
        stats = compute_dataset_stats(json_path, parquet_path)
        assert stats["mean_keywords_all"] == pytest.approx(1.5)
        assert stats["mean_keywords_present"] == pytest.approx(1.5)
    finally:
        json_path.unlink()
        parquet_path.unlink()


def test_compute_dataset_stats_empty_keywords():
    """Documents with no keywords should not crash."""
    docs = [
        {"title": "T1", "abstract": "a b c", "keywords": ""},
        {"title": "T2", "abstract": "d e f", "keywords": "k1"},
    ]
    json_path = _write_jsonl(docs)
    parquet_path = _write_parquet(docs)
    try:
        stats = compute_dataset_stats(json_path, parquet_path)
        assert stats["mean_keywords_all"] == pytest.approx(0.5)
        assert stats["mean_keywords_present"] == pytest.approx(0.5)
        # words_per_keyword should not divide by zero
        assert stats["words_per_keyword_all"] > 0
        assert stats["words_per_keyword_present"] > 0
    finally:
        json_path.unlink()
        parquet_path.unlink()


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
    json_path = _write_jsonl(docs)
    parquet_path = _write_parquet(docs)
    try:
        stats_full = compute_dataset_stats(json_path, parquet_path, vocab_sample=20)
        stats_limited = compute_dataset_stats(json_path, parquet_path, vocab_sample=5)

        assert stats_full["nr_documents"] == 20
        assert stats_limited["nr_documents"] == 20

        # Limited vocab should be smaller (5 unique words vs 20)
        assert stats_limited["vocab_size"] == 5
        assert stats_full["vocab_size"] == 20
    finally:
        json_path.unlink()
        parquet_path.unlink()
