"""Data loading functions for young-haze analysis."""

import json
from pathlib import Path

import polars as pl


def load_seke_row(seke_evals_path: Path) -> pl.DataFrame:
    """Load SEKE baseline metrics from external eval JSON."""
    with open(seke_evals_path) as f:
        seke_data = json.load(f)

    seke_kp20k = seke_data["kp20k"]
    f1_5_str = seke_kp20k["f1@5"]
    f1_10_str = seke_kp20k["f1@10"]
    f1_5 = float(f1_5_str.split("±")[0]) / 100
    f1_10 = float(f1_10_str.split("±")[0]) / 100

    return pl.DataFrame(
        [
            {
                "run": "SEKE (DeBERTA, MOE, BiLSTM)",
                "precision_5": None,
                "precision_10": None,
                "recall_5": None,
                "recall_10": None,
                "f1_5": f1_5,
                "f1_10": f1_10,
                "n_epochs": "20",
            }
        ],
        schema={
            "run": pl.String,
            "precision_5": pl.Float64,
            "precision_10": pl.Float64,
            "recall_5": pl.Float64,
            "recall_10": pl.Float64,
            "f1_5": pl.Float64,
            "f1_10": pl.Float64,
            "n_epochs": pl.String,
        },
    )


def load_young_haze_row(young_haze_dir: Path) -> pl.DataFrame:
    """Load young-haze model metrics from eval JSON."""
    eval_path = young_haze_dir / "evals" / "eval_kp20k_test_20260812T122055Z.json"
    with open(eval_path) as f:
        eval_data = json.load(f)
    m = eval_data["metrics"]

    return pl.DataFrame(
        [
            {
                "run": "Ours (ModernBERT)",
                "precision_5": m["precision@5"],
                "precision_10": m["precision@10"],
                "recall_5": m["recall@5"],
                "recall_10": m["recall@10"],
                "f1_5": m["f1@5"],
                "f1_10": m["f1@10"],
                "n_epochs": "20",
            }
        ]
    )


def load_per_doc_data(young_haze_dir: Path) -> tuple[list[dict], pl.DataFrame]:
    """Load per-document evaluation data with match classification and accuracy tiers."""
    eval_path = young_haze_dir / "evals" / "eval_kp20k_test_20260812T122055Z.json"
    with open(eval_path) as f:
        eval_data = json.load(f)
    per_doc = eval_data["per_doc"]

    rows = []
    for d in per_doc:
        gold_kws = d["gold_keywords"]
        pred_kws = d["pred_keywords"]
        num_matched = sum(1 for p in pred_kws if p.get("matched"))
        num_pred = len(pred_kws)
        num_gold = d["num_gold"]
        has_partial = any(
            _classify_match_type(p["keyword"], gold_kws, p.get("matched")) == "partial"
            for p in pred_kws
        )
        rows.append(
            {
                "doc_row": d["doc_row"],
                "num_gold": num_gold,
                "num_pred": num_pred,
                "precision_5": d["precision@5"],
                "recall_5": d["recall@5"],
                "f1_5": d["f1@5"],
                "precision_10": d["precision@10"],
                "recall_10": d["recall@10"],
                "f1_10": d["f1@10"],
                "gold_keywords": gold_kws,
                "num_matched": num_matched,
                "num_unmatched": num_pred - num_matched,
                "match_rate": num_matched / num_pred if num_pred > 0 else 0.0,
                "f1_gap": d["f1@10"] - d["f1@5"],
                "has_partial_matches": has_partial,
            }
        )

    per_doc_df = pl.DataFrame(
        rows,
        schema_overrides={
            "gold_keywords": pl.List(pl.String),
        },
    )

    return per_doc, per_doc_df


def load_training_metrics(
    young_haze_dir: Path,
) -> tuple[pl.DataFrame, pl.DataFrame, pl.DataFrame]:
    """Load training metrics from metrics.json: all val metrics, raw loss, epoch mean loss."""
    all_metrics = (
        pl.scan_ndjson(young_haze_dir / "metrics.json")
        .filter(pl.col("val_f1@N").is_not_null())
        .select(
            [
                pl.col("global_step"),
                pl.col("val_loss"),
                pl.col("val_f1@5"),
                pl.col("val_f1@10"),
            ]
        )
    ).collect()

    raw_loss = (
        pl.scan_ndjson(young_haze_dir / "metrics.json")
        .filter(pl.col("train_loss").is_not_null())
        .select(["global_step", "train_loss"])
    ).collect()

    epoch_mean_loss = (
        pl.scan_ndjson(young_haze_dir / "metrics.json")
        .filter(pl.col("epoch_mean_loss").is_not_null())
        .select(["epoch", "epoch_mean_loss"])
    ).collect()

    return all_metrics, raw_loss, epoch_mean_loss


def load_all_val_metrics(young_haze_dir: Path) -> pl.DataFrame:
    """Load all validation metrics (precision, recall, f1 at @5 and @10)."""
    return (
        pl.scan_ndjson(young_haze_dir / "metrics.json")
        .filter(pl.col("val_f1@N").is_not_null())
        .select(
            [
                pl.col("global_step"),
                pl.col("val_precision@5"),
                pl.col("val_recall@5"),
                pl.col("val_f1@5"),
                pl.col("val_precision@10"),
                pl.col("val_recall@10"),
                pl.col("val_f1@10"),
            ]
        )
    ).collect()


def _classify_match_type(pred_kw: str, gold_kws: list[str], matched_flag: bool) -> str:
    """Classify a predicted keyword as full, partial, or no match."""
    if matched_flag:
        return "full"
    pred_lower = pred_kw.lower()
    pred_words = set(pred_lower.split())
    for gold in gold_kws:
        gold_lower = gold.lower()
        gold_words = set(gold_lower.split())
        if pred_words & gold_words:
            return "partial"
    return "none"
