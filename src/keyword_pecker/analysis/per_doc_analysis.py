"""Per-document analysis: tier assignment, sample selection, and keyword matching."""

import random

import polars as pl


def assign_accuracy_tiers(per_doc_df: pl.DataFrame) -> pl.DataFrame:
    """Assign accuracy tiers (high/mid/low) based on F1@5 quartiles."""
    import numpy as np

    f1_vals = per_doc_df["f1_5"].to_list()
    q25, q75 = float(np.percentile(f1_vals, 25)), float(np.percentile(f1_vals, 75))

    def tier(f1):
        if f1 >= q75:
            return "high"
        if f1 <= q25:
            return "low"
        return "mid"

    return per_doc_df.with_columns(
        pl.col("f1_5").map_elements(tier, return_dtype=pl.String).alias("accuracy_tier")
    )


def select_samples(
    per_doc_df: pl.DataFrame,
    per_doc: list[dict],
    seed: int = 42,
) -> list[dict]:
    """Select sample documents from each accuracy tier for display."""
    rng = random.Random(seed)
    per_doc_lookup = {d["doc_row"]: d for d in per_doc}

    high = per_doc_df.filter(pl.col("accuracy_tier") == "high")
    low = per_doc_df.filter(pl.col("accuracy_tier") == "low")
    mid_partial = per_doc_df.filter(
        (pl.col("accuracy_tier") == "mid") & (pl.col("has_partial_matches"))
    )

    def pick(df, n):
        idxs = rng.sample(range(len(df)), min(n, len(df)))
        return df[idxs]

    chosen_high = pick(high, 2).sort("f1_10", descending=True)
    chosen_low = pick(low, 2).sort("f1_10", descending=True)
    chosen_mid = pick(mid_partial, 2).sort("f1_10", descending=True)

    def build_pred_list(pred_keywords, gold_kws):
        result = []
        for p in pred_keywords:
            matched = p.get("matched", False)
            if matched:
                mt = "full"
            else:
                pred_lower = p["keyword"].lower()
                pred_words = set(pred_lower.split())
                mt = "none"
                for g in gold_kws:
                    g_lower = g.lower()
                    gold_words = set(g_lower.split())
                    if pred_words & gold_words:
                        mt = "partial"
                        break
            result.append(
                {"keyword": p["keyword"], "score": p["score"], "match_type": mt}
            )
        return result

    samples = []
    idx = 1
    for tier_label, df in [
        ("High", chosen_high),
        ("Mixed", chosen_mid),
        ("Low", chosen_low),
    ]:
        for row in df.iter_rows(named=True):
            raw_doc = per_doc_lookup[row["doc_row"]]
            samples.append(
                {
                    "idx": idx,
                    "tier": tier_label,
                    "doc_row": row["doc_row"],
                    "f1_10_pct": f"{row['f1_10'] * 100:.1f}",
                    "gold_keywords": row["gold_keywords"],
                    "pred_keywords": build_pred_list(
                        raw_doc["pred_keywords"], row["gold_keywords"]
                    ),
                }
            )
            idx += 1

    return samples


def make_comparison_df(
    seke_row: pl.DataFrame, young_haze_row: pl.DataFrame
) -> pl.DataFrame:
    """Concatenate SEKE and young-haze rows into comparison DataFrame."""
    return pl.concat([seke_row, young_haze_row])
