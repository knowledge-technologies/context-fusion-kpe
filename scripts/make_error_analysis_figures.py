"""Generate error analysis figures from a run's eval JSON.

Mirrors the notebook at notebooks/2026-08-05_young-haze/notebook-error-analysis.py
as a standalone, reproducible script that does not require marimo.

Usage:
    python scripts/make_error_analysis_figures.py
    python scripts/make_error_analysis_figures.py --output-dir ./figures
    python scripts/make_error_analysis_figures.py --no-save
"""

import argparse
import json
from pathlib import Path

import numpy as np
import polars as pl
from dotenv import load_dotenv

from keyword_pecker.analysis import (
    assign_accuracy_tiers,
    build_keyphrase_count_summary,
    build_keyphrase_level_df,
    classify_match_type,
    plot_ecdf_confidence,
    plot_stacked_bar_match_type,
)
from keyword_pecker.utils import get_runs_root

load_dotenv()


# ---------------------------------------------------------------------------
# Constants — all data sources defined explicitly at the top
# ---------------------------------------------------------------------------

RUNS_ROOT = get_runs_root()
YOUNG_HAZE_DIR = RUNS_ROOT / "young-haze-50"
EVAL_PATH = YOUNG_HAZE_DIR / "evals" / "eval_kp20k_test_20260812T122055Z.json"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate error analysis figures from an eval JSON."
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="./figures/error-analysis",
        help="Directory to save output files (default: ./figures/error-analysis)",
    )
    parser.add_argument(
        "--no-save",
        action="store_true",
        help="Do not save output files to disk.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    fig_dir = Path(args.output_dir)
    save = not args.no_save

    if save:
        fig_dir.mkdir(parents=True, exist_ok=True)
        print(f"Output directory: {fig_dir.resolve()}")

    # ------------------------------------------------------------------
    # Load eval data
    # ------------------------------------------------------------------

    print(f"Loading eval data from {EVAL_PATH}")
    with open(EVAL_PATH) as f:
        eval_data = json.load(f)
    per_doc = eval_data["per_doc"]

    # ------------------------------------------------------------------
    # Build per-document DataFrame
    # ------------------------------------------------------------------

    print("Building per-document DataFrame...")
    rows = []
    for d in per_doc:
        gold_kws = d["gold_keywords"]
        pred_kws = d["pred_keywords"]
        num_matched = sum(1 for p in pred_kws if p.get("matched"))
        num_pred = len(pred_kws)
        num_gold = d["num_gold"]
        has_partial = any(
            classify_match_type(p["keyword"], gold_kws, p.get("matched")) == "partial"
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

    # ------------------------------------------------------------------
    # Assign accuracy tiers
    # ------------------------------------------------------------------

    per_doc_df = assign_accuracy_tiers(per_doc_df)

    f1_vals = per_doc_df["f1_5"].to_list()
    q25 = float(np.percentile(f1_vals, 25))
    q75 = float(np.percentile(f1_vals, 75))
    print(f"F1@5 quartiles: Q1={q25:.3f}, Q3={q75:.3f}")

    # ------------------------------------------------------------------
    # Build keyphrase-level DataFrame
    # ------------------------------------------------------------------

    print("Building keyphrase-level DataFrame...")
    keyphrase_df = build_keyphrase_level_df(per_doc, per_doc_df)

    # ------------------------------------------------------------------
    # Count summary
    # ------------------------------------------------------------------

    print("Computing count summary...")
    count_summary_df = build_keyphrase_count_summary(per_doc_df)
    print(count_summary_df)

    # ------------------------------------------------------------------
    # Keyphrase length summary by match type
    # ------------------------------------------------------------------

    print("Keyphrase length summary by match type:")
    length_summary = (
        keyphrase_df.group_by("match_type")
        .agg(
            count=pl.col("word_count").count(),
            mean_words=pl.col("word_count").mean(),
            median_words=pl.col("word_count").median(),
            std_words=pl.col("word_count").std(),
            min_words=pl.col("word_count").min(),
            max_words=pl.col("word_count").max(),
        )
        .sort("match_type")
    )
    print(length_summary)

    # ------------------------------------------------------------------
    # Confidence summary by match type
    # ------------------------------------------------------------------

    print("Confidence summary by match type:")
    conf_summary = (
        keyphrase_df.group_by("match_type")
        .agg(
            count=pl.col("score").count(),
            mean_conf=pl.col("score").mean(),
            median_conf=pl.col("score").median(),
            std_conf=pl.col("score").std(),
            min_conf=pl.col("score").min(),
            max_conf=pl.col("score").max(),
        )
        .sort("match_type")
    )
    print(conf_summary)

    # ------------------------------------------------------------------
    # Generate figures
    # ------------------------------------------------------------------

    print("Plotting ECDF of confidence by match type...")
    plot_ecdf_confidence(keyphrase_df, fig_dir=fig_dir, save=save)

    print("Plotting stacked bar of match type by accuracy tier...")
    plot_stacked_bar_match_type(keyphrase_df, q25, q75, fig_dir=fig_dir, save=save)

    print("Done.")


if __name__ == "__main__":
    main()
