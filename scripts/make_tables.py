"""Generate all metrics tables, dataset stats, and analysis figures.

Mirrors the notebook at notebooks/2026-08-05_young-haze/notebook.py as a
standalone, reproducible script that does not require marimo.

Usage:
    python scripts/make_tables.py
    python scripts/make_tables.py --output-dir ./figures
    python scripts/make_tables.py --no-save
"""

import argparse
import json
from pathlib import Path

from dotenv import load_dotenv

from ctx_fusion.analysis import (
    assign_accuracy_tiers,
    build_f1_table_wide,
    build_keyphrase_stats_table,
    build_precision_recall_table,
    compute_dataset_stats,
    load_all_benchmark_per_doc,
    load_all_val_metrics,
    load_external_benchmarks,
    load_per_doc_data,
    load_run_test_row_from_evals,
    load_seke_multi_benchmark,
    load_training_metrics,
    make_dataset_stats_tex,
    make_metrics_table_gt,
    make_metrics_table_tex,
    make_precision_recall_table_tex,
    make_samples_table_tex,
    plot_keyphrase_stats,
    plot_loss_f1_overview,
    plot_loss_overview,
    plot_modernbert_dynamics,
    plot_training_dynamics,
    select_samples,
)
from ctx_fusion.data.paths import get_dataset_folder
from ctx_fusion.data import RUNS_MAPPING_PATH
from ctx_fusion.data.external import (
    BENCHMARKS_EVALS_PATH,
    SEKE_TRAINSET_PATH,
    SEKE_VALSET_PATH,
)
from ctx_fusion.utils import get_runs_root

load_dotenv()


# ---------------------------------------------------------------------------
# Constants — all data sources defined explicitly at the top
# ---------------------------------------------------------------------------

RUNS_ROOT = get_runs_root()

with open(RUNS_MAPPING_PATH) as f:
    MAPPING = json.load(f)

RUN_DIRS = {name: RUNS_ROOT / info["run_id"] for name, info in MAPPING.items()}

DATA_ROOT = get_dataset_folder()
BENCHMARKS = ["kp20k", "kptimes", "inspec"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate metrics tables, dataset stats, and analysis figures."
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="./figures/tables",
        help="Directory to save output files (default: ./figures/tables)",
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
    # Load SEKE baseline data
    # ------------------------------------------------------------------

    seke_trainset_df = load_seke_multi_benchmark(SEKE_TRAINSET_PATH)
    seke_training = {}
    for row in seke_trainset_df.iter_rows(named=True):
        bm = row["benchmark"]
        seke_training.setdefault("SEKE (Martinc et al, 2025)", {})[bm] = {
            "f1_5": row["f1_5"],
            "f1_10": row["f1_10"],
        }

    seke_valset_df = load_seke_multi_benchmark(SEKE_VALSET_PATH)
    seke_validation = {}
    for row in seke_valset_df.iter_rows(named=True):
        bm = row["benchmark"]
        seke_validation.setdefault("SEKE (Martinc et al, 2025)", {})[bm] = {
            "f1_5": row["f1_5"],
            "f1_10": row["f1_10"],
        }

    # ------------------------------------------------------------------
    # Load run results
    # ------------------------------------------------------------------

    yh_row = load_run_test_row_from_evals(
        RUN_DIRS["kp20k_training_set"], MAPPING["kp20k_training_set"]["eval_file"]
    )
    training_set_results = {
        "SEKE (Martinc et al, 2025)": seke_training.get(
            "SEKE (Martinc et al, 2025)", {}
        ),
        "Ours": {
            yh_row["benchmark"]: {
                "f1_5": yh_row["f1_5"],
                "f1_10": yh_row["f1_10"],
                "precision_5": yh_row["precision_5"],
                "recall_5": yh_row["recall_5"],
                "precision_10": yh_row["precision_10"],
                "recall_10": yh_row["recall_10"],
            },
        },
    }

    cs_row = load_run_test_row_from_evals(
        RUN_DIRS["kp20k_validation_set"], MAPPING["kp20k_validation_set"]["eval_file"]
    )
    do_row = load_run_test_row_from_evals(
        RUN_DIRS["kptimes_validation_set"],
        MAPPING["kptimes_validation_set"]["eval_file"],
    )
    sb_row = load_run_test_row_from_evals(
        RUN_DIRS["inspec_validation_set"], MAPPING["inspec_validation_set"]["eval_file"]
    )
    validation_set_results = {
        "SEKE (Martinc et al, 2025)": seke_validation.get(
            "SEKE (Martinc et al, 2025)", {}
        ),
        "Ours": {
            cs_row["benchmark"]: {
                "f1_5": cs_row["f1_5"],
                "f1_10": cs_row["f1_10"],
                "precision_5": cs_row["precision_5"],
                "recall_5": cs_row["recall_5"],
                "precision_10": cs_row["precision_10"],
                "recall_10": cs_row["recall_10"],
            },
            do_row["benchmark"]: {
                "f1_5": do_row["f1_5"],
                "f1_10": do_row["f1_10"],
                "precision_5": do_row["precision_5"],
                "recall_5": do_row["recall_5"],
                "precision_10": do_row["precision_10"],
                "recall_10": do_row["recall_10"],
            },
            sb_row["benchmark"]: {
                "f1_5": sb_row["f1_5"],
                "f1_10": sb_row["f1_10"],
                "precision_5": sb_row["precision_5"],
                "recall_5": sb_row["recall_5"],
                "precision_10": sb_row["precision_10"],
                "recall_10": sb_row["recall_10"],
            },
        },
    }

    external_rows = load_external_benchmarks(BENCHMARKS_EVALS_PATH)

    # ------------------------------------------------------------------
    # Build tables
    # ------------------------------------------------------------------

    wide_df = build_f1_table_wide(training_set_results, validation_set_results)
    performance_stats = build_precision_recall_table(
        training_set_results, validation_set_results
    )

    per_doc, per_doc_df_raw = load_per_doc_data(
        RUN_DIRS["kp20k_training_set"], MAPPING["kp20k_training_set"]["eval_file"]
    )
    per_doc_df = assign_accuracy_tiers(per_doc_df_raw)
    samples = select_samples(per_doc_df, per_doc)

    training_per_doc = load_all_benchmark_per_doc(
        RUN_DIRS["kp20k_training_set"], MAPPING["kp20k_training_set"]["eval_file"]
    )
    validation_per_doc = {}
    for name in [
        "kp20k_validation_set",
        "kptimes_validation_set",
        "inspec_validation_set",
    ]:
        validation_per_doc.update(
            load_all_benchmark_per_doc(RUN_DIRS[name], MAPPING[name]["eval_file"])
        )

    keyphrase_stats = build_keyphrase_stats_table(training_per_doc, validation_per_doc)

    # ------------------------------------------------------------------
    # Generate and save metrics tables
    # ------------------------------------------------------------------

    print("Generating metrics table (Great Tables)...")
    make_metrics_table_gt(
        wide_df, fig_dir=fig_dir, save=save, external_rows=external_rows
    )

    print("Generating metrics table (LaTeX)...")
    make_metrics_table_tex(
        wide_df, fig_dir=fig_dir, save=save, external_rows=external_rows
    )

    # ------------------------------------------------------------------
    # Dataset statistics
    # ------------------------------------------------------------------

    dataset_stats = {}
    for bm in BENCHMARKS:
        json_path = DATA_ROOT / bm / f"{bm}_valid.json"
        parquet_path = DATA_ROOT / bm / "derived" / "valid.parquet"
        if not json_path.exists() or not parquet_path.exists():
            print(f"⚠ Dataset files missing for {bm} — skipping dataset stats")
            break
        dataset_stats[bm] = compute_dataset_stats(json_path, parquet_path)

    if dataset_stats:
        print("Computing dataset statistics...")
        make_dataset_stats_tex(dataset_stats, fig_dir=fig_dir, save=save)
    else:
        print("⚠ Skipping dataset stats (no dataset files found)")

    # ------------------------------------------------------------------
    # Precision/recall table
    # ------------------------------------------------------------------

    print("Generating precision/recall table (LaTeX)...")
    make_precision_recall_table_tex(performance_stats, fig_dir=fig_dir, save=save)

    # ------------------------------------------------------------------
    # Sample predictions table
    # ------------------------------------------------------------------

    print("Generating samples table (LaTeX)...")
    make_samples_table_tex(samples, fig_dir=fig_dir, save=save)

    # ------------------------------------------------------------------
    # Keyphrase statistics figure
    # ------------------------------------------------------------------

    print("Plotting keyphrase statistics...")
    plot_keyphrase_stats(keyphrase_stats, fig_dir=fig_dir, save=save)

    # ------------------------------------------------------------------
    # Training dynamics
    # ------------------------------------------------------------------

    print("Loading training metrics...")
    yh_all_metrics, yh_raw_loss, yh_epoch_mean_loss = load_training_metrics(
        RUN_DIRS["kp20k_training_set"]
    )

    print("Plotting loss overview...")
    plot_loss_overview(
        yh_all_metrics, yh_raw_loss, yh_epoch_mean_loss, fig_dir=fig_dir, save=save
    )

    print("Plotting training dynamics...")
    plot_training_dynamics(yh_all_metrics, fig_dir=fig_dir, save=save)

    print("Plotting loss + F1 overview...")
    plot_loss_f1_overview(
        yh_all_metrics, yh_raw_loss, yh_epoch_mean_loss, fig_dir=fig_dir, save=save
    )

    # ------------------------------------------------------------------
    # ModernBERT detailed dynamics
    # ------------------------------------------------------------------

    print("Loading validation metrics...")
    yh_all_val_metrics = load_all_val_metrics(RUN_DIRS["kp20k_training_set"])

    print("Plotting ModernBERT dynamics...")
    plot_modernbert_dynamics(yh_all_val_metrics, fig_dir=fig_dir, save=save)

    print("Done.")


if __name__ == "__main__":
    main()
