"""Reproduce manuscript artifacts from run folders and bundled data.

Usage:
    reproduce-manuscript-artifacts
    reproduce-manuscript-artifacts --output-dir ./manuscript-artifacts
"""

import argparse
import json
import sys
from pathlib import Path

from dotenv import load_dotenv

from keyword_pecker.data import RUNS_MAPPING_PATH
from keyword_pecker.data.external import (
    BENCHMARKS_EVALS_PATH,
    SEKE_TRAINSET_PATH,
    SEKE_VALSET_PATH,
)
from keyword_pecker.utils import get_runs_root

load_dotenv()


def load_runs_mapping() -> dict:
    with open(RUNS_MAPPING_PATH) as f:
        return json.load(f)


def check_run_folder(
    run_id: str, eval_file: str | None = None, requires_metrics: bool = False
) -> list[str]:
    run_dir = get_runs_root() / run_id
    issues = []

    if not run_dir.exists():
        return [f"Run folder not found: {run_dir}"]

    config_path = run_dir / "config.json"
    if not config_path.exists():
        issues.append(f"Missing config.json in {run_dir}")

    if eval_file is not None:
        eval_path = run_dir / "evals" / eval_file
        if not eval_path.exists():
            issues.append(f"Missing eval file: {eval_file} in {run_dir / 'evals'}")
    else:
        eval_files = list((run_dir / "evals").glob("eval_*_test_*.json"))
        if not eval_files:
            issues.append(f"No eval test files found in {run_dir / 'evals'}")

    if requires_metrics:
        metrics_path = run_dir / "metrics.json"
        if not metrics_path.exists():
            issues.append(f"Missing metrics.json in {run_dir}")

    return issues


def check_external_data() -> list[str]:
    issues = []
    for path, name in [
        (BENCHMARKS_EVALS_PATH, "benchmarks_evals.json"),
        (SEKE_TRAINSET_PATH, "seke_evals_trainset.json"),
        (SEKE_VALSET_PATH, "seke_evals_valset.json"),
    ]:
        if not path.exists():
            issues.append(f"Bundled data missing: {name}")
    return issues


def check_dataset_files(benchmarks: list[str]) -> tuple[bool, list[str]]:
    from keyword_pecker.data.paths import get_dataset_folder

    data_root = get_dataset_folder()
    missing = []
    for bm in benchmarks:
        json_path = data_root / bm / f"{bm}_valid.json"
        parquet_path = data_root / bm / "derived" / "valid.parquet"
        if not json_path.exists():
            missing.append(f"{bm}_valid.json")
        if not parquet_path.exists():
            missing.append(f"{bm}/derived/valid.parquet")
    return len(missing) == 0, missing


def validate_prerequisites() -> bool:
    mapping = load_runs_mapping()
    all_ok = True

    print("Checking run folders...")
    for name, info in mapping.items():
        run_id = info["run_id"]
        eval_file = info.get("eval_file")
        requires_metrics = name == "kp20k_training_set"
        issues = check_run_folder(
            run_id, eval_file=eval_file, requires_metrics=requires_metrics
        )
        if issues:
            all_ok = False
            for issue in issues:
                print(f"  ✗ {issue}")
        else:
            print(f"  ✓ {run_id}")

    print("\nChecking external data...")
    external_issues = check_external_data()
    if external_issues:
        all_ok = False
        for issue in external_issues:
            print(f"  ✗ {issue}")
    else:
        print("  ✓ benchmarks_evals.json (bundled)")
        print("  ✓ seke_evals_trainset.json (bundled)")
        print("  ✓ seke_evals_valset.json (bundled)")

    print("\nChecking optional data...")
    has_datasets, missing = check_dataset_files(["kp20k", "kptimes", "inspec"])
    if has_datasets:
        print("  ✓ Dataset files found — dataset stats will be generated")
    else:
        print(
            f"  ⚠ Dataset files missing ({', '.join(missing)}) — dataset stats will be skipped"
        )

    return all_ok


def run_make_tables(fig_dir: Path, save: bool) -> None:
    from keyword_pecker.analysis import (
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
    from keyword_pecker.data.paths import get_dataset_folder

    runs_root = get_runs_root()
    mapping = load_runs_mapping()
    run_dirs = {name: runs_root / info["run_id"] for name, info in mapping.items()}
    yh_dir = run_dirs["kp20k_training_set"]
    cs_dir = run_dirs["kp20k_validation_set"]
    do_dir = run_dirs["kptimes_validation_set"]
    sb_dir = run_dirs["inspec_validation_set"]
    data_root = get_dataset_folder()
    benchmarks = ["kp20k", "kptimes", "inspec"]

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

    yh_row = load_run_test_row_from_evals(
        yh_dir, mapping["kp20k_training_set"]["eval_file"]
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
        cs_dir, mapping["kp20k_validation_set"]["eval_file"]
    )
    do_row = load_run_test_row_from_evals(
        do_dir, mapping["kptimes_validation_set"]["eval_file"]
    )
    sb_row = load_run_test_row_from_evals(
        sb_dir, mapping["inspec_validation_set"]["eval_file"]
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

    wide_df = build_f1_table_wide(training_set_results, validation_set_results)
    performance_stats = build_precision_recall_table(
        training_set_results, validation_set_results
    )

    per_doc, per_doc_df_raw = load_per_doc_data(
        yh_dir, mapping["kp20k_training_set"]["eval_file"]
    )
    per_doc_df = assign_accuracy_tiers(per_doc_df_raw)
    samples = select_samples(per_doc_df, per_doc)

    training_per_doc = load_all_benchmark_per_doc(
        yh_dir, mapping["kp20k_training_set"]["eval_file"]
    )
    validation_per_doc = {}
    for name in [
        "kp20k_validation_set",
        "kptimes_validation_set",
        "inspec_validation_set",
    ]:
        validation_per_doc.update(
            load_all_benchmark_per_doc(run_dirs[name], mapping[name]["eval_file"])
        )

    keyphrase_stats = build_keyphrase_stats_table(training_per_doc, validation_per_doc)

    print("  ✓ Metrics table (Great Tables)")
    make_metrics_table_gt(
        wide_df, fig_dir=fig_dir, save=save, external_rows=external_rows
    )

    print("  ✓ Metrics table (LaTeX)")
    make_metrics_table_tex(
        wide_df, fig_dir=fig_dir, save=save, external_rows=external_rows
    )

    dataset_stats = {}
    for bm in benchmarks:
        json_path = data_root / bm / f"{bm}_valid.json"
        parquet_path = data_root / bm / "derived" / "valid.parquet"
        if not json_path.exists() or not parquet_path.exists():
            break
        dataset_stats[bm] = compute_dataset_stats(json_path, parquet_path)

    if dataset_stats:
        print("  ✓ Dataset stats (LaTeX)")
        make_dataset_stats_tex(dataset_stats, fig_dir=fig_dir, save=save)
    else:
        print("  ⚠ Dataset stats (skipped)")

    print("  ✓ Precision/recall table (LaTeX)")
    make_precision_recall_table_tex(performance_stats, fig_dir=fig_dir, save=save)

    print("  ✓ Samples table (LaTeX)")
    make_samples_table_tex(samples, fig_dir=fig_dir, save=save)

    print("  ✓ Keyphrase statistics")
    plot_keyphrase_stats(keyphrase_stats, fig_dir=fig_dir, save=save)

    print("  ✓ Loss overview")
    yh_all_metrics, yh_raw_loss, yh_epoch_mean_loss = load_training_metrics(yh_dir)
    plot_loss_overview(
        yh_all_metrics, yh_raw_loss, yh_epoch_mean_loss, fig_dir=fig_dir, save=save
    )

    print("  ✓ Training dynamics")
    plot_training_dynamics(yh_all_metrics, fig_dir=fig_dir, save=save)

    print("  ✓ Loss + F1 overview")
    plot_loss_f1_overview(
        yh_all_metrics, yh_raw_loss, yh_epoch_mean_loss, fig_dir=fig_dir, save=save
    )

    print("  ✓ ModernBERT dynamics")
    yh_all_val_metrics = load_all_val_metrics(yh_dir)
    plot_modernbert_dynamics(yh_all_val_metrics, fig_dir=fig_dir, save=save)


def run_error_analysis(fig_dir: Path, save: bool) -> None:
    import numpy as np
    import polars as pl

    from keyword_pecker.analysis import (
        assign_accuracy_tiers,
        build_keyphrase_level_df,
        classify_match_type,
        plot_ecdf_confidence,
        plot_stacked_bar_match_type,
    )

    runs_root = get_runs_root()
    mapping = load_runs_mapping()
    yh_dir = runs_root / mapping["kp20k_training_set"]["run_id"]
    eval_file = mapping["kp20k_training_set"]["eval_file"]

    eval_path = yh_dir / "evals" / eval_file
    if not eval_path.exists():
        raise FileNotFoundError(f"Missing eval file: {eval_path}")

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
        rows, schema_overrides={"gold_keywords": pl.List(pl.String)}
    )
    per_doc_df = assign_accuracy_tiers(per_doc_df)

    f1_vals = per_doc_df["f1_5"].to_list()
    q25 = float(np.percentile(f1_vals, 25))
    q75 = float(np.percentile(f1_vals, 75))

    keyphrase_df = build_keyphrase_level_df(per_doc, per_doc_df)

    print("  ✓ Error analysis: ECDF")
    plot_ecdf_confidence(keyphrase_df, fig_dir=fig_dir, save=save)

    print("  ✓ Error analysis: stacked bar")
    plot_stacked_bar_match_type(keyphrase_df, q25, q75, fig_dir=fig_dir, save=save)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Reproduce all manuscript tables and figures."
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="./manuscript-artifacts",
        help="Directory to save output files (default: ./manuscript-artifacts)",
    )
    parser.add_argument(
        "--no-save",
        action="store_true",
        help="Do not save output files to disk.",
    )
    args = parser.parse_args()

    fig_dir = Path(args.output_dir)
    save = not args.no_save

    if not validate_prerequisites():
        print("\n✗ Prerequisites not met. Please fix the issues above and try again.")
        sys.exit(1)

    if save:
        fig_dir.mkdir(parents=True, exist_ok=True)
        print(f"\nGenerating artifacts in {fig_dir.resolve()}...")
    else:
        print("\nGenerating artifacts (dry run)...")

    print("\nGenerating tables...")
    run_make_tables(fig_dir, save)

    print("\nGenerating error analysis figures...")
    run_error_analysis(fig_dir, save)

    print("\nDone.")


if __name__ == "__main__":
    main()
