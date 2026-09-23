"""Data loading functions for keyword extraction analysis."""

import json
from pathlib import Path

import polars as pl


BENCHMARKS = ["kp20k", "kptimes", "inspec"]

MODEL_NAME_MAP = {
    "TextRank": "TextRank (unsupervised)",
    "TNT-KID": "TNT-KID (supervised)",
    "Gemini1.5-Flash-mini-8B": "Gemini1.5-Flash-mini-8B (LLM)",
}


def load_run_test_row(run_dir: Path) -> dict:
    """Load test metrics from a single run directory.

    Returns dict with keys: benchmark, f1_5, f1_10
    """
    config_path = run_dir / "config.json"
    with open(config_path) as f:
        config = json.load(f)

    benchmark = config.get("dataset", config.get("benchmark", "unknown"))

    metrics_path = run_dir / "metrics.json"
    with open(metrics_path) as f:
        lines = f.readlines()
    last_line = lines[-1].strip()
    metrics = json.loads(last_line)

    return {
        "benchmark": benchmark,
        "f1_5": metrics.get("test_f1@5"),
        "f1_10": metrics.get("test_f1@10"),
    }


def load_seke_multi_benchmark(
    seke_evals_path: Path,
    benchmarks: list[str] | None = None,
) -> pl.DataFrame:
    """Load SEKE baseline metrics for multiple benchmarks from external eval JSON.

    Returns DataFrame with columns: benchmark, f1_5, f1_10
    """
    if benchmarks is None:
        benchmarks = BENCHMARKS

    with open(seke_evals_path) as f:
        seke_data = json.load(f)

    bm_section = seke_data.get("benchmarks", seke_data)

    bm_lookup = {k.lower(): k for k in bm_section}

    rows = []
    for bm in benchmarks:
        key = bm_lookup.get(bm.lower())
        if key is None:
            continue
        bm_data = bm_section[key]
        f1_5_entry = bm_data.get("F1@5")
        f1_10_entry = bm_data.get("F1@10")
        f1_5 = f1_5_entry["mean"] / 100 if f1_5_entry else None
        f1_10 = f1_10_entry["mean"] / 100 if f1_10_entry else None
        rows.append({"benchmark": bm, "f1_5": f1_5, "f1_10": f1_10})

    return pl.DataFrame(rows)


def load_run_test_row_from_evals(run_dir: Path) -> dict:
    """Load test metrics from a run's evals directory.

    Finds eval_*_test_*.json in run_dir/evals/, extracts precision, recall,
    and F1 at @5 and @10.

    Returns dict with keys: benchmark, precision_5, recall_5, f1_5,
        precision_10, recall_10, f1_10
    """
    config_path = run_dir / "config.json"
    with open(config_path) as f:
        config = json.load(f)

    benchmark = config.get("dataset", config.get("benchmark", "unknown"))

    eval_files = list((run_dir / "evals").glob("eval_*_test_*.json"))
    if not eval_files:
        raise FileNotFoundError(f"No eval test files found in {run_dir / 'evals'}")

    eval_files.sort(reverse=True)
    eval_path = eval_files[0]
    with open(eval_path) as f:
        eval_data = json.load(f)

    metrics = eval_data.get("metrics", {})
    return {
        "benchmark": benchmark,
        "precision_5": metrics.get("precision@5"),
        "recall_5": metrics.get("recall@5"),
        "f1_5": metrics.get("f1@5"),
        "precision_10": metrics.get("precision@10"),
        "recall_10": metrics.get("recall@10"),
        "f1_10": metrics.get("f1@10"),
    }


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
                "run": "Ours",
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


def classify_match_type(pred_kw: str, gold_kws: list[str], matched_flag: bool) -> str:
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


def build_keyphrase_level_df(
    per_doc: list[dict],
    per_doc_df: pl.DataFrame,
) -> pl.DataFrame:
    """Build keyphrase-level DataFrame with match_type, joined with accuracy tiers.

    Args:
        per_doc: List of per-document eval dicts from eval JSON.
        per_doc_df: Per-document DataFrame (must have doc_row, accuracy_tier columns).

    Returns:
        DataFrame with columns: doc_row, keyword, score, matched, match_type,
        word_count, accuracy_tier.
    """
    rows = []
    for d in per_doc:
        doc_row = d["doc_row"]
        gold_kws = d["gold_keywords"]
        for p in d["pred_keywords"]:
            matched = p.get("matched", False)
            mt = classify_match_type(p["keyword"], gold_kws, matched)
            rows.append(
                {
                    "doc_row": doc_row,
                    "keyword": p["keyword"],
                    "score": p["score"],
                    "matched": matched,
                    "match_type": mt,
                    "word_count": len(p["keyword"].split()),
                }
            )

    keyphrase_df = pl.DataFrame(rows).join(
        per_doc_df.select(["doc_row", "accuracy_tier"]), on="doc_row", how="left"
    )
    return keyphrase_df


def build_keyphrase_count_summary(per_doc_df: pl.DataFrame) -> pl.DataFrame:
    """Build summary statistics for gold/pred keyphrase counts and delta.

    Args:
        per_doc_df: Per-document DataFrame with num_gold, num_pred columns.

    Returns:
        DataFrame with columns: metric, mean, median, std, min, max.
    """
    import numpy as np

    delta = per_doc_df.with_columns(
        (pl.col("num_pred") - pl.col("num_gold")).alias("delta")
    )

    summary_rows = [
        {
            "metric": "Gold keyphrases",
            "mean": per_doc_df["num_gold"].mean(),
            "median": float(np.median(per_doc_df["num_gold"])),
            "std": per_doc_df["num_gold"].std(),
            "min": per_doc_df["num_gold"].min(),
            "max": per_doc_df["num_gold"].max(),
        },
        {
            "metric": "Predicted keyphrases",
            "mean": per_doc_df["num_pred"].mean(),
            "median": float(np.median(per_doc_df["num_pred"])),
            "std": per_doc_df["num_pred"].std(),
            "min": per_doc_df["num_pred"].min(),
            "max": per_doc_df["num_pred"].max(),
        },
        {
            "metric": "Delta (pred - gold)",
            "mean": delta["delta"].mean(),
            "median": float(np.median(delta["delta"])),
            "std": delta["delta"].std(),
            "min": delta["delta"].min(),
            "max": delta["delta"].max(),
        },
    ]
    return pl.DataFrame(summary_rows)


def load_all_benchmark_per_doc(run_dir: Path) -> dict[str, list[dict]]:
    """Load per-doc data from all eval JSONs in a run's evals directory.

    Scans run_dir/evals/ for eval_*_test_*.json files, extracts benchmark
    name from filename, and returns {benchmark: per_doc_list, ...}.
    When multiple evals exist for the same benchmark, picks the one with
    the most per-doc entries.
    """
    eval_files = list((run_dir / "evals").glob("eval_*_test_*.json"))
    if not eval_files:
        raise FileNotFoundError(f"No eval test files found in {run_dir / 'evals'}")

    result: dict[str, tuple[int, list[dict]]] = {}
    for eval_path in eval_files:
        parts = eval_path.stem.split("_")
        benchmark = parts[1]

        with open(eval_path) as f:
            eval_data = json.load(f)

        per_doc = eval_data.get("per_doc", [])
        existing_len = len(result.get(benchmark, (0, []))[1])
        if len(per_doc) > existing_len:
            result[benchmark] = (len(per_doc), per_doc)

    return {bm: data for bm, (_, data) in result.items()}


def load_external_benchmarks(
    path: Path | str,
    benchmarks: list[str] | None = None,
) -> pl.DataFrame:
    """Load external benchmark metrics from a JSON file.

    Args:
        path: Path to the benchmarks JSON file.
        benchmarks: List of benchmark names to include (defaults to BENCHMARKS).

    Returns:
        DataFrame with columns: model, group, and per-benchmark f1_5/f1_10.
    """
    if benchmarks is None:
        benchmarks = BENCHMARKS

    with open(path) as f:
        data = json.load(f)

    model_lookup = {k.lower(): v for k, v in data.items()}

    rows = []
    for model_key, model_name in MODEL_NAME_MAP.items():
        model_data = model_lookup.get(model_key.lower())
        if model_data is None:
            continue
        row = {"model": model_name, "group": "Published Baselines"}
        bm_section = model_data.get("benchmarks", {})
        bm_section_lookup = {k.lower(): k for k in bm_section}
        for bm in benchmarks:
            bm_key = bm_section_lookup.get(bm.lower())
            if bm_key is None:
                row[f"{bm}_f1_5"] = None
                row[f"{bm}_f1_10"] = None
                continue
            bm_vals = bm_section[bm_key]
            f1_5 = bm_vals.get("F1@5")
            f1_10 = bm_vals.get("F1@10")
            row[f"{bm}_f1_5"] = f1_5["mean"] / 100 if f1_5 else None
            row[f"{bm}_f1_10"] = f1_10["mean"] / 100 if f1_10 else None
        rows.append(row)

    return pl.DataFrame(rows)
