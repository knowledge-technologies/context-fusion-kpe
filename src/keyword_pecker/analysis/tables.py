"""Table generation functions for metrics and sample displays."""

import json
import re
from pathlib import Path
from typing import TYPE_CHECKING

import polars as pl

if TYPE_CHECKING:
    from great_tables import GT


BENCHMARKS = ["kp20k", "kptimes", "inspec"]

MODEL_NAME_MAP = {
    "TextRank": "TextRank (unsupervised)",
    "TNT-KID": "TNT-KID (supervised)",
    "Gemini1.5-Flash-mini-8B": "Gemini1.5-Flash-mini-8B (LLM)",
}


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
    from keyword_pecker.analysis.data_loading import BENCHMARKS

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


BENCHMARK_COLORS = {
    "kp20k": "#E69F00",
    "kptimes": "#56B4E9",
    "inspec": "#009E73",
}

BENCHMARK_PALETTES = {
    "kp20k": ["#FFF5E0", "#E69F00"],
    "kptimes": ["#E8F4FD", "#56B4E9"],
    "inspec": ["#E0F5F0", "#009E73"],
}

LATEX_COLOR_DEFINITIONS = "\n".join(
    f"\\definecolor{{soft{name}}}{{HTML}}{{{hex_color[1:]}}}"
    for name, hex_color in BENCHMARK_COLORS.items()
)


def make_metrics_table_gt(
    wide_df: pl.DataFrame,
    fig_dir: Path | None = None,
    save: bool = False,
    benchmarks: list[str] | None = None,
    external_rows: pl.DataFrame | None = None,
) -> "GT":
    """Create a Great Tables formatted metrics comparison table with grouped rows."""
    from great_tables import GT, loc, style

    if benchmarks is None:
        benchmarks = BENCHMARKS

    if external_rows is not None:
        wide_df = pl.concat([external_rows, wide_df])

    metric_cols = []
    for bm in benchmarks:
        metric_cols.extend([f"{bm}_f1_5", f"{bm}_f1_10"])

    wide_df = wide_df.with_columns(
        pl.col("group").replace(
            {
                "Validation Set FT": "Validation Set Fine-tuning",
                "Training Set FT": "Training Set Fine-tuning",
            }
        )
    )

    table = (
        GT(wide_df)
        .tab_header(
            title="Evaluation scores across benchmarks",
            subtitle="F1@5 and F1@10 for Training Set FT and Validation Set FT",
        )
        .tab_stub(rowname_col="model", groupname_col="group")
        .cols_label(
            **{f"{bm}_f1_5": "F1@5" for bm in benchmarks},
            **{f"{bm}_f1_10": "F1@10" for bm in benchmarks},
        )
    )

    for bm in benchmarks:
        table = table.tab_spanner(
            label=bm,
            columns=[f"{bm}_f1_5", f"{bm}_f1_10"],
        )

    table = table.tab_style(style.text(weight="bold"), locations=loc.column_labels())
    table = table.tab_style(style.text(style="italic"), locations=loc.row_groups())
    table = table.tab_style(
        style.text(weight="bold"),
        locations=loc.body(columns="model", rows=pl.col("model") == "Ours"),
    )

    for bm in benchmarks:
        bm_cols = [f"{bm}_f1_5", f"{bm}_f1_10"]
        bm_vals = [v for col in bm_cols for v in wide_df[col].drop_nulls().to_list()]
        if bm_vals:
            bm_min = min(bm_vals)
            bm_max = max(bm_vals)
        else:
            bm_min = 0.0
            bm_max = 1.0
        table = table.data_color(
            columns=bm_cols,
            palette=BENCHMARK_PALETTES[bm],
            domain=[bm_min, bm_max],
            na_color="transparent",
        )

    table = table.fmt(
        columns=metric_cols,
        fns=lambda x: "---" if x is None else f"{x * 100:.1f}",
    )

    if save and fig_dir is not None:
        print("Saving metrics_table.png")
        table.gtsave(fig_dir / "metrics_table.png")

    return table


def make_metrics_table_tex(
    wide_df: pl.DataFrame,
    fig_dir: Path | None = None,
    save: bool = False,
    benchmarks: list[str] | None = None,
    external_rows: pl.DataFrame | None = None,
) -> str:
    """Create a LaTeX table string for grouped metrics with cell coloring."""
    if benchmarks is None:
        benchmarks = BENCHMARKS

    if external_rows is not None:
        wide_df = pl.concat([external_rows, wide_df])

    metric_cols = []
    for bm in benchmarks:
        metric_cols.extend([f"{bm}_f1_5", f"{bm}_f1_10"])

    bm_ranges = {}
    for bm in benchmarks:
        bm_cols = [f"{bm}_f1_5", f"{bm}_f1_10"]
        bm_vals = [v for col in bm_cols for v in wide_df[col].drop_nulls().to_list()]
        if bm_vals:
            bm_ranges[bm] = (min(bm_vals), max(bm_vals))
        else:
            bm_ranges[bm] = (0.0, 1.0)

    def cell_color(val, bm):
        if val is None:
            return None
        bm_min, bm_max = bm_ranges[bm]
        if bm_max == bm_min:
            intensity = 50
        else:
            intensity = ((val - bm_min) / (bm_max - bm_min)) * 100
        intensity = max(15, min(100, intensity))
        return intensity

    def fmt_val(val, bm):
        if val is None:
            return "---"
        pct = f"{val * 100:.1f}"
        intensity = cell_color(val, bm)
        color_name = f"soft{bm}"
        return f"\\cellcolor{{{color_name}!{intensity:.0f}}}{pct}"

    n_bm_cols = len(benchmarks) * 2

    header_spanners = []
    header_subs = []
    for bm in benchmarks:
        header_spanners.append(f"\\multicolumn{{2}}{{c}}{{\\textbf{{{bm}}}}}")
        header_subs.extend(["@5", "@10"])

    latex_lines = [
        LATEX_COLOR_DEFINITIONS,
        r"\begin{tabular}{l" + "c" * n_bm_cols + "}",
        r"\toprule",
        r"\textbf{Model/Benchmark} & " + " & ".join(header_spanners) + r" \\",
        r"\cmidrule(lr){2-" + str(n_bm_cols + 1) + r"}",
        r"F1 & " + " & ".join(header_subs) + r" \\",
        r"\midrule",
    ]

    current_group = None
    for row in wide_df.iter_rows(named=True):
        group = row["group"]
        model = row["model"]

        if group != current_group:
            if current_group is not None:
                latex_lines.append(r"\midrule")
            group_label = group.replace(" FT", " Fine-tuning")
            latex_lines.append(
                f"\\multicolumn{{{n_bm_cols + 1}}}{{l}}{{\\textit{{{group_label}}}}} \\\\"
            )
            latex_lines.append(r"\midrule")
            current_group = group

        vals = []
        for bm in benchmarks:
            vals.append(fmt_val(row.get(f"{bm}_f1_5"), bm))
            vals.append(fmt_val(row.get(f"{bm}_f1_10"), bm))
        label = f"\\textbf{{{model}}}" if model == "Ours" else model
        latex_lines.append(f"{label} & {' & '.join(vals)} \\\\")

    latex_lines.append(r"\bottomrule")
    latex_lines.append(r"\end{tabular}")

    latex_table = "\n".join(latex_lines)

    if save and fig_dir is not None:
        print("Saving metrics_table.tex")
        (fig_dir / "metrics_table.tex").write_text(latex_table)

    return latex_table


def _latex_escape(text: str) -> str:
    """Escape special LaTeX characters."""
    chars = {
        "\\": r"\textbackslash{}",
        "{": r"\{",
        "}": r"\}",
        "&": r"\&",
        "%": r"\%",
        "$": r"\$",
        "#": r"\#",
        "_": r"\_",
        "~": r"\textasciitilde{}",
        "^": r"\textasciicircum{}",
    }
    result = []
    for ch in text:
        result.append(chars.get(ch, ch))
    return "".join(result)


def make_samples_table_tex(
    samples: list[dict],
    fig_dir: Path | None = None,
    save: bool = False,
) -> str:
    """Create a LaTeX xltabular table for sample predictions with highlighted keywords."""

    def highlight_kw_full(pred_kw, score):
        escaped = _latex_escape(pred_kw)
        return f"\\keywordFull{{{escaped} ({score:.2f})}}"

    def highlight_kw_partial(pred_kw, score):
        escaped = _latex_escape(pred_kw)
        return f"\\keywordPartial{{{escaped} ({score:.2f})}}"

    def fmt_unmatched(pred_kw, score):
        escaped = _latex_escape(pred_kw)
        return f"{escaped} ({score:.2f})"

    lines = [
        r"% Requires: \usepackage{colortbl}, \usepackage[many]{tcolorbox}, \usepackage{xltabular}, \usepackage{booktabs}",
        r"\clearpage",
        r"\onecolumn",
        r"\newtcbox{\keywordFull}{nobeforeafter, colback=softgreen!40, size=tight, sharp corners, boxrule=0pt, frame hidden, borderline={0.4pt}{0pt}{black}, boxsep=0.5pt}",
        r"\newtcbox{\keywordPartial}{nobeforeafter, colback=softyellow!40, size=tight, enhanced, sharp corners, boxrule=0pt, frame hidden, borderline={0.4pt}{0pt}{black, dash pattern=on 2pt off 1.5pt}, boxsep=0.5pt}",
        r"\newcolumntype{G}{>{\raggedright\arraybackslash\hsize=0.5\hsize}X}",
        r"\newcolumntype{E}{>{\raggedright\arraybackslash\hsize=1.5\hsize}X}",
        r"\begin{xltabular}{\textwidth}{rcGE}",
        r"\caption{Examples of gold truth and extracted key-phrases for five randomly sampled documents from \textsf{kp20k} test set stratified by document-level accuracy. Color legend: \textcolor{softgreen!40}{\rule{0.5em}{0.5em}} Exact match, \textcolor{softyellow!40}{\rule{0.5em}{0.5em}} Partial match (dashed border). Numbers in parentheses indicate model confidence scores.}\label{tab:samples}\\",
        r"\toprule",
        r"\textbf{ID} & \textbf{F1@10} & \textbf{Gold Keywords} & \textbf{Extracted Keywords} \\",
        r"\midrule",
        r"\endfirsthead",
        r"\toprule",
        r"\textbf{ID} & \textbf{F1@10} & \textbf{Gold Keywords} & \textbf{Extracted Keywords} \\",
        r"\midrule",
        r"\endhead",
    ]

    current_tier = None
    for row in samples:
        tier = row["tier"]
        if tier != current_tier:
            tier_label = {
                "High": "High Accuracy (in top quartile)",
                "Low": "Low Accuracy (in bottom quartiles)",
                "Mixed": "Mid Accuracy (in mid quartiles)",
            }[tier]
            lines.append(f"\\multicolumn{{4}}{{l}}{{\\textit{{{tier_label}}}}} \\\\")
            lines.append(r"\midrule")
            current_tier = tier

        idx = row["idx"]
        f1_10_pct = row["f1_10_pct"]
        gold_kws = row["gold_keywords"]
        pred_kws = row["pred_keywords"]

        escaped_gold = (
            ", ".join(_latex_escape(k) for k in gold_kws) if gold_kws else "---"
        )
        highlighted_preds = []
        for p in pred_kws:
            mt = p["match_type"]
            kw = p["keyword"]
            score = p["score"]
            if mt == "full":
                highlighted_preds.append(highlight_kw_full(kw, score))
            elif mt == "partial":
                highlighted_preds.append(highlight_kw_partial(kw, score))
            else:
                highlighted_preds.append(fmt_unmatched(kw, score))
        escaped_pred = (
            ",\\,\\allowbreak ".join(highlighted_preds) if highlighted_preds else "---"
        )

        lines.append(f"{idx} & {f1_10_pct}\\% & {escaped_gold} & {escaped_pred} \\\\")
        lines.append(r"\midrule")

    lines.append(r"\bottomrule")
    lines.append(r"\end{xltabular}")
    lines.append(r"\twocolumn")

    samples_latex = "\n".join(lines)

    if save and fig_dir is not None:
        print("Saving samples_table.tex")
        (fig_dir / "samples_table.tex").write_text(samples_latex)

    return samples_latex


def make_precision_recall_table_tex(
    stats_df: pl.DataFrame,
    fig_dir: Path | None = None,
    save: bool = False,
) -> str:
    """Create a LaTeX table of extraction performance with grouped rows and cell coloring."""
    benchmarks = [b for b in BENCHMARKS if b in stats_df["benchmark"].to_list()]

    metric_rows = [
        (r"Precision", "precision"),
        (r"Recall", "recall"),
    ]

    bm_ranges = {}
    for bm in benchmarks:
        vals = []
        for metric_prefix in ["precision", "recall"]:
            for k in [5, 10]:
                col = f"{metric_prefix}_{k}"
                for group in ["Training Set FT", "Validation Set FT"]:
                    row = stats_df.filter(
                        pl.col("group") == group, pl.col("benchmark") == bm
                    )
                    if len(row) > 0:
                        val = row[col][0]
                        if val is not None:
                            vals.append(val)
        if vals:
            bm_ranges[bm] = (min(vals), max(vals))
        else:
            bm_ranges[bm] = (0.0, 1.0)

    def cell_color(val, bm):
        if val is None:
            return None
        bm_min, bm_max = bm_ranges[bm]
        if bm_max == bm_min:
            intensity = 50
        else:
            intensity = ((val - bm_min) / (bm_max - bm_min)) * 100
        intensity = max(15, min(100, intensity))
        return intensity

    def fmt_val(val, bm, metric_prefix):
        if val is None:
            return "---"
        pct = f"{val * 100:.1f}"
        intensity = cell_color(val, bm)
        color_name = f"soft{bm}"
        return f"\\cellcolor{{{color_name}!{intensity:.0f}}}{pct}"

    n_data_cols = len(benchmarks) * 2

    header_spanners = []
    header_subs = []
    for bm in benchmarks:
        header_spanners.append(f"\\multicolumn{{2}}{{c}}{{\\textbf{{{bm}}}}}")
        header_subs.extend(["@5", "@10"])

    latex_lines = [
        rf"\begin{{tabular}}{{l{'c' * n_data_cols}}}",
        r"\toprule",
        r"\textbf{Metric} & " + " & ".join(header_spanners) + r" \\",
        r"\cmidrule(lr){2-" + str(n_data_cols + 1) + r"}",
        r" & " + " & ".join(header_subs) + r" \\",
        r"\midrule",
    ]

    group_labels = {
        "Training Set FT": "Training Set Fine-tuning",
        "Validation Set FT": "Validation Set Fine-tuning",
    }

    first_group = True
    for group in ["Validation Set FT", "Training Set FT"]:
        if group not in stats_df["group"].to_list():
            continue

        if not first_group:
            latex_lines.append(r"\midrule")
        first_group = False

        label = group_labels.get(group, group)
        latex_lines.append(
            f"\\multicolumn{{{n_data_cols + 1}}}{{l}}{{\\textit{{{label}}}}} \\\\"
        )
        latex_lines.append(r"\midrule")

        for display_label, metric_prefix in metric_rows:
            vals = []
            for bm in benchmarks:
                for k in [5, 10]:
                    col = f"{metric_prefix}_{k}"
                    row = stats_df.filter(
                        pl.col("group") == group, pl.col("benchmark") == bm
                    )
                    if len(row) > 0:
                        val = row[col][0]
                        vals.append(fmt_val(val, bm, metric_prefix))
                    else:
                        vals.append("---")
            latex_lines.append(f"{display_label} & {' & '.join(vals)} \\\\")

    latex_lines.extend(
        [
            r"\bottomrule",
            r"\end{tabular}",
        ]
    )

    latex_table = "\n".join(latex_lines)

    if save and fig_dir is not None:
        print("Saving precision_recall_table.tex")
        (fig_dir / "precision_recall_table.tex").write_text(latex_table)

    return latex_table


def compute_dataset_stats(
    json_path: str | Path,
    parquet_path: str | Path,
    vocab_sample: int = 1000,
) -> dict:
    """Compute dataset statistics from JSON Lines and derived Parquet files.

    Args:
        json_path: Path to raw JSON Lines dataset file (for abstract/vocab stats).
        parquet_path: Path to derived Parquet file (for keyword stats).
        vocab_sample: Number of documents to sample for vocabulary counting.

    Returns:
        Dict with keys: nr_documents, mean_abstract_words, max_abstract_words,
        mean_keywords_all, mean_keywords_present, vocab_size,
        words_per_keyword_all, words_per_keyword_present.
    """
    json_path = Path(json_path)
    parquet_path = Path(parquet_path)

    # --- From JSON: document count, abstract stats, vocabulary ---
    with open(json_path) as f:
        docs = [json.loads(line) for line in f]

    nr_documents = len(docs)

    abs_word_lens = [len(d.get("abstract", "").split()) for d in docs]
    mean_abstract_words = sum(abs_word_lens) / nr_documents
    max_abstract_words = max(abs_word_lens)

    words = set()
    for d in docs[:vocab_sample]:
        text = (d.get("title", "") + " " + d.get("abstract", "")).lower()
        words.update(re.findall(r"[a-z]+", text))
    vocab_size = len(words)

    # --- From Parquet: keyword stats (all vs present) ---
    pq = pl.read_parquet(parquet_path)

    # Group by document to get per-document keyword lists
    doc_groups = pq.group_by(["doc_id", "doc_row"]).agg(
        pl.col("keywords").first().alias("keywords"),
        pl.col("sent_keywords")
        .list.explode(keep_nulls=False, empty_as_null=False)
        .unique()
        .alias("present_keywords"),
    )

    # Mean keywords (all)
    all_kw_counts = doc_groups.with_columns(
        pl.col("keywords").list.len().alias("kw_count")
    )
    mean_keywords_all = all_kw_counts["kw_count"].mean()

    # Mean keywords (present in text)
    present_kw_counts = doc_groups.with_columns(
        pl.col("present_keywords").list.len().alias("present_kw_count")
    )
    mean_keywords_present = present_kw_counts["present_kw_count"].mean()

    # Words per keyword (all)
    all_kws_flat = all_kw_counts.select(
        pl.col("keywords").list.explode(keep_nulls=False, empty_as_null=False)
    ).to_series()
    all_kw_word_counts = [len(kw.split()) for kw in all_kws_flat if kw]
    words_per_keyword_all = sum(all_kw_word_counts) / max(len(all_kw_word_counts), 1)

    # Words per keyword (present)
    present_kws_flat = present_kw_counts.select(
        pl.col("present_keywords").list.explode(keep_nulls=False, empty_as_null=False)
    ).to_series()
    present_kw_word_counts = [len(kw.split()) for kw in present_kws_flat if kw]
    words_per_keyword_present = sum(present_kw_word_counts) / max(
        len(present_kw_word_counts), 1
    )

    return {
        "nr_documents": nr_documents,
        "mean_abstract_words": mean_abstract_words,
        "max_abstract_words": max_abstract_words,
        "mean_keywords_all": mean_keywords_all,
        "mean_keywords_present": mean_keywords_present,
        "vocab_size": vocab_size,
        "words_per_keyword_all": words_per_keyword_all,
        "words_per_keyword_present": words_per_keyword_present,
    }


def make_dataset_stats_tex(
    stats: dict[str, dict],
    fig_dir: Path | None = None,
    save: bool = False,
    benchmarks: list[str] | None = None,
) -> str:
    """Create a LaTeX table of dataset statistics.

    Args:
        stats: Dict mapping benchmark names to stats dicts from compute_dataset_stats.
        fig_dir: Directory to save the output file.
        save: Whether to write the table to disk.
        benchmarks: Ordered list of benchmarks to include (defaults to BENCHMARKS).

    Returns:
        LaTeX table string.
    """
    if benchmarks is None:
        benchmarks = BENCHMARKS

    stats = {bm: stats[bm] for bm in benchmarks if bm in stats}

    row_definitions = [
        ("Nr. Documents", "nr_documents", lambda v: f"{v:,}"),
        ("Mean abstract length (words)", "mean_abstract_words", lambda v: f"{v:.1f}"),
        ("Max abstract length (words)", "max_abstract_words", lambda v: f"{v:,}"),
        (
            "Mean keyphrases/doc (all/in doc)",
            ("mean_keywords_all", "mean_keywords_present"),
            lambda v_all, v_present: (
                f"{v_all:.1f} / {v_present:.1f}"
                if v_present is not None
                else f"{v_all:.1f}"
            ),
        ),
        ("Vocabulary size (first 1k docs)", "vocab_size", lambda v: f"{v:,}"),
        (
            "Words per keyphrase (all/in doc)",
            ("words_per_keyword_all", "words_per_keyword_present"),
            lambda v_all, v_present: (
                f"{v_all:.1f} / {v_present:.1f}"
                if v_present is not None
                else f"{v_all:.1f}"
            ),
        ),
    ]

    n_cols = len(stats)
    header_cells = [f"\\textbf{{{bm}}}" for bm in stats]

    latex_lines = [
        rf"\begin{{tabular}}{{l{'c' * n_cols}}}",
        r"\toprule",
        r"\textbf{Statistic} & " + " & ".join(header_cells) + r" \\",
        r"\midrule",
    ]

    latex_lines.append(r"Domain & " + " CS & news & CS" + r" \\")

    for display_label, key_or_keys, fmt_fn in row_definitions:
        vals = []
        for bm in stats:
            if isinstance(key_or_keys, tuple):
                vals.append(
                    fmt_fn(stats[bm][key_or_keys[0]], stats[bm][key_or_keys[1]])
                )
            else:
                vals.append(fmt_fn(stats[bm][key_or_keys]))
        latex_lines.append(f"{display_label} & {' & '.join(vals)} \\\\")

    latex_lines.extend(
        [
            r"\bottomrule",
            r"\end{tabular}",
        ]
    )

    latex_table = "\n".join(latex_lines)

    if save and fig_dir is not None:
        print("Saving dataset_stats_table.tex")
        (fig_dir / "dataset_stats_table.tex").write_text(latex_table)

    return latex_table
