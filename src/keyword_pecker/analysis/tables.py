"""Table generation functions for metrics and sample displays."""

from pathlib import Path
from typing import TYPE_CHECKING

import polars as pl

if TYPE_CHECKING:
    from great_tables import GT


BENCHMARKS = ["kp20k", "kptimes", "inspec"]

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
) -> "GT":
    """Create a Great Tables formatted metrics comparison table with grouped rows."""
    from great_tables import GT, loc, style

    if benchmarks is None:
        benchmarks = BENCHMARKS

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
            **{f"{bm}_f1_5": "@5" for bm in benchmarks},
            **{f"{bm}_f1_10": "@10" for bm in benchmarks},
        )
    )

    for bm in benchmarks:
        table = table.tab_spanner(
            label=bm,
            columns=[f"{bm}_f1_5", f"{bm}_f1_10"],
        )

    table = table.tab_style(style.text(weight="bold"), locations=loc.column_labels())
    table = table.tab_style(style.text(style="italic"), locations=loc.row_groups())

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
        fns=lambda x: "*" if x is None else f"{x * 100:.1f}",
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
) -> str:
    """Create a LaTeX table string for grouped metrics with cell coloring."""
    if benchmarks is None:
        benchmarks = BENCHMARKS

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
            return "*"
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
        r" & " + " & ".join(header_subs) + r" \\",
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
        latex_lines.append(f"{model} & {' & '.join(vals)} \\\\")

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
