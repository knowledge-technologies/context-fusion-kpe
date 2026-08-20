"""Table generation functions for metrics and sample displays."""

from pathlib import Path
from typing import TYPE_CHECKING

import polars as pl

if TYPE_CHECKING:
    from great_tables import GT


def make_metrics_table_gt(
    comparison_df: pl.DataFrame,
    fig_dir: Path | None = None,
    save: bool = False,
) -> "GT":
    """Create a Great Tables formatted metrics comparison table."""
    from great_tables import GT, loc, md, style

    precision_cols = ["precision_5", "precision_10"]
    recall_cols = ["recall_5", "recall_10"]
    f1_cols = ["f1_5", "f1_10"]
    metric_cols = precision_cols + recall_cols + f1_cols

    min_metric = comparison_df.select(metric_cols).min_horizontal().min()
    max_metric = comparison_df.select(metric_cols).max_horizontal().max()

    table = (
        GT(comparison_df)
        .tab_header(
            title="Evaluation scores on kp20k benchmark",
            subtitle=md("Precision, Recall, and F1@5/@10 across two runs"),
        )
        .tab_stub(rowname_col="run")
        .cols_hide(columns=["n_epochs"])
        .cols_label(
            precision_5="@5",
            precision_10="@10",
            recall_5="@5",
            recall_10="@10",
            f1_5="@5",
            f1_10="@10",
        )
        .tab_spanner(
            label="Precision (%)",
            columns=precision_cols,
        )
        .tab_spanner(
            label="Recall (%)",
            columns=recall_cols,
        )
        .tab_spanner(
            label="F1 (%)",
            columns=f1_cols,
        )
        .tab_style(style.text(weight="bold"), locations=loc.column_labels())
        .data_color(
            columns=metric_cols,
            palette="Greens",
            domain=[min_metric, max_metric],
        )
        .fmt_number(
            columns=metric_cols,
            decimals=1,
            scale_by=100,
        )
    )

    if save and fig_dir is not None:
        print("Saving metrics_table.png")
        table.gtsave(fig_dir / "metrics_table.png")

    return table


def make_metrics_table_tex(
    comparison_df: pl.DataFrame,
    fig_dir: Path | None = None,
    save: bool = False,
) -> str:
    """Create a LaTeX table string for metrics comparison with cell coloring."""
    color_enabled = True

    precision_cols = ["precision_5", "precision_10"]
    recall_cols = ["recall_5", "recall_10"]
    f1_cols = ["f1_5", "f1_10"]
    metric_cols = precision_cols + recall_cols + f1_cols

    all_vals = [
        v for col in metric_cols for v in comparison_df[col].drop_nulls().to_list()
    ]
    global_min = min(all_vals) if all_vals else 0.0
    global_max = max(all_vals) if all_vals else 1.0

    def cell_color(val):
        if val is None:
            return None
        if global_max == global_min:
            intensity = 50
        else:
            intensity = ((val - global_min) / (global_max - global_min)) * 100
        intensity = max(15, min(100, intensity))
        return intensity

    def fmt_val(val):
        if val is None:
            return "---"
        pct = f"{val * 100:.1f}"
        if color_enabled:
            intensity = cell_color(val)
            return f"\\cellcolor{{softgreen!{intensity:.0f}}}{pct}"
        return pct

    runs = comparison_df["run"].to_list()

    rows = []
    for i in range(len(runs)):
        row_parts = [runs[i]]
        for col in metric_cols:
            val = comparison_df[col][i]
            row_parts.append(fmt_val(val))
        rows.append(" & ".join(row_parts))

    latex_lines = [
        r"\begin{tabular}{lcccccc}",
        r"\toprule",
        r"\textbf{Run} & \multicolumn{2}{c}{\textbf{Precision (\%)}} & \multicolumn{2}{c}{\textbf{Recall (\%)}} & \multicolumn{2}{c}{\textbf{F1 (\%)}} \\",
        r"\cmidrule(lr){2-3} \cmidrule(lr){4-5} \cmidrule(lr){6-7}",
        r"& \textbf{@5} & \textbf{@10} & \textbf{@5} & \textbf{@10} & \textbf{@5} & \textbf{@10} \\",
        r"\midrule",
    ]
    for row in rows:
        latex_lines.append(row + r" \\")
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
