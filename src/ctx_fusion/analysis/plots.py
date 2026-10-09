"""Plotting functions for training dynamics and metrics visualization."""

import numpy as np
from pathlib import Path
from typing import TYPE_CHECKING

import polars as pl

from ctx_fusion.analysis.tables import BENCHMARK_COLORS

if TYPE_CHECKING:
    from matplotlib.figure import Figure


def _step_formatter(x, pos):
    """Format step numbers as k notation."""
    return f"{x / 1000:.0f}k" if x >= 1000 else str(int(x))


def _style_ax(ax, xlabel=None, ylabel=None, fontsize=14):
    """Apply consistent styling to a matplotlib axis."""
    if xlabel:
        ax.set_xlabel(xlabel, fontsize=fontsize)
    if ylabel:
        ax.set_ylabel(ylabel, fontsize=fontsize)
    ax.grid(True, alpha=0.3)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.tick_params(axis="both", labelsize=12)


def plot_loss_overview(
    all_metrics: pl.DataFrame,
    raw_loss: pl.DataFrame,
    epoch_mean_loss: pl.DataFrame,
    fig_dir: Path | None = None,
    save: bool = False,
) -> "Figure":
    """Plot training loss overview with raw, mean, and validation loss."""
    import matplotlib.pyplot as plt
    from matplotlib.ticker import FuncFormatter

    val_steps = all_metrics["global_step"].to_list()
    val_losses = all_metrics["val_loss"].to_list()
    epoch_mean_losses = epoch_mean_loss["epoch_mean_loss"].to_list()
    raw_steps = raw_loss["global_step"].to_list()
    raw_losses = raw_loss["train_loss"].to_list()

    fig, ax = plt.subplots(figsize=(8, 4))

    ax.plot(
        raw_steps,
        raw_losses,
        color="red",
        linewidth=0.8,
        alpha=0.5,
        label="Raw Training loss",
    )
    ax.plot(
        val_steps,
        epoch_mean_losses,
        color="tab:blue",
        marker="o",
        linewidth=1.5,
        label="Mean Training Loss",
    )
    ax.plot(
        val_steps,
        val_losses,
        color="tab:orange",
        marker="s",
        linewidth=1.5,
        label="Mean Validation Loss",
    )

    for i, step in enumerate(val_steps):
        if i == 0:
            ax.axvline(
                step,
                color="black",
                linestyle="--",
                alpha=0.4,
                linewidth=1,
                label="Epoch end",
            )
        else:
            ax.axvline(step, color="black", linestyle="--", alpha=0.4, linewidth=1)

    _style_ax(ax, xlabel="Step", ylabel="Loss")
    ax.set_xlim(-2500, int(raw_steps[-1] * 1.05))
    ax.xaxis.set_major_formatter(FuncFormatter(_step_formatter))
    ax.legend(fontsize=12)

    if save and fig_dir is not None:
        print("Saving loss_overview.png")
        fig.savefig(fig_dir / "loss_overview.png", dpi=200, bbox_inches="tight")
        print("Saving loss_overview.pdf")
        fig.savefig(fig_dir / "loss_overview.pdf", bbox_inches="tight")

    return fig


def plot_training_dynamics(
    all_metrics: pl.DataFrame,
    fig_dir: Path | None = None,
    save: bool = False,
) -> "Figure":
    """Plot training dynamics with validation loss and F1 scores."""
    import matplotlib.pyplot as plt
    from matplotlib.ticker import FuncFormatter

    val_steps = sorted(all_metrics["global_step"].to_list())
    total_steps = int(all_metrics["global_step"].max() * 1.05)

    dyn_fig, (ax_loss, ax_f1) = plt.subplots(2, 1, figsize=(10, 8), sharex=True)

    mask_loss = all_metrics["val_loss"].is_not_null()
    ax_loss.plot(
        all_metrics.filter(mask_loss)["global_step"],
        all_metrics.filter(mask_loss)["val_loss"],
        linestyle="--",
        marker="o",
        linewidth=1.5,
        markersize=2.5,
        label="Validation Loss",
    )
    _style_ax(ax_loss, ylabel="Validation Loss")

    mask_f1_5 = all_metrics["val_f1@5"].is_not_null()
    mask_f1_10 = all_metrics["val_f1@10"].is_not_null()
    ax_f1.plot(
        all_metrics.filter(mask_f1_5)["global_step"],
        all_metrics.filter(mask_f1_5)["val_f1@5"],
        linestyle="--",
        marker="o",
        linewidth=1.5,
        markersize=2.5,
        label="F1@5",
    )
    ax_f1.plot(
        all_metrics.filter(mask_f1_10)["global_step"],
        all_metrics.filter(mask_f1_10)["val_f1@10"],
        linestyle="-",
        marker="s",
        linewidth=1.5,
        markersize=2.5,
        label="F1@10",
    )
    _style_ax(ax_f1, xlabel="Step", ylabel="F1 Score")
    ax_f1.xaxis.set_major_formatter(FuncFormatter(_step_formatter))

    ax_f1.set_xlim(0, total_steps)
    ax_loss.set_xlim(0, total_steps)

    for i, step in enumerate(val_steps):
        if i == 0:
            ax_f1.axvline(
                step,
                color="tab:red",
                linestyle=":",
                alpha=0.5,
                linewidth=1,
                label="Epoch end",
            )
        else:
            ax_f1.axvline(step, color="tab:red", linestyle=":", alpha=0.5, linewidth=1)
        ax_loss.axvline(step, color="tab:red", linestyle=":", alpha=0.5, linewidth=1)

    ax_f1.legend(fontsize=12)
    dyn_fig.suptitle(
        "Training dynamics: ModernBERT (young-haze-50)",
        fontsize=16,
    )

    if save and fig_dir is not None:
        print("Saving training_dynamics.png")
        dyn_fig.savefig(fig_dir / "training_dynamics.png", dpi=200, bbox_inches="tight")

    return dyn_fig


def plot_loss_f1_overview(
    all_metrics: pl.DataFrame,
    raw_loss: pl.DataFrame,
    epoch_mean_loss: pl.DataFrame,
    fig_dir: Path | None = None,
    save: bool = False,
) -> "Figure":
    """Plot combined loss and F1 overview side by side."""
    import matplotlib.pyplot as plt
    from matplotlib.ticker import FuncFormatter

    val_steps = all_metrics["global_step"].to_list()
    val_losses = all_metrics["val_loss"].to_list()
    epoch_mean_losses = epoch_mean_loss["epoch_mean_loss"].to_list()
    raw_steps = raw_loss["global_step"].to_list()
    raw_losses = raw_loss["train_loss"].to_list()

    fig, (ax_loss, ax_f1) = plt.subplots(1, 2, figsize=(14, 4.5), sharex=True)

    ax_loss.plot(
        raw_steps,
        raw_losses,
        color="red",
        linewidth=0.8,
        alpha=0.5,
        label="Raw Training loss",
    )
    ax_loss.plot(
        val_steps,
        epoch_mean_losses,
        color="tab:blue",
        marker="o",
        linewidth=1.5,
        label="Mean Training Loss",
    )
    ax_loss.plot(
        val_steps,
        val_losses,
        color="tab:orange",
        marker="s",
        linewidth=1.5,
        label="Mean Validation Loss",
    )

    for i, step in enumerate(val_steps):
        if i == 0:
            ax_loss.axvline(
                step,
                color="black",
                linestyle="--",
                alpha=0.4,
                linewidth=1,
                label="Epoch end",
            )
        else:
            ax_loss.axvline(step, color="black", linestyle="--", alpha=0.4, linewidth=1)

    _style_ax(ax_loss, xlabel="Step", ylabel="Loss")
    ax_loss.set_title("Loss curves", fontsize=14)
    ax_loss.xaxis.set_major_formatter(FuncFormatter(_step_formatter))
    ax_loss.legend(fontsize=12)

    mask_f1_5 = all_metrics["val_f1@5"].is_not_null()
    mask_f1_10 = all_metrics["val_f1@10"].is_not_null()
    ax_f1.plot(
        all_metrics.filter(mask_f1_5)["global_step"],
        all_metrics.filter(mask_f1_5)["val_f1@5"],
        linestyle="--",
        marker="o",
        linewidth=1.5,
        markersize=2.5,
        label="F1@5",
    )
    ax_f1.plot(
        all_metrics.filter(mask_f1_10)["global_step"],
        all_metrics.filter(mask_f1_10)["val_f1@10"],
        linestyle="-",
        marker="s",
        linewidth=1.5,
        markersize=2.5,
        label="F1@10",
    )

    _style_ax(ax_f1, xlabel="Step", ylabel="F1 Score (validation set)")
    ax_f1.set_title("Key-phrase extraction metrics", fontsize=14)
    ax_f1.xaxis.set_major_formatter(FuncFormatter(_step_formatter))
    ax_f1.set_xlim(0, int(raw_steps[-1] * 1.05))

    for i, step in enumerate(val_steps):
        if i == 0:
            ax_f1.axvline(
                step,
                color="tab:red",
                linestyle=":",
                alpha=0.5,
                linewidth=1,
                label="Epoch end",
            )
        else:
            ax_f1.axvline(step, color="tab:red", linestyle=":", alpha=0.5, linewidth=1)

    ax_f1.legend(fontsize=12)
    fig.suptitle("Training Loss and F1 Score Over Training", fontsize=16)
    fig.tight_layout()

    if save and fig_dir is not None:
        print("Saving loss_f1_overview.png")
        fig.savefig(fig_dir / "loss_f1_overview.png", dpi=200, bbox_inches="tight")
        print("Saving loss_f1_overview.pdf")
        fig.savefig(fig_dir / "loss_f1_overview.pdf", bbox_inches="tight")

    return fig


def plot_modernbert_dynamics(
    all_val_metrics: pl.DataFrame,
    fig_dir: Path | None = None,
    save: bool = False,
) -> "Figure":
    """Plot detailed ModernBERT training dynamics with precision, recall, and F1."""
    import matplotlib.pyplot as plt
    from matplotlib.ticker import FuncFormatter

    val_steps = sorted(all_val_metrics["global_step"].to_list())
    max_step = int(all_val_metrics["global_step"].max() * 1.05)

    metrics_at_5 = ["val_precision@5", "val_recall@5", "val_f1@5"]
    metrics_at_10 = ["val_precision@10", "val_recall@10", "val_f1@10"]
    metric_labels = ["Precision", "Recall", "F1"]

    dyn2_fig, axes = plt.subplots(2, 3, figsize=(14, 5.5), sharex=True, sharey=True)

    for col_idx, (m5, m10, label) in enumerate(
        zip(metrics_at_5, metrics_at_10, metric_labels)
    ):
        mask_5 = all_val_metrics[m5].is_not_null()
        mask_10 = all_val_metrics[m10].is_not_null()
        steps_5 = all_val_metrics.filter(mask_5)["global_step"]
        steps_10 = all_val_metrics.filter(mask_10)["global_step"]

        axes[0, col_idx].plot(
            steps_5,
            all_val_metrics.filter(mask_5)[m5],
            linestyle="--",
            marker="o",
            linewidth=1.5,
            markersize=2.5,
            color="tab:blue",
        )
        axes[1, col_idx].plot(
            steps_10,
            all_val_metrics.filter(mask_10)[m10],
            linestyle="--",
            marker="o",
            linewidth=1.5,
            markersize=2.5,
            color="tab:blue",
        )

        for row in range(2):
            axes[row, col_idx].set_title(
                f"{label}@{5 if row == 0 else 10}", fontsize=14
            )
            axes[row, col_idx].grid(True, alpha=0.3)
            axes[row, col_idx].spines["top"].set_visible(False)
            axes[row, col_idx].spines["right"].set_visible(False)
            axes[row, col_idx].tick_params(axis="both", labelsize=12)

    for i, step in enumerate(val_steps):
        for row in range(2):
            for col in range(3):
                if row == 0 and col == 0 and i == 0:
                    axes[row, col].axvline(
                        step,
                        color="tab:red",
                        linestyle=":",
                        alpha=0.5,
                        linewidth=1.5,
                        label="Epoch end",
                    )
                else:
                    axes[row, col].axvline(
                        step, color="tab:red", linestyle=":", alpha=0.5, linewidth=1.5
                    )

    axes[0, 0].legend(fontsize=12)
    axes[1, 0].xaxis.set_major_formatter(FuncFormatter(_step_formatter))
    axes[0, 0].set_xlim(0, max_step)

    dyn2_fig.supxlabel("Global Step", fontsize=14)
    dyn2_fig.supylabel("Metric Value", fontsize=14)
    dyn2_fig.suptitle(
        "ModernBERT Training Dynamics: Precision, Recall, and F1", fontsize=16
    )

    if save and fig_dir is not None:
        print("Saving modernbert_dynamics.png")
        dyn2_fig.savefig(
            fig_dir / "modernbert_dynamics.png", dpi=200, bbox_inches="tight"
        )

    return dyn2_fig


def plot_keyphrase_stats(
    stats_df: pl.DataFrame,
    fig_dir: Path | None = None,
    save: bool = False,
) -> "Figure":
    """Create a bar chart comparing gold vs predicted keyphrase median counts."""
    import matplotlib.pyplot as plt

    val_benchmarks = [
        b
        for b in BENCHMARK_COLORS
        if b
        in stats_df.filter(pl.col("group") == "Validation Set FT")[
            "benchmark"
        ].to_list()
    ]
    train_benchmarks = [
        b
        for b in BENCHMARK_COLORS
        if b
        in stats_df.filter(pl.col("group") == "Training Set FT")["benchmark"].to_list()
    ]

    if not val_benchmarks and not train_benchmarks:
        fig, ax = plt.subplots(figsize=(4, 2))
        ax.text(
            0.5,
            0.5,
            "No data available",
            ha="center",
            va="center",
            transform=ax.transAxes,
        )
        ax.axis("off")
        return fig

    fig, (ax_val, ax_train) = plt.subplots(
        1,
        2,
        figsize=(5, 3.5),
        gridspec_kw={"width_ratios": [3, 1], "wspace": 0.15},
        sharey=True,
        layout="constrained",
    )

    def _plot_facet(ax, benchmarks, group_name, is_val=False):
        group_df = stats_df.filter(pl.col("group") == group_name)
        if len(group_df) == 0:
            ax.axis("off")
            return

        x = range(len(benchmarks))
        width = 0.35

        gold_vals = []
        pred_vals = []
        for bm in benchmarks:
            row = group_df.filter(pl.col("benchmark") == bm)
            if len(row) > 0:
                gold_vals.append(row["gold_median"][0])
                pred_vals.append(row["pred_median"][0])
            else:
                gold_vals.append(0)
                pred_vals.append(0)

        for i, bm in enumerate(benchmarks):
            color = BENCHMARK_COLORS[bm]
            ax.bar(
                i - width / 2,
                gold_vals[i],
                width,
                label="Gold" if i == 0 and is_val else "",
                color=color,
                edgecolor="black",
                linewidth=0.5,
            )
            ax.bar(
                i + width / 2,
                pred_vals[i],
                width,
                label="Predicted" if i == 0 and is_val else "",
                facecolor="white",
                edgecolor=color,
                linewidth=1.5,
                hatch="///",
            )

        ax.set_xticks(x)
        ax.set_xticklabels([bm.capitalize() for bm in benchmarks], fontsize=11)
        ax.set_xlim(-0.5, len(benchmarks) - 0.5)
        ax.set_title(f"{group_name.replace(' Set', ' set')}\nFine-tuning", fontsize=11)
        ax.grid(True, axis="y", alpha=0.3)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.tick_params(axis="both", labelsize=10)

        if is_val:
            ax.legend(fontsize=9, loc="upper right")

    if val_benchmarks:
        _plot_facet(ax_val, val_benchmarks, "Validation Set FT", is_val=True)

    if train_benchmarks:
        _plot_facet(ax_train, train_benchmarks, "Training Set FT")

    fig.supylabel("Median keyphrase count", fontsize=11)
    fig.supxlabel("Benchmark", fontsize=11)

    if save and fig_dir is not None:
        print("Saving keyphrase_stats.png")
        fig.savefig(fig_dir / "keyphrase_stats.png", dpi=200, bbox_inches="tight")
        print("Saving keyphrase_stats.pdf")
        fig.savefig(fig_dir / "keyphrase_stats.pdf", bbox_inches="tight")

    return fig


def plot_ecdf_confidence(
    keyphrase_df: pl.DataFrame,
    fig_dir: Path | None = None,
    save: bool = False,
) -> "Figure":
    """Plot ECDF of confidence scores by match type with rug plot.

    Args:
        keyphrase_df: DataFrame with score and match_type columns.
        fig_dir: Directory to save output files.
        save: Whether to write files to disk.

    Returns:
        The matplotlib Figure.
    """
    import matplotlib
    import matplotlib.pyplot as plt

    match_types = ["none", "partial", "full"]
    match_types_nice = {"none": "Non-match", "partial": "Partial", "full": "Exact"}
    colors = {"none": "#e74c3c", "partial": "#f39c12", "full": "#27ae60"}
    rug_levels = {"none": 0.15, "partial": 0.5, "full": 0.85}
    rug_height = 0.12

    ecdf_fig = plt.figure(figsize=(8, 6))
    gs = matplotlib.gridspec.GridSpec(
        2, 1, height_ratios=[1, 4], hspace=0.05, figure=ecdf_fig
    )
    rug_ax = ecdf_fig.add_subplot(gs[0])
    ecdf_ax = ecdf_fig.add_subplot(gs[1], sharex=rug_ax)

    scores_by_mt = {}
    for mt in match_types:
        scores = keyphrase_df.filter(pl.col("match_type") == mt)["score"].sort()
        scores_by_mt[mt] = scores
        n = len(scores)
        y = rug_levels[mt]
        rug_ax.vlines(
            scores,
            y - rug_height,
            y + rug_height,
            colors=colors[mt],
            alpha=0.8,
            linewidth=0.3,
        )
        ecdf_ax.ecdf(scores, label=f"{match_types_nice[mt]} (n={n})", color=colors[mt])

    threshold = 0.9
    rug_ax.axvline(x=threshold, color="gray", linestyle="--", alpha=0.5)
    ecdf_ax.axvline(x=threshold, color="gray", linestyle="--", alpha=0.5)

    for mt in match_types:
        scores = scores_by_mt[mt]
        n = len(scores)
        fraction = float((scores <= threshold).sum() / n)
        ecdf_ax.hlines(
            fraction, 0.5, threshold, colors=colors[mt], linestyles="--", alpha=0.6
        )
        ecdf_ax.text(
            0.51,
            fraction + 0.01,
            f"{fraction * 100:.1f}%",
            color="dimgray",
            fontsize=11,
            ha="left",
            va="bottom",
        )

    rug_ax.set_yticks([])
    rug_ax.set_ylim(0, 1)
    rug_ax.spines["top"].set_visible(False)
    rug_ax.spines["right"].set_visible(False)
    rug_ax.spines["bottom"].set_visible(False)
    rug_ax.spines["left"].set_visible(False)
    rug_ax.set_xlim(0.5, 1.0)

    ecdf_ax.set_xlabel("Confidence Score", fontsize=14)
    ecdf_ax.set_ylabel("ECDF\nFraction keyphrases <= score", fontsize=14)
    ecdf_ax.set_xlim(0.5, 1.0)
    ecdf_ax.legend(title="Keyphrase Match", fontsize=12, title_fontsize=13)
    ecdf_ax.grid(True, alpha=0.3)

    if save and fig_dir is not None:
        print("Saving ecdf_confidence_by_match_type.png")
        ecdf_fig.savefig(fig_dir / "ecdf_confidence_by_match_type.png", dpi=150)
        print("Saving ecdf_confidence_by_match_type.pdf")
        ecdf_fig.savefig(
            fig_dir / "ecdf_confidence_by_match_type.pdf", bbox_inches="tight"
        )

    return ecdf_fig


def plot_stacked_bar_match_type(
    keyphrase_df: pl.DataFrame,
    q25: float,
    q75: float,
    fig_dir: Path | None = None,
    save: bool = False,
) -> "Figure":
    """Plot stacked bar of match type proportions by accuracy tier.

    Args:
        keyphrase_df: DataFrame with accuracy_tier and match_type columns.
        q25: F1@5 25th percentile value (for axis labels).
        q75: F1@5 75th percentile value (for axis labels).
        fig_dir: Directory to save output files.
        save: Whether to write files to disk.

    Returns:
        The matplotlib Figure.
    """
    import matplotlib.pyplot as plt

    tier_order = ["low", "mid", "high"]

    proportions = (
        keyphrase_df.group_by(["accuracy_tier", "match_type"])
        .agg(count=pl.col("keyword").count())
        .with_columns(
            (pl.col("count") / pl.col("count").sum().over("accuracy_tier")).alias(
                "proportion"
            )
        )
    )

    bar_fig, bar_ax = plt.subplots(figsize=(6.5, 5))

    colors = {"none": "#e74c3c", "partial": "#f39c12", "full": "#27ae60"}
    labels = {"none": "Non-match", "partial": "Partial", "full": "Exact"}
    bottom = np.zeros(3)

    for mt in ["none", "partial", "full"]:
        vals = []
        counts = []
        for tier in tier_order:
            row = proportions.filter(
                (pl.col("accuracy_tier") == tier) & (pl.col("match_type") == mt)
            )
            vals.append(row["proportion"][0] if len(row) > 0 else 0.0)
            counts.append(int(row["count"][0] if len(row) > 0 else 0))
        bar_ax.bar(tier_order, vals, bottom=bottom, label=labels[mt], color=colors[mt])
        for i, (v, c) in enumerate(zip(vals, counts)):
            if v > 0.05:
                bar_ax.text(
                    i,
                    bottom[i] + v / 2,
                    str(c),
                    ha="center",
                    va="center",
                    fontsize=12,
                    fontweight="bold",
                    color="white",
                )
        bottom += np.array(vals)

    q25_str = f"{q25:.2f}"
    q75_str = f"{q75:.2f}"
    bar_ax.set_xticks(range(3))
    bar_ax.set_xticklabels(
        [
            f"Low (Q1)\nF1@5 <= {q25_str}",
            "Mid (Q2-Q3)",
            f"High (Q4)\nF1@5 >= {q75_str}",
        ]
    )

    bar_ax.set_ylabel("Proportion of Keyphrases (%)", fontsize=14)
    bar_ax.set_xlabel("Accuracy Tier", fontsize=14)
    bar_ax.legend(title="Match Status", fontsize=12, title_fontsize=13)
    bar_ax.grid(True, alpha=0.3, axis="y")
    bar_ax.spines["right"].set_visible(False)
    bar_ax.spines["top"].set_visible(False)

    bar_fig.tight_layout()

    if save and fig_dir is not None:
        print("Saving stacked_bar_match_type_by_tier.png")
        bar_fig.savefig(fig_dir / "stacked_bar_match_type_by_tier.png", dpi=150)
        print("Saving stacked_bar_match_type_by_tier.pdf")
        bar_fig.savefig(
            fig_dir / "stacked_bar_match_type_by_tier.pdf", bbox_inches="tight"
        )

    return bar_fig
