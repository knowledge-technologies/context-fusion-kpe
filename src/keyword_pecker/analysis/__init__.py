"""Analysis module for young-haze keyword extraction evaluation."""

from keyword_pecker.analysis.data_loading import (
    load_all_val_metrics,
    load_per_doc_data,
    load_seke_row,
    load_training_metrics,
    load_young_haze_row,
)
from keyword_pecker.analysis.per_doc_analysis import (
    assign_accuracy_tiers,
    make_comparison_df,
    select_samples,
)
from keyword_pecker.analysis.plots import (
    plot_loss_f1_overview,
    plot_loss_overview,
    plot_modernbert_dynamics,
    plot_training_dynamics,
)
from keyword_pecker.analysis.tables import (
    make_metrics_table_gt,
    make_metrics_table_tex,
    make_samples_table_tex,
)

__all__ = [
    # Data loading
    "load_seke_row",
    "load_young_haze_row",
    "load_per_doc_data",
    "load_training_metrics",
    "load_all_val_metrics",
    # Analysis
    "assign_accuracy_tiers",
    "make_comparison_df",
    "select_samples",
    # Tables
    "make_metrics_table_gt",
    "make_metrics_table_tex",
    "make_samples_table_tex",
    # Plots
    "plot_loss_overview",
    "plot_training_dynamics",
    "plot_loss_f1_overview",
    "plot_modernbert_dynamics",
]
