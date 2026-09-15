"""Analysis module for young-haze keyword extraction evaluation."""

from keyword_pecker.analysis.data_loading import (
    BENCHMARKS,
    build_f1_table_wide,
    build_precision_recall_table,
    load_all_benchmark_per_doc,
    load_all_val_metrics,
    load_per_doc_data,
    load_run_test_row,
    load_run_test_row_from_evals,
    load_seke_multi_benchmark,
    load_seke_row,
    load_training_metrics,
    load_young_haze_row,
)
from keyword_pecker.analysis.per_doc_analysis import (
    assign_accuracy_tiers,
    build_keyphrase_stats_table,
    make_comparison_df,
    select_samples,
)
from keyword_pecker.analysis.plots import (
    plot_keyphrase_stats,
    plot_loss_f1_overview,
    plot_loss_overview,
    plot_modernbert_dynamics,
    plot_training_dynamics,
)
from keyword_pecker.analysis.tables import (
    BENCHMARK_COLORS,
    LATEX_COLOR_DEFINITIONS,
    compute_dataset_stats,
    load_external_benchmarks,
    make_dataset_stats_tex,
    make_metrics_table_gt,
    make_metrics_table_tex,
    make_precision_recall_table_tex,
    make_samples_table_tex,
)

__all__ = [
    # Data loading
    "BENCHMARKS",
    "build_f1_table_wide",
    "build_precision_recall_table",
    "load_all_benchmark_per_doc",
    "load_run_test_row",
    "load_run_test_row_from_evals",
    "load_seke_multi_benchmark",
    "load_seke_row",
    "load_young_haze_row",
    "load_per_doc_data",
    "load_training_metrics",
    "load_all_val_metrics",
    # Analysis
    "assign_accuracy_tiers",
    "build_keyphrase_stats_table",
    "make_comparison_df",
    "select_samples",
    # Tables
    "BENCHMARK_COLORS",
    "LATEX_COLOR_DEFINITIONS",
    "compute_dataset_stats",
    "load_external_benchmarks",
    "make_dataset_stats_tex",
    "make_metrics_table_gt",
    "make_metrics_table_tex",
    "make_precision_recall_table_tex",
    "make_samples_table_tex",
    # Plots
    "plot_keyphrase_stats",
    "plot_loss_overview",
    "plot_training_dynamics",
    "plot_loss_f1_overview",
    "plot_modernbert_dynamics",
]
