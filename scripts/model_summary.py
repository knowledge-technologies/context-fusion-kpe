"""Model parameter summary script.

Instantiates the KeyphraseModel with parameters from config files,
counts parameters by component, and outputs a CSV + LaTeX table.

Usage:
    python scripts/model_summary.py
    python scripts/model_summary.py --output model_params.csv --latex-output model_params.tex
"""

import argparse
from pathlib import Path

import polars as pl
import torch

from ctx_fusion.models import KeyphraseModel
from ctx_fusion.utils.config import load_config


def count_params(module: torch.nn.Module) -> tuple[int, int, int]:
    """Count total, trainable, and frozen parameters in a module.

    Returns:
        Tuple of (total_params, trainable_params, frozen_params).
    """
    total = sum(p.numel() for p in module.parameters())
    trainable = sum(p.numel() for p in module.parameters() if p.requires_grad)
    frozen = total - trainable
    return total, trainable, frozen


def latex_escape(text: str) -> str:
    """Escape special LaTeX characters in a string."""
    chars = {
        "\\": r"\textbackslash{}",
        "&": r"\&",
        "%": r"\%",
        "$": r"\$",
        "#": r"\#",
        "_": r"\_",
        "{": r"\{",
        "}": r"\}",
        "~": r"\textasciitilde{}",
        "^": r"\textasciicircum{}",
    }
    result = []
    for ch in text:
        result.append(chars.get(ch, ch))
    return "".join(result)


def format_num(n: int) -> str:
    """Format integer with comma separators."""
    return f"{n:,}"


def format_millions(n: int) -> str:
    """Format integer as millions with 1 decimal place."""
    return f"{n / 1_000_000:.1f}"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Summarize model parameters and output CSV + LaTeX table."
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Path to write CSV output (default: print to stdout)",
    )
    parser.add_argument(
        "--latex-output",
        type=str,
        default=None,
        help="Path to write LaTeX table (default: print to stdout)",
    )
    parser.add_argument(
        "--caption",
        type=str,
        default="Model parameter breakdown. All parameters are trainable (base encoder unfrozen).",
        help="LaTeX table caption (default: model parameter breakdown)",
    )
    parser.add_argument(
        "--label",
        type=str,
        default="tab:model_params",
        help="LaTeX table label (default: tab:model_params)",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    # Load configuration
    model_config = load_config("configs/models.yaml")["keyphrase_model"]
    training_config = load_config("configs/training.yaml")["training"]
    model_config["num_classes"] = training_config["num_classes"]

    # Instantiate model (downloads ModernBERT-base weights)
    print("Loading KeyphraseModel with config:")
    for k, v in model_config.items():
        print(f"  {k}: {v}")
    print()

    model = KeyphraseModel(**model_config)

    # Count parameters
    base_total, base_trainable, base_frozen = count_params(model.base_encoder)
    head_total, head_trainable, head_frozen = count_params(model.refinement_head)
    total_total = base_total + head_total
    total_trainable = base_trainable + head_trainable
    total_frozen = base_frozen + head_frozen

    # Build DataFrame
    rows = [
        {
            "component": f"Base Encoder ({model_config['base_encoder_name']})",
            "total_params": base_total,
            "trainable_params": base_trainable,
            "frozen_params": base_frozen,
            "trainable_pct": round(base_trainable / base_total * 100, 1)
            if base_total > 0
            else 0.0,
        },
        {
            "component": "Refinement Head",
            "total_params": head_total,
            "trainable_params": head_trainable,
            "frozen_params": head_frozen,
            "trainable_pct": round(head_trainable / head_total * 100, 1)
            if head_total > 0
            else 0.0,
        },
        {
            "component": "Total Model",
            "total_params": total_total,
            "trainable_params": total_trainable,
            "frozen_params": total_frozen,
            "trainable_pct": round(total_trainable / total_total * 100, 1)
            if total_total > 0
            else 0.0,
        },
    ]

    df = pl.DataFrame(rows)

    # Print summary to stdout
    print("Parameter Summary:")
    print("-" * 80)
    for row in rows:
        print(f"  {row['component']}")
        print(f"    Total:       {format_num(row['total_params']):>12}")
        print(f"    Trainable:   {format_num(row['trainable_params']):>12}")
        print(f"    Frozen:      {format_num(row['frozen_params']):>12}")
        print(f"    Trainable:   {row['trainable_pct']:>6.1f}%")
        print()

    # Write CSV
    csv_content = df.write_csv()
    if args.output:
        Path(args.output).write_text(csv_content)
        print(f"CSV written to: {args.output}")
    else:
        print("CSV output:")
        print(csv_content)

    # Generate LaTeX table (booktabs style, matching notebook convention)
    # Numbers in millions to fit within a single column
    caption = latex_escape(args.caption)
    latex_lines = [
        r"\begin{table}[htbp]",
        r"  \centering",
        f"  \\caption{{{caption}}}",
        f"  \\label{{{args.label}}}",
        r"  \begin{tabular}{lr}",
        r"    \toprule",
        r"    \textbf{Component} & \textbf{Parameters (M)} \\",
        r"    \midrule",
    ]

    for row in rows:
        component = latex_escape(row["component"])
        if component == "Total Model":
            latex_lines.append(r"    \midrule")
        latex_lines.append(
            f"    {component} & {format_millions(row['total_params'])} \\\\"
        )

    latex_lines.append(r"    \bottomrule")
    latex_lines.append(r"  \end{tabular}")
    latex_lines.append(r"\end{table}")

    latex_table = "\n".join(latex_lines)

    if args.latex_output:
        Path(args.latex_output).write_text(latex_table)
        print(f"LaTeX table written to: {args.latex_output}")
    else:
        print("LaTeX table:")
        print(latex_table)


if __name__ == "__main__":
    main()
