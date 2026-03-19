"""Standalone evaluation script for keyphrase extraction model checkpoints.

Evaluates a trained model checkpoint on a validation or test split of a
benchmark dataset, computing loss and keyphrase extraction metrics.

Usage:
    python scripts/evaluate.py --checkpoint path/to/model.pt
    python scripts/evaluate.py --checkpoint model.pt --dataset kp20k --split test
    python scripts/evaluate.py --checkpoint model.pt --from_config --device cuda
"""

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path

import polars as pl
import torch

from keyword_pecker.data import get_derived_path
from keyword_pecker.models import KeyphraseModel
from keyword_pecker.trainer import evaluate_model
from keyword_pecker.utils import get_logger, load_config

logger = get_logger("evaluate")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate a trained keyphrase model checkpoint."
    )
    parser.add_argument(
        "--checkpoint",
        required=True,
        help="Path to .pt checkpoint file",
    )
    parser.add_argument(
        "--dataset",
        default="kp20k",
        help="Dataset name (default: kp20k)",
    )
    parser.add_argument(
        "--split",
        choices=["val", "test"],
        default="val",
        help="Dataset split to evaluate (default: val)",
    )
    parser.add_argument(
        "--from-config",
        action="store_true",
        help="Load evaluation params from configs/training.yaml",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=100,
        help="Evaluation batch size (default: 100)",
    )
    parser.add_argument(
        "--device",
        default="cpu",
        help="Device for inference: cpu, cuda, mps (default: cpu)",
    )
    parser.add_argument(
        "--pos-weight",
        type=float,
        default=5.0,
        help="Positive class weight for BCEWithLogitsLoss (default: 5.0)",
    )
    parser.add_argument(
        "--k-values",
        type=int,
        nargs="+",
        default=[5, 10],
        help="k values for @k metrics (default: 5 10)",
    )
    parser.add_argument(
        "--output",
        help="Output JSON file path (auto-generated if not provided)",
    )
    return parser.parse_args()


def load_eval_config(args: argparse.Namespace) -> dict:
    """Load config from training.yaml if --from_config, merge with CLI args."""
    if args.from_config:
        training_cfg = load_config("configs/training.yaml").get("training", {})
        return {
            "batch_size": training_cfg.get("val_batch_size", args.batch_size),
            "device": training_cfg.get("device", args.device),
            "pos_weight": training_cfg.get("pos_weight", args.pos_weight),
            "k_values": training_cfg.get("k_values", args.k_values),
        }
    return {
        "batch_size": args.batch_size,
        "device": args.device,
        "pos_weight": args.pos_weight,
        "k_values": args.k_values,
    }


def validate_inputs(checkpoint_path: Path, data_path: Path, split: str) -> None:
    """Validate that checkpoint and dataset split exist."""
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")
    if not data_path.exists():
        raise FileNotFoundError(
            f"Dataset split '{split}' not found: {data_path}. "
            f"Run 'python scripts/prepare_data.py --dataset {data_path.parent.name}' first."
        )


def generate_output_filename(dataset: str, split: str) -> str:
    """Generate output filename with ISO 8601 timestamp."""
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    return f"eval_{dataset}_{split}_{timestamp}.json"


def main() -> None:
    args = parse_args()
    config = load_eval_config(args)

    checkpoint_path = Path(args.checkpoint)
    data_path = get_derived_path(args.dataset, args.split)

    validate_inputs(checkpoint_path, data_path, args.split)

    logger.info(f"Loading checkpoint: {checkpoint_path}")
    model_config = load_config("configs/models.yaml")["keyphrase_model"]
    model = KeyphraseModel(**model_config)
    model.load_state_dict(
        torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    )
    device = torch.device(config["device"])
    model = model.to(device)
    model.eval()

    logger.info(f"Loading dataset: {args.dataset}/{args.split}")
    df = pl.read_parquet(data_path)

    logger.info(
        f"Evaluating (batch_size={config['batch_size']}, device={config['device']})..."
    )
    metrics = evaluate_model(
        model,
        df,
        batch_size=config["batch_size"],
        pos_weight=config["pos_weight"],
        device=device,
        split=args.split,
        k_values=config["k_values"],
    )

    output_path = (
        Path(args.output)
        if args.output
        else Path(generate_output_filename(args.dataset, args.split))
    )

    output_data = {
        "metadata": {
            "checkpoint": str(checkpoint_path),
            "dataset": args.dataset,
            "split": args.split,
            "device": config["device"],
            "batch_size": config["batch_size"],
            "pos_weight": config["pos_weight"],
            "k_values": config["k_values"],
            "timestamp": datetime.now(timezone.utc).isoformat(),
        },
        "metrics": {k: v for k, v in metrics.items() if k != "samples"},
        "samples": metrics["samples"],
    }

    output_path.write_text(json.dumps(output_data, indent=2))

    logger.info(f"Checkpoint: {checkpoint_path}")
    logger.info(f"Dataset: {args.dataset}/{args.split}")
    logger.info(f"Loss: {metrics[f'{args.split}_loss']:.6f}")
    for k in config["k_values"]:
        logger.info(f"F1@{k}: {metrics[f'f1@{k}']:.4f}")
    logger.info(f"Results saved to: {output_path}")


if __name__ == "__main__":
    main()
