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

from keyword_pecker.constants import DEFAULT_DISALLOWED_PUNCTUATION
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
        choices=["valid", "test"],
        default="valid",
        help="Dataset split to evaluate (default: valid)",
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
        "--scheme",
        type=str,
        default="IOB",
        choices=["binary", "IOB"],
        help="Labeling scheme: binary (0/1) or IOB (0/1/2) (default: IOB)",
    )
    parser.add_argument(
        "--num-classes",
        type=int,
        default=3,
        help="Number of output classes (2 for binary, 3 for IOB) (default: 3)",
    )
    parser.add_argument(
        "--k-values",
        nargs="+",
        default=[5, 10, "N", "M"],
        help="k values for @k metrics: N = num gold, M = num pred (default: 5 10 N M)",
    )
    parser.add_argument(
        "--scoring-strategy",
        type=str,
        default="confidence",
        choices=["confidence", "unordered"],
        help="Scoring strategy (default: confidence)",
    )
    parser.add_argument(
        "--score-method",
        type=str,
        default="average",
        choices=["average", "max", "min", "geometric"],
        help="Score aggregation method (default: average)",
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
        eval_cfg = load_config("configs/training.yaml").get("evaluation", {})
        return {
            "batch_size": training_cfg.get("val_batch_size", args.batch_size),
            "device": training_cfg.get("device", args.device),
            "scheme": training_cfg.get("scheme", args.scheme),
            "num_classes": training_cfg.get("num_classes", args.num_classes),
            "k_values": training_cfg.get("k_values", args.k_values),
            "scoring_strategy": eval_cfg.get("scoring_strategy", "confidence"),
            "score_method": eval_cfg.get("score_method", "average"),
            "filter_punctuation": eval_cfg.get("filter_punctuation", True),
            "disallowed_punctuation": eval_cfg.get(
                "disallowed_punctuation", DEFAULT_DISALLOWED_PUNCTUATION
            ),
            "dedup_strategy": eval_cfg.get("dedup_strategy", "stem"),
        }
    return {
        "batch_size": args.batch_size,
        "device": args.device,
        "scheme": args.scheme,
        "num_classes": args.num_classes,
        "k_values": args.k_values,
        "scoring_strategy": args.scoring_strategy,
        "score_method": args.score_method,
        "filter_punctuation": True,
        "disallowed_punctuation": DEFAULT_DISALLOWED_PUNCTUATION,
        "dedup_strategy": "stem",
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
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True)

    if isinstance(checkpoint, dict) and "metadata" in checkpoint:
        num_classes = checkpoint["metadata"].get("num_classes", 2)
        class_weights = checkpoint["metadata"].get("class_weights")
        state_dict = checkpoint["state_dict"]
    else:
        num_classes = 2
        class_weights = None
        state_dict = checkpoint

    model_config = load_config("configs/models.yaml")["keyphrase_model"]
    model = KeyphraseModel(**model_config, num_classes=num_classes)
    model.load_state_dict(state_dict)
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
        num_classes=config["num_classes"],
        class_weights=class_weights,
        device=device,
        split=args.split,
        k_values=config["k_values"],
        scoring_strategy=config.get("scoring_strategy", "confidence"),
        score_method=config["score_method"],
        filter_punctuation=config["filter_punctuation"],
        disallowed_punctuation=config["disallowed_punctuation"],
        dedup_strategy=config["dedup_strategy"],
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
            "eval_config": {**config},
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
