"""Batch evaluation script for keyphrase extraction model checkpoints.

Evaluates a trained model checkpoint on multiple benchmark datasets,
saving results to the run's evals/ directory.

Usage:
    python scripts/evaluate_all.py --run-id worthy-firefly-11
    python scripts/evaluate_all.py --run-id worthy-firefly-11 --datasets inspec,kp20k
    python scripts/evaluate_all.py --run-id worthy-firefly-11 --checkpoint model_e5.pt --device cuda
"""

import argparse
import json
import os
from datetime import datetime, timezone
from pathlib import Path

import polars as pl
import torch

from keyword_pecker.data import get_derived_path
from keyword_pecker.models import KeyphraseModel
from keyword_pecker.trainer import evaluate_model
from keyword_pecker.utils import get_logger, load_config

logger = get_logger("evaluate_all")

DEFAULT_DATASETS = ["kp20k", "kptimes", "inspec", "krapivin", "jptimes"]


def get_runs_root() -> Path:
    """Return the root directory for all experiment runs."""
    runs_env = os.environ.get("EXPERIMENT_RUNS_FOLDER")
    if runs_env:
        return Path(runs_env)

    dataset_env = os.environ.get("DATASET_FOLDER")
    if dataset_env:
        return Path(dataset_env) / "runs"

    return Path(__file__).parent.parent / "runs"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate a trained keyphrase model on multiple datasets."
    )
    parser.add_argument(
        "--run-id",
        required=True,
        help="W&B run name (directory name in runs/)",
    )
    parser.add_argument(
        "--datasets",
        help=f"Comma-separated dataset names (default: {','.join(DEFAULT_DATASETS)})",
    )
    parser.add_argument(
        "--checkpoint",
        default="model.pt",
        help="Checkpoint filename relative to run dir (default: model.pt)",
    )
    parser.add_argument(
        "--split",
        choices=["val", "test"],
        default="test",
        help="Dataset split to evaluate (default: test)",
    )
    parser.add_argument(
        "--from-config",
        action="store_true",
        help="Load evaluation params from configs/training.yaml",
    )
    parser.add_argument(
        "--batch-size",
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
    return parser.parse_args()


def load_eval_config(args: argparse.Namespace) -> dict:
    """Load config from training.yaml if --from-config, merge with CLI args."""
    if args.from_config:
        training_cfg = load_config("configs/training.yaml").get("training", {})
        return {
            "batch_size": training_cfg.get("val_batch_size", args.batch_size),
            "device": training_cfg.get("device", args.device),
            "scheme": training_cfg.get("scheme", args.scheme),
            "num_classes": training_cfg.get("num_classes", args.num_classes),
            "k_values": training_cfg.get("k_values", args.k_values),
        }
    return {
        "batch_size": args.batch_size,
        "device": args.device,
        "scheme": args.scheme,
        "num_classes": args.num_classes,
        "k_values": args.k_values,
    }


def main() -> None:
    args = parse_args()
    config = load_eval_config(args)

    runs_root = get_runs_root()
    run_dir = runs_root / args.run_id

    if not run_dir.exists():
        raise FileNotFoundError(f"Run directory not found: {run_dir}")

    checkpoint_path = run_dir / args.checkpoint
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

    datasets = args.datasets.split(",") if args.datasets else DEFAULT_DATASETS

    evals_dir = run_dir / "evals"
    evals_dir.mkdir(exist_ok=True)

    logger.info(f"Run directory: {run_dir}")
    logger.info(f"Checkpoint: {checkpoint_path}")
    logger.info(f"Datasets: {', '.join(datasets)}")
    logger.info(f"Split: {args.split}")
    logger.info(f"Device: {config['device']}")

    logger.info(f"Loading model from: {checkpoint_path}")
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

    results = []
    for dataset in datasets:
        data_path = get_derived_path(dataset, args.split)

        if not data_path.exists():
            logger.warning(f"Dataset split not found: {data_path}. Skipping {dataset}.")
            continue

        logger.info(f"Evaluating {dataset}/{args.split}...")
        df = pl.read_parquet(data_path)

        metrics = evaluate_model(
            model,
            df,
            batch_size=config["batch_size"],
            num_classes=config["num_classes"],
            class_weights=class_weights,
            device=device,
            split=args.split,
            k_values=config["k_values"],
        )

        output_path = evals_dir / f"{dataset}_{args.split}.json"
        output_data = {
            "metadata": {
                "checkpoint": str(checkpoint_path),
                "dataset": dataset,
                "split": args.split,
                "device": config["device"],
                "batch_size": config["batch_size"],
                "scheme": config["scheme"],
                "num_classes": config["num_classes"],
                "k_values": config["k_values"],
                "timestamp": datetime.now(timezone.utc).isoformat(),
            },
            "metrics": {k: v for k, v in metrics.items() if k != "samples"},
            "samples": metrics["samples"],
        }

        output_path.write_text(json.dumps(output_data, indent=2))
        logger.info(f"  Loss: {metrics[f'{args.split}_loss']:.6f}")
        for k in config["k_values"]:
            logger.info(f"  F1@{k}: {metrics[f'f1@{k}']:.4f}")
        logger.info(f"  Saved to: {output_path}")

        results.append(
            {
                "dataset": dataset,
                "loss": metrics[f"{args.split}_loss"],
                **{f"f1@{k}": metrics[f"f1@{k}"] for k in config["k_values"]},
            }
        )

    logger.info("\nSummary:")
    logger.info("-" * 60)
    for r in results:
        f1_str = " ".join(f"F1@{k}={r[f'f1@{k}']:.4f}" for k in config["k_values"])
        logger.info(f"{r['dataset']:12} Loss={r['loss']:.6f} {f1_str}")
    logger.info("-" * 60)
    logger.info(f"Results saved to: {evals_dir}")


if __name__ == "__main__":
    main()
