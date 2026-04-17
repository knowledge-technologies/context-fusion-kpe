"""Training script for the keyphrase extraction model.

Trains a KeyphraseModel (SBERT base encoder + refinement head) using the
two-level batching strategy from the original training loop:
  - Outer batch (gradient accumulation unit): outer_batch_size sentences
  - Inner batch (forward pass unit):          inner_batch_size sentences

Local metrics are always written to a JSONL file under:
  $EXPERIMENT_RUNS_FOLDER/<run_name>/metrics.jsonl

When --wandb is passed (and the ``tracking`` extra is installed), metrics are
also streamed to Weights & Biases and the run name is taken from W&B.

Usage:
    python scripts/train.py
    python scripts/train.py --epochs 10 --device cuda --dataset kp20k
    python scripts/train.py --base-lr 2e-5 --head-lr 5e-5 --pos-weight 5.0
    python scripts/train.py --wandb
"""

import argparse
import json

import polars as pl
import torch
from dotenv import load_dotenv

from keyword_pecker.data import compute_class_weights, get_derived_path
from keyword_pecker.models import KeyphraseModel
from keyword_pecker.trainer import evaluate_model, train
from keyword_pecker.utils import get_logger, init_run_dir, load_config, setup_wandb

load_dotenv()

logger = get_logger("train")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train the keyphrase extraction model."
    )
    parser.add_argument(
        "--dataset",
        default="kp20k",
        help="Dataset name (default: kp20k)",
    )
    parser.add_argument(
        "--from_config",
        action="store_true",
        default=False,
        help="Whether to load training params from `configs/training.yaml`. If false, CLI arguments are used.",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=5,
        help="Number of training epochs (default: 5)",
    )
    parser.add_argument(
        "--outer-batch-size",
        type=int,
        default=140,
        help="Sentences per gradient accumulation unit (default: 140)",
    )
    parser.add_argument(
        "--inner-batch-size",
        type=int,
        default=70,
        help="Sentences per forward pass (default: 70)",
    )
    parser.add_argument(
        "--base-lr",
        type=float,
        default=2e-5,
        help="Learning rate for base encoder parameters (default: 2e-5)",
    )
    parser.add_argument(
        "--head-lr",
        type=float,
        default=5e-5,
        help="Learning rate for refinement head parameters (default: 5e-5)",
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
        "--warmup-ratio",
        type=float,
        default=0.1,
        help="Fraction of training steps for warmup (default: 0.1)",
    )
    parser.add_argument(
        "--log-every",
        type=int,
        default=1000,
        help="Print loss every N optimizer steps (default: 1000)",
    )
    parser.add_argument(
        "--device",
        default="cpu",
        help="Device to train on: cpu, cuda, mps (default: cpu)",
    )
    parser.add_argument(
        "--val-batch-size",
        type=int,
        default=100,
        help="Batch size for validation (default: 100)",
    )
    parser.add_argument(
        "--wandb",
        action="store_true",
        default=False,
        help="Enable Weights & Biases experiment tracking (requires the 'tracking' extra).",
    )
    parser.add_argument(
        "--no-test-eval",
        action="store_true",
        default=False,
        help="Skip final evaluation on the test split after training completes.",
    )
    parser.add_argument(
        "--save-weights",
        action="store_true",
        default=False,
        help="Save trained model weights (model.pt) to the run directory after training.",
    )
    parser.add_argument(
        "--k-values",
        nargs="+",
        default=[5, 10, "N", "M"],
        help="k values for @k metrics: N = num gold, M = num pred (default: 5 10 N M)",
    )
    parser.add_argument(
        "--early-stopping-patience",
        type=int,
        default=0,
        help="Epochs without improvement before stopping (0 = disabled, default: 0)",
    )
    parser.add_argument(
        "--early-stopping-delta",
        type=float,
        default=0.0,
        help="Minimum improvement to reset patience counter (default: 0.0)",
    )
    parser.add_argument(
        "--early-stopping-metric",
        default="val_loss",
        help="Metric to monitor for early stopping (default: val_loss)",
    )
    parser.add_argument(
        "--early-stopping-mode",
        default="min",
        choices=["min", "max"],
        help="'min' for loss-like metrics, 'max' for F1-like metrics (default: min)",
    )
    parser.add_argument(
        "--early-stopping-check-every",
        type=int,
        default=0,
        help="Run validation every N optimizer steps for early stopping (0 = epoch-only, default: 0)",
    )
    return parser.parse_args()


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------


def main() -> None:
    args = parse_args()

    # Load model and training configuration
    model_config = load_config("configs/models.yaml")["keyphrase_model"]

    train_cfg_tmp = load_config("configs/training.yaml")
    wandb_config = train_cfg_tmp.get("wandb", {})
    eval_cfg = train_cfg_tmp.get("evaluation", {})

    # Flatten all hyperparams into one dict for W&B config and training_onfig.json
    if args.from_config:
        training_config = train_cfg_tmp.get("training", {})
        hyperparams: dict = {
            "dataset": args.dataset,
            **training_config,
            **model_config,
            "run_test_eval": not args.no_test_eval,
            "save_weights": args.save_weights,
            "max_keyphrases": eval_cfg.get("max_keyphrases", 10),
            "max_keyphrase_length": eval_cfg.get("max_keyphrase_length", 5),
            "filter_punctuation": eval_cfg.get("filter_punctuation", True),
            "disallowed_punctuation": eval_cfg.get(
                "disallowed_punctuation", "!#$%&'()*+,.:;<=>?@[]^_`{|}~"
            ),
            "dedup_strategy": eval_cfg.get("dedup_strategy", "stem"),
        }
    else:
        hyperparams: dict = {
            "dataset": args.dataset,
            "epochs": args.epochs,
            "outer_batch_size": args.outer_batch_size,
            "inner_batch_size": args.inner_batch_size,
            "base_lr": args.base_lr,
            "head_lr": args.head_lr,
            "scheme": args.scheme,
            "num_classes": args.num_classes,
            "warmup_ratio": args.warmup_ratio,
            "log_every": args.log_every,
            "device": args.device,
            "val_batch_size": args.val_batch_size,
            "k_values": args.k_values,
            "early_stopping_patience": args.early_stopping_patience,
            "early_stopping_delta": args.early_stopping_delta,
            "early_stopping_metric": args.early_stopping_metric,
            "early_stopping_mode": args.early_stopping_mode,
            "early_stopping_check_every": args.early_stopping_check_every,
            **model_config,
            "run_test_eval": not args.no_test_eval,
            "save_weights": args.save_weights,
            "max_keyphrases": 10,
            "max_keyphrase_length": 5,
            "filter_punctuation": True,
            "disallowed_punctuation": "!#$%&'()*+,.:;<=>?@[]^_`{|}~",
            "dedup_strategy": "stem",
        }

    device = torch.device(hyperparams["device"])

    # Single W&B setup call. Returns (wandb_module, run_name) — wandb_module is
    # None if W&B is disabled or not installed. run_name is used for the local
    # run directory in both cases.
    wandb, run_name = setup_wandb(
        enabled=args.wandb,
        config=hyperparams,
        project=wandb_config.get("project"),
        entity=wandb_config.get("entity"),
        tags=wandb_config.get("tags") or [],
        notes=wandb_config.get("notes", ""),
    )

    # Set up local run directory and files
    run_dir = init_run_dir(run_name)
    metrics_filename = run_dir / "metrics.json"
    config_filename = run_dir / "config.json"

    config_filename.write_text(json.dumps(hyperparams, indent=2))
    logger.info(f"Run directory: {run_dir}")

    # Load data and model (defer test_df until needed to reduce memory during training)
    train_df = pl.read_parquet(get_derived_path(args.dataset, "train"))
    val_df = pl.read_parquet(get_derived_path(args.dataset, "valid"))
    model = KeyphraseModel(**model_config).to(device)

    # Compute class weights from training data (used for loss weighting)
    num_classes = hyperparams["num_classes"]
    class_weights = compute_class_weights(train_df, num_classes)
    hyperparams["class_weights"] = class_weights
    logger.info(f"Class weights (from train split): {class_weights}")

    try:
        train(
            model,
            train_df,
            epochs=hyperparams["epochs"],
            outer_batch_size=hyperparams["outer_batch_size"],
            inner_batch_size=hyperparams["inner_batch_size"],
            base_lr=hyperparams["base_lr"],
            head_lr=hyperparams["head_lr"],
            num_classes=hyperparams["num_classes"],
            class_weights=hyperparams["class_weights"],
            warmup_ratio=hyperparams["warmup_ratio"],
            log_every=hyperparams["log_every"],
            device=device,
            metrics_filename=metrics_filename,
            wandb=wandb,
            run_dir=run_dir,
            save_weights=hyperparams["save_weights"],
            val_df=val_df,
            val_batch_size=hyperparams["val_batch_size"],
            k_values=hyperparams.get("k_values"),
            early_stopping_patience=hyperparams.get("early_stopping_patience", 0),
            early_stopping_delta=hyperparams.get("early_stopping_delta", 0.0),
            early_stopping_metric=hyperparams.get("early_stopping_metric", "val_loss"),
            early_stopping_mode=hyperparams.get("early_stopping_mode", "min"),
            early_stopping_check_every=hyperparams.get("early_stopping_check_every", 0),
            max_keyphrases=hyperparams["max_keyphrases"],
            max_keyphrase_length=hyperparams["max_keyphrase_length"],
            filter_punctuation=hyperparams["filter_punctuation"],
            disallowed_punctuation=hyperparams["disallowed_punctuation"],
            dedup_strategy=hyperparams["dedup_strategy"],
        )

        # Final evaluation on the test split
        if hyperparams["run_test_eval"]:
            logger.info("Running final evaluation on test split...")
            # Load test data only when needed (deferred from startup to save memory)
            test_df = pl.read_parquet(get_derived_path(args.dataset, "test"))
            test_metrics = evaluate_model(
                model,
                test_df,
                batch_size=hyperparams["val_batch_size"],
                num_classes=hyperparams["num_classes"],
                class_weights=hyperparams["class_weights"],
                device=device,
                split="test",
                k_values=hyperparams.get("k_values"),
                metrics_filename=metrics_filename,
                wandb=wandb,
                global_step=0,
                epoch=hyperparams["epochs"],
                max_keyphrases=hyperparams["max_keyphrases"],
                max_keyphrase_length=hyperparams["max_keyphrase_length"],
                filter_punctuation=hyperparams["filter_punctuation"],
                disallowed_punctuation=hyperparams["disallowed_punctuation"],
                dedup_strategy=hyperparams["dedup_strategy"],
            )
            k_val = hyperparams.get("k_values", [5])[0]
            logger.info(
                f"Test loss: {test_metrics['test_loss']:.6f}, "
                f"F1@{k_val}: {test_metrics[f'f1@{k_val}']:.4f}"
            )

    finally:
        if wandb:
            wandb.finish()


if __name__ == "__main__":
    main()
