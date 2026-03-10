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
import math
import os
import types
from pathlib import Path

import polars as pl
import torch
import tqdm
from dotenv import load_dotenv

from keyword_pecker.data import create_keyphrase_dataloader, get_derived_path
from keyword_pecker.models import KeyphraseModel
from keyword_pecker.trainer import validate
from keyword_pecker.utils import get_logger, load_config, setup_wandb

load_dotenv()

logger = get_logger("train")


# ---------------------------------------------------------------------------
# Local run directory helpers
# ---------------------------------------------------------------------------


def get_runs_root() -> Path:
    """Return the root directory for all experiment runs.

    Reads ``EXPERIMENT_RUNS_FOLDER`` from the environment (set in ``.env``).
    Falls back to ``$DATASET_FOLDER/runs`` if not defined.
    """
    runs_env = os.environ.get("EXPERIMENT_RUNS_FOLDER")
    if runs_env:
        return Path(runs_env)

    dataset_env = os.environ.get("DATASET_FOLDER")
    if dataset_env:
        return Path(dataset_env) / "runs"

    # Last resort: write next to the project root
    return Path(__file__).parent.parent / "runs"


def init_run_dir(run_name: str) -> Path:
    """Create and return the directory for a single run."""
    run_dir = get_runs_root() / run_name
    run_dir.mkdir(parents=True, exist_ok=True)
    return run_dir


# ---------------------------------------------------------------------------
# Training loop
# ---------------------------------------------------------------------------


def train(
    model: KeyphraseModel,
    train_df: pl.DataFrame,
    *,
    epochs: int,
    outer_batch_size: int,
    inner_batch_size: int,
    base_lr: float,
    head_lr: float,
    pos_weight: float,
    log_every: int,
    device: torch.device,
    metrics_filename: Path,
    wandb: types.ModuleType | None,
    val_df: pl.DataFrame | None = None,
    val_batch_size: int = 100,
) -> None:
    """Full training routine: optimizer, loss, dataloader creation, and epoch loop.

    Replicates the two-level batching semantics of the original training loop
    (tmp/train-model.py lines 209-305):
      - Gradients accumulate across inner batches within each outer chunk
      - optimizer.step() is called once per outer chunk (every outer_batch_size sentences)
      - The dataloader is re-created each epoch to re-shuffle at document level
        (equivalent to the legacy train_df.sample(frac=1.0))

    Args:
        model: KeyphraseModel to train (already moved to device).
        train_df: Polars DataFrame with columns [doc_id, doc_row, sent_num,
            token_ids, keyword_mask], sorted by (doc_id, doc_row, sent_num).
        epochs: Number of full passes over the training data.
        outer_batch_size: Sentences per gradient accumulation unit.
        inner_batch_size: Sentences per forward pass.
        base_lr: Learning rate for base encoder parameters.
        head_lr: Learning rate for refinement head parameters.
        pos_weight: Positive class weight for BCEWithLogitsLoss.
        log_every: Print loss every this many optimizer steps.
        device: Device to run on.
        metrics_filename: Path to the JSONL file where metrics are appended.
        wandb: The wandb module if W&B tracking is active, or ``None``.
        val_df: Optional Polars DataFrame for validation (same schema as train_df).
        val_batch_size: Batch size for validation forward passes (default: 100).
    """
    optimizer = torch.optim.AdamW(
        model.get_param_groups(base_lr=base_lr, head_lr=head_lr)
    )

    # reduction="none" so we can apply the attention mask and compute a masked
    # mean: (loss * attn_mask).sum() / attn_mask.sum()
    loss_fn = torch.nn.BCEWithLogitsLoss(
        reduction="none",
        pos_weight=torch.tensor([[pos_weight]], device=device),
    )

    inner_per_outer = math.ceil(outer_batch_size / inner_batch_size)

    global_step = 0

    metrics_file = metrics_filename.open("a")

    for epoch in range(epochs):
        logger.info(f"Epoch {epoch + 1}/{epochs}")
        if wandb:
            wandb.log({"epoch": epoch + 1}, step=global_step)

        # Re-create each epoch to re-shuffle documents
        train_loader = create_keyphrase_dataloader(
            df=train_df,
            outer_batch_size=outer_batch_size,
            inner_batch_size=inner_batch_size,
            shuffle=True,
            drop_last=False,
            num_workers=0,
        )

        model.train()
        optimizer.zero_grad()

        outer_step = 0
        inner_count = 0
        last_loss: float | None = None
        epoch_loss_sum = 0.0
        epoch_loss_count = 0

        total_outer = math.ceil(len(train_loader) / inner_per_outer)
        progress = tqdm.trange(total_outer, desc="steps")

        for batch in train_loader:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            target = batch["target"].to(device)
            sentence_counts = batch["sentence_counts"]

            logits = model(input_ids, attention_mask, sentence_counts)

            # Masked mean loss: average only over non-padding token positions
            loss = loss_fn(logits, target) * attention_mask
            loss = loss.sum() / attention_mask.sum()
            loss.backward()

            last_loss = loss.detach().cpu().item()

            # Eagerly free activations
            del logits, loss, input_ids, attention_mask, target

            inner_count += 1

            if inner_count == inner_per_outer:
                optimizer.step()
                optimizer.zero_grad()

                if last_loss is not None:
                    epoch_loss_sum += last_loss
                    epoch_loss_count += 1

                    if outer_step % log_every == 0:
                        logger.info(f"step {global_step} loss: {last_loss:.6f}")

                        record = {
                            "epoch": epoch + 1,
                            "outer_step": outer_step,
                            "global_step": global_step,
                            "train_loss": last_loss,
                        }
                        metrics_file.write(json.dumps(record) + "\n")
                        metrics_file.flush()

                        if wandb:
                            wandb.log({"train/loss": last_loss}, step=global_step)

                outer_step += 1
                global_step += 1
                inner_count = 0
                progress.update(1)

        # Step on any remaining accumulated gradients (last incomplete outer chunk)
        if inner_count > 0:
            optimizer.step()
            optimizer.zero_grad()

            if last_loss is not None:
                epoch_loss_sum += last_loss
                epoch_loss_count += 1

                if outer_step % log_every == 0:
                    logger.info(f"step {global_step} loss: {last_loss:.6f}")

                    record = {
                        "epoch": epoch + 1,
                        "outer_step": outer_step,
                        "global_step": global_step,
                        "train_loss": last_loss,
                    }
                    metrics_file.write(json.dumps(record) + "\n")
                    metrics_file.flush()

                    if wandb:
                        wandb.log({"train/loss": last_loss}, step=global_step)

                global_step += 1
                progress.update(1)

        progress.close()

        # Epoch-level summary
        if epoch_loss_count > 0:
            mean_loss = epoch_loss_sum / epoch_loss_count
            logger.info(f"Epoch {epoch + 1} mean loss: {mean_loss:.6f}")

            metrics_file.write(
                json.dumps({"epoch": epoch + 1, "epoch_mean_loss": mean_loss}) + "\n"
            )
            metrics_file.flush()

            if wandb:
                wandb.log({"train/epoch_mean_loss": mean_loss}, step=global_step)

        # Run validation at end of each epoch
        if val_df is not None:
            val_metrics = validate(
                model,
                val_df,
                batch_size=val_batch_size,
                pos_weight=pos_weight,
                device=device,
                metrics_filename=metrics_filename,
                wandb=wandb,
                global_step=global_step,
                epoch=epoch + 1,
            )
            logger.info(
                f"Epoch {epoch + 1} val loss: {val_metrics['val_loss']:.6f}, "
                f"P/R/F1: {val_metrics['precision']:.4f}/"
                f"{val_metrics['recall']:.4f}/{val_metrics['f1']:.4f}"
            )

    metrics_file.close()


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
        "--pos-weight",
        type=float,
        default=5.0,
        help="Positive class weight for BCEWithLogitsLoss (default: 5.0)",
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

    # Flatten all hyperparams into one dict for W&B config and training_onfig.json
    if args.from_config:
        training_config = train_cfg_tmp.get("training", {})
        hyperparams: dict = {
            "dataset": args.dataset,
            **training_config,
            **model_config,
        }
    else:
        hyperparams: dict = {
            "dataset": args.dataset,
            "epochs": args.epochs,
            "outer_batch_size": args.outer_batch_size,
            "inner_batch_size": args.inner_batch_size,
            "base_lr": args.base_lr,
            "head_lr": args.head_lr,
            "pos_weight": args.pos_weight,
            "log_every": args.log_every,
            "device": args.device,
            "val_batch_size": args.val_batch_size,
            **model_config,
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

    # Load data and model
    train_df = pl.read_parquet(get_derived_path(args.dataset, "train"))
    val_df = pl.read_parquet(get_derived_path(args.dataset, "valid"))
    model = KeyphraseModel(**model_config).to(device)

    try:
        train(
            model,
            train_df,
            epochs=hyperparams["epochs"],
            outer_batch_size=hyperparams["outer_batch_size"],
            inner_batch_size=hyperparams["inner_batch_size"],
            base_lr=hyperparams["base_lr"],
            head_lr=hyperparams["head_lr"],
            pos_weight=hyperparams["pos_weight"],
            log_every=hyperparams["log_every"],
            device=device,
            metrics_filename=metrics_filename,
            wandb=wandb,
            val_df=val_df,
            val_batch_size=hyperparams["val_batch_size"],
        )
    finally:
        if wandb:
            wandb.finish()


if __name__ == "__main__":
    main()
