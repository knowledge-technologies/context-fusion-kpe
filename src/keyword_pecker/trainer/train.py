"""Training loop for keyphrase extraction model.

This module provides the core training routine for the KeyphraseModel,
implementing the two-level batching strategy with gradient accumulation
and support for within-epoch early stopping.
"""

from __future__ import annotations

import json
import math
import types
from pathlib import Path

import polars as pl
import torch
import tqdm
from transformers import get_linear_schedule_with_warmup

from keyword_pecker.constants import DEFAULT_DISALLOWED_PUNCTUATION, IGNORE_IDX
from keyword_pecker.data import compute_class_weights, create_keyphrase_dataloader
from keyword_pecker.models import KeyphraseModel
from keyword_pecker.trainer.early_stopping import EarlyStopping
from keyword_pecker.trainer.validate import evaluate_model
from keyword_pecker.utils import get_logger, set_random_seeds

logger = get_logger("train")


def _save_checkpoint(
    state_dict: dict,
    run_dir: Path,
    filename: str,
    num_classes: int,
    class_weights: list[float] | None,
    best_step: int | None = None,
) -> None:
    """Save a model checkpoint with minimal metadata."""
    metadata: dict = {
        "num_classes": num_classes,
        "class_weights": class_weights,
    }
    if best_step is not None:
        metadata["best_step"] = best_step
    checkpoint = {
        "state_dict": state_dict,
        "metadata": metadata,
    }
    torch.save(checkpoint, run_dir / filename)
    logger.info(f"Model saved to {run_dir / filename}")


def train(
    model: KeyphraseModel,
    train_df: pl.DataFrame,
    *,
    epochs: int,
    outer_batch_size: int,
    inner_batch_size: int,
    base_lr: float,
    head_lr: float,
    num_classes: int = 3,
    use_class_weights: bool = True,
    class_weights: list[float] | None = None,
    log_every: int,
    device: torch.device,
    metrics_filename: Path,
    wandb: types.ModuleType | None,
    run_dir: Path | None = None,
    save_weights: bool = False,
    val_df: pl.DataFrame | None = None,
    val_batch_size: int = 100,
    warmup_ratio: float = 0.1,
    use_scheduler: bool = True,
    k_values: list[int | str] | None = None,
    early_stopping_patience: int = 0,
    early_stopping_delta: float = 0.0,
    early_stopping_metric: str = "val_loss",
    early_stopping_mode: str = "min",
    early_stopping_check_every: int = 0,
    scoring_strategy: str = "confidence",
    score_method: str = "average",
    filter_punctuation: bool = True,
    disallowed_punctuation: str = DEFAULT_DISALLOWED_PUNCTUATION,
    dedup_strategy: str = "stem",
    seed: int = 42,
) -> None:
    """Full training routine: optimizer, loss, dataloader creation, and epoch loop.

    Replicates the two-level batching semantics of the original training loop:
      - Gradients accumulate across inner batches within each outer chunk
      - optimizer.step() is called once per outer chunk (every outer_batch_size sentences)
      - The dataloader is re-created each epoch to re-shuffle at document level

    Args:
        model: KeyphraseModel to train (already moved to device).
        train_df: Polars DataFrame with columns [doc_id, doc_row, sent_num,
            token_ids, keyword_mask], sorted by (doc_id, doc_row, sent_num).
        epochs: Number of full passes over the training data.
        outer_batch_size: Sentences per gradient accumulation unit.
        inner_batch_size: Sentences per forward pass.
        base_lr: Learning rate for base encoder parameters.
        head_lr: Learning rate for refinement head parameters.
        num_classes: Number of output classes (2 for binary, 3 for IOB).
        use_class_weights: Whether to apply class weights in loss function.
            If False, uses equal weighting (CrossEntropyLoss with weight=None).
            If True, computes or uses provided class_weights.
        class_weights: Optional class weights for loss function. If None and
            use_class_weights=True, computed from training data for IOB scheme,
            or defaults to [1.0, 5.0] for binary.
        log_every: Print loss every this many optimizer steps.
        device: Device to run on.
        metrics_filename: Path to the JSONL file where metrics are appended.
        wandb: The wandb module if W&B tracking is active, or ``None``.
        run_dir: Directory to save epoch checkpoint weights (model_e1.pt, model_e2.pt, ...).
        save_weights: If True, save model weights after each epoch (except the last).
        val_df: Optional Polars DataFrame for validation (same schema as train_df).
        val_batch_size: Batch size for validation forward passes (default: 100).
        warmup_ratio: Fraction of training steps for warmup (default: 0.1).
        use_scheduler: Whether to use a learning rate scheduler (default: True).
        k_values: List of k values for @k metrics (default: [5, 10]).
        early_stopping_patience: Checks without improvement before stopping (0 = disabled).
        early_stopping_delta: Minimum improvement to reset patience counter.
        early_stopping_metric: Metric name to monitor (e.g., "val_loss", "f1@5").
        early_stopping_mode: "min" for lower-is-better, "max" for higher-is-better.
        early_stopping_check_every: Run validation every N optimizer steps (0 = epoch-only).
        scoring_strategy: "confidence" for model probability scores, "ordinal" for extraction order (default: "confidence").
        score_method: How to aggregate token probabilities ("average", "max", "min", "geometric") (default: "average").
        filter_punctuation: If True, remove keyphrases with disallowed punctuation (default: True).
        disallowed_punctuation: String of punctuation characters to filter (default excludes dash/apostrophe).
        dedup_strategy: "stem" for Porter stemmer-based dedup, or "lowercase" (default: "stem").
        seed: Random seed for reproducibility (default: 42). Set to None to disable seeding.
    """
    if seed is not None:
        set_random_seeds(seed)

    optimizer = torch.optim.AdamW(
        model.get_param_groups(base_lr=base_lr, head_lr=head_lr)
    )

    total_steps = math.ceil(len(train_df) / outer_batch_size) * epochs
    warmup_steps = int(total_steps * warmup_ratio)

    if use_scheduler:
        scheduler = get_linear_schedule_with_warmup(
            optimizer,
            num_warmup_steps=warmup_steps,
            num_training_steps=total_steps,
        )
        logger.info(
            f"LR scheduler: {warmup_steps} warmup steps / {total_steps} total steps"
        )
    else:
        scheduler = None
        logger.info("LR scheduler: disabled")

    def get_lr() -> float:
        if scheduler is not None:
            return float(scheduler.get_last_lr()[0])
        return float(optimizer.param_groups[0]["lr"])

    # Setup loss function based on num_classes
    if num_classes == 2:
        # Binary classification: BCEWithLogitsLoss
        if class_weights is None:
            class_weights = [1.0, 5.0]
        pos_weight = class_weights[1]
        loss_fn = torch.nn.BCEWithLogitsLoss(
            reduction="none",
            pos_weight=torch.tensor([[pos_weight]], device=device),
        )
        logger.info(f"Using BCEWithLogitsLoss with pos_weight={pos_weight}")
    else:
        # IOB classification: CrossEntropyLoss with optional class weights
        if use_class_weights:
            if class_weights is None:
                class_weights, _ = compute_class_weights(train_df, num_classes)
            class_weights_tensor = torch.tensor(
                class_weights, dtype=torch.float32, device=device
            )
            loss_fn = torch.nn.CrossEntropyLoss(
                weight=class_weights_tensor,
                reduction="none",
                ignore_index=IGNORE_IDX,
            )
            logger.info(f"Using CrossEntropyLoss with class_weights={class_weights}")
        else:
            loss_fn = torch.nn.CrossEntropyLoss(
                weight=None,
                reduction="none",
                ignore_index=IGNORE_IDX,
            )
            logger.info("Using CrossEntropyLoss with equal class weights (weight=None)")

    # Log class weights to W&B
    if wandb and class_weights and use_class_weights:
        wandb.log(
            {f"class_weights/class_{i}": w for i, w in enumerate(class_weights)},
            step=0,
        )

    inner_per_outer = math.ceil(outer_batch_size / inner_batch_size)

    # Two-level batching strategy:
    # - Inner batch: forward/backward pass on `inner_batch_size` sentences
    # - Outer batch: gradient accumulation unit of `outer_batch_size` sentences
    # - Forward/backward pass executed per inner batch (gradients accumulate)
    # - optimizer.step() (weight update) executed once per outer batch

    early_stopping: EarlyStopping | None = None
    if early_stopping_patience > 0:
        early_stopping = EarlyStopping(
            metric=early_stopping_metric,
            patience=early_stopping_patience,
            delta=early_stopping_delta,
            mode=early_stopping_mode,
        )
        check_msg = (
            f"every {early_stopping_check_every} steps"
            if early_stopping_check_every > 0
            else "at epoch end"
        )
        logger.info(
            f"Early stopping enabled: patience={early_stopping_patience}, "
            f"metric={early_stopping_metric}, delta={early_stopping_delta}, "
            f"mode={early_stopping_mode}, check={check_msg}"
        )

    global_step = 0
    best_model_state: dict | None = None
    best_val_score: float | None = None
    best_step: int | None = None
    last_val_metrics: dict | None = None
    exit_reason = "completed"

    metrics_file = metrics_filename.open("a")

    try:
        for epoch in range(epochs):
            logger.info(f"Epoch {epoch + 1}/{epochs}")
            if wandb:
                wandb.log({"epoch": epoch + 1}, step=global_step)

            train_loader = create_keyphrase_dataloader(
                df=train_df,
                outer_batch_size=outer_batch_size,
                inner_batch_size=inner_batch_size,
                shuffle=True,
                drop_last=False,
                num_workers=0,
            )

            model.train()
            optimizer.zero_grad()  # Clear gradients at start of each epoch before accumulation begins

            outer_step = 0
            inner_count = 0
            last_loss: float | None = None
            epoch_raw_loss_sum = 0.0
            epoch_token_count = 0

            total_outer = math.ceil(len(train_loader) / inner_per_outer)
            progress = tqdm.trange(total_outer, desc="steps")

            for batch in train_loader:
                input_ids = batch["input_ids"].to(device)
                attention_mask = batch["attention_mask"].to(device)
                target = batch["target"].to(device)
                sentence_counts = batch["sentence_counts"]

                logits = model(input_ids, attention_mask, sentence_counts)

                if num_classes == 2:
                    loss = loss_fn(logits, target) * attention_mask
                else:
                    ignore_mask = target != IGNORE_IDX
                    full_mask = attention_mask & ignore_mask
                    loss = loss_fn(logits.transpose(1, 2), target) * full_mask
                    raw_loss_sum = loss.sum().item()
                    raw_token_count = full_mask.sum().item()
                    loss = loss.sum() / full_mask.sum().clamp(min=1)
                    loss.backward()

                last_loss = loss.detach().cpu().item()

                if num_classes != 2:
                    epoch_raw_loss_sum += raw_loss_sum
                    epoch_token_count += raw_token_count

                del logits, loss, input_ids, attention_mask, target
                if device.type == "cuda":
                    torch.cuda.empty_cache()

                inner_count += 1

                # Complete outer batch: gradients accumulated for `inner_per_outer` inner batches
                # Step optimizer (update weights) and scheduler, then zero gradients
                # Note: zero_grad() called AFTER step() to prepare for next accumulation cycle
                if inner_count == inner_per_outer:
                    optimizer.step()
                    if scheduler is not None:
                        scheduler.step()
                    optimizer.zero_grad()

                    if last_loss is not None:
                        if outer_step % log_every == 0:
                            logger.info(f"step {global_step} loss: {last_loss:.6f}")

                            record = {
                                "epoch": epoch + 1,
                                "outer_step": outer_step,
                                "global_step": global_step,
                                "train_loss": last_loss,
                                "learning_rate": get_lr(),
                            }
                            metrics_file.write(json.dumps(record) + "\n")
                            metrics_file.flush()

                            if wandb:
                                wandb.log(
                                    {
                                        "train/loss": last_loss,
                                        "train/learning_rate": get_lr(),
                                    },
                                    step=global_step,
                                )

                    outer_step += 1
                    global_step += 1
                    inner_count = 0
                    progress.update(1)

                    if (
                        early_stopping_check_every > 0
                        and global_step % early_stopping_check_every == 0
                        and val_df is not None
                    ):
                        val_metrics = evaluate_model(
                            model,
                            val_df,
                            batch_size=val_batch_size,
                            num_classes=num_classes,
                            use_class_weights=use_class_weights,
                            class_weights=class_weights,
                            device=device,
                            split="val",
                            k_values=k_values,
                            metrics_filename=metrics_filename,
                            wandb=wandb,
                            global_step=global_step,
                            epoch=epoch + 1,
                            scoring_strategy=scoring_strategy,
                            score_method=score_method,
                            filter_punctuation=filter_punctuation,
                            disallowed_punctuation=disallowed_punctuation,
                            dedup_strategy=dedup_strategy,
                        )
                        k_val = k_values[0] if k_values else 5
                        logger.info(
                            f"Step {global_step} val loss: {val_metrics['val_loss']:.6f}, "
                            f"F1@{k_val}: {val_metrics[f'f1@{k_val}']:.4f}"
                        )

                        current_score = val_metrics.get(early_stopping_metric)
                        if current_score is not None:
                            if (
                                best_val_score is None
                                or (
                                    early_stopping_mode == "min"
                                    and current_score < best_val_score
                                )
                                or (
                                    early_stopping_mode == "max"
                                    and current_score > best_val_score
                                )
                            ):
                                best_val_score = current_score
                                best_step = global_step
                                best_model_state = {
                                    k: v.cpu() for k, v in model.state_dict().items()
                                }
                                if save_weights and run_dir is not None:
                                    _save_checkpoint(
                                        best_model_state,
                                        run_dir,
                                        "model_best.pt",
                                        num_classes,
                                        class_weights,
                                        best_step=best_step,
                                    )

                        if early_stopping is not None:
                            should_stop = early_stopping(val_metrics, step=global_step)

                            if should_stop:
                                logger.info(
                                    f"Early stopping triggered at step {global_step} "
                                    f"(best {early_stopping_metric}: {early_stopping.best_score:.6f} "
                                    f"at step {early_stopping.best_step})"
                                )
                                exit_reason = "early_stopping"
                                progress.close()
                                metrics_file.close()
                                return

            # Remainder handling: process leftover samples that don't fill a complete outer batch
            # This ensures we don't lose gradients on the final partial batch
            # Note: outer_step not incremented since this is a partial batch
            if inner_count > 0:
                optimizer.step()
                if scheduler is not None:
                    scheduler.step()
                optimizer.zero_grad()

                if last_loss is not None:
                    if outer_step % log_every == 0:
                        logger.info(f"step {global_step} loss: {last_loss:.6f}")

                        record = {
                            "epoch": epoch + 1,
                            "outer_step": outer_step,
                            "global_step": global_step,
                            "train_loss": last_loss,
                            "learning_rate": get_lr(),
                        }
                        metrics_file.write(json.dumps(record) + "\n")
                        metrics_file.flush()

                        if wandb:
                            wandb.log(
                                {
                                    "train/loss": last_loss,
                                    "train/learning_rate": get_lr(),
                                },
                                step=global_step,
                            )

                global_step += 1
                progress.update(1)

            progress.close()

            if epoch_token_count > 0:
                mean_loss = epoch_raw_loss_sum / epoch_token_count
                logger.info(f"Epoch {epoch + 1} mean loss: {mean_loss:.6f}")

                metrics_file.write(
                    json.dumps({"epoch": epoch + 1, "epoch_mean_loss": mean_loss})
                    + "\n"
                )
                metrics_file.flush()

                if wandb:
                    wandb.log({"train/epoch_mean_loss": mean_loss}, step=global_step)

            if val_df is not None:
                val_metrics = evaluate_model(
                    model,
                    val_df,
                    batch_size=val_batch_size,
                    num_classes=num_classes,
                    use_class_weights=use_class_weights,
                    class_weights=class_weights,
                    device=device,
                    split="val",
                    k_values=k_values,
                    metrics_filename=metrics_filename,
                    wandb=wandb,
                    global_step=global_step,
                    epoch=epoch + 1,
                    scoring_strategy=scoring_strategy,
                    score_method=score_method,
                    filter_punctuation=filter_punctuation,
                    disallowed_punctuation=disallowed_punctuation,
                    dedup_strategy=dedup_strategy,
                )
                k_val = k_values[0] if k_values else 5
                logger.info(
                    f"Epoch {epoch + 1} val loss: {val_metrics['val_loss']:.6f}, "
                    f"F1@{k_val}: {val_metrics[f'f1@{k_val}']:.4f}"
                )
                last_val_metrics = val_metrics

                current_score = val_metrics.get(early_stopping_metric)
                if current_score is not None:
                    if (
                        best_val_score is None
                        or (
                            early_stopping_mode == "min"
                            and current_score < best_val_score
                        )
                        or (
                            early_stopping_mode == "max"
                            and current_score > best_val_score
                        )
                    ):
                        best_val_score = current_score
                        best_step = global_step
                        best_model_state = {
                            k: v.cpu() for k, v in model.state_dict().items()
                        }
                        if save_weights and run_dir is not None:
                            _save_checkpoint(
                                best_model_state,
                                run_dir,
                                "model_best.pt",
                                num_classes,
                                class_weights,
                                best_step=best_step,
                            )

                if early_stopping is not None and early_stopping_check_every == 0:
                    should_stop = early_stopping(val_metrics, step=global_step)

                    if should_stop:
                        logger.info(
                            f"Early stopping triggered at epoch {epoch + 1} "
                            f"(best {early_stopping_metric}: {early_stopping.best_score:.6f} "
                            f"at step {early_stopping.best_step})"
                        )
                        exit_reason = "early_stopping"
                        break

        if save_weights and run_dir is not None:
            _save_checkpoint(
                model.state_dict(),
                run_dir,
                "model.pt",
                num_classes,
                class_weights,
            )

    except Exception:
        exit_reason = "error"
        raise
    finally:
        metrics_file.close()

        if wandb and wandb.run:
            run_name = wandb.run.name
            alert_level = (
                wandb.AlertLevel.WARN
                if exit_reason == "early_stopping"
                else wandb.AlertLevel.INFO
            )
            if exit_reason == "error":
                alert_level = wandb.AlertLevel.ERROR

            parts = [f"Training {exit_reason}"]
            if last_val_metrics is not None:
                k_val = k_values[0] if k_values else 5
                parts.append(
                    f"val_loss={last_val_metrics['val_loss']:.4f}, "
                    f"F1@{k_val}={last_val_metrics[f'f1@{k_val}']:.4f}"
                )
            text = " | ".join(parts)

            wandb.run.alert(
                title=f"Training {exit_reason}: {run_name}",
                text=text,
                wait_duration=300,
                level=alert_level,
            )
