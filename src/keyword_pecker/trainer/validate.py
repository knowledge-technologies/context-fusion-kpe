"""Evaluation module for keyphrase extraction model.

This module provides evaluation functionality for the keyphrase extraction model,
computing both loss and keyphrase extraction metrics (precision, recall, F1).
It supports evaluating any dataset split (e.g. "val", "test") by passing the
``split`` parameter to :func:`evaluate_model`.

The evaluation logic is adapted from the legacy training script (tmp/train-model.py
lines 302-531), refactored to use Polars DataFrames and the project's established
data loading patterns.
"""

from __future__ import annotations

import json
import re
import types
from pathlib import Path

import polars as pl
import torch
import tqdm

from keyword_pecker.data import create_keyphrase_dataloader
from keyword_pecker.models import KeyphraseModel
from keyword_pecker.utils import get_logger

logger = get_logger("validate")


def extract_keyphrases_from_predictions(
    predictions: torch.Tensor,
    input_ids: torch.Tensor,
    attention_mask: torch.Tensor,
    tokenizer,
) -> list[list[str]]:
    """Extract keyphrase strings from model predictions.

    This function replicates the keyphrase extraction logic from the legacy code
    (tmp/train-model.py lines 362-397). It converts model logits to binary
    predictions and decodes the predicted token spans back to keyword strings.

    Args:
        predictions: Model logits of shape (batch_size, max_seq_len).
        input_ids: Token IDs of shape (batch_size, max_seq_len).
        attention_mask: Attention mask of shape (batch_size, max_seq_len).
        tokenizer: HuggingFace tokenizer for decoding tokens.

    Returns:
        List of keyword lists, one per sentence in the batch.
    """
    batch_size = predictions.shape[0]
    all_keywords: list[list[str]] = []

    # Binary predictions: logits > 0 means keyword
    binary_preds = (predictions > 0).long()

    # Mask out padding tokens
    masked_preds = binary_preds * input_ids

    for i in range(batch_size):
        # Get sequence length from attention mask
        seq_len = attention_mask[i].sum().item()

        # Decode predicted keyword tokens
        # Skip BOS (index 0) and EOS (last token before padding)
        # The legacy code uses res[i, 1 : len(row.sent_tokens) - 1]
        pred_tokens = masked_preds[i, 1 : seq_len - 1]

        # Decode to string - predicted tokens are split by separator token (<s>)
        decoded = tokenizer.decode(pred_tokens)

        # Split by separator and clean up
        # Legacy pattern: split by "<s>", strip whitespace, normalize " - " to "-"
        keywords = decoded.split("<s>")
        keywords = [re.sub(r" - ", "-", kw.strip()) for kw in keywords if kw.strip()]

        all_keywords.append(keywords)

    return all_keywords


def compute_keyphrase_metrics(
    gold_keywords: list[set[str]],
    pred_keywords: list[set[str]],
) -> dict[str, float]:
    """Compute precision, recall, and F1 using exact set matching.

    This function replicates the metrics computation from the legacy code
    (tmp/train-model.py lines 404-414).

    Args:
        gold_keywords: List of gold keyword sets, one per document.
        pred_keywords: List of predicted keyword sets, one per document.

    Returns:
        dict with keys: precision, recall, f1
    """
    tp = fp = fn = 0

    for gold, pred in zip(gold_keywords, pred_keywords):
        tp += len(gold & pred)
        fp += len(pred - gold)
        fn += len(gold - pred)

    # Avoid division by zero
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = (2 * tp) / (2 * tp + fp + fn) if (2 * tp + fp + fn) > 0 else 0.0

    return {
        "precision": precision,
        "recall": recall,
        "f1": f1,
    }


def evaluate_model(
    model: KeyphraseModel,
    val_df: pl.DataFrame,
    *,
    batch_size: int,
    pos_weight: float,
    device: torch.device,
    split: str = "val",
    metrics_filename: Path | None = None,
    wandb: types.ModuleType | None = None,
    global_step: int = 0,
    epoch: int = 0,
) -> dict[str, float]:
    """Run evaluation and compute loss + keyphrase metrics for any dataset split.

    This function runs the model in evaluation mode over the given dataset,
    computing both the BCE loss and keyphrase extraction metrics (precision,
    recall, F1) using exact set matching at the document level.

    The evaluation logic is adapted from the legacy training script
    (tmp/train-model.py lines 302-531), using the project's established
    data loading patterns with Polars DataFrames.

    Args:
        model: KeyphraseModel to evaluate (will be set to eval mode).
        val_df: Polars DataFrame with columns [doc_id, doc_row, sent_num,
            token_ids, keyword_mask], sorted by (doc_id, doc_row, sent_num).
        batch_size: Number of sentences per forward pass.
        pos_weight: Positive class weight for BCEWithLogitsLoss.
        device: Device to run inference on.
        split: Dataset split label used for metric key prefixes and logging,
            e.g. ``"val"`` or ``"test"`` (default: ``"val"``).
        metrics_filename: Optional Path to JSON metrics file (appends records).
        wandb: The wandb module if W&B tracking is active, or None.
        global_step: Current global training step (for logging).
        epoch: Current epoch number (for logging).

    Returns:
        dict with keys:
            - ``{split}_loss``: Mean loss over all batches.
            - precision: Keyphrase precision (exact match).
            - recall: Keyphrase recall (exact match).
            - f1: Keyphrase F1 score (exact match).
    """
    model.eval()

    # Create dataloader for validation (no shuffling, use batch_size for both levels)
    val_loader = create_keyphrase_dataloader(
        df=val_df,
        outer_batch_size=batch_size,
        inner_batch_size=batch_size,
        shuffle=False,
        drop_last=False,
        num_workers=0,
    )

    loss_fn = torch.nn.BCEWithLogitsLoss(
        reduction="none",
        pos_weight=torch.tensor([[pos_weight]], device=device),
    )

    total_loss = 0.0
    total_tokens = 0

    # Collect predictions per sentence
    # We'll aggregate to document level after the loop
    all_predictions: list[list[str]] = []

    num_batches = len(val_loader)

    with torch.no_grad():
        for batch in tqdm.tqdm(
            val_loader, desc=f"Evaluating ({split})", total=num_batches
        ):
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            target = batch["target"].to(device)
            sentence_counts = batch["sentence_counts"]

            # Forward pass
            logits = model(input_ids, attention_mask, sentence_counts)

            # Compute masked loss
            loss = loss_fn(logits, target) * attention_mask
            batch_loss = loss.sum().item()
            batch_tokens = attention_mask.sum().item()

            total_loss += batch_loss
            total_tokens += batch_tokens

            # Extract keyphrases from predictions
            batch_keywords = extract_keyphrases_from_predictions(
                logits.cpu(),
                input_ids.cpu(),
                attention_mask.cpu(),
                model.tokenizer,
            )
            all_predictions.extend(batch_keywords)

            # Free batch tensors to reduce memory fragmentation
            del input_ids, attention_mask, target, logits, loss, batch_keywords
            if device.type == "cuda":
                torch.cuda.empty_cache()

    # Compute mean loss
    val_loss = total_loss / total_tokens if total_tokens > 0 else 0.0

    # Now we need to aggregate predictions and gold labels at document level
    # Get document identifiers for each sentence
    doc_ids = val_df["doc_id"].to_list()
    doc_rows = val_df["doc_row"].to_list()

    # Extract gold keywords from each sentence using the same logic as predictions
    gold_per_sentence: list[list[str]] = []
    token_ids_list = val_df["token_ids"].to_list()
    keyword_masks = val_df["keyword_mask"].to_list()

    for token_ids, kw_mask in zip(token_ids_list, keyword_masks):
        # Create tensors for gold extraction
        seq_len = len(token_ids)
        gold_input_ids = torch.tensor(token_ids, dtype=torch.long).unsqueeze(0)
        gold_mask = torch.tensor(kw_mask, dtype=torch.float).unsqueeze(0)
        gold_attn = torch.ones(1, seq_len, dtype=torch.long)

        # Extract gold keywords using the same function
        gold_kws = extract_keyphrases_from_predictions(
            gold_mask,  # Use keyword_mask as "predictions" (already 0/1)
            gold_input_ids,
            gold_attn,
            model.tokenizer,
        )
        gold_per_sentence.append(gold_kws[0])

    # Free intermediate data no longer needed
    del token_ids_list, keyword_masks

    # Aggregate to document level
    doc_to_gold: dict[tuple[str, int], set[str]] = {}
    doc_to_pred: dict[tuple[str, int], set[str]] = {}

    for i, (doc_id, doc_row) in enumerate(zip(doc_ids, doc_rows)):
        key = (doc_id, doc_row)

        if key not in doc_to_gold:
            doc_to_gold[key] = set()
            doc_to_pred[key] = set()

        doc_to_gold[key].update(gold_per_sentence[i])

        if i < len(all_predictions):
            doc_to_pred[key].update(all_predictions[i])

    # Convert to lists for metric computation (preserve order)
    doc_keys = list(doc_to_gold.keys())
    gold_sets = [doc_to_gold[k] for k in doc_keys]
    pred_sets = [doc_to_pred[k] for k in doc_keys]

    # Free aggregation intermediates
    del gold_per_sentence, all_predictions, doc_ids, doc_rows, doc_to_gold, doc_to_pred

    # Compute keyphrase metrics
    metrics = compute_keyphrase_metrics(gold_sets, pred_sets)
    metrics[f"{split}_loss"] = val_loss

    # Log metrics to file
    if metrics_filename is not None:
        record = {
            "epoch": epoch,
            "global_step": global_step,
            f"{split}_loss": val_loss,
            f"{split}_precision": metrics["precision"],
            f"{split}_recall": metrics["recall"],
            f"{split}_f1": metrics["f1"],
        }
        with metrics_filename.open("a") as f:
            f.write(json.dumps(record) + "\n")

    # Log metrics to W&B
    if wandb is not None:
        wandb.log(
            {
                f"{split}/loss": val_loss,
                f"{split}/precision": metrics["precision"],
                f"{split}/recall": metrics["recall"],
                f"{split}/f1": metrics["f1"],
            },
            step=global_step,
        )

    # Restore model to training mode
    model.train()

    return metrics
