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
import random
import re
import types
from pathlib import Path

import polars as pl
import torch
import tqdm

from keyword_pecker.data import create_keyphrase_dataloader
from keyword_pecker.models import KeyphraseModel
from keyword_pecker.models.inference import extract_ranked_keyphrases
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


def compute_metrics_at_k(
    gold_keywords: list[set[str]],
    pred_keywords_ranked: list[list[tuple[str, float]]],
    k_values: list[int],
) -> dict[str, float]:
    """Compute precision@k, recall@k, and F1@k for each k value.

    For each document, takes the top-k predicted keywords (ranked by confidence)
    and computes metrics against the gold keywords.

    Args:
        gold_keywords: List of gold keyword sets, one per document.
        pred_keywords_ranked: List of predicted keyword lists with confidence scores,
            one per document. Each list contains (keyword, score) tuples sorted
            by confidence descending.
        k_values: List of k values to compute metrics for (e.g., [5, 10]).

    Returns:
        dict with keys like precision@5, recall@5, f1@5, etc.
    """
    metrics: dict[str, float] = {}

    for k in k_values:
        tp = fp = fn = 0

        for gold, pred_ranked in zip(gold_keywords, pred_keywords_ranked):
            # Take top-k predictions
            top_k = set(kw for kw, _ in pred_ranked[:k])

            tp += len(gold & top_k)
            fp += len(top_k - gold)
            fn += len(gold - top_k)

        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = (2 * tp) / (2 * tp + fp + fn) if (2 * tp + fp + fn) > 0 else 0.0

        metrics[f"precision@{k}"] = precision
        metrics[f"recall@{k}"] = recall
        metrics[f"f1@{k}"] = f1

    return metrics


def evaluate_model(
    model: KeyphraseModel,
    val_df: pl.DataFrame,
    *,
    batch_size: int,
    pos_weight: float,
    device: torch.device,
    split: str = "val",
    k_values: list[int] | None = None,
    metrics_filename: Path | None = None,
    wandb: types.ModuleType | None = None,
    global_step: int = 0,
    epoch: int = 0,
    sample_seed: int = 42,
) -> dict[str, float | list[dict[str, str | int | list[str]]]]:
    """Run evaluation and compute loss + keyphrase metrics for any dataset split.

    This function runs the model in evaluation mode over the given dataset,
    computing both the BCE loss and keyphrase extraction metrics (precision@k,
    recall@k, F1@k) using exact set matching at the document level.

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
        k_values: List of k values for @k metrics (default: [5, 10]).
        metrics_filename: Optional Path to JSON metrics file (appends records).
        wandb: The wandb module if W&B tracking is active, or None.
        global_step: Current global training step (for logging).
        epoch: Current epoch number (for logging).
        sample_seed: Random seed for sampling documents (default: 42). Set to None
            for non-reproducible sampling.

    Returns:
        dict with keys:
            - ``{split}_loss``: Mean loss over all batches.
            - precision@k, recall@k, f1@k for each k in k_values.
            - samples: List of dicts with random document samples for qualitative
              analysis. Each dict has keys: doc_id, doc_row, gold_keywords,
              pred_keywords (keyword lists are sorted).
    """
    if k_values is None:
        k_values = [5, 10]

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

    # Collect ranked predictions per sentence (with confidence scores)
    all_ranked_predictions: list[list[tuple[str, float]]] = []

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

            # Extract ranked keyphrases from predictions
            batch_ranked_keywords = extract_ranked_keyphrases(
                logits.cpu(),
                input_ids.cpu(),
                attention_mask.cpu(),
                model.tokenizer,
            )
            all_ranked_predictions.extend(batch_ranked_keywords)

            # Free batch tensors to reduce memory fragmentation
            del input_ids, attention_mask, target, logits, loss, batch_ranked_keywords
            if device.type == "cuda":
                torch.cuda.empty_cache()

    # Compute mean loss
    val_loss = total_loss / total_tokens if total_tokens > 0 else 0.0

    # Now we need to aggregate predictions and gold labels at document level
    # Get document identifiers for each sentence
    doc_ids = val_df["doc_id"].to_list()
    doc_rows = val_df["doc_row"].to_list()

    # Extract gold keywords from each sentence
    gold_per_sentence: list[list[str]] = []
    token_ids_list = val_df["token_ids"].to_list()
    keyword_masks = val_df["keyword_mask"].to_list()

    for token_ids, kw_mask in zip(token_ids_list, keyword_masks):
        # Create tensors for gold extraction
        seq_len = len(token_ids)
        gold_input_ids = torch.tensor(token_ids, dtype=torch.long).unsqueeze(0)
        gold_mask = torch.tensor(kw_mask, dtype=torch.float).unsqueeze(0)
        gold_attn = torch.ones(1, seq_len, dtype=torch.long)

        # Extract gold keywords
        gold_kws = extract_keyphrases_from_predictions(
            gold_mask,
            gold_input_ids,
            gold_attn,
            model.tokenizer,
        )
        gold_per_sentence.append(gold_kws[0])

    # Free intermediate data no longer needed
    del token_ids_list, keyword_masks

    # Aggregate to document level
    doc_to_gold: dict[tuple[str, int], set[str]] = {}
    doc_to_pred_ranked: dict[tuple[str, int], list[tuple[str, float]]] = {}

    for i, (doc_id, doc_row) in enumerate(zip(doc_ids, doc_rows)):
        key = (doc_id, doc_row)

        if key not in doc_to_gold:
            doc_to_gold[key] = set()
            doc_to_pred_ranked[key] = []

        doc_to_gold[key].update(gold_per_sentence[i])

        if i < len(all_ranked_predictions):
            doc_to_pred_ranked[key].extend(all_ranked_predictions[i])

    # Sort predictions by confidence and deduplicate per document
    for key in doc_to_pred_ranked:
        seen: set[str] = set()
        deduped: list[tuple[str, float]] = []
        for kw, score in doc_to_pred_ranked[key]:
            if kw not in seen:
                seen.add(kw)
                deduped.append((kw, score))
        deduped.sort(key=lambda x: x[1], reverse=True)
        doc_to_pred_ranked[key] = deduped

    # Convert to lists for metric computation (preserve order)
    doc_keys = list(doc_to_gold.keys())
    gold_sets = [doc_to_gold[k] for k in doc_keys]
    pred_ranked = [doc_to_pred_ranked[k] for k in doc_keys]

    # Capture random samples for qualitative analysis
    num_samples = min(15, len(doc_keys))
    rng = random.Random(sample_seed)
    sample_indices = rng.sample(range(len(doc_keys)), num_samples)
    samples = []
    for idx in sample_indices:
        doc_key = doc_keys[idx]
        samples.append(
            {
                "doc_id": doc_key[0],
                "doc_row": doc_key[1],
                "gold_keywords": sorted(gold_sets[idx]),
                "pred_keywords": [kw for kw, _ in pred_ranked[idx]],
            }
        )

    # Free aggregation intermediates
    del (
        gold_per_sentence,
        all_ranked_predictions,
        doc_ids,
        doc_rows,
        doc_to_gold,
        doc_to_pred_ranked,
    )

    # Compute @k metrics
    at_k_metrics = compute_metrics_at_k(gold_sets, pred_ranked, k_values)
    metrics: dict[str, float | list[dict[str, str | int | list[str]]]] = dict(
        at_k_metrics
    )
    metrics[f"{split}_loss"] = val_loss
    metrics["samples"] = samples

    # Log metrics to file
    if metrics_filename is not None:
        record: dict = {
            "epoch": epoch,
            "global_step": global_step,
            f"{split}_loss": val_loss,
        }
        for k in k_values:
            record[f"{split}_precision@{k}"] = at_k_metrics[f"precision@{k}"]
            record[f"{split}_recall@{k}"] = at_k_metrics[f"recall@{k}"]
            record[f"{split}_f1@{k}"] = at_k_metrics[f"f1@{k}"]
        record[f"{split}_samples"] = metrics["samples"]
        with metrics_filename.open("a") as f:
            f.write(json.dumps(record) + "\n")

    # Log metrics to W&B
    if wandb is not None:
        wandb_log: dict = {f"{split}/loss": val_loss}
        for k in k_values:
            wandb_log[f"{split}/precision@{k}"] = at_k_metrics[f"precision@{k}"]
            wandb_log[f"{split}/recall@{k}"] = at_k_metrics[f"recall@{k}"]
            wandb_log[f"{split}/f1@{k}"] = at_k_metrics[f"f1@{k}"]
        wandb.log(wandb_log, step=global_step)
        samples_table = wandb.Table(
            columns=["doc_id", "doc_row", "gold_keywords", "pred_keywords"],
            data=[
                [s["doc_id"], s["doc_row"], s["gold_keywords"], s["pred_keywords"]]
                for s in metrics["samples"]
            ],
        )
        wandb.log({f"{split}/samples": samples_table}, step=global_step)

    # Restore model to training mode
    model.train()

    return metrics
