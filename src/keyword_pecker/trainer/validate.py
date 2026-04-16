"""Evaluation module for keyphrase extraction model.

This module provides evaluation functionality for the keyphrase extraction model,
computing both loss and keyphrase extraction metrics (precision@k, recall@k, F1@k).
It supports evaluating any dataset split (e.g. "val", "test") by passing the
``split`` parameter to :func:`evaluate_model`.

Metrics are computed using the external eval module from TNT-KID
(:mod:`keyword_pecker.trainer.external.eval`), which applies Porter stemming
before exact word-by-word matching and uses macro-averaging (average per-document
P/R, then compute F1). This matches the evaluation methodology used in prior
keyword extraction benchmarks.
"""

from __future__ import annotations

import contextlib
import io
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
from keyword_pecker.models.inference import (
    aggregate_document_keyphrases,
    extract_ranked_keyphrases,
)
from keyword_pecker.trainer.external import eval as eval_martinc
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

    .. deprecated::
        Use :func:`compute_metrics_at_k_external` instead, which uses Porter
        stemming and macro-averaging to match prior benchmark evaluations.

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
    k_values: list[int | str],
) -> dict[str, float]:
    """Compute precision@k, recall@k, and F1@k for each k value.

    .. deprecated::
        Use :func:`compute_metrics_at_k_external` instead, which uses Porter
        stemming and macro-averaging to match prior benchmark evaluations.

    For each document, takes the top-k predicted keywords (ranked by confidence)
    and computes metrics against the gold keywords.

    Args:
        gold_keywords: List of gold keyword sets, one per document.
        pred_keywords_ranked: List of predicted keyword lists with confidence scores,
            one per document. Each list contains (keyword, score) tuples sorted
            by confidence descending.
        k_values: List of k values to compute metrics for (e.g., [5, 10]).
            Use the string "N" to compute @N metrics where k equals the number
            of gold keywords for each document.

    Returns:
        dict with keys like precision@5, recall@5, f1@5, etc.
        For "N" sentinel, keys are precision@N, recall@N, f1@N.
    """
    metrics: dict[str, float] = {}

    for k_spec in k_values:
        tp = fp = fn = 0

        for gold, pred_ranked in zip(gold_keywords, pred_keywords_ranked):
            k = len(gold) if k_spec == "N" else k_spec
            top_k = set(kw for kw, _ in pred_ranked[:k])

            tp += len(gold & top_k)
            fp += len(top_k - gold)
            fn += len(gold - top_k)

        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = (2 * tp) / (2 * tp + fp + fn) if (2 * tp + fp + fn) > 0 else 0.0

        key_suffix = "N" if k_spec == "N" else str(k_spec)
        metrics[f"precision@{key_suffix}"] = precision
        metrics[f"recall@{key_suffix}"] = recall
        metrics[f"f1@{key_suffix}"] = f1

    return metrics


_EVAL_RESULT_MAP: dict[int | str, dict[str, int]] = {
    5: {"precision": 0, "recall": 1, "f1": 2},
    10: {"precision": 3, "recall": 4, "f1": 5},
    "N": {"precision": 6, "recall": 7, "f1": 8},
    "M": {"precision": 9, "recall": 10, "f1": 11},
}


def compute_metrics_at_k_martinc(
    gold_keywords: list[list[str]],
    pred_keywords: list[list[str]],
    k_values: list[int | str] | None = None,
    remove_docs_with_no_kw: bool = True,
) -> dict[str, float]:
    """Compute stemmed, macro-averaged @k metrics using the external eval module.

    This function wraps :func:`keyword_pecker.trainer.external.eval.eval` to compute
    precision@k, recall@k, and F1@k using Porter stemming and macro-averaging,
    matching the evaluation methodology from prior keyword extraction benchmarks.

    The external eval module applies Porter stemming before exact word-by-word
    matching and uses macro-averaging (average per-document P/R, then compute F1).

    Args:
        gold_keywords: List of gold keyword lists, one per document.
            Each inner list contains keyword strings (may be multi-word like
            ``"neural network"``).
        pred_keywords: List of predicted keyword lists, one per document.
            Each inner list contains keyword strings in ranked order (highest
            confidence first).
        k_values: List of k values to compute metrics for. Supported values:
            ``5``, ``10``, ``"N"`` (number of gold keywords per document),
            ``"M"`` (number of predicted keywords per document).
            Default: ``[5, 10, "N", "M"]``.
        remove_docs_with_no_kw: If True, skip documents with empty gold keywords
            (matching eval.py's default behavior).

    Returns:
        dict with keys:
            - ``precision@{k}``, ``recall@{k}``, ``f1@{k}`` for each k in k_values
            - ``num_docs_total``: Total number of documents in input
            - ``num_docs_removed``: Documents skipped due to empty gold keywords
            - ``num_docs_evaluated``: Documents actually used in metric computation
    """
    if k_values is None:
        k_values = [5, 10, "N", "M"]

    num_docs_total = len(gold_keywords)
    num_docs_removed = (
        sum(1 for g in gold_keywords if len(g) == 0) if remove_docs_with_no_kw else 0
    )
    num_docs_evaluated = num_docs_total - num_docs_removed

    with contextlib.redirect_stdout(io.StringIO()):
        result = eval_martinc.eval(pred_keywords, gold_keywords, remove_docs_with_no_kw)

    metrics: dict[str, float] = {}
    for k_spec in k_values:
        key_suffix = str(k_spec)
        for metric_name in ("precision", "recall", "f1"):
            idx = _EVAL_RESULT_MAP[k_spec][metric_name]
            metrics[f"{metric_name}@{key_suffix}"] = float(result[idx])

    metrics["num_docs_total"] = float(num_docs_total)
    metrics["num_docs_removed"] = float(num_docs_removed)
    metrics["num_docs_evaluated"] = float(num_docs_evaluated)

    return metrics


def evaluate_model(
    model: KeyphraseModel,
    val_df: pl.DataFrame,
    *,
    batch_size: int,
    num_classes: int = 3,
    class_weights: list[float] | None = None,
    device: torch.device,
    split: str = "val",
    k_values: list[int | str] | None = None,
    metrics_filename: Path | None = None,
    wandb: types.ModuleType | None = None,
    global_step: int = 0,
    epoch: int = 0,
    sample_seed: int = 42,
) -> dict[str, float | list[dict[str, str | int | list[str]]]]:
    """Run evaluation and compute loss + keyphrase metrics for any dataset split.

    This function runs the model in evaluation mode over the given dataset,
    computing both the loss and keyphrase extraction metrics (precision@k,
    recall@k, F1@k) using the external eval module which applies Porter stemming
    and macro-averaging.

    Args:
        model: KeyphraseModel to evaluate (will be set to eval mode).
        val_df: Polars DataFrame with columns [doc_id, doc_row, sent_num,
            token_ids, keyword_mask], sorted by (doc_id, doc_row, sent_num).
        batch_size: Number of sentences per forward pass.
        num_classes: Number of output classes (2 for binary, 3 for IOB).
        class_weights: Class weights for loss function. Required for IOB scheme;
            defaults to [1.0, 5.0] for binary scheme.
        device: Device to run inference on.
        split: Dataset split label used for metric key prefixes and logging,
            e.g. ``"val"`` or ``"test"`` (default: ``"val"``).
        k_values: List of k values for @k metrics. Supported: ``5``, ``10``,
            ``"N"`` (num gold keywords per doc), ``"M"`` (num predicted keywords).
            Default: ``[5, 10, "N", "M"]``.
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
            - num_docs_total, num_docs_removed, num_docs_evaluated: Document counts.
            - samples: List of dicts with random document samples for qualitative
              analysis. Each dict has keys: doc_id, doc_row, gold_keywords,
              pred_keywords (keyword lists are sorted).
    """
    if k_values is None:
        k_values = [5, 10, "N", "M"]

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

    if num_classes == 2:
        if class_weights is None:
            class_weights = [1.0, 5.0]
        pos_weight = class_weights[1]
        loss_fn = torch.nn.BCEWithLogitsLoss(
            reduction="none",
            pos_weight=torch.tensor([[pos_weight]], device=device),
        )
    else:
        class_weights_tensor = torch.tensor(
            class_weights, dtype=torch.float32, device=device
        )
        loss_fn = torch.nn.CrossEntropyLoss(
            weight=class_weights_tensor,
            reduction="none",
        )

    total_loss = 0.0
    total_tokens = 0

    # Collect ranked predictions per sentence (with confidence scores)
    all_ranked_predictions: list[list[tuple[str, float]]] = []

    # Track gold keywords per document (doc_id, doc_row) -> keywords
    doc_to_gold: dict[tuple[str, int], list[str]] = {}

    num_batches = len(val_loader)

    with torch.no_grad():
        for batch in tqdm.tqdm(
            val_loader, desc=f"Evaluating ({split})", total=num_batches
        ):
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            target = batch["target"].to(device)
            sentence_counts = batch["sentence_counts"]
            doc_ids = batch["doc_ids"]
            doc_rows = batch["doc_rows"]
            keywords = batch["keywords"]

            # Forward pass
            logits = model(input_ids, attention_mask, sentence_counts)

            # Compute masked loss
            if num_classes == 2:
                loss = loss_fn(logits, target) * attention_mask
            else:
                loss = loss_fn(logits.transpose(1, 2), target) * attention_mask
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

            # Track gold keywords per document
            doc_to_gold.update(
                {
                    (doc_id, doc_row): kw_list
                    for doc_id, doc_row, kw_list in zip(doc_ids, doc_rows, keywords)
                }
            )

            # Free batch tensors to reduce memory fragmentation
            del input_ids, attention_mask, target, logits, loss, batch_ranked_keywords
            if device.type == "cuda":
                torch.cuda.empty_cache()

    # Compute mean loss
    val_loss = total_loss / total_tokens if total_tokens > 0 else 0.0

    # Aggregate predictions to document level with filtering, deduplication, and capping
    doc_ids = val_df["doc_id"].to_list()
    doc_rows = val_df["doc_row"].to_list()

    doc_to_pred_ranked = aggregate_document_keyphrases(
        all_ranked_predictions,
        doc_ids,
        doc_rows,
        max_keywords=10,
        filter_punctuation=True,
    )

    # Convert to lists for external eval module (preserve order)
    doc_keys = list(doc_to_gold.keys())
    gold_lists = [doc_to_gold[k] for k in doc_keys]
    pred_lists = [[kw for kw, _ in doc_to_pred_ranked[k]] for k in doc_keys]

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
                "gold_keywords": sorted(gold_lists[idx]),
                "pred_keywords": pred_lists[idx],
            }
        )

    # Free aggregation intermediates
    del (
        all_ranked_predictions,
        doc_ids,
        doc_rows,
        doc_to_gold,
        doc_to_pred_ranked,
    )

    # Compute @k metrics using external eval module (stemmed, macro-averaged)
    at_k_metrics = compute_metrics_at_k_martinc(gold_lists, pred_lists, k_values)
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
