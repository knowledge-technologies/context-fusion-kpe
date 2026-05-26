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
import types
from pathlib import Path

import polars as pl
import torch
import tqdm

from keyword_pecker.constants import DEFAULT_DISALLOWED_PUNCTUATION
from keyword_pecker.data import create_keyphrase_dataloader
from keyword_pecker.models import KeyphraseModel
from keyword_pecker.models.inference import (
    aggregate_document_keyphrases,
    extract_ranked_keyphrases,
)
from keyword_pecker.trainer.external import eval as eval_martinc
from keyword_pecker.utils import get_logger

logger = get_logger("validate")


def _normalize_keyphrases(keyphrases: list[list[str]]) -> list[list[str]]:
    """Apply legacy-style normalization: replace -, /, ∗ with spaces and lowercase."""
    return [
        [
            kw.replace("-", " ").replace("/", " ").replace("∗", " ").lower()
            for kw in doc_kws
        ]
        for doc_kws in keyphrases
    ]


def _compute_confusion_matrix(
    predictions: list[int], targets: list[int], num_classes: int = 3
) -> list[list[int]]:
    """Compute confusion matrix from flattened predictions and targets."""
    cm = [[0] * num_classes for _ in range(num_classes)]
    for pred, target in zip(predictions, targets):
        cm[target][pred] += 1
    return cm


def _log_confusion_matrix_to_wandb(
    cm: list[list[int]],
    class_labels: list[str],
    split: str,
    wandb,
    global_step: int,
) -> None:
    """Log confusion matrix as a W&B Table."""
    columns = ["actual \\ predicted"] + class_labels
    data = [[f"actual_{label}"] + row for label, row in zip(class_labels, cm)]
    wandb.log(
        {f"{split}/confusion_matrix": wandb.Table(columns=columns, data=data)},
        step=global_step,
    )


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
    5: {"precision": 6, "recall": 7, "f1": 8},
    10: {"precision": 9, "recall": 10, "f1": 11},
    "N": {"precision": 12, "recall": 13, "f1": 14},
    "M": {"precision": 15, "recall": 16, "f1": 17},
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
    use_class_weights: bool = True,
    class_weights: list[float] | None = None,
    device: torch.device,
    split: str = "val",
    k_values: list[int | str] | None = None,
    metrics_filename: Path | None = None,
    wandb: types.ModuleType | None = None,
    global_step: int = 0,
    epoch: int = 0,
    sample_seed: int = 42,
    scoring_strategy: str = "confidence",
    score_method: str = "average",
    iob_threshold: float = 0.5,
    filter_punctuation: bool = True,
    disallowed_punctuation: str = DEFAULT_DISALLOWED_PUNCTUATION,
    dedup_strategy: str = "stem",
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
        use_class_weights: Whether to apply class weights in loss function.
            If False, uses equal weighting (CrossEntropyLoss with weight=None).
        class_weights: Class weights for loss function. Required for IOB scheme
            when use_class_weights=True; defaults to [1.0, 5.0] for binary scheme.
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
        scoring_strategy: "confidence" for model probability scores, "ordinal" for extraction order (default: "confidence").
        score_method: How to aggregate token-specific probabilities within the keyphrase ("average", "max", "min", "geometric").
        iob_threshold: Minimum confidence for B/I predictions in IOB scheme.
            Predictions below this threshold are treated as O. Default 0.5.
        filter_punctuation: If True, remove keyphrases containing disallowed
            punctuation characters (default: True).
        disallowed_punctuation: String of punctuation characters to filter from
            keyphrases. Copied from legacy TNT-KID eval code. Only dash (-) is
            allowed by default. Note: apostrophe (') IS included (filtered),
            which differs from the TNT-KID paper that says apostrophes should
            be kept.
        dedup_strategy: "stem" for Porter stemmer-based deduplication (matches
            legacy TNT-KID behavior), or "lowercase" for simple case folding.

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
        if use_class_weights and class_weights is not None:
            class_weights_tensor = torch.tensor(
                class_weights, dtype=torch.float32, device=device
            )
            loss_fn = torch.nn.CrossEntropyLoss(
                weight=class_weights_tensor,
                reduction="none",
            )
        else:
            loss_fn = torch.nn.CrossEntropyLoss(
                weight=None,
                reduction="none",
            )

    total_loss = 0.0
    total_tokens = 0

    # Collect ranked predictions per sentence (with confidence scores)
    all_ranked_predictions: list[list[tuple[str, float]]] = []

    # Compute gold keywords per document from sent_keywords union.
    # This matches legacy kw_in_paper: the union of all sentence-level
    # keyword matches across the document.
    gold_agg = val_df.group_by(["doc_id", "doc_row"]).agg(
        pl.col("sent_keywords").flatten().unique().alias("keywords")
    )
    doc_to_gold: dict[tuple[str, int], list[str]] = {
        (row["doc_id"], row["doc_row"]): row["keywords"]
        for row in gold_agg.iter_rows(named=True)
    }

    # Collect token-level predictions/targets for confusion matrix (IOB only)
    cm_preds: list[int] = []
    cm_targets: list[int] = []

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

            # Forward pass
            logits = model(input_ids, attention_mask, sentence_counts)

            # Collect confusion matrix data (IOB scheme only)
            if num_classes == 3:
                preds = logits.argmax(dim=2)
                mask = attention_mask.bool()
                cm_preds.extend(preds[mask].cpu().tolist())
                cm_targets.extend(target[mask].cpu().tolist())

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
                num_classes=num_classes,
                scoring_strategy=scoring_strategy,
                score_method=score_method,
                word_ids=batch.get("word_ids"),
                word_lists=batch.get("word_lists"),
                iob_threshold=iob_threshold,
            )
            all_ranked_predictions.extend(batch_ranked_keywords)

            # Free batch tensors to reduce memory fragmentation
            del input_ids, attention_mask, target, logits, loss, batch_ranked_keywords
            if device.type == "cuda":
                torch.cuda.empty_cache()

    # Compute mean loss
    val_loss = total_loss / total_tokens if total_tokens > 0 else 0.0

    # Aggregate predictions to document level with filtering and deduplication
    doc_ids = val_df["doc_id"].to_list()
    doc_rows = val_df["doc_row"].to_list()

    doc_to_pred_ranked = aggregate_document_keyphrases(
        all_ranked_predictions,
        doc_ids,
        doc_rows,
        max_keyphrases=None,
        filter_punctuation=filter_punctuation,
        disallowed_punctuation=disallowed_punctuation,
        dedup_strategy=dedup_strategy,
    )

    # Convert to lists for external eval module (preserve order)
    doc_keys = list(doc_to_gold.keys())
    gold_lists = [doc_to_gold[k] for k in doc_keys]
    pred_lists = [[kw for kw, _ in doc_to_pred_ranked[k]] for k in doc_keys]

    gold_lists_norm = _normalize_keyphrases(keyphrases=gold_lists)
    pred_lists_norm = _normalize_keyphrases(keyphrases=pred_lists)

    _N_SAMPLES = 10
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
                "gold_keywords": sorted(gold_lists_norm[idx]),
                "pred_keywords": pred_lists_norm[idx][:_N_SAMPLES],
            }
        )

    # Free aggregation intermediates
    del (
        all_ranked_predictions,
        doc_ids,
        doc_rows,
        doc_to_gold,
        doc_to_pred_ranked,
        gold_lists,
        pred_lists,
    )

    # Compute @k metrics using external eval module (stemmed, macro-averaged)
    at_k_metrics = compute_metrics_at_k_martinc(
        gold_lists_norm, pred_lists_norm, k_values
    )
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

        # log evaluation metadata
        wandb_log["num_docs_total"] = at_k_metrics["num_docs_total"]
        wandb_log["num_docs_removed"] = at_k_metrics["num_docs_removed"]
        wandb_log["num_docs_evaluated"] = at_k_metrics["num_docs_evaluated"]

        wandb.log(wandb_log, step=global_step)

        # log samples too
        samples_table = wandb.Table(
            columns=["doc_id", "doc_row", "gold_keywords", "pred_keywords"],
            data=[
                [s["doc_id"], s["doc_row"], s["gold_keywords"], s["pred_keywords"]]
                for s in metrics["samples"]
            ],
        )
        wandb.log({f"{split}/samples": samples_table}, step=global_step)

        # log confusion matrix (IOB scheme only)
        if num_classes == 3 and cm_preds and cm_targets:
            cm = _compute_confusion_matrix(cm_preds, cm_targets, num_classes=3)
            _log_confusion_matrix_to_wandb(
                cm,
                class_labels=["O", "B", "I"],
                split=split,
                wandb=wandb,
                global_step=global_step,
            )

    # Restore model to training mode
    model.train()

    return metrics
