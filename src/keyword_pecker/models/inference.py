"""Inference utilities for keyphrase extraction.

This module provides user-friendly inference functions for extracting
keyphrases from text using a trained KeyphraseModel.
"""

from __future__ import annotations

import math
import re
from typing import NamedTuple

import spacy
import torch
from nltk.stem.porter import PorterStemmer

from keyword_pecker.data import extract_word_level_data, load_spacy_model
from .keyphrase_model import KeyphraseModel


class KeyphraseResult(NamedTuple):
    keyword: str
    score: float


def _decode_binary_keyphrases(
    predictions: torch.Tensor,
    input_ids: torch.Tensor,
    attention_mask: torch.Tensor,
    tokenizer,
    scoring_strategy: str = "confidence",
    score_method: str = "average",
) -> list[list[tuple[str, float]]]:
    """Extract keyphrases from binary predictions using sigmoid + threshold.

    Args:
        predictions: Model logits of shape (batch_size, max_seq_len).
        input_ids: Token IDs of shape (batch_size, max_seq_len).
        attention_mask: Attention mask of shape (batch_size, max_seq_len).
        tokenizer: HuggingFace tokenizer for decoding tokens.

    Returns:
        List of keyword lists with confidence scores, one per sentence.
    """
    batch_size = predictions.shape[0]
    all_keyphrases: list[list[tuple[str, float]]] = []

    probs = torch.sigmoid(predictions)

    for i in range(batch_size):
        seq_len = attention_mask[i].sum().item()

        token_ids = input_ids[i, 1 : seq_len - 1]
        token_probs = probs[i, 1 : seq_len - 1]

        keywords_with_scores: list[tuple[str, float]] = []
        keyword_order = 0

        current_tokens: list[int] = []
        current_probs: list[float] = []

        for j in range(len(token_ids)):
            token_id = int(token_ids[j].item())
            prob = token_probs[j].item()

            if prob > 0.5:
                current_tokens.append(token_id)
                current_probs.append(prob)
            else:
                if current_tokens:
                    decoded = tokenizer.decode(torch.tensor(current_tokens))
                    keyword = re.sub(r" - ", "-", decoded.strip())
                    if keyword:
                        if scoring_strategy == "ordinal":
                            prob_score = -float(keyword_order)
                            keyword_order += 1
                        else:
                            prob_score = _score_keyphrase(
                                probs=current_probs, method=score_method
                            )

                        keywords_with_scores.append((keyword, prob_score))

                    current_tokens = []
                    current_probs = []

        # Flush any remaining tokens at the end of the sequence
        if current_tokens:
            decoded = tokenizer.decode(torch.tensor(current_tokens))
            keyword = re.sub(r" - ", "-", decoded.strip())
            if keyword:
                if scoring_strategy == "ordinal":
                    kw_score = -float(keyword_order)
                else:
                    kw_score = _score_keyphrase(
                        probs=current_probs, method=score_method
                    )
                keywords_with_scores.append((keyword, kw_score))

        keywords_with_scores.sort(key=lambda x: x[1], reverse=True)

        all_keyphrases.append(keywords_with_scores)

    return all_keyphrases


def _aggregate_tokens_to_words(
    token_probs: torch.Tensor,
    word_ids: torch.Tensor,
) -> tuple[list[int], list[float]]:
    """Aggregate token-level predictions to word-level by grouping subwords.

    For each word (grouped by word_id), the word-level prediction is derived
    by averaging softmax probabilities across all subword tokens, then taking
    argmax. The confidence score is the averaged P(predicted_class).

    This approach is more robust than using only the first subword's label,
    as it prevents a weak or noisy prefix from overriding a confident root.

    Args:
        token_probs: (seq_len, num_classes) tensor of softmax probabilities.
        word_ids: (seq_len,) tensor mapping tokens to word indices
            (-1 for special/padding).

    Returns:
        Tuple of (word_preds, word_probs) as plain Python lists.
    """
    word_token_indices: dict[int, list[int]] = {}
    for idx in range(len(word_ids)):
        wid = int(word_ids[idx].item())
        if wid == -1:
            continue
        if wid not in word_token_indices:
            word_token_indices[wid] = []
        word_token_indices[wid].append(idx)

    num_classes = token_probs.shape[1]
    word_preds: list[int] = []
    word_probs: list[float] = []

    for wid in sorted(word_token_indices.keys()):
        indices = word_token_indices[wid]
        avg_probs = [
            sum(token_probs[idx, c].item() for idx in indices) / len(indices)
            for c in range(num_classes)
        ]
        pred = int(max(range(num_classes), key=lambda c: avg_probs[c]))
        word_preds.append(pred)
        word_probs.append(avg_probs[pred])

    return word_preds, word_probs


def _extract_iob_keywords_word_level(
    word_preds: list[int],
    word_probs: list[float],
    word_list: list[str],
    scoring_strategy: str = "confidence",
    score_method: str = "average",
) -> list[tuple[str, float]]:
    """Extract keyphrases from word-level IOB predictions.

    Operates on pre-aggregated word-level data (subword tokens already merged
    by word_id). Each word has a single B/I/O prediction and a confidence score.

    Args:
        word_preds: Word-level class IDs (0=O, 1=B, 2=I), one per word.
        word_probs: Word-level confidence scores, one per word.
        word_list: The actual word strings, indexed to match word_preds.
        scoring_strategy: "confidence" for model probability scores, "ordinal" for extraction order.
        score_method: How to aggregate word probs ("average", "max", "min", "geometric").

    Returns:
        List of (keyphrase, score) tuples. No deduplication — handled at
        document level by aggregate_document_keyphrases.
    """
    keyphrases = []
    keyphrase_words = []
    keyphrase_probs = []
    keyphrase_order = 0

    for idx in range(len(word_preds)):
        pred = word_preds[idx]
        prob = word_probs[idx]

        if pred == 1:  # B-tag: start of a new keyphrase
            if keyphrase_words:
                kp = " ".join(keyphrase_words)
                if scoring_strategy == "ordinal":
                    score = -float(keyphrase_order)
                    keyphrase_order += 1
                else:
                    score = _score_keyphrase(keyphrase_probs, score_method)
                keyphrases.append((kp, score))
                keyphrase_words = []
                keyphrase_probs = []

            keyphrase_words.append(word_list[idx])
            keyphrase_probs.append(prob)

        elif pred == 2:  # I-tag: continuation of current keyphrase
            keyphrase_words.append(word_list[idx])
            keyphrase_probs.append(prob)

        else:  # O-tag: outside any keyphrase
            if keyphrase_words:
                kp = " ".join(keyphrase_words)
                if scoring_strategy == "ordinal":
                    score = -float(keyphrase_order)
                    keyphrase_order += 1
                else:
                    score = _score_keyphrase(keyphrase_probs, score_method)
                keyphrases.append((kp, score))
                keyphrase_words = []
                keyphrase_probs = []

    if keyphrase_words:
        kp = " ".join(keyphrase_words)
        if scoring_strategy == "ordinal":
            score = -float(keyphrase_order)
        else:
            score = _score_keyphrase(keyphrase_probs, score_method)
        keyphrases.append((kp, score))

    return keyphrases


def _decode_iob_keyphrases(
    predictions: torch.Tensor,
    input_ids: torch.Tensor,
    attention_mask: torch.Tensor,
    tokenizer,
    scoring_strategy: str = "confidence",
    score_method: str = "average",
    word_ids: torch.Tensor | None = None,
    word_lists: list[list[str]] | None = None,
) -> list[list[tuple[str, float]]]:
    """Extract keyphrases from IOB predictions using argmax + B/I/O decoding.

    When word_ids and word_lists are provided, uses word-boundary logic that
    matches the legacy TNT-KID extraction. Otherwise falls back to subword-level
    extraction for backward compatibility.

    Args:
        predictions: Model logits of shape (batch_size, max_seq_len, 3).
        input_ids: Token IDs of shape (batch_size, max_seq_len).
        attention_mask: Attention mask of shape (batch_size, max_seq_len).
        tokenizer: HuggingFace tokenizer for decoding tokens.
        score_method: How to aggregate token probs ("average", "max", "min", or "geometric").
        word_ids: Optional (batch_size, max_seq_len) tensor mapping tokens to word indices.
        word_lists: Optional list of word lists, one per batch item.

    Returns:
        List of keyword lists with confidence scores, one per sentence.
    """
    batch_size = predictions.shape[0]
    all_keyphrases: list[list[tuple[str, float]]] = []

    class_preds = torch.argmax(predictions, dim=-1)
    probs = torch.softmax(predictions, dim=-1)

    # Decide whether to use word-boundary or subword-level extraction
    use_word_level = word_ids is not None and word_lists is not None

    for i in range(batch_size):
        seq_len = attention_mask[i].sum().item()

        if use_word_level:
            token_probs = probs[i, :seq_len]
            sentence_word_ids = word_ids[i, :seq_len]

            word_preds, word_probs = _aggregate_tokens_to_words(
                token_probs, sentence_word_ids
            )
            sentence_keyphrases = _extract_iob_keywords_word_level(
                word_preds,
                word_probs,
                word_lists[i],
                scoring_strategy,
                score_method,
            )
        else:
            # Fallback: subword-level extraction (backward compatibility)
            token_ids = input_ids[i, 1 : seq_len - 1]
            token_preds = class_preds[i, 1 : seq_len - 1]
            token_probs = probs[i, 1 : seq_len - 1]

            keywords_with_scores: list[tuple[str, float]] = []
            keyword_order = 0

            current_tokens: list[int] = []
            current_probs: list[float] = []

            for j in range(len(token_ids)):
                pred = token_preds[j].item()
                prob = (1.0 - token_probs[j, 0]).item()

                if pred == 1:
                    if current_tokens:
                        decoded = tokenizer.decode(torch.tensor(current_tokens))
                        keyword = re.sub(r" - ", "-", decoded.strip())
                        if keyword:
                            if scoring_strategy == "ordinal":
                                prob_score = -float(keyword_order)
                                keyword_order += 1
                            else:
                                prob_score = _score_keyphrase(
                                    probs=current_probs, method=score_method
                                )
                            keywords_with_scores.append((keyword, prob_score))

                    current_tokens = [int(token_ids[j].item())]
                    current_probs = [prob]

                elif pred == 2:
                    if current_tokens:
                        current_tokens.append(int(token_ids[j].item()))
                        current_probs.append(prob)

                else:
                    if current_tokens:
                        decoded = tokenizer.decode(torch.tensor(current_tokens))
                        keyword = re.sub(r" - ", "-", decoded.strip())
                        if keyword:
                            if scoring_strategy == "ordinal":
                                prob_score = -float(keyword_order)
                                keyword_order += 1
                            else:
                                prob_score = _score_keyphrase(
                                    probs=current_probs, method=score_method
                                )

                            keywords_with_scores.append((keyword, prob_score))
                        current_tokens = []
                        current_probs = []

            # Flush any remaining tokens at the end of the sequence
            if current_tokens:
                decoded = tokenizer.decode(torch.tensor(current_tokens))
                keyword = re.sub(r" - ", "-", decoded.strip())
                if keyword:
                    if scoring_strategy == "ordinal":
                        prob_score = -float(keyword_order)
                    else:
                        prob_score = _score_keyphrase(
                            probs=current_probs, method=score_method
                        )
                    keywords_with_scores.append((keyword, prob_score))

            keywords_with_scores.sort(key=lambda x: x[1], reverse=True)

            sentence_keyphrases = keywords_with_scores

        all_keyphrases.append(sentence_keyphrases)

    return all_keyphrases


def extract_ranked_keyphrases(
    predictions: torch.Tensor,
    input_ids: torch.Tensor,
    attention_mask: torch.Tensor,
    tokenizer,
    num_classes: int | None = None,
    scoring_strategy: str = "confidence",
    score_method: str = "average",
    word_ids: torch.Tensor | None = None,
    word_lists: list[list[str]] | None = None,
) -> list[list[tuple[str, float]]]:
    """Extract keyphrases ranked by model confidence.

    Supports both binary and IOB classification schemes:
    - Binary (num_classes=2): Uses sigmoid + 0.5 threshold
    - IOB (num_classes=3): Uses argmax to decode B/I/O tags

    Args:
        predictions: Model logits.
            - Binary: shape (batch_size, max_seq_len)
            - IOB: shape (batch_size, max_seq_len, 3)
        input_ids: Token IDs of shape (batch_size, max_seq_len).
        attention_mask: Attention mask of shape (batch_size, max_seq_len).
        tokenizer: HuggingFace tokenizer for decoding tokens.
        num_classes: Number of output classes. If None, auto-detect from shape.
            2 for binary, 3 for IOB scheme.
        word_ids: Optional (batch_size, max_seq_len) tensor mapping tokens to word indices.
            Only used for IOB scheme.
        word_lists: Optional list of word lists, one per batch item.
            Only used for IOB scheme.

    Returns:
        List of keyword lists with confidence scores, one per sentence.
        Each element is a list of (keyword, confidence) tuples sorted by
        confidence descending.
    """
    if num_classes is None:
        num_classes = 3 if predictions.dim() == 3 else 2

    if num_classes == 2:
        return _decode_binary_keyphrases(
            predictions,
            input_ids,
            attention_mask,
            tokenizer,
            scoring_strategy,
            score_method,
        )
    else:
        return _decode_iob_keyphrases(
            predictions,
            input_ids,
            attention_mask,
            tokenizer,
            scoring_strategy,
            score_method,
            word_ids=word_ids,
            word_lists=word_lists,
        )


# Punctuation characters to disallow in keyphrases.
# Copied exactly from the legacy TNT-KID eval code (train_and_eval.py line 858).
# NOTE: This includes apostrophe ('), which contradicts the original TNT-KID paper
# stating "we remove keyphrases containing punctuation (except dashes and apostrophes)".
# We keep the legacy behavior as-is for reproducible evaluation comparisons.
_DEFAULT_DISALLOWED_PUNCTUATION = "!#$%&'()*+,.:;<=>?@[\\]^_`{|}~"

_stemmer = PorterStemmer()


def _score_keyphrase(probs: list[float], method: str = "average") -> float:
    """Score a keyphrase from its token-level probabilities.

    Args:
        probs: Token-level probabilities for the keyphrase.
        method: "average" (mean of token probs),
            "max" (highest single token prob),
            "min" (lowest token prob, conservative),
            "geometric" (geometric mean, penalizes low-confidence tokens).

    Returns:
        Scalar confidence score for the keyphrase.
    """
    if method == "max":
        return max(probs)
    if method == "min":
        return min(probs)
    if method == "geometric":
        log_sum = sum(math.log(p) if p > 0 else float("-inf") for p in probs)
        return math.exp(log_sum / len(probs))
    return sum(probs) / len(probs)


def aggregate_document_keyphrases(
    ranked_predictions: list[list[tuple[str, float]]],
    doc_ids: list[str],
    doc_rows: list[int],
    max_keyphrases: int | None = 10,
    max_keyphrase_length: int = 5,
    filter_punctuation: bool = True,
    disallowed_punctuation: str = _DEFAULT_DISALLOWED_PUNCTUATION,
    dedup_strategy: str = "stem",
) -> dict[tuple[str, int], list[tuple[str, float]]]:
    """Aggregate sentence-level keyphrase predictions to document level.

    Applies the post-processing pipeline aligned with the legacy TNT-KID
    evaluation logic:
    1. Aggregate predictions by document
    2. Filter keyphrases containing disallowed punctuation (only dash is
       allowed; apostrophe is filtered)
    3. Filter keyphrases with >= max_keyphrase_length words
    4. Deduplicate (stem-based by default to match legacy, or lowercase)
    5. Sort by confidence score (descending)
    6. Cap at max_keyphrases per document (skip if None)

    Args:
        ranked_predictions: List of keyphrase lists with confidence scores,
            one per sentence. Each inner list contains (keyword, score) tuples.
        doc_ids: List of document IDs for each sentence.
        doc_rows: List of document row indices for each sentence.
        max_keyphrases: Maximum number of keyphrases to return per document.
            If None, no cap is applied.
        max_keyphrase_length: Filter out keyphrases with >= this many words.
        filter_punctuation: If True, remove keyphrases containing disallowed
            punctuation characters.
        disallowed_punctuation: String of punctuation characters to filter.
            Copied from legacy TNT-KID eval code. Only dash (-) is allowed by
            default. Note: apostrophe (') IS included (filtered), which differs
            from the TNT-KID paper that says apostrophes should be kept.
        dedup_strategy: "stem" for Porter stemmer-based deduplication (matches
            legacy TNT-KID behavior), or "lowercase" for simple case folding.

    Returns:
        Dict mapping (doc_id, doc_row) -> list of (keyword, score) tuples,
        sorted by confidence descending, capped at max_keyphrases (if set).
    """
    # Step 1: Aggregate predictions by document
    doc_to_pred: dict[tuple[str, int], list[tuple[str, float]]] = {}
    for i, (doc_id, doc_row) in enumerate(zip(doc_ids, doc_rows)):
        key = (doc_id, doc_row)
        if key not in doc_to_pred:
            doc_to_pred[key] = []
        if i < len(ranked_predictions):
            doc_to_pred[key].extend(ranked_predictions[i])

    # Steps 2-6: Process each document
    for key in doc_to_pred:
        predictions = doc_to_pred[key]

        # Step 2: Filter punctuation (only dash allowed; apostrophe is filtered,
        # which differs from the TNT-KID paper but matches legacy code behavior)
        if filter_punctuation:
            predictions = [
                (kw, score)
                for kw, score in predictions
                if kw.strip() and not any(p in kw for p in disallowed_punctuation)
            ]

        # Step 3: Filter keyphrases that are too long (>= max_keyphrase_length words)
        predictions = [
            (kw, score)
            for kw, score in predictions
            if len(kw.split()) < max_keyphrase_length
        ]

        # Step 4: Deduplicate - stem-based (legacy) or lowercase
        seen: dict[str, tuple[str, float]] = {}
        for kw, score in predictions:
            if dedup_strategy == "stem":
                # Normalize hyphens to spaces, then lowercase and stem each word
                normalized = kw.replace("-", " ")
                kw_key = " ".join(
                    [_stemmer.stem(w.lower()) for w in normalized.split()]
                )
                # Skip keyphrases containing repeated stemmed words
                if len(kw_key.split()) != len(set(kw_key.split())):
                    continue
            else:
                kw_key = kw.lower()

            # Keep the occurrence with the highest confidence score
            if kw_key not in seen or score > seen[kw_key][1]:
                seen[kw_key] = (kw, score)

        # Step 5: Sort by score descending
        deduped = list(seen.values())
        deduped.sort(key=lambda x: x[1], reverse=True)

        # Step 6: Cap at max_keyphrases (skip if None)
        if max_keyphrases is not None:
            doc_to_pred[key] = deduped[:max_keyphrases]
        else:
            doc_to_pred[key] = deduped

    return doc_to_pred


_spacy_model: spacy.Language | None = None


def _get_spacy_model() -> spacy.Language:
    """Get or lazy-load the spaCy model for sentence segmentation."""
    global _spacy_model
    if _spacy_model is None:
        _spacy_model = load_spacy_model()
    return _spacy_model


def predict_keyphrases(
    model: KeyphraseModel,
    text: str,
    nlp: spacy.Language | None = None,
    device: torch.device | str | None = None,
    nr_keyphrases: int | None = None,
    scoring_strategy: str = "confidence",
    score_method: str = "average",
    return_scores: bool = False,
) -> list[str] | list[KeyphraseResult]:
    """Extract keyphrases from text using a trained KeyphraseModel.

    This function handles the full inference pipeline:
    1. Sentence segmentation (using spaCy)
    2. Tokenization (using the model's tokenizer)
    3. Model inference
    4. Keyphrase extraction, aggregation, and deduplication

    Args:
        model: Trained KeyphraseModel.
        text: Input text to extract keyphrases from.
        nlp: Optional spaCy model for sentence segmentation.
            If None, loads en_core_web_sm on first call (cached for reuse).
        device: Device for inference. If None, uses the device of model parameters.
        nr_keyphrases: Maximum number of keyphrases to return. If None, uses
            the default from evaluation config (typically 10).
        return_scores: If True, returns list of KeyphraseResult with confidence
            scores. If False, returns list of keyphrase strings only.

    Returns:
        If return_scores=False: List of unique keyphrase strings sorted by
            confidence (highest first).
        If return_scores=True: List of KeyphraseResult(keyword, score) tuples
            sorted by confidence (highest first).

    Example:
        >>> from keyword_pecker.models import KeyphraseModel, predict_keyphrases
        >>> model = KeyphraseModel.from_pretrained("path/to/checkpoint")
        >>> keyphrases = predict_keyphrases(model, "Machine learning is a subset of AI.")
        >>> print(keyphrases)
        ['machine learning', 'ai']
        >>> results = predict_keyphrases(model, text, return_scores=True)
        >>> for result in results:
        ...     print(f"{result.keyword}: {result.score:.2f}")
        machine learning: 0.95
        ai: 0.87
    """
    model.eval()

    if device is None:
        device = next(model.parameters()).device
    elif isinstance(device, str):
        device = torch.device(device)

    if nlp is None:
        nlp = _get_spacy_model()

    doc = nlp(text)
    sentences = [s.text for s in doc.sents if s.text.strip()]

    if not sentences:
        return [] if not return_scores else []

    tokenized = model.tokenizer(
        sentences,
        return_tensors="pt",
        return_offsets_mapping=True,
        truncation=True,
        padding=True,
    )

    input_ids = tokenized["input_ids"].to(device)
    attention_mask = tokenized["attention_mask"].to(device)
    sentence_counts = [len(sentences)]

    # Compute word_ids and word_lists using tokenizer word boundaries
    word_ids_batch, word_lists = extract_word_level_data(tokenized, sentences)

    # Pad word_ids to match input_ids length
    max_len = input_ids.shape[1]
    padded_word_ids = torch.full((len(sentences), max_len), -1, dtype=torch.long)
    for i, wid_seq in enumerate(word_ids_batch):
        padded_word_ids[i, : len(wid_seq)] = torch.tensor(wid_seq, dtype=torch.long)

    with torch.no_grad():
        logits = model(input_ids, attention_mask, sentence_counts)

    ranked_keyphrases = extract_ranked_keyphrases(
        logits,
        input_ids,
        attention_mask,
        model.tokenizer,
        num_classes=model.refinement_head.num_classes,
        scoring_strategy=scoring_strategy,
        score_method=score_method,
        word_ids=padded_word_ids.to(device),
        word_lists=word_lists,
    )

    # Aggregate to document level with filtering, deduplication, and capping
    doc_ids = ["doc_0"] * len(sentences)
    doc_rows = [0] * len(sentences)
    max_kp = nr_keyphrases or 10

    doc_results = aggregate_document_keyphrases(
        ranked_keyphrases,
        doc_ids,
        doc_rows,
        max_keyphrases=max_kp,
    )

    sorted_keyphrases = doc_results.get(("doc_0", 0), [])

    if return_scores:
        return [KeyphraseResult(kw, score) for kw, score in sorted_keyphrases]
    else:
        return [kw for kw, _ in sorted_keyphrases]
