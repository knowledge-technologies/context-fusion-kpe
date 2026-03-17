"""Inference utilities for keyphrase extraction.

This module provides user-friendly inference functions for extracting
keyphrases from text using a trained KeyphraseModel.
"""

from __future__ import annotations

import re
from typing import NamedTuple

import spacy
import torch

from keyword_pecker.data import load_spacy_model

from .keyphrase_model import KeyphraseModel


class KeyphraseResult(NamedTuple):
    keyword: str
    score: float


def extract_ranked_keyphrases(
    predictions: torch.Tensor,
    input_ids: torch.Tensor,
    attention_mask: torch.Tensor,
    tokenizer,
) -> list[list[tuple[str, float]]]:
    """Extract keyphrases ranked by model confidence.

    Converts logits to probabilities via sigmoid, then extracts contiguous
    spans of tokens where probability > 0.5, treating each span as a keyword.

    Args:
        predictions: Model logits of shape (batch_size, max_seq_len).
        input_ids: Token IDs of shape (batch_size, max_seq_len).
        attention_mask: Attention mask of shape (batch_size, max_seq_len).
        tokenizer: HuggingFace tokenizer for decoding tokens.

    Returns:
        List of keyword lists with confidence scores, one per sentence.
        Each element is a list of (keyword, confidence) tuples sorted by
        confidence descending.
    """
    batch_size = predictions.shape[0]
    all_ranked_keywords: list[list[tuple[str, float]]] = []

    probs = torch.sigmoid(predictions)

    for i in range(batch_size):
        seq_len = attention_mask[i].sum().item()

        token_ids = input_ids[i, 1 : seq_len - 1]
        token_probs = probs[i, 1 : seq_len - 1]

        keywords_with_scores: list[tuple[str, float]] = []

        current_tokens: list[int] = []
        current_probs: list[float] = []

        for j in range(len(token_ids)):
            token_id = token_ids[j].item()
            prob = token_probs[j].item()

            if prob > 0.5:
                current_tokens.append(token_id)
                current_probs.append(prob)
            else:
                if current_tokens:
                    decoded = tokenizer.decode(torch.tensor(current_tokens))
                    keyword = re.sub(r" - ", "-", decoded.strip())
                    if keyword:
                        max_prob = max(current_probs)
                        keywords_with_scores.append((keyword, max_prob))
                    current_tokens = []
                    current_probs = []

        if current_tokens:
            decoded = tokenizer.decode(torch.tensor(current_tokens))
            keyword = re.sub(r" - ", "-", decoded.strip())
            if keyword:
                max_prob = max(current_probs)
                keywords_with_scores.append((keyword, max_prob))

        keywords_with_scores.sort(key=lambda x: x[1], reverse=True)

        seen: set[str] = set()
        deduped: list[tuple[str, float]] = []
        for kw, score in keywords_with_scores:
            if kw not in seen:
                seen.add(kw)
                deduped.append((kw, score))

        all_ranked_keywords.append(deduped)

    return all_ranked_keywords


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
    nr_keywords: int | None = None,
    return_scores: bool = False,
) -> list[str] | list[KeyphraseResult]:
    """Extract keyphrases from text using a trained KeyphraseModel.

    This function handles the full inference pipeline:
    1. Sentence segmentation (using spaCy)
    2. Tokenization (using the model's tokenizer)
    3. Model inference
    4. Keyphrase extraction and deduplication

    Args:
        model: Trained KeyphraseModel.
        text: Input text to extract keyphrases from.
        nlp: Optional spaCy model for sentence segmentation.
            If None, loads en_core_web_sm on first call (cached for reuse).
        device: Device for inference. If None, uses the device of model parameters.
        nr_keywords: Maximum number of keywords to return. If None, returns all
            detected keywords.
        return_scores: If True, returns list of KeyphraseResult with confidence
            scores. If False, returns list of keyword strings only.

    Returns:
        If return_scores=False: List of unique keyphrase strings sorted by
            confidence (highest first).
        If return_scores=True: List of KeyphraseResult(keyword, score) tuples
            sorted by confidence (highest first).

    Example:
        >>> from keyword_pecker.models import KeyphraseModel, predict_keyphrases
        >>> model = KeyphraseModel.from_pretrained("path/to/checkpoint")
        >>> # Get keywords only (backward compatible)
        >>> keyphrases = predict_keyphrases(model, "Machine learning is a subset of AI.")
        >>> print(keyphrases)
        ['machine learning', 'ai']
        >>> # Get keywords with confidence scores
        >>> results = predict_keyphrases(model, "Machine learning is a subset of AI.",
        ...                              return_scores=True)
        >>> for result in results:
        ...     print(f"{result.keyword}: {result.score:.2f}")
        machine learning: 0.95
        ai: 0.87
        >>> # Limit to top 3 keywords
        >>> top_3 = predict_keyphrases(model, text, nr_keywords=3, return_scores=True)
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
        truncation=True,
        padding=True,
    )

    input_ids = tokenized["input_ids"].to(device)
    attention_mask = tokenized["attention_mask"].to(device)
    sentence_counts = [len(sentences)]

    with torch.no_grad():
        logits = model(input_ids, attention_mask, sentence_counts)

    ranked_keyphrases = extract_ranked_keyphrases(
        logits, input_ids, attention_mask, model.tokenizer
    )

    all_keyphrases: dict[str, float] = {}
    for sent_keyphrases in ranked_keyphrases:
        for keyword, score in sent_keyphrases:
            if keyword not in all_keyphrases or score > all_keyphrases[keyword]:
                all_keyphrases[keyword] = score

    sorted_keyphrases = sorted(all_keyphrases.items(), key=lambda x: x[1], reverse=True)

    if nr_keywords is not None:
        sorted_keyphrases = sorted_keyphrases[:nr_keywords]

    if return_scores:
        return [KeyphraseResult(kw, score) for kw, score in sorted_keyphrases]
    else:
        return [kw for kw, _ in sorted_keyphrases]
