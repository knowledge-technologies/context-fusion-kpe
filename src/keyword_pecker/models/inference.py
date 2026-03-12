"""Inference utilities for keyphrase extraction.

This module provides user-friendly inference functions for extracting
keyphrases from text using a trained KeyphraseModel.
"""

from __future__ import annotations

import re

import spacy
import torch

from keyword_pecker.data import load_spacy_model

from .keyphrase_model import KeyphraseModel

_spacy_model: spacy.Language | None = None


def _get_spacy_model() -> spacy.Language:
    """Get or lazy-load the spaCy model for sentence segmentation."""
    global _spacy_model
    if _spacy_model is None:
        _spacy_model = load_spacy_model()
    return _spacy_model


def _extract_keyphrases_from_logits(
    logits: torch.Tensor,
    input_ids: torch.Tensor,
    attention_mask: torch.Tensor,
    tokenizer,
) -> list[str]:
    """Extract keyphrase strings from model logits for a single sentence.

    Args:
        logits: Model logits of shape (seq_len,).
        input_ids: Token IDs of shape (seq_len,).
        attention_mask: Attention mask of shape (seq_len,).
        tokenizer: HuggingFace tokenizer for decoding tokens.

    Returns:
        List of keyphrase strings found in this sentence.
    """
    seq_len = attention_mask.sum().item()

    binary_pred = (logits > 0).long()
    masked_pred = binary_pred * input_ids

    pred_tokens = masked_pred[1 : seq_len - 1]

    decoded = tokenizer.decode(pred_tokens)

    keywords = decoded.split("<s>")
    keywords = [re.sub(r" - ", "-", kw.strip()) for kw in keywords if kw.strip()]

    return keywords


def predict_keyphrases(
    model: KeyphraseModel,
    text: str,
    nlp: spacy.Language | None = None,
    device: torch.device | str | None = None,
) -> list[str]:
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

    Returns:
        List of unique keyphrase strings in order of first appearance in text.

    Example:
        >>> from keyword_pecker.models import KeyphraseModel, predict_keyphrases
        >>> model = KeyphraseModel.from_pretrained("path/to/checkpoint")
        >>> keyphrases = predict_keyphrases(model, "Machine learning is a subset of AI.")
        >>> print(keyphrases)
        ['machine learning', 'ai']
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
        return []

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

    all_keyphrases: list[str] = []
    seen: set[str] = set()

    for i, sentence in enumerate(sentences):
        sent_keyphrases = _extract_keyphrases_from_logits(
            logits[i],
            input_ids[i],
            attention_mask[i],
            model.tokenizer,
        )

        for kw in sent_keyphrases:
            if kw not in seen:
                seen.add(kw)
                all_keyphrases.append(kw)

    return all_keyphrases
