"""Tests for inference module functions."""

import torch
from transformers import AutoTokenizer

from keyword_pecker.models.inference import extract_ranked_keyphrases


def _create_mock_tokenizer():
    tokenizer = AutoTokenizer.from_pretrained("sentence-transformers/all-mpnet-base-v2")
    return tokenizer


def test_extract_ranked_keyphrases_single_keyword():
    tokenizer = _create_mock_tokenizer()
    text = "machine learning is great"
    encoded = tokenizer(text, return_tensors="pt")
    input_ids = encoded["input_ids"]
    attention_mask = encoded["attention_mask"]

    logits = torch.zeros_like(input_ids, dtype=torch.float)
    logits[0, 1:3] = 2.0

    results = extract_ranked_keyphrases(logits, input_ids, attention_mask, tokenizer)

    assert len(results) == 1
    assert len(results[0]) == 1
    keyword, score = results[0][0]
    assert "machine" in keyword.lower()
    assert score > 0.5


def test_extract_ranked_keyphrases_multiple_keywords():
    tokenizer = _create_mock_tokenizer()
    text = "machine learning and deep learning are popular"
    encoded = tokenizer(text, return_tensors="pt")
    input_ids = encoded["input_ids"]
    attention_mask = encoded["attention_mask"]

    logits = torch.zeros_like(input_ids, dtype=torch.float)
    logits[0, 1:3] = 2.0
    logits[0, 5:7] = 1.5

    results = extract_ranked_keyphrases(logits, input_ids, attention_mask, tokenizer)

    assert len(results) == 1
    assert len(results[0]) == 2
    assert results[0][0][1] > results[0][1][1]


def test_extract_ranked_keyphrases_no_keywords():
    tokenizer = _create_mock_tokenizer()
    text = "this is a simple sentence"
    encoded = tokenizer(text, return_tensors="pt")
    input_ids = encoded["input_ids"]
    attention_mask = encoded["attention_mask"]

    logits = torch.zeros_like(input_ids, dtype=torch.float)

    results = extract_ranked_keyphrases(logits, input_ids, attention_mask, tokenizer)

    assert len(results) == 1
    assert len(results[0]) == 0


def test_extract_ranked_keyphrases_multiword_keyword():
    tokenizer = _create_mock_tokenizer()
    text = "natural language processing is useful"
    encoded = tokenizer(text, return_tensors="pt")
    input_ids = encoded["input_ids"]
    attention_mask = encoded["attention_mask"]

    logits = torch.zeros_like(input_ids, dtype=torch.float)
    logits[0, 1:4] = 2.0

    results = extract_ranked_keyphrases(logits, input_ids, attention_mask, tokenizer)

    assert len(results) == 1
    assert len(results[0]) == 1
    keyword, score = results[0][0]
    assert "natural" in keyword.lower() or "language" in keyword.lower()
    assert score > 0.5


def test_extract_ranked_keyphrases_batch():
    tokenizer = _create_mock_tokenizer()
    texts = ["machine learning", "deep neural networks"]
    encoded = tokenizer(texts, return_tensors="pt", padding=True)
    input_ids = encoded["input_ids"]
    attention_mask = encoded["attention_mask"]

    logits = torch.zeros_like(input_ids, dtype=torch.float)
    logits[0, 1:3] = 2.0
    logits[1, 1:3] = 1.5

    results = extract_ranked_keyphrases(logits, input_ids, attention_mask, tokenizer)

    assert len(results) == 2
    assert len(results[0]) == 1
    assert len(results[1]) == 1


def test_extract_ranked_keyphrases_deduplication():
    tokenizer = _create_mock_tokenizer()
    text = "python python python"
    encoded = tokenizer(text, return_tensors="pt")
    input_ids = encoded["input_ids"]
    attention_mask = encoded["attention_mask"]

    logits = torch.zeros_like(input_ids, dtype=torch.float)
    seq_len = attention_mask.sum().item()
    logits[0, 1:seq_len] = 2.0

    results = extract_ranked_keyphrases(logits, input_ids, attention_mask, tokenizer)

    assert len(results) == 1
    keywords = [kw for kw, _ in results[0]]
    assert len(set(keywords)) == len(keywords)


def test_extract_ranked_keyphrases_keyword_at_boundary():
    tokenizer = _create_mock_tokenizer()
    text = "python is great"
    encoded = tokenizer(text, return_tensors="pt")
    input_ids = encoded["input_ids"]
    attention_mask = encoded["attention_mask"]

    logits = torch.zeros_like(input_ids, dtype=torch.float)
    logits[0, 1] = 2.0

    results = extract_ranked_keyphrases(logits, input_ids, attention_mask, tokenizer)

    assert len(results) == 1
    assert len(results[0]) == 1
    keyword, score = results[0][0]
    assert "python" in keyword.lower()
