"""Tests for inference module functions."""

import torch
from transformers import AutoTokenizer

from keyword_pecker.models.inference import (
    aggregate_document_keyphrases,
    extract_ranked_keyphrases,
)


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


# ============================================================================
# Tests for aggregate_document_keyphrases
# ============================================================================


def test_aggregate_document_keyphrases_basic():
    """Test basic aggregation of multiple sentences to single document."""
    predictions = [
        [("machine learning", 0.9), ("neural networks", 0.8)],
        [("deep learning", 0.85)],
    ]
    doc_ids = ["doc1", "doc1"]
    doc_rows = [0, 0]

    result = aggregate_document_keyphrases(predictions, doc_ids, doc_rows)

    assert ("doc1", 0) in result
    keywords = [kw for kw, _ in result[("doc1", 0)]]
    assert len(keywords) == 3
    assert "machine learning" in keywords
    assert "neural networks" in keywords
    assert "deep learning" in keywords


def test_aggregate_document_keyphrases_multi_document():
    """Test aggregation across multiple documents."""
    predictions = [
        [("keyword1", 0.9)],
        [("keyword2", 0.8)],
        [("keyword3", 0.7)],
    ]
    doc_ids = ["doc1", "doc1", "doc2"]
    doc_rows = [0, 0, 0]

    result = aggregate_document_keyphrases(predictions, doc_ids, doc_rows)

    assert len(result) == 2
    assert ("doc1", 0) in result
    assert ("doc2", 0) in result
    assert len(result[("doc1", 0)]) == 2
    assert len(result[("doc2", 0)]) == 1


def test_aggregate_document_keyphrases_punctuation_filtering():
    """Test that keyphrases with disallowed punctuation are filtered out."""
    predictions = [
        [("machine learning", 0.9), ("data, science", 0.8)],
        [("neural-network", 0.85), ("don't", 0.7)],
        [("test (keyword)", 0.6)],
    ]
    doc_ids = ["doc1", "doc1", "doc1"]
    doc_rows = [0, 0, 0]

    result = aggregate_document_keyphrases(predictions, doc_ids, doc_rows)

    keywords = [kw for kw, _ in result[("doc1", 0)]]
    assert "machine learning" in keywords
    assert "neural-network" in keywords
    assert "don't" in keywords
    assert "data, science" not in keywords
    assert "test (keyword)" not in keywords


def test_aggregate_document_keyphrases_deduplication_case_insensitive():
    """Test stem-based deduplication groups morphological variants."""
    predictions = [
        [("Machine Learning", 0.9), ("machine learning", 0.95)],
        [("MACHINE LEARNING", 0.8)],
        [("neural network", 0.7)],
    ]
    doc_ids = ["doc1", "doc1", "doc1"]
    doc_rows = [0, 0, 0]

    # Default dedup_strategy="stem" groups stem variants
    result = aggregate_document_keyphrases(predictions, doc_ids, doc_rows)

    keywords = [kw for kw, _ in result[("doc1", 0)]]
    assert len(keywords) == 2
    ml_result = next(
        (kw, score)
        for kw, score in result[("doc1", 0)]
        if kw.lower().startswith("machine")
    )
    assert ml_result[1] == 0.95


def test_aggregate_document_keyphrases_capping():
    """Test that results are capped at max_keyphrases."""
    predictions = [[(f"keyword{i}", 0.9 - i * 0.01) for i in range(15)]]
    doc_ids = ["doc1"]
    doc_rows = [0]

    result = aggregate_document_keyphrases(
        predictions, doc_ids, doc_rows, max_keyphrases=10
    )

    assert len(result[("doc1", 0)]) == 10


def test_aggregate_document_keyphrases_no_cap():
    """Test that a large max_keyphrases returns all keywords."""
    predictions = [[(f"keyword{i}", 0.9 - i * 0.01) for i in range(15)]]
    doc_ids = ["doc1"]
    doc_rows = [0]

    result = aggregate_document_keyphrases(
        predictions, doc_ids, doc_rows, max_keyphrases=100
    )

    assert len(result[("doc1", 0)]) == 15


def test_aggregate_document_keyphrases_empty_predictions():
    """Test handling of empty predictions."""
    predictions = [[], []]
    doc_ids = ["doc1", "doc1"]
    doc_rows = [0, 0]

    result = aggregate_document_keyphrases(predictions, doc_ids, doc_rows)

    assert ("doc1", 0) in result
    assert len(result[("doc1", 0)]) == 0


def test_aggregate_document_keyphrases_all_filtered():
    """Test when all keywords are filtered out due to punctuation."""
    predictions = [[("test, keyword", 0.9), ("another (one)", 0.8)]]
    doc_ids = ["doc1"]
    doc_rows = [0]

    result = aggregate_document_keyphrases(predictions, doc_ids, doc_rows)

    assert len(result[("doc1", 0)]) == 0


def test_aggregate_document_keyphrases_score_ranking():
    """Test that results are sorted by score descending."""
    predictions = [
        [("keyword1", 0.5), ("keyword2", 0.9), ("keyword3", 0.7)],
    ]
    doc_ids = ["doc1"]
    doc_rows = [0]

    result = aggregate_document_keyphrases(predictions, doc_ids, doc_rows)

    scores = [score for _, score in result[("doc1", 0)]]
    assert scores == sorted(scores, reverse=True)


def test_aggregate_document_keyphrases_no_punctuation_filter():
    """Test that filter_punctuation=False keeps all keywords."""
    predictions = [[("test, keyword", 0.9), ("another (one)", 0.8)]]
    doc_ids = ["doc1"]
    doc_rows = [0]

    result = aggregate_document_keyphrases(
        predictions, doc_ids, doc_rows, filter_punctuation=False
    )

    assert len(result[("doc1", 0)]) == 2


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


def test_extract_ranked_keyphrases_iob_single_keyword():
    """Test IOB decoding for a single multi-token keyword."""
    tokenizer = _create_mock_tokenizer()
    text = "machine learning is great"
    encoded = tokenizer(text, return_tensors="pt")
    input_ids = encoded["input_ids"]
    attention_mask = encoded["attention_mask"]

    logits = torch.zeros(input_ids.shape[0], input_ids.shape[1], 3, dtype=torch.float)
    logits[0, 1, 1] = 5.0
    logits[0, 2, 2] = 5.0

    results = extract_ranked_keyphrases(
        logits, input_ids, attention_mask, tokenizer, num_classes=3
    )

    assert len(results) == 1
    assert len(results[0]) == 1
    keyword, score = results[0][0]
    assert "machine" in keyword.lower()
    assert score > 0.5


def test_extract_ranked_keyphrases_iob_multiple_keywords():
    """Test IOB decoding for multiple separate keywords."""
    tokenizer = _create_mock_tokenizer()
    text = "machine learning and deep learning are popular"
    encoded = tokenizer(text, return_tensors="pt")
    input_ids = encoded["input_ids"]
    attention_mask = encoded["attention_mask"]

    # place logits
    logits = torch.zeros(input_ids.shape[0], input_ids.shape[1], 3, dtype=torch.float)
    logits[0, 1, 1] = 5.0  # 'machine'
    logits[0, 2, 2] = 5.0  # 'learning'
    logits[0, 4, 1] = 4.0  # 'deep'
    logits[0, 5, 2] = 4.0  # 'learning'

    results = extract_ranked_keyphrases(
        logits, input_ids, attention_mask, tokenizer, num_classes=3
    )

    assert len(results) == 1
    assert len(results[0]) == 2
    assert "machine learning" == results[0][0][0]
    assert "deep learning" == results[0][1][0]
    assert results[0][0][1] > results[0][1][1]  # check score ranking


def test_extract_ranked_keyphrases_iob_no_keywords():
    """Test IOB decoding when all tokens are Outside (O)."""
    tokenizer = _create_mock_tokenizer()
    text = "this is a simple sentence"
    encoded = tokenizer(text, return_tensors="pt")
    input_ids = encoded["input_ids"]
    attention_mask = encoded["attention_mask"]

    logits = torch.zeros(input_ids.shape[0], input_ids.shape[1], 3, dtype=torch.float)
    logits[:, :, 0] = 5.0

    results = extract_ranked_keyphrases(
        logits, input_ids, attention_mask, tokenizer, num_classes=3
    )

    assert len(results) == 1
    assert len(results[0]) == 0


def test_extract_ranked_keyphrases_iob_orphan_i_tag():
    """Test that orphan I tags (without preceding B) are ignored."""
    tokenizer = _create_mock_tokenizer()
    text = "this is great"
    encoded = tokenizer(text, return_tensors="pt")
    input_ids = encoded["input_ids"]
    attention_mask = encoded["attention_mask"]

    logits = torch.zeros(input_ids.shape[0], input_ids.shape[1], 3, dtype=torch.float)
    logits[0, 2, 2] = 5.0

    results = extract_ranked_keyphrases(
        logits, input_ids, attention_mask, tokenizer, num_classes=3
    )

    assert len(results) == 1
    assert len(results[0]) == 0


def test_extract_ranked_keyphrases_iob_single_token_keyword():
    """Test IOB decoding for a single-token keyword (B only, no I)."""
    tokenizer = _create_mock_tokenizer()
    text = "AI is great"
    encoded = tokenizer(text, return_tensors="pt")
    input_ids = encoded["input_ids"]
    attention_mask = encoded["attention_mask"]

    logits = torch.zeros(input_ids.shape[0], input_ids.shape[1], 3, dtype=torch.float)
    logits[0, 1, 1] = 5.0

    results = extract_ranked_keyphrases(
        logits, input_ids, attention_mask, tokenizer, num_classes=3
    )

    assert len(results) == 1
    assert len(results[0]) == 1
    keyword, score = results[0][0]
    assert "ai" in keyword.lower()
    assert score > 0.5


def test_extract_ranked_keyphrases_iob_adjacent_keywords():
    """Test IOB decoding for adjacent keywords (B followed by B)."""
    tokenizer = _create_mock_tokenizer()
    text = "AI machine learning"
    encoded = tokenizer(text, return_tensors="pt")
    input_ids = encoded["input_ids"]
    attention_mask = encoded["attention_mask"]

    logits = torch.zeros(input_ids.shape[0], input_ids.shape[1], 3, dtype=torch.float)
    logits[0, 1, 1] = 5.0
    logits[0, 2, 1] = 4.0
    logits[0, 3, 2] = 4.0

    results = extract_ranked_keyphrases(
        logits, input_ids, attention_mask, tokenizer, num_classes=3
    )

    assert len(results) == 1
    assert len(results[0]) == 2


def test_extract_ranked_keyphrases_auto_detect_binary():
    """Test auto-detection of binary scheme from 2D logits."""
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


def test_extract_ranked_keyphrases_auto_detect_iob():
    """Test auto-detection of IOB scheme from 3D logits."""
    tokenizer = _create_mock_tokenizer()
    text = "machine learning is great"
    encoded = tokenizer(text, return_tensors="pt")
    input_ids = encoded["input_ids"]
    attention_mask = encoded["attention_mask"]

    logits = torch.zeros(input_ids.shape[0], input_ids.shape[1], 3, dtype=torch.float)
    logits[0, 1, 1] = 5.0
    logits[0, 2, 2] = 5.0

    results = extract_ranked_keyphrases(logits, input_ids, attention_mask, tokenizer)

    assert len(results) == 1
    assert len(results[0]) == 1
    keyword, score = results[0][0]
    assert "machine" in keyword.lower()
    assert score > 0.5


# ============================================================================
# Tests for legacy-aligned evaluation logic
# ============================================================================


def test_aggregate_document_keyphrases_stem_dedup():
    """Test that stem-based deduplication groups morphological variants."""
    # "learning" and "Learning" stem to the same thing
    predictions = [[("machine learning", 0.9), ("machine Learning", 0.8)]]
    result = aggregate_document_keyphrases(
        predictions, ["doc1"], [0], dedup_strategy="stem"
    )
    assert len(result[("doc1", 0)]) == 1
    assert result[("doc1", 0)][0][0] == "machine learning"
    assert result[("doc1", 0)][0][1] == 0.9


def test_aggregate_document_keyphrases_lowercase_dedup():
    """Test that lowercase deduplication is case-insensitive."""
    predictions = [[("Machine Learning", 0.9), ("machine learning", 0.8)]]
    result = aggregate_document_keyphrases(
        predictions, ["doc1"], [0], dedup_strategy="lowercase"
    )
    assert len(result[("doc1", 0)]) == 1
    assert result[("doc1", 0)][0][0] == "Machine Learning"
    assert result[("doc1", 0)][0][1] == 0.9


def test_aggregate_document_keyphrases_repeated_words():
    """Test that keywords with repeated stemmed words are filtered out."""
    predictions = [[("machine learning machine", 0.9), ("deep learning", 0.8)]]
    result = aggregate_document_keyphrases(
        predictions, ["doc1"], [0], dedup_strategy="stem"
    )
    keywords = [kw for kw, _ in result[("doc1", 0)]]
    assert "deep learning" in keywords
    # "machine learning machine" has repeated stem "machin" so it's filtered
    assert "machine learning machine" not in keywords


def test_aggregate_document_keyphrases_length_filter():
    """Test that keywords with >= max_keyphrase_length words are filtered out."""
    predictions = [[("a b c d e", 0.9), ("a b c d", 0.8), ("a b c", 0.7)]]
    result = aggregate_document_keyphrases(
        predictions, ["doc1"], [0], max_keyphrase_length=5
    )
    keywords = [kw for kw, _ in result[("doc1", 0)]]
    # "a b c d e" has 5 words, which is >= max_keyphrase_length=5, so it's filtered
    assert "a b c d e" not in keywords
    assert "a b c d" in keywords
    assert "a b c" in keywords


def test_aggregate_document_keyphrases_dash_and_apostrophe_allowed():
    """Test that dash and apostrophe are allowed in keyphrases."""
    predictions = [[("machine-learning", 0.9), ("don't stop", 0.8)]]
    result = aggregate_document_keyphrases(predictions, ["doc1"], [0])
    keywords = [kw for kw, _ in result[("doc1", 0)]]
    assert "machine-learning" in keywords
    assert "don't stop" in keywords
