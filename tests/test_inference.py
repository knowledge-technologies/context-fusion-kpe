"""Tests for inference module functions."""

import torch
from transformers import AutoTokenizer

from keyword_pecker.data import extract_word_level_data
from keyword_pecker.models.inference import (
    aggregate_document_keyphrases,
    extract_ranked_keyphrases,
    _aggregate_tokens_to_words,
    _extract_iob_keywords_word_level,
    _score_keyphrase,
)

_TOKENIZER = AutoTokenizer.from_pretrained("sentence-transformers/all-mpnet-base-v2")


def test_extract_ranked_keyphrases_single_keyword():
    tokenizer = _TOKENIZER
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
    tokenizer = _TOKENIZER
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
    tokenizer = _TOKENIZER
    text = "this is a simple sentence"
    encoded = tokenizer(text, return_tensors="pt")
    input_ids = encoded["input_ids"]
    attention_mask = encoded["attention_mask"]

    logits = torch.zeros_like(input_ids, dtype=torch.float)

    results = extract_ranked_keyphrases(logits, input_ids, attention_mask, tokenizer)

    assert len(results) == 1
    assert len(results[0]) == 0


def test_extract_ranked_keyphrases_multiword_keyword():
    tokenizer = _TOKENIZER
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
    """Test that keyphrases with disallowed punctuation are filtered out.

    Note: apostrophe (') IS filtered per legacy TNT-KID code, even though the
    TNT-KID paper says apostrophes should be kept.
    """
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
    assert "don't" not in keywords  # apostrophe is filtered per legacy code
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
    tokenizer = _TOKENIZER
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
    tokenizer = _TOKENIZER
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
    tokenizer = _TOKENIZER
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
    tokenizer = _TOKENIZER
    text = "machine learning is great"
    encoded = tokenizer(text, return_tensors="pt")
    input_ids = encoded["input_ids"]
    attention_mask = encoded["attention_mask"]

    # <s> machine learning is great <s>
    logits = torch.zeros(input_ids.shape[0], input_ids.shape[1], 3, dtype=torch.float)
    logits[0, 1, 1] = 5.0  # B (machine)
    logits[0, 2, 2] = 5.0  # I (learning)

    results = extract_ranked_keyphrases(
        logits, input_ids, attention_mask, tokenizer, num_classes=3
    )

    assert len(results) == 1
    assert len(results[0]) == 1
    keyword, score = results[0][0]
    assert "machine learning" in keyword.lower()
    assert score > 0.5


def test_extract_ranked_keyphrases_iob_multiple_keywords():
    """Test IOB decoding for multiple separate keywords."""
    tokenizer = _TOKENIZER
    text = "machine learning and deep learning are popular"
    encoded = tokenizer(text, return_tensors="pt")
    input_ids = encoded["input_ids"]
    attention_mask = encoded["attention_mask"]

    # place logits
    logits = torch.zeros(input_ids.shape[0], input_ids.shape[1], 3, dtype=torch.float)
    logits[0, 1, 1] = 5.0  # 'machine'
    logits[0, 2, 2] = 3  # 'learning'
    logits[0, 4, 1] = 4.0  # 'deep'
    logits[0, 5, 2] = 2  # 'learning'

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
    tokenizer = _TOKENIZER
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
    tokenizer = _TOKENIZER
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
    tokenizer = _TOKENIZER
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
    tokenizer = _TOKENIZER
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

    assert len(results[0]) == 2
    assert "ai" in results[0][0]
    assert "machine learning" in results[0][1]


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


def test_aggregate_document_keyphrases_case_insensitive_stem():
    """Test that stem dedup handles full case differences like 'Value Management' vs 'value management'."""
    predictions = [[("Value Management", 0.7), ("value management", 0.9)]]
    result = aggregate_document_keyphrases(
        predictions, ["doc1"], [0], dedup_strategy="stem"
    )
    assert len(result[("doc1", 0)]) == 1
    assert result[("doc1", 0)][0][1] == 0.9


def test_aggregate_document_keyphrases_hyphen_normalization():
    """Test that hyphenated and space-separated variants are deduplicated."""
    predictions = [[("event related", 0.8), ("event-related", 0.9)]]
    result = aggregate_document_keyphrases(
        predictions, ["doc1"], [0], dedup_strategy="stem"
    )
    assert len(result[("doc1", 0)]) == 1
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


def test_aggregate_document_keyphrases_dash_allowed_apostrophe_filtered():
    """Test that dash is allowed but apostrophe is filtered (per legacy TNT-KID code).

    The paper says apostrophes should be kept, but the legacy code filters them.
    We follow the legacy code for reproducible evaluation.
    """
    predictions = [[("machine-learning", 0.9), ("don't stop", 0.8)]]
    result = aggregate_document_keyphrases(predictions, ["doc1"], [0])
    keywords = [kw for kw, _ in result[("doc1", 0)]]
    assert "machine-learning" in keywords
    assert "don't stop" not in keywords


# ============================================================================
# Tests for word-level IOB extraction
# ============================================================================


def test_iob_word_level_single_keyword():
    """Test word-level extraction with B/I/O transitions."""
    word_preds = [1, 2, 0, 0]
    word_probs = [0.9, 0.8, 0.1, 0.1]
    word_list = ["machine", "learning", "is", "great"]

    result = _extract_iob_keywords_word_level(word_preds, word_probs, word_list)

    assert len(result) == 1
    assert result[0][0] == "machine learning"
    assert result[0][1] > 0.5


def test_iob_word_level_multiple_keywords():
    """Test extraction of two separate keyphrases."""
    word_preds = [1, 0, 1, 2, 0]
    word_probs = [0.9, 0.1, 0.8, 0.7, 0.1]
    word_list = ["neural", "and", "deep", "learning", "stuff"]

    result = _extract_iob_keywords_word_level(word_preds, word_probs, word_list)

    assert len(result) == 2
    assert result[0][0] == "neural"
    assert result[1][0] == "deep learning"


def test_iob_word_level_eager_method():
    """Test that eager scoring orders by occurrence, not score."""
    word_preds = [1, 1, 1]
    word_probs = [0.7, 0.9, 0.8]
    word_list = ["neural", "networks", "machines"]

    result = _extract_iob_keywords_word_level(
        word_preds, word_probs, word_list, score_method="eager"
    )

    assert len(result) == 3
    assert result[0][0] == "neural"
    assert result[1][0] == "networks"
    assert result[2][0] == "machines"


# ============================================================================
# Tests for _aggregate_tokens_to_words
# ============================================================================


def test_aggregate_tokens_to_words_basic():
    """Test that subwords are grouped by word_id and averaged probabilities determine the label."""
    # [CLS] ma chine learn ing [SEP]
    # Word IDs: -1   0   0    1    1   -1
    # 3 classes: P(O), P(B), P(I)
    word_ids = torch.tensor([-1, 0, 0, 1, 1, -1])
    token_probs = torch.tensor(
        [
            [0.9, 0.05, 0.05],
            [0.1, 0.85, 0.05],
            [0.1, 0.1, 0.8],
            [0.05, 0.05, 0.9],
            [0.1, 0.1, 0.8],
            [0.9, 0.05, 0.05],
        ]
    )

    word_preds, word_probs = _aggregate_tokens_to_words(token_probs, word_ids)

    assert word_preds == [1, 2]
    assert len(word_probs) == 2
    # Word 0: avg P(O)=0.1, P(B)=0.475, P(I)=0.425 → pred=B(1), conf=0.475
    assert abs(word_probs[0] - 0.475) < 1e-6
    # Word 1: avg P(O)=0.075, P(B)=0.075, P(I)=0.85 → pred=I(2), conf=0.85
    assert abs(word_probs[1] - 0.85) < 1e-6


def test_aggregate_tokens_to_words_skips_special_tokens():
    """Test that word_id=-1 tokens are skipped."""
    word_ids = torch.tensor([-1, 0, -1])
    token_probs = torch.tensor(
        [
            [0.9, 0.1],
            [0.1, 0.9],
            [0.9, 0.1],
        ]
    )

    word_preds, word_probs = _aggregate_tokens_to_words(token_probs, word_ids)

    assert word_preds == [1]
    assert len(word_probs) == 1


def test_iob_subword_fallback():
    """Test backward compatibility when word_ids=None uses subword-level extraction."""
    tokenizer = AutoTokenizer.from_pretrained("sentence-transformers/all-mpnet-base-v2")
    text = "machine learning is great"
    encoded = tokenizer(text, return_tensors="pt")
    input_ids = encoded["input_ids"]
    attention_mask = encoded["attention_mask"]

    logits = torch.zeros(input_ids.shape[0], input_ids.shape[1], 3, dtype=torch.float)
    logits[0, 1, 1] = 5.0  # B-tag
    logits[0, 2, 2] = 5.0  # I-tag

    # Without word_ids → subword fallback
    results = extract_ranked_keyphrases(
        logits, input_ids, attention_mask, tokenizer, num_classes=3
    )

    assert len(results) == 1
    assert len(results[0]) == 1
    keyword, score = results[0][0]
    assert "machine" in keyword.lower()
    assert score > 0.5


def test_extract_ranked_keyphrases_with_word_ids():
    """Test that word_ids/word_lists are passed through to word-level extraction."""
    text = "machine learning is great"
    encoded = _TOKENIZER(text, return_tensors="pt", return_offsets_mapping=True)
    input_ids = encoded["input_ids"]
    attention_mask = encoded["attention_mask"]

    # Build word_ids and word_list using the shared helper
    word_ids_batch, word_lists = extract_word_level_data(encoded, [text])
    word_ids = torch.tensor(word_ids_batch)

    logits = torch.zeros(input_ids.shape[0], input_ids.shape[1], 3, dtype=torch.float)
    # Set B-tag on first subword of "machine", I-tag on first subword of "learning"
    logits[0, 1, 1] = 5.0
    logits[0, 2, 2] = 5.0

    results = extract_ranked_keyphrases(
        logits,
        input_ids,
        attention_mask,
        _TOKENIZER,
        num_classes=3,
        word_ids=word_ids,
        word_lists=word_lists,
    )

    assert len(results) == 1
    assert len(results[0]) == 1
    keyword, score = results[0][0]
    assert keyword == "machine learning"
    assert score > 0.5


def test_extract_ranked_keyphrases_with_hyphenated_keyword():
    """Test word-level extraction with hyphenated keyword like 'state-of-the-art'.

    The tokenizer splits hyphens into separate word_ids, so the word_list
    contains ['state', '-', 'of', '-', 'the', '-', 'art', 'methods'].
    When all are tagged as part of a keyphrase, the result is space-joined.
    """
    text = "state-of-the-art methods are popular"
    encoded = _TOKENIZER(text, return_tensors="pt", return_offsets_mapping=True)
    input_ids = encoded["input_ids"]
    attention_mask = encoded["attention_mask"]

    word_ids_batch, word_lists = extract_word_level_data(encoded, [text])
    word_ids = torch.tensor(word_ids_batch)

    # Set B-tag on first token (word 0, "state"), I-tags on all tokens
    # through "methods" (word 7) to form a single keyphrase
    logits = torch.zeros(input_ids.shape[0], input_ids.shape[1], 3, dtype=torch.float)
    wid_list = encoded.word_ids(batch_index=0)
    seq_len = attention_mask[0].sum().item()
    for tok_idx in range(1, seq_len - 1):  # skip [CLS] and [SEP]
        wid = wid_list[tok_idx]
        if wid == 0:
            logits[0, tok_idx, 1] = 5.0  # B-tag on first word
        elif wid is not None and wid <= 7:
            logits[0, tok_idx, 2] = 5.0  # I-tag on remaining words

    results = extract_ranked_keyphrases(
        logits,
        input_ids,
        attention_mask,
        _TOKENIZER,
        num_classes=3,
        word_ids=word_ids,
        word_lists=word_lists,
    )

    assert len(results) == 1
    assert len(results[0]) == 1
    keyword, score = results[0][0]
    # word_list contains hyphens as separate entries, so result is space-joined
    assert keyword == "state - of - the - art methods"
    assert score > 0.5


# ============================================================================
# Tests for _score_keyphrase methods
# ============================================================================


def test_score_keyphrase_average():
    """Test average scoring method."""
    result = _score_keyphrase([0.9, 0.7, 0.8], method="average")
    assert abs(result - 0.8) < 1e-6


def test_score_keyphrase_max():
    """Test max scoring method."""
    result = _score_keyphrase([0.9, 0.7, 0.8], method="max")
    assert abs(result - 0.9) < 1e-6


def test_score_keyphrase_min():
    """Test min scoring method."""
    result = _score_keyphrase([0.9, 0.7, 0.8], method="min")
    assert abs(result - 0.7) < 1e-6


def test_score_keyphrase_geometric():
    """Test geometric mean scoring method."""
    result = _score_keyphrase([0.9, 0.7, 0.8], method="geometric")
    expected = (0.9 * 0.7 * 0.8) ** (1 / 3)
    assert abs(result - expected) < 1e-6


def test_score_keyphrase_geometric_penalizes_low():
    """Test that geometric mean penalizes low-confidence tokens more than average."""
    probs = [0.95, 0.95, 0.1]
    avg = _score_keyphrase(probs, method="average")
    geo = _score_keyphrase(probs, method="geometric")
    assert geo < avg


def test_score_keyphrase_default_is_average():
    """Test that default method is average."""
    result = _score_keyphrase([0.6, 0.8])
    expected = _score_keyphrase([0.6, 0.8], method="average")
    assert abs(result - expected) < 1e-6
