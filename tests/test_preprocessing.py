import numpy as np
import regex
import pytest
import torch
from unittest.mock import MagicMock
from transformers import AutoTokenizer

from keyword_pecker.constants import IGNORE_IDX
from keyword_pecker.data import (
    build_keyword_regex,
    create_keyword_mask,
    pretokenize_sentence,
    process_document,
    resolve_keyword_spans_stem,
    word_ranges_to_token_ranges,
)
from nltk.stem.porter import PorterStemmer


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


ABSTRACT = (
    "In this paper, we present a new method of estimating the novelty of rules "
    + "discovered by data-mining methods using WordNet, a lexical knowledge-base of English words. "
    + "We assess the novelty of a rule by the average semantic distance in a knowledge hierarchy between the words in the antecedent and the consequent of the rule "
    + "- the more the average distance, more is the novelty of the rule."
)
KEYWORDS = [
    "interesting rules",
    "novelty",
    "semantic distance",
    "knowledge hierarchy",
    "wordnet",
]
TITLE = "evaluating the novelty of text-mined rules using lexical knowledge"


def test_build_keyword_regex():

    TEST_KEYWORDS = ["language models", "machine learning"]

    pattern = build_keyword_regex(keywords=TEST_KEYWORDS, stemmer=PorterStemmer())

    assert isinstance(pattern, regex.Pattern)

    text = "language models and machine learning are popular"
    matches = [m.group(0) for m in pattern.finditer(text)]
    assert "language models" in matches
    assert "machine learning" in matches

    text2 = "language modelsing should not match"
    matches2 = [m.group(0) for m in pattern.finditer(text2)]
    assert "language models" not in matches2


def test_keyword_regex_word_boundaries():
    """Test that regex matches only complete words, not substrings.

    This test reproduces a bug where keywords like "ai" incorrectly match
    words containing "ai" as a substring (e.g., "aims", "cocktails", "explained").

    Gold keywords: ai, aibo, ibm, london, robots, teamlab
    False positives to avoid: aims, cocktails, explained
    """
    abstract = (
        "An ambitious London exhibition aims to prove that AI can help humans. "
        "teamLab created an immersive space. There are robots from Sony's Aibo "
        "to a mechanical arm that serves cocktails. IBM's research explores "
        "AI ethics. The exhibition explained the limits of AI."
    )

    keywords = ["ai", "aibo", "ibm", "london", "robots", "teamlab"]

    pattern = build_keyword_regex(keywords, PorterStemmer())

    matches = [m.group(0) for m in pattern.finditer(abstract.lower())]

    # Should match actual keywords
    assert "ai" in matches
    assert "robots" in matches
    assert "ibm" in matches
    assert "london" in matches
    assert "teamlab" in matches
    assert "aibo" in matches

    # Should NOT match false positives (words containing keyword stems)
    assert "aims" not in matches
    assert "cocktails" not in matches
    assert "explained" not in matches


def test_create_keyword_mask():

    SENT_LEN = 10
    KEYWORD_RANGES = [(2, 5), (6, 8)]
    word_ids = [-1] + list(range(SENT_LEN - 2)) + [-1]

    kw_mask = create_keyword_mask(
        keyword_token_ranges=KEYWORD_RANGES,
        sent_tok_len=SENT_LEN,
        word_ids=word_ids,
        scheme="binary",
    )

    assert len(kw_mask) == SENT_LEN
    assert kw_mask[0].item() == IGNORE_IDX
    assert kw_mask[-1].item() == IGNORE_IDX
    assert sum(1 for v in kw_mask if v == 1) == 5

    kw_mask = create_keyword_mask(
        keyword_token_ranges=KEYWORD_RANGES,
        sent_tok_len=SENT_LEN,
        word_ids=word_ids,
        scheme="IOB",
    )

    assert len(kw_mask) == SENT_LEN
    assert kw_mask[0].item() == IGNORE_IDX
    assert kw_mask[-1].item() == IGNORE_IDX
    assert sum(1 for v in kw_mask if v in (1, 2)) == 5


def test_create_keyword_mask_iob_detailed():
    """Test IOB labeling scheme with detailed assertions."""

    SENT_LEN = 10
    word_ids = [-1] + list(range(SENT_LEN - 2)) + [-1]

    kw_mask_single = create_keyword_mask(
        keyword_token_ranges=[(2, 3)],
        sent_tok_len=SENT_LEN,
        word_ids=word_ids,
        scheme="IOB",
    )
    assert kw_mask_single[2].item() == 1
    assert sum(1 for v in kw_mask_single if v in (1, 2)) == 1

    kw_mask_multi = create_keyword_mask(
        keyword_token_ranges=[(2, 5)],
        sent_tok_len=SENT_LEN,
        word_ids=word_ids,
        scheme="IOB",
    )
    assert kw_mask_multi[2].item() == 1
    assert kw_mask_multi[3].item() == 2
    assert kw_mask_multi[4].item() == 2
    assert sum(1 for v in kw_mask_multi if v in (1, 2)) == 3

    kw_mask_adjacent = create_keyword_mask(
        keyword_token_ranges=[(2, 3), (3, 5)],
        sent_tok_len=SENT_LEN,
        word_ids=word_ids,
        scheme="IOB",
    )
    assert kw_mask_adjacent[2].item() == 1
    assert kw_mask_adjacent[3].item() == 1
    assert kw_mask_adjacent[4].item() == 2
    assert sum(1 for v in kw_mask_adjacent if v in (1, 2)) == 3

    kw_mask_none = create_keyword_mask(
        keyword_token_ranges=[],
        sent_tok_len=SENT_LEN,
        word_ids=word_ids,
        scheme="IOB",
    )
    assert all(v == 0 or v == IGNORE_IDX for v in kw_mask_none)


# ---------------------------------------------------------------------------
# resolve_keyword_spans_stem tests
# ---------------------------------------------------------------------------


def test_resolve_keyword_spans_stem_exact_match():
    """Keyword present verbatim in sentence is matched."""
    sentence = "novelty and semantic distance are key concepts"
    keywords = ["novelty", "semantic distance"]
    stemmer = PorterStemmer()
    words = pretokenize_sentence(sentence)

    kwds, word_ranges = resolve_keyword_spans_stem(words, keywords, stemmer)

    assert "novelty" in kwds
    assert "semantic distance" in kwds
    assert len(word_ranges) == 2


def test_resolve_keyword_spans_stem_inflected_forms():
    """Inflected forms of keywords (e.g. plurals, past tense) are matched."""
    stemmer = PorterStemmer()

    sentence = "the program aided researchers"
    keywords = ["aid"]
    words = pretokenize_sentence(sentence)
    kwds, word_ranges = resolve_keyword_spans_stem(words, keywords, stemmer)
    assert len(kwds) == 1, f"Expected 'aided' to match keyword 'aid', got: {kwds}"
    assert word_ranges[0][1] > word_ranges[0][0]

    sentence2 = "machines learning techniques are powerful"
    keywords2 = ["machine learning"]
    words2 = pretokenize_sentence(sentence2)
    kwds2, word_ranges2 = resolve_keyword_spans_stem(words2, keywords2, stemmer)
    assert len(kwds2) == 1, (
        f"Expected 'machines learning' to match 'machine learning', got: {kwds2}"
    )


def test_resolve_keyword_spans_stem_no_false_positive_ai_aims():
    """Keyword 'ai' must NOT match 'aims'."""
    stemmer = PorterStemmer()
    sentence = "an ambitious exhibition aims to prove that ai can help"
    keywords_ai_only = ["ai"]
    words = pretokenize_sentence(sentence)
    kwds, word_ranges = resolve_keyword_spans_stem(words, keywords_ai_only, stemmer)

    assert not any("aims" in s for s in kwds), (
        f"'aims' should not be matched by keyword 'ai', but got: {kwds}"
    )
    assert any(s.strip() == "ai" for s in kwds), (
        f"'ai' should be matched but got: {kwds}"
    )


def test_resolve_keyword_spans_stem_no_false_positive_substring():
    """Keyword matching must not match inside a longer word."""
    stemmer = PorterStemmer()
    sentence = "a mechanical arm serves cocktails"
    keywords = ["ai"]
    words = pretokenize_sentence(sentence)
    kwds, _ = resolve_keyword_spans_stem(words, keywords, stemmer)
    assert kwds == [], f"Expected no matches, got: {kwds}"


def test_resolve_keyword_spans_stem_word_ranges_correct():
    """Word index ranges returned must cover exactly the right words.

    Sentence: "the machine learning model"
    Words (0-indexed):
      0: the
      1: machine
      2: learning
      3: model
    Keyword "machine learning" should map to word range (1, 3) [start incl, end excl].
    """
    stemmer = PorterStemmer()
    sentence = "the machine learning model"
    keywords = ["machine learning"]
    words = pretokenize_sentence(sentence)
    kwds, word_ranges = resolve_keyword_spans_stem(words, keywords, stemmer)

    assert len(word_ranges) == 1
    start_word, end_word = word_ranges[0]
    assert start_word == 1, f"Expected start_word=1, got {start_word}"
    assert end_word == 3, f"Expected end_word=3, got {end_word}"


def test_resolve_keyword_spans_stem_overlapping_skipped():
    """When two keywords overlap, the longer one wins and the shorter is skipped."""
    stemmer = PorterStemmer()
    sentence = "we measure semantic distance between concepts"
    keywords = ["semantic", "semantic distance"]
    words = pretokenize_sentence(sentence)
    kwds, word_ranges = resolve_keyword_spans_stem(words, keywords, stemmer)

    assert len(kwds) == 1
    assert kwds[0] == "semantic distance"


def test_resolve_keyword_spans_stem_multiword_inflected():
    """Multi-word keyword with inflected words matches in sentence."""
    stemmer = PorterStemmer()
    assert stemmer.stem("hierarchies") == stemmer.stem("hierarchy")

    sentence = "we study knowledge hierarchies in ontologies"
    keywords = ["knowledge hierarchy"]
    words = pretokenize_sentence(sentence)
    kwds, word_ranges = resolve_keyword_spans_stem(words, keywords, stemmer)

    assert len(kwds) == 1, (
        f"Expected 'knowledge hierarchies' to match keyword 'knowledge hierarchy', got: {kwds}"
    )


# ---------------------------------------------------------------------------
# Subtoken-aware IOB labeling tests
# ---------------------------------------------------------------------------


def test_iob_labels_with_subtoken_split_on_non_initial_keyword_word():
    """IOB labels must use IGNORE_IDX for continuation subtokens.

    Example: keyword "neural networks" matches "neural, networks" where
    "networks" is tokenized as "net", "##works" (2 subtokens).

    Expected: [-100, B, I, -100, O, O, -100]
    """
    stemmer = PorterStemmer()

    sentence = "neural networks are powerful"
    keywords = ["neural networks"]
    words = pretokenize_sentence(sentence)

    kwds, word_ranges = resolve_keyword_spans_stem(words, keywords, stemmer)
    assert len(kwds) == 1, f"Expected 1 match, got: {kwds}"

    # Simulate subtoken word_ids: BOS(-1), neural(0), net(1), ##works(1), are(2), powerful(3), EOS(-1)
    word_ids = [-1, 0, 1, 1, 2, 3, -1]
    tok_ranges = word_ranges_to_token_ranges(word_ranges, word_ids)

    assert len(tok_ranges) == 1
    start_tok, end_tok = tok_ranges[0]
    assert start_tok == 1, f"Expected start_tok=1, got {start_tok}"
    assert end_tok == 4, f"Expected end_tok=4, got {end_tok}"

    sent_tok_len = len(word_ids)
    iob_mask = create_keyword_mask(
        tok_ranges, sent_tok_len, word_ids=word_ids, scheme="IOB"
    )

    assert iob_mask[0].item() == IGNORE_IDX
    assert iob_mask[1].item() == 1
    assert iob_mask[2].item() == 2
    assert iob_mask[3].item() == IGNORE_IDX
    assert iob_mask[4].item() == 0
    assert iob_mask[5].item() == 0
    assert iob_mask[6].item() == IGNORE_IDX


def test_iob_labels_with_subtoken_split_on_initial_keyword_word():
    """IOB labels must use IGNORE_IDX for continuation subtokens on initial word.

    Example: keyword "bioinformatics research" where "bioinformatics" is
    tokenized as "bio", "##in", "##for", "##matics" (4 subtokens).

    Expected: [-100, B, -100, -100, -100, I, O, -100]
    """
    stemmer = PorterStemmer()

    sentence = "bioinformatics research advances"
    keywords = ["bioinformatics research"]
    words = pretokenize_sentence(sentence)

    kwds, word_ranges = resolve_keyword_spans_stem(words, keywords, stemmer)
    assert len(kwds) == 1, f"Expected 1 match, got: {kwds}"

    word_ids = [-1, 0, 0, 0, 0, 1, 2, -1]
    tok_ranges = word_ranges_to_token_ranges(word_ranges, word_ids)

    assert len(tok_ranges) == 1
    start_tok, end_tok = tok_ranges[0]
    assert start_tok == 1, f"Expected start_tok=1, got {start_tok}"
    assert end_tok == 6, f"Expected end_tok=6, got {end_tok}"

    sent_tok_len = len(word_ids)
    iob_mask = create_keyword_mask(
        tok_ranges, sent_tok_len, word_ids=word_ids, scheme="IOB"
    )

    assert iob_mask[0].item() == IGNORE_IDX
    assert iob_mask[1].item() == 1
    assert iob_mask[2].item() == IGNORE_IDX
    assert iob_mask[3].item() == IGNORE_IDX
    assert iob_mask[4].item() == IGNORE_IDX
    assert iob_mask[5].item() == 2
    assert iob_mask[6].item() == 0
    assert iob_mask[7].item() == IGNORE_IDX


# ---------------------------------------------------------------------------
# Keyword matching with NLTK pre-tokenization (replaces DeBERTA sentinel tests)
# ---------------------------------------------------------------------------


def test_keyword_matching_with_nltk_pretokenization():
    """resolve_keyword_spans_stem must work correctly with NLTK pre-tokenized words.

    With is_split_into_words=True, word_ids() is reliable regardless of tokenizer.
    This test verifies keyword matching works on word lists.
    """
    stemmer = PorterStemmer()
    sentence = "the machine learning model"
    keywords = ["machine learning"]
    words = pretokenize_sentence(sentence)

    kwds, word_ranges = resolve_keyword_spans_stem(words, keywords, stemmer)

    assert len(kwds) == 1
    assert kwds[0] == "machine learning"
    assert word_ranges[0] == (1, 3)


def test_keyword_at_end_of_sentence_not_truncated():
    """Keywords at the end of a sentence must not be incorrectly skipped."""
    stemmer = PorterStemmer()
    sentence = "we study machine learning and deep learning"
    keywords = ["machine learning", "deep learning"]
    words = pretokenize_sentence(sentence)

    kwds, word_ranges = resolve_keyword_spans_stem(words, keywords, stemmer)

    assert len(kwds) == 2, f"Expected 2 keywords matched, got {len(kwds)}: {kwds}"
    assert "machine learning" in kwds
    assert "deep learning" in kwds


def test_keyword_at_start_of_sentence():
    """Keywords starting at the first word must be matched correctly."""
    stemmer = PorterStemmer()
    sentence = "machine learning is powerful"
    keywords = ["machine learning"]
    words = pretokenize_sentence(sentence)

    kwds, word_ranges = resolve_keyword_spans_stem(words, keywords, stemmer)

    assert len(kwds) == 1
    assert kwds[0] == "machine learning"
    assert word_ranges[0] == (0, 2)


def test_keyword_mask_alignment_with_simulated_subtokens():
    """create_keyword_mask must produce correct labels with simulated subtoken word_ids.

    Sentence: "a feedback vertex set of 2-degenerate graphs"
    Keywords: "feedback vertex set", "2 degenerate graphs" (hyphens normalized)

    NLTK pre-tokenizes to: ['a', 'feedback', 'vertex', 'set', 'of', '2', 'degenerate', 'graphs']
    HF tokenizer may split "degenerate" into subtokens.

    Expected IOB labels (with "degenerate" split into "de", "generate"):
      a          → 0 (Outside)
      feedback   → 1 (Beginning)
      vertex     → 2 (Inside)
      set        → 2 (Inside)
      of         → 0 (Outside)
      2          → 1 (Beginning)
      de         → 2 (Inside, first subtoken of word 6)
      generate   → -100 (continuation subtoken of word 6)
      graphs     → 2 (Inside)
    """
    stemmer = PorterStemmer()
    sentence = "a feedback vertex set of 2-degenerate graphs"
    keywords = ["feedback vertex set", "2 degenerate graphs"]
    words = pretokenize_sentence(sentence)

    kwds, word_ranges = resolve_keyword_spans_stem(words, keywords, stemmer)
    assert len(kwds) == 2, f"Expected 2 keywords, got {len(kwds)}: {kwds}"

    # word_ids: each NLTK word gets an index, "degenerate" split into 2 subtokens
    # words: a(0), feedback(1), vertex(2), set(3), of(4), 2(5), degenerate(6→6), graphs(7)
    word_ids = [0, 1, 2, 3, 4, 5, 6, 6, 7]
    tok_ranges = word_ranges_to_token_ranges(word_ranges, word_ids)

    sent_tok_len = len(word_ids)
    iob_mask = create_keyword_mask(
        tok_ranges, sent_tok_len, word_ids=word_ids, scheme="IOB"
    )

    assert iob_mask[0].item() == 0
    assert iob_mask[1].item() == 1
    assert iob_mask[2].item() == 2
    assert iob_mask[3].item() == 2
    assert iob_mask[4].item() == 0
    assert iob_mask[5].item() == 1
    assert iob_mask[6].item() == 2
    assert iob_mask[7].item() == IGNORE_IDX
    assert iob_mask[8].item() == 2


def test_iob_subtoken_continuation_non_initial_word():
    """IOB labels must use IGNORE_IDX for continuation subtokens (non-initial word)."""
    stemmer = PorterStemmer()
    sentence = "neural networks are powerful"
    keywords = ["neural networks"]
    words = pretokenize_sentence(sentence)

    kwds, word_ranges = resolve_keyword_spans_stem(words, keywords, stemmer)
    assert len(kwds) == 1

    word_ids = [0, 1, 1, 2, 3]
    tok_ranges = word_ranges_to_token_ranges(word_ranges, word_ids)

    sent_tok_len = len(word_ids)
    iob_mask = create_keyword_mask(
        tok_ranges, sent_tok_len, word_ids=word_ids, scheme="IOB"
    )

    assert iob_mask[0].item() == 1
    assert iob_mask[1].item() == 2
    assert iob_mask[2].item() == IGNORE_IDX
    assert iob_mask[3].item() == 0
    assert iob_mask[4].item() == 0


def test_iob_subtoken_continuation_initial_word():
    """IOB labels must use IGNORE_IDX for continuation subtokens on initial word."""
    stemmer = PorterStemmer()
    sentence = "bioinformatics research advances"
    keywords = ["bioinformatics research"]
    words = pretokenize_sentence(sentence)

    kwds, word_ranges = resolve_keyword_spans_stem(words, keywords, stemmer)
    assert len(kwds) == 1

    word_ids = [0, 0, 0, 0, 1, 2]
    tok_ranges = word_ranges_to_token_ranges(word_ranges, word_ids)

    sent_tok_len = len(word_ids)
    iob_mask = create_keyword_mask(
        tok_ranges, sent_tok_len, word_ids=word_ids, scheme="IOB"
    )

    assert iob_mask[0].item() == 1
    assert iob_mask[1].item() == IGNORE_IDX
    assert iob_mask[2].item() == IGNORE_IDX
    assert iob_mask[3].item() == IGNORE_IDX
    assert iob_mask[4].item() == 2
    assert iob_mask[5].item() == 0


# ---------------------------------------------------------------------------
# process_document integration test: DeBERTa vs ModernBERT parity
# ---------------------------------------------------------------------------


def _make_mock_batch(offsets, word_ids, n_tokens):
    """Build a mocked BatchEncoding return value for process_document.

    Args:
        offsets: List of [char_start, char_end] pairs for each token.
        word_ids: List of word indices (integers, no None values).
        n_tokens: Number of tokens.
    """
    offset_tensor = torch.tensor([offsets])
    offset_tensor.numpy = MagicMock(return_value=np.array(offsets, dtype=np.int64))

    batch = MagicMock()
    batch.input_ids = torch.tensor([[100] * n_tokens])
    batch.attention_mask = torch.tensor([[1] * n_tokens])
    batch.__getitem__ = MagicMock(
        side_effect=lambda key: offset_tensor if key == "offset_mapping" else None
    )
    batch.pop = MagicMock(
        side_effect=lambda key: offset_tensor if key == "offset_mapping" else None
    )
    batch.word_ids = MagicMock(return_value=word_ids)
    return batch


def test_process_document_deberta_modernbert_parity():
    """process_document must produce identical results for DeBERTa and ModernBERT tokenizers.

    DeBERTa returns offset_mapping without (0,0) sentinels and word_ids without None.
    ModernBERT includes sentinels. process_document must normalize DeBERTa output
    so the downstream resolution functions work correctly.

    Regression test for: plans/deberta_tokenizer_postmortem.md
    """
    stemmer = PorterStemmer()

    sentence = "a feedback vertex set of 2-degenerate graphs"
    sentences = [sentence]
    keywords = ["feedback vertex set", "2-degenerate graphs"]

    # Build DeBERTa-style batch (no sentinels)
    # Tokens: a(0,1) feedback(2,10) vertex(11,17) set(18,21) of(22,24) 2(25,26) -(26,27) de(27,29) generate(29,37) graphs(38,44)
    deberta_offsets = [
        [0, 1],  # a
        [2, 10],  # feedback
        [11, 17],  # vertex
        [18, 21],  # set
        [22, 24],  # of
        [25, 26],  # 2
        [26, 27],  # -
        [27, 29],  # de
        [29, 37],  # generate
        [38, 44],  # graphs
    ]
    deberta_word_ids = [0, 1, 2, 3, 4, 5, 5, 5, 5, 6]
    deberta_n = len(deberta_offsets)

    # Build ModernBERT-style batch (with [CLS]/[SEP] sentinels)
    modernbert_offsets = [[0, 0]] + deberta_offsets + [[0, 0]]
    modernbert_word_ids = [None] + deberta_word_ids + [None]
    modernbert_n = len(modernbert_offsets)

    deberta_tokenizer = MagicMock(
        return_value=_make_mock_batch(deberta_offsets, deberta_word_ids, deberta_n)
    )
    modernbert_tokenizer = MagicMock(
        return_value=_make_mock_batch(
            modernbert_offsets, modernbert_word_ids, modernbert_n
        )
    )

    deberta_result = process_document(
        doc_id="test",
        doc_row=0,
        sentences=sentences,
        keywords=keywords,
        tokenizer=deberta_tokenizer,
        stemmer=stemmer,
        scheme="IOB",
        match_mode="stem_equality",
    )

    modernbert_result = process_document(
        doc_id="test",
        doc_row=0,
        sentences=sentences,
        keywords=keywords,
        tokenizer=modernbert_tokenizer,
        stemmer=stemmer,
        scheme="IOB",
        match_mode="stem_equality",
    )

    # Both should match the same keywords
    assert (
        deberta_result[0]["sent_keywords"] == modernbert_result[0]["sent_keywords"]
    ), (
        f"Keyword mismatch: DeBERTa={deberta_result[0]['sent_keywords']}, "
        f"ModernBERT={modernbert_result[0]['sent_keywords']}"
    )

    # Both should produce identical keyword masks for real tokens.
    # ModernBERT mask includes sentinel entries (-100) at start/end;
    # DeBERTa mask does not. Compare DeBERTa mask with ModernBERT[1:-1].
    deberta_mask = deberta_result[0]["keyword_mask"]
    modernbert_mask = modernbert_result[0]["keyword_mask"]
    assert deberta_mask == modernbert_mask[1:-1], (
        f"Mask mismatch: DeBERTa={deberta_mask}, "
        f"ModernBERT (stripped)={modernbert_mask[1:-1]}"
    )

    # Verify the expected keywords are present (hyphens normalized to spaces)
    assert "feedback vertex set" in deberta_result[0]["sent_keywords"]
    assert "2 degenerate graphs" in deberta_result[0]["sent_keywords"]


# ---------------------------------------------------------------------------
# Regression tests: NLTK pre-tokenization for reliable word_ids
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def deberta_tokenizer():
    return AutoTokenizer.from_pretrained("microsoft/deberta-v3-base")


@pytest.fixture(scope="module")
def modernbert_tokenizer():
    return AutoTokenizer.from_pretrained("answerdotai/ModernBERT-base")


def test_hyphenated_keyphrases(deberta_tokenizer, modernbert_tokenizer):
    """Hyphenated keyphrases must produce correct IOB labels with both tokenizers.

    Regression test for: ModernBERT assigns different word_ids to hyphen vs
    word parts, causing the hyphen to be labeled as "Inside" instead of ignored.

    Sentence: "This work presents a selective finite element refinement strategy based on the h-refinement type"
    Keyword: "h-refinement type" (normalized to "h refinement type")

    Expected: "h" → B, "refinement" → I, "type" → I
    All continuation subtokens → -100
    """
    stemmer = PorterStemmer()
    sentence = "This work presents a selective finite element refinement strategy based on the h-refinement type"
    keywords = ["h-refinement type"]

    for tokenizer, name in [
        (deberta_tokenizer, "DeBERTa"),
        (modernbert_tokenizer, "ModernBERT"),
    ]:
        result = process_document(
            doc_id="test",
            doc_row=0,
            sentences=[sentence],
            keywords=keywords,
            tokenizer=tokenizer,
            stemmer=stemmer,
            scheme="IOB",
            match_mode="stem_equality",
        )

        assert len(result) == 1, f"{name}: Expected 1 sentence, got {len(result)}"
        assert "h refinement type" in result[0]["sent_keywords"], (
            f"{name}: Keyword 'h refinement type' not matched, got {result[0]['sent_keywords']}"
        )

        mask = result[0]["keyword_mask"]
        word_ids = result[0]["word_ids"]

        # Count B/I labels — should be exactly 3 (one per word in the keyphrase)
        b_count = sum(1 for v in mask if v == 1)
        i_count = sum(1 for v in mask if v == 2)
        assert b_count == 1, f"{name}: Expected 1 B label, got {b_count}"
        assert i_count == 2, f"{name}: Expected 2 I labels, got {i_count}"

        # No subtoken should be labeled I or B if it's a continuation subtoken
        # (continuation subtokens must be -100)
        for i, (wid, label) in enumerate(zip(word_ids, mask)):
            if i > 0 and wid == word_ids[i - 1] and wid != -1:
                assert label == IGNORE_IDX, (
                    f"{name}: Continuation subtoken at index {i} (word_id={wid}) "
                    f"has label {label}, expected {IGNORE_IDX}"
                )


def test_acronym_in_parentheses(deberta_tokenizer, modernbert_tokenizer):
    """Acronyms in parentheses must produce correct IOB labels with both tokenizers.

    Regression test for: DeBERTa assigns same word_id to parentheses and acronym,
    causing acronym subtokens to be masked as -100 (continuation of '(').

    Sentence: "Autoimmune polyendocrinopathy candidiasis ectodermal dystrophy (APECED) is a monogenic autosomal recessive disease caused by mutations in the autoimmune regulator (AIRE) gene"
    Keywords: "APECED", "AIRE"

    Expected: "APECED" → B, "AIRE" → B
    All continuation subtokens → -100
    """
    stemmer = PorterStemmer()
    sentence = (
        "Autoimmune polyendocrinopathy candidiasis ectodermal dystrophy (APECED) "
        "is a monogenic autosomal recessive disease caused by mutations in the "
        "autoimmune regulator (AIRE) gene"
    )
    keywords = ["APECED", "AIRE"]

    for tokenizer, name in [
        (deberta_tokenizer, "DeBERTa"),
        (modernbert_tokenizer, "ModernBERT"),
    ]:
        result = process_document(
            doc_id="test",
            doc_row=0,
            sentences=[sentence],
            keywords=keywords,
            tokenizer=tokenizer,
            stemmer=stemmer,
            scheme="IOB",
            match_mode="stem_equality",
        )

        assert len(result) == 1, f"{name}: Expected 1 sentence, got {len(result)}"
        assert "apeced" in result[0]["sent_keywords"], (
            f"{name}: Keyword 'apeced' not matched, got {result[0]['sent_keywords']}"
        )
        assert "aire" in result[0]["sent_keywords"], (
            f"{name}: Keyword 'aire' not matched, got {result[0]['sent_keywords']}"
        )

        mask = result[0]["keyword_mask"]

        # Each single-word keyword should produce exactly 1 B label
        b_count = sum(1 for v in mask if v == 1)
        assert b_count == 2, (
            f"{name}: Expected 2 B labels (one per acronym), got {b_count}"
        )

        # No I labels expected (single-word keywords)
        i_count = sum(1 for v in mask if v == 2)
        assert i_count == 0, f"{name}: Expected 0 I labels, got {i_count}"


def test_deberta_modernbert_parity_with_nltk_pretokenization(
    deberta_tokenizer, modernbert_tokenizer
):
    """DeBERTa and ModernBERT must produce identical IOB labels after NLTK pre-tokenization.

    This is the core parity test: both tokenizers should produce identical
    keyword masks when using NLTK pre-tokenization, because word_ids are
    now reliable and independent of tokenizer subword splitting behavior.
    """
    stemmer = PorterStemmer()
    sentence = "a feedback vertex set of 2-degenerate graphs"
    keywords = ["feedback vertex set", "2-degenerate graphs"]

    deberta_result = process_document(
        doc_id="test",
        doc_row=0,
        sentences=[sentence],
        keywords=keywords,
        tokenizer=deberta_tokenizer,
        stemmer=stemmer,
        scheme="IOB",
        match_mode="stem_equality",
    )

    modernbert_result = process_document(
        doc_id="test",
        doc_row=0,
        sentences=[sentence],
        keywords=keywords,
        tokenizer=modernbert_tokenizer,
        stemmer=stemmer,
        scheme="IOB",
        match_mode="stem_equality",
    )

    # Both should match the same keywords
    assert (
        deberta_result[0]["sent_keywords"] == modernbert_result[0]["sent_keywords"]
    ), (
        f"Keyword mismatch: DeBERTa={deberta_result[0]['sent_keywords']}, "
        f"ModernBERT={modernbert_result[0]['sent_keywords']}"
    )

    # Both should produce identical keyword masks (excluding ModernBERT sentinels)
    deberta_mask = deberta_result[0]["keyword_mask"]
    modernbert_mask = modernbert_result[0]["keyword_mask"]
    assert deberta_mask == modernbert_mask[1:-1], (
        f"Mask mismatch: DeBERTa={deberta_mask}, "
        f"ModernBERT (stripped)={modernbert_mask[1:-1]}"
    )
