import numpy as np
import regex

from keyword_pecker.data import (
    build_keyword_regex,
    create_keyword_mask,
    resolve_keyword_spans_stem,
)
from nltk.stem.porter import PorterStemmer


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_offsets(sentence: str) -> tuple[np.ndarray, int]:
    """Build a fake tokenizer offset array from a sentence.

    Treats each ``\\b\\w+\\b`` word as a single token, surrounded by BOS/EOS
    sentinel entries (both [0, 0]), matching the shape that HuggingFace
    tokenizers return.

    Returns:
        offsets: np.ndarray of shape [num_tokens + 2, 2] (includes BOS/EOS).
        num_total_tokens: len(offsets), i.e. the value passed to
            resolve_keyword_spans_stem as num_total_tokens.
    """
    word_matches = list(regex.finditer(r"\b\w+\b", sentence))
    # BOS sentinel
    spans = [[0, 0]]
    for m in word_matches:
        spans.append([m.start(), m.end()])
    # EOS sentinel
    spans.append([0, 0])
    offsets = np.array(spans, dtype=np.int64)
    return offsets, len(offsets)


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
    import regex

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

    kw_mask = create_keyword_mask(
        keyword_token_ranges=KEYWORD_RANGES,
        sent_tok_len=SENT_LEN,
        scheme="binary",
    )

    assert len(kw_mask) == SENT_LEN
    assert sum(kw_mask) == 5

    kw_mask = create_keyword_mask(
        keyword_token_ranges=KEYWORD_RANGES,
        sent_tok_len=SENT_LEN,
        scheme="IOB",
    )

    assert len(kw_mask) == SENT_LEN
    assert sum(kw_mask) == 8


def test_create_keyword_mask_iob_detailed():
    """Test IOB labeling scheme with detailed assertions."""

    SENT_LEN = 10

    kw_mask_single = create_keyword_mask(
        keyword_token_ranges=[(2, 3)],
        sent_tok_len=SENT_LEN,
        scheme="IOB",
    )
    assert kw_mask_single[2].item() == 1
    assert sum(kw_mask_single).item() == 1

    kw_mask_multi = create_keyword_mask(
        keyword_token_ranges=[(2, 5)],
        sent_tok_len=SENT_LEN,
        scheme="IOB",
    )
    assert kw_mask_multi[2].item() == 1
    assert kw_mask_multi[3].item() == 2
    assert kw_mask_multi[4].item() == 2
    assert sum(kw_mask_multi).item() == 5

    kw_mask_adjacent = create_keyword_mask(
        keyword_token_ranges=[(2, 3), (3, 5)],
        sent_tok_len=SENT_LEN,
        scheme="IOB",
    )
    assert kw_mask_adjacent[2].item() == 1
    assert kw_mask_adjacent[3].item() == 1
    assert kw_mask_adjacent[4].item() == 2
    assert sum(kw_mask_adjacent).item() == 4

    kw_mask_none = create_keyword_mask(
        keyword_token_ranges=[],
        sent_tok_len=SENT_LEN,
        scheme="IOB",
    )
    assert all(kw_mask_none == 0)


# ---------------------------------------------------------------------------
# resolve_keyword_spans_stem tests
# ---------------------------------------------------------------------------


def test_resolve_keyword_spans_stem_exact_match():
    """Keyword present verbatim in sentence is matched."""
    sentence = "novelty and semantic distance are key concepts"
    keywords = ["novelty", "semantic distance"]
    stemmer = PorterStemmer()
    offsets, n = _make_offsets(sentence)

    kwds, ranges = resolve_keyword_spans_stem(sentence, offsets, n, keywords, stemmer)

    assert "novelty" in kwds
    assert "semantic distance" in kwds
    assert len(ranges) == 2


def test_resolve_keyword_spans_stem_inflected_forms():
    """Inflected forms of keywords (e.g. plurals, past tense) are matched.

    This is the core regression that the exact-word regex approach fails:
    keyword "machine learning" should match "machines learning" and
    keyword "aid" should match "aided".
    """
    stemmer = PorterStemmer()

    # Single-word inflection: "aided" matches keyword "aid"
    sentence = "the program aided researchers"
    keywords = ["aid"]
    offsets, n = _make_offsets(sentence)
    kwds, ranges = resolve_keyword_spans_stem(sentence, offsets, n, keywords, stemmer)
    assert len(kwds) == 1, f"Expected 'aided' to match keyword 'aid', got: {kwds}"
    assert ranges[0][1] > ranges[0][0]

    # Multi-word: "machines learning" matches keyword "machine learning"
    sentence2 = "machines learning techniques are powerful"
    keywords2 = ["machine learning"]
    offsets2, n2 = _make_offsets(sentence2)
    kwds2, ranges2 = resolve_keyword_spans_stem(
        sentence2, offsets2, n2, keywords2, stemmer
    )
    assert len(kwds2) == 1, (
        f"Expected 'machines learning' to match 'machine learning', got: {kwds2}"
    )


def test_resolve_keyword_spans_stem_no_false_positive_ai_aims():
    """Keyword 'ai' must NOT match 'aims'.

    stem('aims') == 'aim' != 'ai' == stem('ai'), so stem-equality
    correctly rejects the match. This is the failure mode of the
    stem-prefix regex approach \\bai\\w*\\b.
    """
    stemmer = PorterStemmer()
    sentence = "an ambitious exhibition aims to prove that ai can help"

    # Use only "ai" as a keyword — "aims" in the sentence should not be matched
    keywords_ai_only = ["ai"]
    offsets, n = _make_offsets(sentence)
    kwds, ranges = resolve_keyword_spans_stem(
        sentence, offsets, n, keywords_ai_only, stemmer
    )

    # "aims" appears in the sentence but stem("aims")="aim" != stem("ai")="ai"
    assert not any("aims" in s for s in kwds), (
        f"'aims' should not be matched by keyword 'ai', but got: {kwds}"
    )
    # "ai" itself should be matched
    assert any(s.strip() == "ai" for s in kwds), (
        f"'ai' should be matched but got: {kwds}"
    )


def test_resolve_keyword_spans_stem_no_false_positive_substring():
    """Keyword matching must not match inside a longer word.

    Keyword 'ai' should not match 'cocktails' (where 'ai' is a substring).
    """
    stemmer = PorterStemmer()
    sentence = "a mechanical arm serves cocktails"
    keywords = ["ai"]
    offsets, n = _make_offsets(sentence)
    kwds, _ = resolve_keyword_spans_stem(sentence, offsets, n, keywords, stemmer)
    assert kwds == [], f"Expected no matches, got: {kwds}"


def test_resolve_keyword_spans_stem_token_ranges_correct():
    """Token index ranges returned must cover exactly the right tokens.

    Sentence: "the machine learning model"
    Words/tokens (1-indexed after BOS):
      1: the
      2: machine
      3: learning
      4: model
    Keyword "machine learning" should map to token range (2, 4)  [start incl, end excl].
    """
    stemmer = PorterStemmer()
    sentence = "the machine learning model"
    keywords = ["machine learning"]
    offsets, n = _make_offsets(sentence)
    kwds, ranges = resolve_keyword_spans_stem(sentence, offsets, n, keywords, stemmer)

    assert len(ranges) == 1
    start_tok, end_tok = ranges[0]
    # token 1 = "the", token 2 = "machine", token 3 = "learning", token 4 = "model"
    # BOS is at index 0, so "machine" is token index 2, "learning" is 3 → range (2, 4)
    assert start_tok == 2, f"Expected start_tok=2, got {start_tok}"
    assert end_tok == 4, f"Expected end_tok=4, got {end_tok}"


def test_resolve_keyword_spans_stem_overlapping_skipped():
    """When two keywords overlap, the longer one wins and the shorter is skipped."""
    stemmer = PorterStemmer()
    # "semantic distance" (2 words) and "semantic" (1 word) both match;
    # longest-first ordering means "semantic distance" is tried first and wins.
    sentence = "we measure semantic distance between concepts"
    keywords = ["semantic", "semantic distance"]
    offsets, n = _make_offsets(sentence)
    kwds, ranges = resolve_keyword_spans_stem(sentence, offsets, n, keywords, stemmer)

    # Only one of the two should be matched (no overlap)
    assert len(kwds) == 1
    assert kwds[0] == "semantic distance"


def test_resolve_keyword_spans_stem_multiword_inflected():
    """Multi-word keyword with inflected words matches in sentence.

    Keyword "knowledge hierarchy" vs sentence containing "knowledge hierarchies".
    stem("hierarchies") == stem("hierarchy") == "hiearchi" (Porter).
    """
    stemmer = PorterStemmer()
    # Confirm stems match (documents the assumption)
    assert stemmer.stem("hierarchies") == stemmer.stem("hierarchy")

    sentence = "we study knowledge hierarchies in ontologies"
    keywords = ["knowledge hierarchy"]
    offsets, n = _make_offsets(sentence)
    kwds, ranges = resolve_keyword_spans_stem(sentence, offsets, n, keywords, stemmer)

    assert len(kwds) == 1, (
        f"Expected 'knowledge hierarchies' to match keyword 'knowledge hierarchy', got: {kwds}"
    )


# ---------------------------------------------------------------------------
# Subtoken-aware IOB labeling tests
# ---------------------------------------------------------------------------


def _make_offsets_with_subtokens(
    words: list[str],
    subtoken_splits: list[list[tuple[int, int]]],
) -> tuple[np.ndarray, int]:
    """Build offset array where words may be split into multiple subtokens.

    Args:
        words: List of word strings (the original sentence).
        subtoken_splits: For each word, a list of (char_start, char_end) tuples
            for each subtoken that the word is split into.

    Returns:
        offsets: np.ndarray of shape [num_subtokens + 2, 2] (includes BOS/EOS).
        num_total_tokens: len(offsets).
    """
    spans = [[0, 0]]  # BOS sentinel
    for word_subtokens in subtoken_splits:
        for span in word_subtokens:
            spans.append(list(span))
    spans.append([0, 0])  # EOS sentinel
    offsets = np.array(spans, dtype=np.int64)
    return offsets, len(offsets)


def test_iob_labels_with_subtoken_split_on_non_initial_keyword_word():
    """IOB labels must propagate correctly when a keyword's non-initial word is split into subtokens.

    Example: keyword "neural networks" matches "neural, networks" where
    "networks" is tokenized as "net", "##works" (2 subtokens).

    Expected IOB: [O, B, I, I, O, ...]
      - "neural" → B (Beginning)
      - "net"    → I (Inside, part of "networks")
      - "##works"→ I (Inside, part of "networks")
    """
    stemmer = PorterStemmer()

    # Sentence: "neural networks are powerful"
    # Keyword: "neural networks" (2 words)
    # Simulate tokenizer splitting "networks" into "net" + "##works"
    sentence = "neural networks are powerful"
    keywords = ["neural networks"]

    # Build offsets simulating subtokenization:
    # Token 0: BOS [0, 0]
    # Token 1: "neural" [0, 6]
    # Token 2: "net" [7, 10]   (first subtoken of "networks")
    # Token 3: "##works" [10, 17]  (second subtoken of "networks")
    # Token 4: "are" [18, 21]
    # Token 5: "powerful" [22, 30]
    # Token 6: EOS [0, 0]
    offsets = np.array(
        [
            [0, 0],  # BOS
            [0, 6],  # neural
            [7, 10],  # net (subtoken 1 of "networks")
            [10, 17],  # ##works (subtoken 2 of "networks")
            [18, 21],  # are
            [22, 30],  # powerful
            [0, 0],  # EOS
        ],
        dtype=np.int64,
    )
    num_total_tokens = len(offsets)

    kwds, ranges = resolve_keyword_spans_stem(
        sentence, offsets, num_total_tokens, keywords, stemmer
    )

    # Keyword should be matched
    assert len(kwds) == 1, f"Expected 1 match, got: {kwds}"

    # Token range should cover tokens 1-4 (neural, net, ##works)
    # i.e., range (1, 4) [start incl, end excl]
    assert len(ranges) == 1
    start_tok, end_tok = ranges[0]
    assert start_tok == 1, f"Expected start_tok=1, got {start_tok}"
    assert end_tok == 4, f"Expected end_tok=4, got {end_tok}"

    # Now create IOB mask and verify labels
    sent_tok_len = num_total_tokens  # 7 tokens including BOS/EOS
    iob_mask = create_keyword_mask(ranges, sent_tok_len, scheme="IOB")

    # Expected: [O, B, I, I, O, O, O]
    # BOS=O, neural=B(1), net=I(2), ##works=I(2), are=O(0), powerful=O(0), EOS=O(0)
    assert iob_mask[0].item() == 0, "BOS should be O"
    assert iob_mask[1].item() == 1, "'neural' should be B (Beginning)"
    assert iob_mask[2].item() == 2, "'net' should be I (Inside)"
    assert iob_mask[3].item() == 2, "'##works' should be I (Inside)"
    assert iob_mask[4].item() == 0, "'are' should be O (Outside)"
    assert iob_mask[5].item() == 0, "'powerful' should be O (Outside)"
    assert iob_mask[6].item() == 0, "EOS should be O"


def test_iob_labels_with_subtoken_split_on_initial_keyword_word():
    """IOB labels must propagate correctly when a keyword's initial word is split into subtokens.

    Example: keyword "bioinformatics research" where "bioinformatics" is
    tokenized as "bio", "##in", "##for", "##matics" (4 subtokens).

    Expected IOB: [B, B, B, B, I, O, ...]
      - "bio"       → B (Beginning of first word)
      - "##in"      → B (Beginning of first word, new subtoken)
      - "##for"     → B (Beginning of first word, new subtoken)
      - "##matics"  → B (Beginning of first word, new subtoken)
      - "research"  → I (Inside, second word of keyword)

    Note: The current implementation uses word-level stem matching, so it
    operates on word boundaries, not subtoken boundaries. When a word is
    split into subtokens, the token range covers ALL subtokens of matched
    words, and create_keyword_mask assigns B to the first token and I to
    all subsequent tokens in the range.
    """
    stemmer = PorterStemmer()

    # Sentence: "bioinformatics research advances"
    # Keyword: "bioinformatics research" (2 words)
    # Simulate tokenizer splitting "bioinformatics" into 4 subtokens
    sentence = "bioinformatics research advances"
    keywords = ["bioinformatics research"]

    # Build offsets simulating subtokenization:
    # Token 0: BOS [0, 0]
    # Token 1: "bio" [0, 3]
    # Token 2: "##in" [3, 6]
    # Token 3: "##for" [6, 10]
    # Token 4: "##matics" [10, 18]
    # Token 5: "research" [19, 27]
    # Token 6: "advances" [28, 36]
    # Token 7: EOS [0, 0]
    offsets = np.array(
        [
            [0, 0],  # BOS
            [0, 3],  # bio (subtoken 1 of "bioinformatics")
            [3, 6],  # ##in (subtoken 2)
            [6, 10],  # ##for (subtoken 3)
            [10, 18],  # ##matics (subtoken 4)
            [19, 27],  # research
            [28, 36],  # advances
            [0, 0],  # EOS
        ],
        dtype=np.int64,
    )
    num_total_tokens = len(offsets)

    kwds, ranges = resolve_keyword_spans_stem(
        sentence, offsets, num_total_tokens, keywords, stemmer
    )

    # Keyword should be matched
    assert len(kwds) == 1, f"Expected 1 match, got: {kwds}"

    # Token range should cover tokens 1-6 (bio, ##in, ##for, ##matics, research)
    # i.e., range (1, 6) [start incl, end excl]
    assert len(ranges) == 1
    start_tok, end_tok = ranges[0]
    assert start_tok == 1, f"Expected start_tok=1, got {start_tok}"
    assert end_tok == 6, f"Expected end_tok=6, got {end_tok}"

    # Create IOB mask and verify labels
    sent_tok_len = num_total_tokens
    iob_mask = create_keyword_mask(ranges, sent_tok_len, scheme="IOB")

    # Expected: [O, B, I, I, I, I, O, O]
    # BOS=O, bio=B, ##in=I, ##for=I, ##matics=I, research=I, advances=O, EOS=O
    assert iob_mask[0].item() == 0, "BOS should be O"
    assert iob_mask[1].item() == 1, "'bio' should be B (Beginning)"
    assert iob_mask[2].item() == 2, "'##in' should be I (Inside)"
    assert iob_mask[3].item() == 2, "'##for' should be I (Inside)"
    assert iob_mask[4].item() == 2, "'##matics' should be I (Inside)"
    assert iob_mask[5].item() == 2, "'research' should be I (Inside)"
    assert iob_mask[6].item() == 0, "'advances' should be O (Outside)"
    assert iob_mask[7].item() == 0, "EOS should be O"
