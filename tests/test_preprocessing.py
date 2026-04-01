from keyword_pecker.data import build_keyword_regex, create_keyword_mask
from nltk.stem.porter import PorterStemmer

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
