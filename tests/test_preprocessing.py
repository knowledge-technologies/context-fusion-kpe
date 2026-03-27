from keyword_pecker.data import build_keyword_regex
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

    TEST_KEYWORDS = ["language models", "language models", "machine learning"]
    EXPECTED_N_STEMMED_KEYWORDS = 2

    regex_patter, stemmed_keywords = build_keyword_regex(
        keywords=TEST_KEYWORDS, stemmer=PorterStemmer()
    )

    assert len(stemmed_keywords) == len(TEST_KEYWORDS)
    assert len(set(stemmed_keywords)) == EXPECTED_N_STEMMED_KEYWORDS
    assert set(stemmed_keywords) == {"languag model", "machin learn"}


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

    pattern, stemmed = build_keyword_regex(keywords, PorterStemmer())

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
