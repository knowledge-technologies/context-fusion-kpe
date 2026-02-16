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
