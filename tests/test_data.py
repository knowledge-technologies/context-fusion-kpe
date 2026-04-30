import polars as pl
from numpy.testing import assert_approx_equal
from transformers import AutoTokenizer
from keyword_pecker.data import compute_class_weights, extract_word_level_data
from keyword_pecker.data.dataset import KeyphraseDataset, keyphrase_collate_fn

_TOKENIZER = AutoTokenizer.from_pretrained("sentence-transformers/all-mpnet-base-v2")


def test_compute_class_weights():

    CLASS_1_SAMPLES = 10
    CLASS_2_SAMPLES = 15
    CLASS_3_SAMPLES = 4
    N_CLASSES = 3

    N_SAMPLES = CLASS_1_SAMPLES + CLASS_2_SAMPLES + CLASS_3_SAMPLES

    # expected weights
    CLASS_1_EXP_WEIGHT = N_SAMPLES / (N_CLASSES * CLASS_1_SAMPLES)
    CLASS_2_EXP_WEIGHT = N_SAMPLES / (N_CLASSES * CLASS_2_SAMPLES)
    CLASS_3_EXP_WEIGHT = N_SAMPLES / (N_CLASSES * CLASS_3_SAMPLES)

    CLASS_1_EXP_FREQ = CLASS_1_SAMPLES / N_SAMPLES * 100
    CLASS_2_EXP_FREQ = CLASS_2_SAMPLES / N_SAMPLES * 100
    CLASS_3_EXP_FREQ = CLASS_3_SAMPLES / N_SAMPLES * 100

    # input array
    test_record = {
        "keyword_mask": [
            [0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 2],
            [0, 0, 1, 1, 1, 0, 0, 2],
            [1, 1, 0, 0, 0, 1, 1, 2, 2],
        ]
    }

    test_df = pl.DataFrame(test_record)

    weights, frequencies = compute_class_weights(df=test_df, num_classes=N_CLASSES)

    assert_approx_equal(weights[0], CLASS_1_EXP_WEIGHT, significant=3)
    assert_approx_equal(weights[1], CLASS_2_EXP_WEIGHT, significant=3)
    assert_approx_equal(weights[2], CLASS_3_EXP_WEIGHT, significant=3)

    assert_approx_equal(frequencies[0], CLASS_1_EXP_FREQ, significant=3)
    assert_approx_equal(frequencies[1], CLASS_2_EXP_FREQ, significant=3)
    assert_approx_equal(frequencies[2], CLASS_3_EXP_FREQ, significant=3)


def test_dataset_includes_word_ids():
    """Verify __getitem__ returns word_ids and word_list."""
    df = pl.DataFrame(
        {
            "doc_id": ["doc1"],
            "doc_row": [0],
            "sent_num": [0],
            "sent_txt": ["hello world"],
            "keywords": [["hello"]],
            "sent_keywords": [["hello"]],
            "token_ids": [[1, 2, 3]],
            "attention_mask": [[1, 1, 1]],
            "keyword_mask": [[0, 1, 0]],
            "word_ids": [[-1, 0, -1]],
            "word_list": [["hello", "world"]],
        }
    )
    dataset = KeyphraseDataset(df)
    item = dataset[0]

    assert "word_ids" in item
    assert "word_list" in item
    assert item["word_ids"] == [-1, 0, -1]
    assert item["word_list"] == ["hello", "world"]


def test_collate_pads_word_ids_with_minus_one():
    """Verify collate_fn pads word_ids with -1."""
    batch = [
        {
            "doc_id": "doc1",
            "doc_row": 0,
            "token_ids": [1, 2, 3],
            "keyword_mask": [0, 1, 0],
            "keywords": ["kw1"],
            "word_ids": [-1, 0, -1],
            "word_list": ["hello", "world"],
        },
        {
            "doc_id": "doc1",
            "doc_row": 0,
            "token_ids": [1, 2, 3, 4, 5],
            "keyword_mask": [0, 1, 0, 1, 0],
            "keywords": ["kw2"],
            "word_ids": [-1, 0, 1, 2, -1],
            "word_list": ["a", "b", "c"],
        },
    ]

    result = keyphrase_collate_fn(batch)

    assert "word_ids" in result
    assert "word_lists" in result
    assert result["word_ids"].shape == (2, 5)
    # First row padded with -1
    assert result["word_ids"][0, 3].item() == -1
    assert result["word_ids"][0, 4].item() == -1
    # Second row not padded
    assert result["word_ids"][1, 4].item() == -1  # original -1 at end
    assert result["word_lists"] == [["hello", "world"], ["a", "b", "c"]]


def test_extract_word_level_data_simple_and_indexing():
    """Verify word_list content and that word_ids are sequential 0..N-1."""
    sentences = ["machine learning is great"]
    tokenized = _TOKENIZER(sentences, return_tensors="pt", return_offsets_mapping=True)

    word_ids, word_lists = extract_word_level_data(tokenized, sentences)

    assert len(word_ids) == 1
    assert len(word_lists) == 1
    assert word_lists[0] == ["machine", "learning", "is", "great"]
    assert word_ids[0][0] == -1  # [CLS]
    assert word_ids[0][-1] == -1  # [SEP]

    # Verify word_ids are sequential and index into word_list correctly
    unique_wids = [w for w in word_ids[0] if w >= 0]
    assert unique_wids == [0, 1, 2, 3]
    for wid in unique_wids:
        assert word_lists[0][wid] == ["machine", "learning", "is", "great"][wid]


def test_extract_word_level_data_hyphenated_and_punctuation():
    """Verify word_list handles hyphenated words and punctuation with separate word_ids."""
    sentences = ["state-of-the-art methods", "hello, world!"]
    tokenized = _TOKENIZER(
        sentences, return_tensors="pt", return_offsets_mapping=True, padding=True
    )

    word_ids, word_lists = extract_word_level_data(tokenized, sentences)

    # "state-of-the-art methods" — tokenizer assigns separate word_ids to each component
    max_wid_0 = max(w for w in word_ids[0] if w != -1)
    assert len(word_lists[0]) == max_wid_0 + 1
    assert "state" in word_lists[0]
    assert "art" in word_lists[0]
    assert "methods" in word_lists[0]
    # Verify every word_id is indexable
    for wid in [w for w in word_ids[0] if w >= 0]:
        _ = word_lists[0][wid]

    # "hello, world!" — punctuation gets its own word_ids
    max_wid_1 = max(w for w in word_ids[1] if w != -1)
    assert len(word_lists[1]) == max_wid_1 + 1
    for wid in [w for w in word_ids[1] if w >= 0]:
        _ = word_lists[1][wid]


def test_extract_word_level_data_batch():
    """Verify batch processing: both elements correctly indexed from 0 independently."""
    sentences = ["hello world", "foo-bar baz"]
    tokenized = _TOKENIZER(
        sentences, return_tensors="pt", return_offsets_mapping=True, padding=True
    )

    word_ids, word_lists = extract_word_level_data(tokenized, sentences)

    assert len(word_ids) == 2
    assert len(word_lists) == 2

    # First sentence: "hello world" → word_ids 0, 1
    assert word_lists[0] == ["hello", "world"]
    unique_wids_0 = [w for w in word_ids[0] if w >= 0]
    assert unique_wids_0 == [0, 1]
    assert word_lists[0][0] == "hello"
    assert word_lists[0][1] == "world"

    # Second sentence: "foo-bar baz" → word_ids 0..3 (hyphen splits, "baz" subwords share word_id 3)
    max_wid_1 = max(w for w in word_ids[1] if w != -1)
    assert max_wid_1 > 1  # hyphenated word creates extra word_ids
    assert len(word_lists[1]) == max_wid_1 + 1
    # All word_ids should be in range [0, max_wid_1]
    assert all(0 <= w <= max_wid_1 for w in word_ids[1] if w != -1)
    for wid in range(max_wid_1 + 1):
        _ = word_lists[1][wid]


def test_extract_word_level_data_subword_aggregation():
    """Verify multi-subtoken words are fully reconstructed, not truncated."""
    sentence = (
        "This paper describes a method for detecting event trigger words "
        "in biomedical text based on a word sense disambiguation (WSD) approach."
    )
    tokenized = _TOKENIZER([sentence], return_tensors="pt", return_offsets_mapping=True)

    word_ids, word_lists = extract_word_level_data(tokenized, [sentence])

    assert "disambiguation" in word_lists[0], (
        f"Expected 'disambiguation' in word_list, got: {word_lists[0]}"
    )
    assert "di" not in word_lists[0] or word_lists[0].count("di") == 0, (
        f"'di' should not appear as a separate entry; word_list: {word_lists[0]}"
    )
    assert "WSD" in word_lists[0] or "wsd" in word_lists[0], (
        f"Expected 'WSD'/'wsd' in word_list, got: {word_lists[0]}"
    )
    assert "W" not in word_lists[0], (
        f"'W' should not appear as a separate entry; word_list: {word_lists[0]}"
    )

    max_wid = max(w for w in word_ids[0] if w != -1)
    assert len(word_lists[0]) == max_wid + 1
    for wid in [w for w in word_ids[0] if w >= 0]:
        _ = word_lists[0][wid]
