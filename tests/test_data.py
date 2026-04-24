import polars as pl
from numpy.testing import assert_approx_equal
from keyword_pecker.data import compute_class_weights
from keyword_pecker.data.dataset import KeyphraseDataset, keyphrase_collate_fn


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
