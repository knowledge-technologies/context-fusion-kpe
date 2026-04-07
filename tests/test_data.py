import polars as pl
from numpy.testing import assert_approx_equal
from keyword_pecker.data import compute_class_weights


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

    # input array
    test_record = {
        "keyword_mask": [
            [0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 2],
            [0, 0, 1, 1, 1, 0, 0, 2],
            [1, 1, 0, 0, 0, 1, 1, 2, 2],
        ]
    }

    test_df = pl.DataFrame(test_record)

    weights = compute_class_weights(df=test_df, num_classes=N_CLASSES)

    assert_approx_equal(weights[0], CLASS_1_EXP_WEIGHT, significant=3)
    assert_approx_equal(weights[1], CLASS_2_EXP_WEIGHT, significant=3)
    assert_approx_equal(weights[2], CLASS_3_EXP_WEIGHT, significant=3)
