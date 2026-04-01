from keyword_pecker.trainer import EarlyStopping


def test_early_stopping_max_mode():
    es = EarlyStopping(metric="f1", patience=2, delta=0.0, mode="max")

    assert es({"f1": 0.5}, step=1) is False
    assert es.best_score == 0.5
    assert es.counter == 0

    assert es({"f1": 0.6}, step=2) is False
    assert es.best_score == 0.6
    assert es.counter == 0

    assert es({"f1": 0.55}, step=3) is False
    assert es.counter == 1

    assert es({"f1": 0.55}, step=4) is True
    assert es.counter == 2


def test_early_stopping_min_mode():
    es = EarlyStopping(metric="loss", patience=1, delta=0.0, mode="min")

    assert es({"loss": 1.0}, step=1) is False
    assert es.best_score == 1.0

    assert es({"loss": 0.9}, step=2) is False
    assert es.best_score == 0.9
    assert es.counter == 0

    assert es({"loss": 0.95}, step=3) is True
    assert es.counter == 1


def test_early_stopping_delta():
    es = EarlyStopping(metric="f1", patience=1, delta=0.1, mode="max")

    assert es({"f1": 0.5}, step=1) is False

    assert es({"f1": 0.55}, step=2) is True
    assert es.counter == 1

    es.reset()
    assert es({"f1": 0.5}, step=1) is False
    assert es({"f1": 0.61}, step=2) is False
    assert es.best_score == 0.61
    assert es.counter == 0


def test_early_stopping_reset():
    es = EarlyStopping(metric="loss", patience=2, delta=0.0, mode="min")

    es({"loss": 1.0}, step=1)
    es({"loss": 1.1}, step=2)
    assert es.counter == 1

    es.reset()
    assert es.best_score is None
    assert es.counter == 0
    assert es.best_step == 0

    assert es({"loss": 0.5}, step=1) is False
    assert es.best_score == 0.5
