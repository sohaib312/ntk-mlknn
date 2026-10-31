import numpy as np
import pytest
from sklearn.metrics import (
    coverage_error,
    f1_score,
    hamming_loss,
    label_ranking_average_precision_score,
    label_ranking_loss,
)

from ntk_mlknn.metrics import evaluate


@pytest.mark.parametrize("seed", [0, 1, 2])
def test_matches_sklearn_including_ties(seed):
    rng = np.random.default_rng(seed)
    Y = (rng.random((300, 12)) < 0.3).astype(np.int8)
    Y[0], Y[1] = 0, 1  # rows the ranking metrics must skip
    scores = rng.integers(0, 4, size=Y.shape) / 3.0  # coarse scores -> many ties
    pred = (scores >= 0.5).astype(np.int8)
    got = evaluate(Y, pred, scores)

    ok = (Y.sum(1) > 0) & (Y.sum(1) < Y.shape[1])
    assert got["hamming_loss"] == pytest.approx(hamming_loss(Y, pred))
    assert got["micro_f1"] == pytest.approx(f1_score(Y, pred, average="micro", zero_division=0))
    assert got["macro_f1"] == pytest.approx(f1_score(Y, pred, average="macro", zero_division=0))
    assert got["ranking_loss"] == pytest.approx(label_ranking_loss(Y[ok], scores[ok]))
    assert got["avg_precision"] == pytest.approx(label_ranking_average_precision_score(Y[ok], scores[ok]))
    assert got["coverage"] == pytest.approx(coverage_error(Y[ok], scores[ok]) - 1)
