"""Multi-label metrics, computed the way the MULAN / ML-kNN literature does.

Bipartition metrics (Hamming loss, subset accuracy, micro/macro F1) use 0/1 predictions.
Ranking metrics (ranking loss, average precision, one-error, coverage) use real-valued label
scores -- here the ML-kNN posterior. Passing 0/1 predictions to ranking metrics instead
would make every predicted label tie.

Ranking metrics are averaged over test instances that have at least one relevant and one
irrelevant label (they are undefined otherwise; MULAN skips such instances too). Ties count
against the model, as in scikit-learn. The implementations are vectorised versions of
scikit-learn's (the grid search scores thousands of configurations); tests/test_metrics.py
checks they agree with scikit-learn.
"""

from __future__ import annotations

import numpy as np
from scipy.stats import rankdata

BIPARTITION = ["hamming_loss", "subset_accuracy", "micro_f1", "macro_f1"]
RANKING = ["ranking_loss", "avg_precision", "one_error", "coverage"]
ALL = BIPARTITION + RANKING
LOWER_IS_BETTER = {"hamming_loss", "ranking_loss", "one_error", "coverage"}


def ranking_per_instance(Y: np.ndarray, S: np.ndarray) -> dict[str, np.ndarray]:
    """Per-instance ranking metrics; rows must have >= 1 relevant and >= 1 irrelevant label."""
    y = Y.astype(bool)
    # rank[n, j]: labels scoring at least as high as j (itself included), i.e. its pessimistic rank;
    # rel_above[n, j]: the same count among relevant labels only (irrelevant ones pushed to +inf).
    rank = rankdata(-S, method="max", axis=1)
    rel_above = rankdata(np.where(y, -S, np.inf), method="max", axis=1)
    n_rel, n_irr = y.sum(1), (~y).sum(1)
    return {
        "ranking_loss": np.where(y, rank - rel_above, 0).sum(1) / (n_rel * n_irr),
        "avg_precision": np.where(y, rel_above / rank, 0).sum(1) / n_rel,
        "one_error": (Y[np.arange(len(Y)), S.argmax(1)] == 0).astype(float),
        "coverage": np.where(y, rank, 0).max(1) - 1.0,
    }


def rankable(Y: np.ndarray) -> np.ndarray:
    n_pos = Y.sum(1)
    return (n_pos > 0) & (n_pos < Y.shape[1])


def evaluate(Y: np.ndarray, pred: np.ndarray, scores: np.ndarray) -> dict[str, float]:
    Y, pred = Y.astype(bool), pred.astype(bool)
    tp = (Y & pred).sum(0)
    fp = (~Y & pred).sum(0)
    fn = (Y & ~pred).sum(0)
    denom = 2 * tp + fp + fn
    out = {
        "hamming_loss": (Y != pred).mean(),
        "subset_accuracy": (Y == pred).all(1).mean(),
        "micro_f1": 2 * tp.sum() / max(denom.sum(), 1),
        "macro_f1": np.where(denom > 0, 2 * tp / np.maximum(denom, 1), 0.0).mean(),
    }
    ok = rankable(Y)
    out.update({k: v.mean() for k, v in ranking_per_instance(Y[ok], scores[ok]).items()})
    return {k: float(v) for k, v in out.items()}
