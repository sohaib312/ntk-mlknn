"""ML-kNN (Zhang & Zhou, 2007) over any neighbourhood.

Neighbourhoods are passed in as index arrays, so the same estimator runs on Euclidean
neighbours, RBF/NTK-kernel neighbours or a fused kernel. Whether a training point counts
among its own neighbours is decided by whoever builds `train_nbrs`: standard ML-kNN excludes
it; scikit-multilearn's default does not.

`fit_counts` / `predict_proba_counts` take precomputed neighbour-label counts, so a grid
search can reuse one set of counts across smoothing values.
"""

from __future__ import annotations

import numpy as np


def neighbour_label_counts(nbrs: np.ndarray, Y_train: np.ndarray) -> np.ndarray:
    """For each query row, how many of its neighbours carry each label: (n, L) in 0..k."""
    Y = Y_train.astype(np.int32)
    C = np.zeros((len(nbrs), Y.shape[1]), dtype=np.int32)
    for j in range(nbrs.shape[1]):
        C += Y[nbrs[:, j]]
    return C


class MLkNN:
    """Standard ML-kNN posterior with Laplace smoothing `s`."""

    def __init__(self, k: int = 10, s: float = 1.0):
        self.k, self.s = k, s

    def fit(self, train_nbrs: np.ndarray, Y_train: np.ndarray) -> "MLkNN":
        """`train_nbrs` (n, >=k): each training point's neighbours among the training set."""
        self.Y_train = Y_train
        return self.fit_counts(neighbour_label_counts(train_nbrs[:, : self.k], Y_train), Y_train)

    def fit_counts(self, C: np.ndarray, Y_train: np.ndarray) -> "MLkNN":
        k, s = self.k, self.s
        Y = Y_train.astype(bool)
        n, L = Y.shape
        self.prior1 = (s + Y.sum(0)) / (2 * s + n)
        self.prior0 = 1.0 - self.prior1
        c1 = np.zeros((L, k + 1))
        c0 = np.zeros((L, k + 1))
        for h in range(k + 1):
            hit = C == h
            c1[:, h] = (hit & Y).sum(0)
            c0[:, h] = (hit & ~Y).sum(0)
        self.like1 = (s + c1) / (s * (k + 1) + c1.sum(1, keepdims=True))
        self.like0 = (s + c0) / (s * (k + 1) + c0.sum(1, keepdims=True))
        return self

    def predict_proba(self, query_nbrs: np.ndarray) -> np.ndarray:
        """Posterior P(label j | neighbour counts), shape (n_query, L)."""
        return self.predict_proba_counts(neighbour_label_counts(query_nbrs[:, : self.k], self.Y_train))

    def predict_proba_counts(self, C: np.ndarray) -> np.ndarray:
        cols = np.arange(C.shape[1])[None, :]
        p1 = self.prior1[None, :] * self.like1[cols, C]
        p0 = self.prior0[None, :] * self.like0[cols, C]
        return p1 / (p1 + p0)

    def predict(self, query_nbrs: np.ndarray) -> np.ndarray:
        # Predict 1 when p1 >= p0, i.e. posterior >= 0.5.
        return (self.predict_proba(query_nbrs) >= 0.5).astype(np.int8)

