"""Our vectorised ML-kNN against scikit-multilearn's reference implementation."""

import numpy as np
import pytest
import scipy.sparse as sp
from sklearn.neighbors import NearestNeighbors

from ntk_mlknn.kernels import sq_euclidean, topk_smallest
from ntk_mlknn.mlknn import MLkNN

mlknn_mod = pytest.importorskip("skmultilearn.adapt.mlknn")


@pytest.fixture(autouse=True)
def _patch_nearest_neighbors(monkeypatch):
    # scikit-multilearn 0.2.0 calls NearestNeighbors(self.k) positionally, which
    # scikit-learn >= 1.0 rejects.
    monkeypatch.setattr(mlknn_mod, "NearestNeighbors", lambda k: NearestNeighbors(n_neighbors=k))


def _data(seed=0, n=240, d=8, L=5):
    rng = np.random.default_rng(seed)
    X = rng.normal(size=(n, d))
    W = rng.normal(size=(d, L))
    Y = (X @ W + rng.normal(size=(n, L)) > 0.8).astype(np.int8)
    return X[:180], Y[:180], X[180:], Y[180:]


@pytest.mark.parametrize("k,s", [(5, 1.0), (10, 0.5), (10, 2.0)])
def test_predictions_match_skmultilearn_default(k, s):
    # scikit-multilearn's default keeps each training point as its own first neighbour.
    Xtr, Ytr, Xte, Yte = _data()
    ref = mlknn_mod.MLkNN(k=k, s=s).fit(Xtr, sp.csr_matrix(Ytr))
    ours = MLkNN(k=k, s=s).fit(topk_smallest(sq_euclidean(Xtr, Xtr), k), Ytr)
    np.testing.assert_allclose(ours.like1, ref._cond_prob_true.toarray())
    np.testing.assert_allclose(ours.like0, ref._cond_prob_false.toarray())
    np.testing.assert_array_equal(ours.predict(topk_smallest(sq_euclidean(Xte, Xtr), k)), ref.predict(Xte).toarray())


def test_self_excluded_likelihoods_match_skmultilearn_ignore_first():
    Xtr, Ytr, _, _ = _data(seed=1)
    ref = mlknn_mod.MLkNN(k=10, s=1.0, ignore_first_neighbours=1).fit(Xtr, sp.csr_matrix(Ytr))
    ours = MLkNN(k=10, s=1.0).fit(topk_smallest(sq_euclidean(Xtr, Xtr), 10, exclude_diagonal=True), Ytr)
    np.testing.assert_allclose(ours.prior1, ref._prior_prob_true)
    np.testing.assert_allclose(ours.like1, ref._cond_prob_true.toarray())
    np.testing.assert_allclose(ours.like0, ref._cond_prob_false.toarray())
