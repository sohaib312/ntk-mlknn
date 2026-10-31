import numpy as np
import pytest

from ntk_mlknn.kernels import neural_tangents_ntk, relu_ntk, relu_ntk_diag, sq_euclidean, topk_smallest


def _X(seed=0, n=60, d=20):
    return np.random.default_rng(seed).normal(size=(n, d)) * 2.0


@pytest.mark.parametrize("widths", [(64, 32), (128, 64), (7, 3)])
def test_closed_form_ntk_equals_neural_tangents_for_any_width(widths):
    pytest.importorskip("neural_tangents")
    X1, X2 = _X(0), _X(1, n=40)
    np.testing.assert_allclose(relu_ntk(X1, X2), neural_tangents_ntk(X1, X2, widths), rtol=2e-4, atol=1e-5)


def test_ntk_diag():
    X = _X(2)
    np.testing.assert_allclose(relu_ntk_diag(X), np.diag(relu_ntk(X, X)), rtol=1e-6)


@pytest.mark.parametrize("gamma", [1e-3, 1e-2, 1e-1])
def test_rbf_neighbours_are_euclidean_neighbours(gamma):
    # exp(-gamma * d^2) is decreasing in d, so the top-k by RBF similarity is the top-k by
    # Euclidean distance. (For large gamma the exponentials underflow to 0 and the ranking
    # degenerates into ties; that is a numerical artefact, not a different neighbourhood.)
    Xq, Xtr = _X(3, n=50, d=10), _X(4, n=200, d=10)
    D = sq_euclidean(Xq, Xtr).astype(np.float64)
    np.testing.assert_array_equal(topk_smallest(-np.exp(-gamma * D), 10), topk_smallest(D, 10))
