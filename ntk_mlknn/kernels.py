"""Similarities and neighbourhoods: Euclidean, RBF, infinite-width NTK and their fusion.

Two facts this module makes explicit (and tests/ checks):

* An RBF kernel exp(-gamma * ||x - y||^2) is a decreasing function of Euclidean distance, so
  "the k most RBF-similar points" are exactly the k Euclidean nearest neighbours, for every
  gamma. RBF changes a kNN neighbourhood only once it is mixed with another kernel.
* The NTK of an infinitely wide ReLU MLP depends on depth and weight variance, not on the
  hidden widths: `relu_ntk` below has no width argument, and it equals what
  neural_tangents' stax.serial(Dense(w1), Relu(), Dense(w2), Relu(), Dense(1)) returns for
  any w1, w2 -- (64, 32) and (128, 64), for example, give the same kernel.

Neighbourhoods can be ranked two ways. "raw" ranks by kernel value. "distance" ranks by the kernel-induced distance K(x,x) + K(y,y) - 2 K(x,y), the usual
way to turn a kernel into a metric; raw NTK values grow with input norm, so raw ranking
favours large-norm training points regardless of where the query is.
"""

from __future__ import annotations

import numpy as np

# Pairwise matrices are computed in float64 one block of rows at a time and stored as
# float32, so bibtex / corel5k (~5k training points) fit in a laptop's memory.
BLOCK = 512
DTYPE = np.float32


def sq_euclidean(X1: np.ndarray, X2: np.ndarray) -> np.ndarray:
    n2 = (X2**2).sum(1)
    out = np.empty((len(X1), len(X2)), dtype=DTYPE)
    for i in range(0, len(X1), BLOCK):
        A = X1[i : i + BLOCK]
        out[i : i + BLOCK] = np.maximum((A**2).sum(1)[:, None] + n2[None, :] - 2.0 * A @ X2.T, 0.0)
    return out


def _relu_mlp_ntk(S, norm, depth: int, w2: float):
    """Push (NNGP S, sqrt(q1 q2) norm) after the first Dense through depth x [ReLU, Dense]."""
    T = S
    for _ in range(depth):
        cos = np.clip(S / np.where(norm > 0, norm, 1.0), -1.0, 1.0)
        theta = np.arccos(cos)
        S_relu = norm / (2 * np.pi) * (np.sin(theta) + (np.pi - theta) * cos)
        T = w2 * (S_relu + T * (np.pi - theta) / (2 * np.pi))
        S, norm = w2 * S_relu, w2 * norm / 2
    return T


def relu_ntk(X1: np.ndarray, X2: np.ndarray, depth: int = 2, W_std: float = 1.0) -> np.ndarray:
    """Closed-form infinite-width NTK of [Dense, ReLU] x depth -> Dense(1), no biases,
    NTK parameterisation (neural_tangents' stax defaults)."""
    w2, d = W_std**2, X1.shape[1]
    n2 = (X2**2).sum(1)
    out = np.empty((len(X1), len(X2)), dtype=DTYPE)
    for i in range(0, len(X1), BLOCK):
        A = X1[i : i + BLOCK]
        S = w2 * A @ X2.T / d
        norm = w2 * np.sqrt(np.outer((A**2).sum(1), n2)) / d
        out[i : i + BLOCK] = _relu_mlp_ntk(S, norm, depth, w2)
    return out


def relu_ntk_diag(X: np.ndarray, depth: int = 2, W_std: float = 1.0) -> np.ndarray:
    """K(x, x) for each row of X, without forming the full matrix."""
    w2 = W_std**2
    q = w2 * (X**2).sum(1) / X.shape[1]
    return _relu_mlp_ntk(q, q.copy(), depth, w2)


def neural_tangents_ntk(X1: np.ndarray, X2: np.ndarray, widths=(64, 32)) -> np.ndarray:
    """The same NTK computed by neural_tangents; the reference the tests check `relu_ntk` against."""
    import jax
    import jax.numpy as jnp
    from neural_tangents import stax

    _, _, ntk_fn = stax.serial(
        stax.Dense(widths[0]), stax.Relu(), stax.Dense(widths[1]), stax.Relu(), stax.Dense(1)
    )
    ntk_fn = jax.jit(ntk_fn, static_argnames=("get",))
    return np.asarray(ntk_fn(jnp.array(X1, jnp.float32), jnp.array(X2, jnp.float32), get="ntk"), np.float64)


def topk_smallest(D: np.ndarray, k: int, exclude_diagonal: bool = False) -> np.ndarray:
    """Column indices of the k smallest entries of each row, sorted ascending.

    `exclude_diagonal` (train-vs-train) bars each point from being its own neighbour.
    """
    out = np.empty((len(D), k), dtype=np.int64)
    for i in range(0, len(D), BLOCK):
        B = D[i : i + BLOCK]
        if exclude_diagonal:
            B = B.copy()
            r = np.arange(len(B))
            B[r, i + r] = np.inf
        idx = np.argpartition(B, kth=k - 1, axis=1)[:, :k]
        order = np.take_along_axis(B, idx, axis=1).argsort(axis=1, kind="stable")
        out[i : i + BLOCK] = np.take_along_axis(idx, order, axis=1)
    return out
