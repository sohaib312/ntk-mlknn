"""Why does ranking neighbours by raw NTK value do so badly?

Raw NTK values grow with input norm (for this ReLU MLP, K(x, x) is proportional to ||x||^2),
so ranking by K(q, x) favours training points with large norm whatever the query. This
measures that on each dataset's test queries (standardised features, PCA 50, k = 10):

* top_decile_share  share of all neighbour slots taken by the 10% of training points with
                    the largest norm (10% would be neutral)
* unused_train      share of training points that are nobody's neighbour
* max_in_degree     how many test queries the most popular training point serves

Usage: python -m ntk_mlknn.diagnostics [dataset ...]  ->  results/neighbour_diagnostics.csv
"""

from __future__ import annotations

import argparse

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

from .data import DATASETS, load
from .experiment import RESULTS
from .kernels import relu_ntk, relu_ntk_diag, sq_euclidean, topk_smallest

K_NEIGHBOURS, N_COMP = 10, 50


def diagnose(name: str) -> list[dict]:
    ds = load(name)
    scaler = StandardScaler().fit(ds.train.X)
    Xtr, Xte = scaler.transform(ds.train.X), scaler.transform(ds.test.X)
    pca = PCA(n_components=min(N_COMP, Xtr.shape[1]), svd_solver="full").fit(Xtr)
    Ztr, Zte = pca.transform(Xtr), pca.transform(Xte)
    K = relu_ntk(Zte, Ztr)
    g_tr, g_te = relu_ntk_diag(Ztr), relu_ntk_diag(Zte)
    rankings = {
        "euclidean": sq_euclidean(Zte, Ztr),
        "ntk_raw": -K,
        "ntk_distance": g_te[:, None] + g_tr[None, :] - 2.0 * K,
    }
    norm = np.linalg.norm(Ztr, axis=1)
    top_decile = norm >= np.quantile(norm, 0.9)
    rows = []
    for how, D in rankings.items():
        nbrs = topk_smallest(D, K_NEIGHBOURS)
        degree = np.bincount(nbrs.ravel(), minlength=len(Ztr))
        rows.append({
            "dataset": name, "ranking": how,
            "top_decile_share": float(top_decile[nbrs].mean()),
            "unused_train": float((degree == 0).mean()),
            "max_in_degree": int(degree.max()), "n_test": len(Zte),
        })
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("datasets", nargs="*", default=DATASETS)
    df = pd.DataFrame([r for name in ap.parse_args().datasets for r in diagnose(name)])
    df.to_csv(RESULTS / "neighbour_diagnostics.csv", index=False)
    print(df.round(3).to_string(index=False))


if __name__ == "__main__":
    main()
