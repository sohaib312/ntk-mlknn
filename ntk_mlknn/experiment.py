"""Cross-validated evaluation of NTK / RBF neighbourhoods for ML-kNN on MULAN benchmarks.

For each dataset:

1. Every configuration in the grid is scored by 5-fold cross-validation on the standard
   training split (scaler and PCA refit inside each fold; training points never count as
   their own neighbours).
2. Each method family picks one configuration by mean CV Hamming loss (and, separately,
   by mean CV average precision). Exact ties go to the simpler configuration (smaller NTK weight first), so NTK has to win outright to be selected.
3. Every configuration is refit on the full training split and scored on the test split
   once. Only the CV-selected configurations' test scores are reported as results; the rest
   are kept to measure how optimistic choosing by test score would be.

Neighbourhoods (all on standardised features; "PCA" ones on PCA-projected features):

  euclid    Euclidean distance. On PCA features this is also every RBF kernel's ranking,
            i.e. the hybrid at alpha = 0.
  ntk_raw   rank by raw NTK value (the hybrid at alpha = 1)
  ntk_dist  rank by NTK-induced distance K(x,x) + K(y,y) - 2K(x,y)
  mix_raw   alpha * NTK + (1 - alpha) * RBF, ranked by value ("value fusion")
  mix_dist  same fusion with NTK rescaled to unit mean diagonal, ranked by induced distance

Usage: python -m ntk_mlknn.experiment [dataset ...]   ->  results/<dataset>/
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler

from . import metrics
from .data import DATASETS, load
from .kernels import BLOCK, relu_ntk, relu_ntk_diag, sq_euclidean, topk_smallest
from .mlknn import MLkNN, neighbour_label_counts

RESULTS = Path(__file__).resolve().parents[1] / "results"

PCA_DIMS = (30, 50, 80)
GAMMAS = tuple(np.logspace(-3, 1, 5))
MIX_ALPHAS = (0.1, 0.25, 0.5, 0.75, 0.9)  # alpha = 0 and 1 are the euclid / ntk endpoints
KS = (5, 10, 15)
SS = (0.5, 1.0, 2.0)
FOLDS = 5
SEED = 0

KEY = ["n_comp", "kind", "gamma", "alpha", "k", "s"]

FAMILIES = {
    "ML-kNN": lambda c: (c.n_comp == 0) & (c.kind == "euclid"),
    "ML-kNN + PCA": lambda c: (c.n_comp > 0) & (c.kind == "euclid"),
    "NTK (raw)": lambda c: c.kind == "ntk_raw",
    "NTK (distance)": lambda c: c.kind == "ntk_dist",
    "Hybrid (value fusion)": lambda c: ((c.n_comp > 0) & (c.kind == "euclid")) | c.kind.isin(["ntk_raw", "mix_raw"]),
    "Hybrid (kernel distance)": lambda c: ((c.n_comp > 0) & (c.kind == "euclid")) | c.kind.isin(["ntk_dist", "mix_dist"]),
}
CRITERIA = {"hamming_loss": True, "avg_precision": False}  # metric -> ascending


def neighbour_sets(Z_tr: np.ndarray, Z_q: np.ndarray):
    """Yield (params, train-vs-train dissimilarity, query-vs-train dissimilarity).

    Per side (train, query) only three matrices live at once -- Euclidean D, NTK K and a work
    buffer W (RBF values are recomputed block by block) -- and each yielded pair is written
    into W, so a yielded matrix is only valid until the next one is requested.
    """
    sides = {"tr": Z_tr, "q": Z_q}
    D = {s: sq_euclidean(Z, Z_tr) for s, Z in sides.items()}
    yield {"kind": "euclid", "gamma": 0.0, "alpha": 0.0}, D["tr"], D["q"]

    K = {s: relu_ntk(Z, Z_tr) for s, Z in sides.items()}
    g = {s: relu_ntk_diag(Z) for s, Z in sides.items()}
    c = g["tr"].mean()
    W = {s: np.empty_like(K[s]) for s in sides}

    def ntk_dist(s, rows):  # NTK-induced squared distance, rescaled to unit mean self-similarity
        return (g[s][rows][:, None] + g["tr"][None, :] - 2.0 * K[s][rows]) / c

    def fill(out, f):
        for s in sides:
            for i in range(0, len(out[s]), BLOCK):
                rows = slice(i, i + BLOCK)
                out[s][rows] = f(s, rows)
        return out["tr"], out["q"]

    yield ({"kind": "ntk_raw", "gamma": 0.0, "alpha": 1.0}, *fill(W, lambda s, r: -K[s][r]))
    yield ({"kind": "ntk_dist", "gamma": 0.0, "alpha": 1.0}, *fill(W, ntk_dist))

    def rbf(s, rows):
        return np.exp(-gamma * D[s][rows])

    for gamma in GAMMAS:
        for a in MIX_ALPHAS:
            yield ({"kind": "mix_raw", "gamma": gamma, "alpha": a},
                   *fill(W, lambda s, r: -(a * K[s][r] + (1 - a) * rbf(s, r))))
            yield ({"kind": "mix_dist", "gamma": gamma, "alpha": a},
                   *fill(W, lambda s, r: a * ntk_dist(s, r) + (1 - a) * (2 - 2 * rbf(s, r))))


def score_all(X_tr, Y_tr, X_q, Y_q, keep=frozenset()):
    """Score every grid configuration fitted on (X_tr, Y_tr) against (X_q, Y_q).

    Returns metric rows, plus the query posteriors of configurations whose key is in `keep`.
    """
    scaler = StandardScaler().fit(X_tr)
    X_tr, X_q = scaler.transform(X_tr), scaler.transform(X_q)
    kmax = max(KS)
    rows, kept = [], {}

    def run(params, D_tr, D_q):
        nb_tr = topk_smallest(D_tr, kmax, exclude_diagonal=True)
        nb_q = topk_smallest(D_q, kmax)
        for k in KS:
            C_tr, C_q = neighbour_label_counts(nb_tr[:, :k], Y_tr), neighbour_label_counts(nb_q[:, :k], Y_tr)
            for s in SS:
                P = MLkNN(k=k, s=s).fit_counts(C_tr, Y_tr).predict_proba_counts(C_q)
                key = (params["n_comp"], params["kind"], params["gamma"], params["alpha"], k, s)
                rows.append({**dict(zip(KEY, key)), **metrics.evaluate(Y_q, P >= 0.5, P)})
                if key in keep:
                    kept[key] = P

    run({"n_comp": 0, "kind": "euclid", "gamma": 0.0, "alpha": 0.0}, sq_euclidean(X_tr, X_tr), sq_euclidean(X_q, X_tr))
    dims = sorted({min(n, X_tr.shape[1]) for n in PCA_DIMS})
    for n in dims:
        pca = PCA(n_components=n, svd_solver="full").fit(X_tr)
        Z_tr, Z_q = pca.transform(X_tr), pca.transform(X_q)
        for params, D_tr, D_q in neighbour_sets(Z_tr, Z_q):
            run({"n_comp": n, **params}, D_tr, D_q)
        del D_tr, D_q  # release the last work buffers before the next PCA fit
    return rows, kept


def select(cv: pd.DataFrame) -> pd.DataFrame:
    """One configuration per (family, criterion), by mean CV score; ties -> simpler config."""
    picks = []
    for fam, mask in FAMILIES.items():
        sub = cv[mask(cv)]
        for metric, asc in CRITERIA.items():
            best = sub.sort_values([f"cv_{metric}", "alpha", "n_comp", "k", "s", "gamma"],
                                   ascending=[asc, True, True, True, True, True]).iloc[0]
            picks.append({"family": fam, "criterion": metric, **best.to_dict()})
    return pd.DataFrame(picks)


def run_dataset(name: str, out_root: Path = RESULTS) -> None:
    out = out_root / name
    out.mkdir(parents=True, exist_ok=True)
    ds = load(name)
    X, Y = ds.train.X, ds.train.Y
    t0 = time.time()

    # Each fold is checkpointed, so an interrupted run resumes where it stopped.
    fold_frames = []
    for fold, (tr, va) in enumerate(KFold(FOLDS, shuffle=True, random_state=SEED).split(X)):
        path = out / f"cv_fold{fold}.csv"
        if not path.exists():
            rows, _ = score_all(X[tr], Y[tr], X[va], Y[va])
            pd.DataFrame(rows).assign(fold=fold).to_csv(path, index=False)
            print(f"[{name}] fold {fold} done ({time.time() - t0:.0f}s)", flush=True)
        fold_frames.append(pd.read_csv(path))
    folds = pd.concat(fold_frames, ignore_index=True)
    g = folds.groupby(KEY, sort=False)[metrics.ALL]
    cv = g.mean().add_prefix("cv_").join(g.std().add_prefix("cv_std_")).reset_index()

    picks = select(cv)
    keep = frozenset(tuple(r) for r in picks[KEY].itertuples(index=False))
    rows, kept = score_all(X, Y, ds.test.X, ds.test.Y, keep=keep)
    test = pd.DataFrame(rows)
    grid = cv.merge(test.rename(columns={m: f"test_{m}" for m in metrics.ALL}), on=KEY)
    picks = picks.merge(test.rename(columns={m: f"test_{m}" for m in metrics.ALL}), on=KEY)

    grid.to_csv(out / "grid.csv", index=False)
    picks.to_csv(out / "selected.csv", index=False)
    np.savez_compressed(
        out / "selected_scores.npz",
        Y_test=ds.test.Y,
        **{f"{r.family}|{r.criterion}": kept[tuple(getattr(r, k) for k in KEY)] for r in picks.itertuples()},
    )
    meta = {
        "dataset": name, "n_train": len(X), "n_test": len(ds.test.X), "n_features": X.shape[1],
        "n_labels": Y.shape[1], "label_cardinality": float(Y.sum(1).mean()), "n_configs": len(grid),
        "seconds": round(time.time() - t0, 1),
    }
    (out / "meta.json").write_text(json.dumps(meta, indent=2))
    print(f"[{name}] done in {meta['seconds']}s, {len(grid)} configs", flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("datasets", nargs="*", default=DATASETS)
    for name in ap.parse_args().datasets:
        run_dataset(name)


if __name__ == "__main__":
    main()
