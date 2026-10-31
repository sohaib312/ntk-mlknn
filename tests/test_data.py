import pytest

from ntk_mlknn.data import DATA_DIR, load

# (train, test, features, labels) of MULAN's standard splits
EXPECTED = {
    "emotions": (391, 202, 72, 6),
    "scene": (1211, 1196, 294, 6),
    "yeast": (1500, 917, 103, 14),
    "medical": (333, 645, 1449, 45),
    "enron": (1123, 579, 1001, 53),
    "corel5k": (4500, 500, 499, 374),
    "bibtex": (4880, 2515, 1836, 159),
}


@pytest.mark.parametrize("name", EXPECTED)
def test_standard_split_shapes(name):
    if not (DATA_DIR / name).exists():
        pytest.skip(f"run scripts/download_data.py first ({DATA_DIR / name} missing)")
    ds = load(name)
    n_tr, n_te, d, L = EXPECTED[name]
    assert ds.train.X.shape == (n_tr, d) and ds.test.X.shape == (n_te, d)
    assert ds.train.Y.shape == (n_tr, L) and ds.test.Y.shape == (n_te, L)
    assert set(ds.train.Y.ravel()) <= {0, 1}
