"""Load MULAN multi-label benchmarks with their standard train/test splits.

The datasets come from the MULAN repository (https://mulan.sourceforge.net/datasets-mlc.html):
each `<name>.rar` holds `<name>-train.arff`, `<name>-test.arff` and `<name>.xml`, whose
`<label name=...>` entries say which ARFF attributes are labels. `scripts/download_data.py`
fetches and unpacks them into `data/<name>/` (or `$MULAN_DATA_DIR/<name>/`).
"""

from __future__ import annotations

import os
import xml.etree.ElementTree as ET
from dataclasses import dataclass
from pathlib import Path

import arff  # liac-arff: reads both dense and sparse ARFF rows
import numpy as np

DATA_DIR = Path(os.environ.get("MULAN_DATA_DIR", Path(__file__).resolve().parents[1] / "data"))

DATASETS = ["emotions", "scene", "yeast", "medical", "enron", "corel5k", "bibtex"]
# MULAN capitalises some file names inside the archives
FILE_STEM = {"corel5k": "Corel5k"}


@dataclass
class Split:
    X: np.ndarray  # (n, d) float64 features
    Y: np.ndarray  # (n, L) int8 labels in {0, 1}


@dataclass
class Dataset:
    name: str
    train: Split
    test: Split
    label_names: list[str]


def _label_names(xml_path: Path) -> list[str]:
    root = ET.parse(xml_path).getroot()
    return [el.attrib["name"] for el in root.iter() if el.tag.endswith("label")]


def _read_arff(path: Path, labels: list[str]) -> Split:
    # Rows are streamed one at a time (sparse rows densified, nominals as strings), so the
    # larger files never exist as millions of Python objects at once.
    with open(path, encoding="utf-8", errors="replace") as f:
        raw = arff.load(f, return_type=arff.DENSE_GEN)
        names = [a[0] for a in raw["attributes"]]
        data = np.stack([np.array([float(v) for v in row]) for row in raw["data"]])
    label_idx = [names.index(n) for n in labels]
    is_label = np.zeros(len(names), dtype=bool)
    is_label[label_idx] = True
    return Split(X=data[:, ~is_label], Y=data[:, label_idx].astype(np.int8))


def load(name: str, data_dir: Path = DATA_DIR) -> Dataset:
    """Load `name`'s standard MULAN train/test split from `data_dir/<name>/`."""
    folder = Path(data_dir) / name
    stem = FILE_STEM.get(name, name)
    labels = _label_names(folder / f"{stem}.xml")
    return Dataset(
        name=name,
        train=_read_arff(folder / f"{stem}-train.arff", labels),
        test=_read_arff(folder / f"{stem}-test.arff", labels),
        label_names=labels,
    )
