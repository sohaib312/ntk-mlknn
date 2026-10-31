"""Download the seven MULAN benchmarks used in the experiments into data/<name>/.

The archives are RAR files hosted by the MULAN project on SourceForge. They are unpacked
with `tar` (bsdtar, which ships with Windows 10+ and macOS; on Linux install
libarchive-tools) or, failing that, `unrar`. Checksums pin the exact files the results
in results/ were produced from.

Usage: python scripts/download_data.py [name ...]
"""

from __future__ import annotations

import hashlib
import shutil
import subprocess
import sys
import urllib.request
from pathlib import Path

URL = "https://sourceforge.net/projects/mulan/files/datasets/{name}.rar/download"
SHA256 = {
    "emotions": "0501053829a00b3c368b150b55bbbfcd2ddb153b8eba788021b521e764c4739f",
    "scene": "8dae3b1bb244544ff478dd85980fc92d8e6472b4b68dbc2fc915ceeb4d5f1bcc",
    "yeast": "d795520e33e4c8e3449359f9d39e20cf194da76c18c423f6f779142e15e6a684",
    "medical": "7b773528f58a918dc82c1f71cc30c036949f691fd6c4ad02e836448f511dbfa1",
    "enron": "36a1f9502be3aaea600d0b5ce834595b21b72165d66b4c5b81d289caff34732a",
    "corel5k": "1f2d37ce72febf224ea04a89129ceadf800ee3e4ec8057deeff4bdc48783a528",
    "bibtex": "549d7be574352764d410043663d2d1747653ec3a1e9378c4b27412b9505aa310",
}
DATA = Path(__file__).resolve().parents[1] / "data"


def extract(archive: Path, dest: Path) -> None:
    for cmd in (["tar", "-xf", str(archive), "-C", str(dest)], ["unrar", "x", "-o+", str(archive), str(dest) + "/"]):
        if shutil.which(cmd[0]) and subprocess.run(cmd, capture_output=True).returncode == 0:
            return
    sys.exit(f"Could not unpack {archive}: install bsdtar (libarchive-tools) or unrar.")


def fetch(name: str) -> None:
    dest = DATA / name
    if any(dest.glob("*-train.arff")):
        print(f"{name}: already present")
        return
    dest.mkdir(parents=True, exist_ok=True)
    archive = DATA / f"{name}.rar"
    print(f"{name}: downloading")
    with urllib.request.urlopen(URL.format(name=name), timeout=120) as r:
        archive.write_bytes(r.read())
    digest = hashlib.sha256(archive.read_bytes()).hexdigest()
    if digest != SHA256[name]:
        sys.exit(f"{name}: checksum mismatch ({digest}); the hosted file has changed.")
    extract(archive, dest)
    archive.unlink()
    print(f"{name}: ok")


if __name__ == "__main__":
    for n in sys.argv[1:] or SHA256:
        fetch(n)
