"""Index the raw datasets, verify split counts, build val splits and dedup lists.

Outputs ``data/manifest.json`` with one entry per benchmark::

    {"lolv1": {"train": [[low, high], ...], "val_a": [...], "val_b": [...], "test": [...]}, ...}

Paths are relative to the repo root. Val splits are carved from the official train split only
(never from test) and are grouped by scene so that multiple exposures of one scene never straddle
train and val. Also writes ``data/dedup.json``: train images of each benchmark whose ground truth
is a near-duplicate (perceptual hash) of a test image of another LOL benchmark. These must be
excluded whenever a model is trained on one benchmark and evaluated on another (generalist model).

Usage::

    python training/data_prep.py --root data/raw --out data
"""

from __future__ import annotations

import argparse
import hashlib
import json
import random
import re
from collections import defaultdict
from pathlib import Path

IMG_EXT = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}

EXPECTED = {
    "lolv1": {"train": 485, "test": 15},
    "lolv2_real": {"train": 689, "test": 100},
    "lolv2_syn": {"train": 900, "test": 100},
}
VAL_A_FRAC = 0.06  # checkpoint selection
VAL_B_FRAC = 0.04  # controller / ensemble fitting
SEED = 2026


def _images(d: Path) -> list[Path]:
    return sorted(p for p in d.iterdir() if p.suffix.lower() in IMG_EXT)


def _find_dir(root: Path, *names: str) -> Path:
    """Find a directory whose path ends with the given components (case-insensitive)."""
    want = [n.lower() for n in names]
    for p in root.rglob("*"):
        if p.is_dir() and [x.lower() for x in p.parts[-len(want):]] == want:
            return p
    raise FileNotFoundError(f"no directory matching {'/'.join(names)} under {root}")


def _pair(low_dir: Path, high_dir: Path) -> list[tuple[Path, Path]]:
    lows, highs = _images(low_dir), _images(high_dir)
    hmap = {h.stem.lower(): h for h in highs}
    pairs = []
    for lo in lows:
        key = lo.stem.lower()
        cand = hmap.get(key) or hmap.get(key.replace("low", "normal")) or hmap.get(
            re.sub(r"^low", "", key))
        if cand is None:  # LOL-v2 names: low00001 <-> normal00001
            digits = re.sub(r"\D", "", key)
            cand = next((h for k, h in hmap.items() if re.sub(r"\D", "", k) == digits), None)
        if cand is None:
            raise ValueError(f"no GT for {lo}")
        pairs.append((lo, cand))
    return pairs


def _scene_id(bench: str, low: Path) -> str:
    # LOL-v2-Real stores several exposures per scene with consecutive ids; group in blocks
    # of 10 ids (conservative: may merge two scenes, never splits one). Others are 1:1.
    num = int(re.sub(r"\D", "", low.stem) or 0)
    return f"{num // 10}" if bench == "lolv2_real" else low.stem


def _split_val(bench: str, train: list[tuple[Path, Path]]):
    groups = defaultdict(list)
    for p in train:
        groups[_scene_id(bench, p[0])].append(p)
    keys = sorted(groups)
    random.Random(SEED).shuffle(keys)
    n = len(train)
    val_a, val_b, rest = [], [], []
    for k in keys:
        if len(val_a) < VAL_A_FRAC * n:
            val_a += groups[k]
        elif len(val_b) < VAL_B_FRAC * n:
            val_b += groups[k]
        else:
            rest += groups[k]
    return sorted(rest), sorted(val_a), sorted(val_b)


def _phash(path: Path) -> str:
    import imagehash
    from PIL import Image

    with Image.open(path) as im:
        return str(imagehash.phash(im.convert("L"), hash_size=16))


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()[:16]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default="data/raw")
    ap.add_argument("--out", default="data")
    ap.add_argument("--dedup-threshold", type=int, default=12, help="max phash hamming distance")
    args = ap.parse_args()
    root, out = Path(args.root), Path(args.out)
    repo = Path.cwd()

    lol = root / "lol"
    raw = {
        "lolv1": {
            "train": _pair(_find_dir(lol, "LOLv1", "Train", "Low"),
                           _find_dir(lol, "LOLv1", "Train", "Normal")),
            "test": _pair(_find_dir(lol, "LOLv1", "Test", "Low"),
                          _find_dir(lol, "LOLv1", "Test", "Normal")),
        },
        "lolv2_real": {
            "train": _pair(_find_dir(lol, "Real_captured", "Train", "Low"),
                           _find_dir(lol, "Real_captured", "Train", "Normal")),
            "test": _pair(_find_dir(lol, "Real_captured", "Test", "Low"),
                          _find_dir(lol, "Real_captured", "Test", "Normal")),
        },
        "lolv2_syn": {
            "train": _pair(_find_dir(lol, "Synthetic", "Train", "Low"),
                           _find_dir(lol, "Synthetic", "Train", "Normal")),
            "test": _pair(_find_dir(lol, "Synthetic", "Test", "Low"),
                          _find_dir(lol, "Synthetic", "Test", "Normal")),
        },
    }
    for bench, exp in EXPECTED.items():
        for split, n in exp.items():
            got = len(raw[bench][split])
            if got != n:
                raise SystemExit(f"{bench}/{split}: expected {n} pairs, found {got}")
            print(f"ok {bench}/{split}: {got}")

    # LSRW (optional, used for the generalist model)
    lsrw_root = root / "lsrw"
    try:
        _find_dir(lsrw_root, "Training data", "Huawei")
        has_lsrw = True
    except FileNotFoundError:
        has_lsrw = False
        print("LSRW not found (or still downloading): skipped")
    if has_lsrw:
        train = []
        for cam in ("Huawei", "Nikon"):
            d = _find_dir(lsrw_root, "Training data", cam)
            train += _pair(d / "low", d / "high")
        test = []
        for cam in ("Huawei", "Nikon"):
            d = _find_dir(lsrw_root, "Eval", cam)
            test += _pair(d / "low", d / "high")
        raw["lsrw"] = {"train": train, "test": test}
        print(f"ok lsrw: train {len(train)} test {len(test)}")

    rel = lambda p: p.resolve().relative_to(repo.resolve()).as_posix()  # noqa: E731
    manifest = {}
    for bench, splits in raw.items():
        tr, va, vb = _split_val(bench, splits["train"])
        manifest[bench] = {k: [[rel(a), rel(b)] for a, b in v]
                           for k, v in {"train": tr, "val_a": va, "val_b": vb,
                                        "test": splits["test"]}.items()}
        print(f"{bench}: train {len(tr)} val_a {len(va)} val_b {len(vb)} "
              f"test {len(splits['test'])}")

    # unpaired sets
    up = root / "unpaired"
    if up.exists():
        manifest["unpaired"] = {}
        for name in ("DICM", "LIME", "MEF", "NPE", "VV"):
            try:
                d = _find_dir(up, name)
            except FileNotFoundError:
                continue
            manifest["unpaired"][name] = [rel(p) for p in _images(d)]
            print(f"ok unpaired/{name}: {len(manifest['unpaired'][name])}")

    # cross-benchmark dedup against every LOL test GT
    test_hashes = {}
    for bench in ("lolv1", "lolv2_real", "lolv2_syn"):
        for _, gt in raw[bench]["test"]:
            test_hashes[rel(gt)] = _phash(gt)
    import imagehash

    th = {k: imagehash.hex_to_hash(v) for k, v in test_hashes.items()}
    dedup = {}
    for bench, splits in raw.items():
        bad = []
        for low, gt in splits["train"]:
            h = imagehash.hex_to_hash(_phash(gt))
            if any(h - t <= args.dedup_threshold for t in th.values()):
                bad.append(rel(low))
        dedup[bench] = bad
        print(f"dedup {bench}: {len(bad)} train images near-duplicate a LOL test GT")

    out.mkdir(parents=True, exist_ok=True)
    (out / "manifest.json").write_text(json.dumps(manifest, indent=1))
    (out / "dedup.json").write_text(json.dumps(dedup, indent=1))
    checks = {rel(p): _sha(p) for b in ("lolv1", "lolv2_real", "lolv2_syn")
              for pair in raw[b]["test"] for p in pair}
    (out / "test_sha256.json").write_text(json.dumps(checks, indent=1))
    print("wrote", out / "manifest.json", out / "dedup.json")


if __name__ == "__main__":
    main()
