"""Weight registry: download on first use, verify sha256, cache locally.

Resolution order for a weight name: ``$RESENH_WEIGHTS/<file>`` -> ``./weights/**/<file>`` ->
download from ``url`` into the user cache (``~/.cache/resenh``).
"""

from __future__ import annotations

import hashlib
import os
from pathlib import Path

RELEASE = "https://huggingface.co/SharadhNaiduTrains/resenh/resolve/main"

REGISTRY: dict[str, dict] = {
    # official Retinexformer checkpoints (Cai et al., ICCV 2023, MIT licence)
    "retinexformer-lolv1": {"file": "LOL_v1.pth", "sha256": "9b4810dc51322fab70be8658ee0ad1df64a54aa814497b7849767baa2ff32cd1", "priors": False},
    "retinexformer-lolv2-real": {"file": "LOL_v2_real.pth", "sha256": "539bd16c4da6179e45616329f249c4672951b1045193428e1d042c50d4b65a0b", "priors": False},
    "retinexformer-lolv2-syn": {"file": "LOL_v2_synthetic.pth", "sha256": "2f793853c6b1ba0b4a3ff12b7b10f0f2c851eaefda1006437eb3f35229838c4a", "priors": False},
    # resenh hybrid models (filled in after training; sha256 pinned at release time)
    "resenh-light-lolv1": {"file": "resenh_light_lolv1.pth", "sha256": "17859e38b8db9a63b9011a035132d2f2ff15465b6b87d18910276a94e4b00c75", "priors": True},
    "resenh-light-lolv2-real": {"file": "resenh_light_lolv2_real.pth", "sha256": "ad15ee6b49189dfc5c363ee56ced2b16b00210fd81c8a00660399bc8a0ac6640", "priors": True},
    "resenh-light-lolv2-syn": {"file": "resenh_light_lolv2_syn.pth", "sha256": "a773a0c72fcd2b98590fb54cd3b16a2faa2736c2fb5bdcb60821e627fb30989e", "priors": True},
    "resenh-light-general": {"file": "resenh_light_general.pth", "sha256": "cf67009d2611b84dbbfde65d31c124479fdad7d1bd4e3c93e39a6f12a5c6a373", "priors": True},
}

DEFAULT_MODEL = "resenh-light-general"
FALLBACK_MODEL = "retinexformer-lolv2-real"


def _sha256(p: Path) -> str:
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def cache_dir() -> Path:
    d = Path(os.environ.get("RESENH_CACHE", Path.home() / ".cache" / "resenh"))
    d.mkdir(parents=True, exist_ok=True)
    return d


def resolve(name: str, download: bool = True) -> Path:
    if name not in REGISTRY:
        p = Path(name)
        if p.exists():
            return p
        raise KeyError(f"unknown model '{name}'; known: {sorted(REGISTRY)}")
    spec = REGISTRY[name]
    candidates = []
    if os.environ.get("RESENH_WEIGHTS"):
        candidates.append(Path(os.environ["RESENH_WEIGHTS"]) / spec["file"])
    local = Path("weights")
    if local.is_dir():
        candidates += list(local.rglob(spec["file"]))
    candidates.append(cache_dir() / spec["file"])
    for c in candidates:
        if c.exists():
            return c
    if not download:
        raise FileNotFoundError(spec["file"])
    import urllib.request  # noqa: PLC0415

    dst = cache_dir() / spec["file"]
    url = spec.get("url", f"{RELEASE}/{spec['file']}")
    tmp = dst.with_suffix(".part")
    urllib.request.urlretrieve(url, tmp)
    if spec.get("sha256") and _sha256(tmp) != spec["sha256"]:
        tmp.unlink(missing_ok=True)
        raise RuntimeError(f"checksum mismatch for {spec['file']}")
    tmp.replace(dst)
    if url.startswith(RELEASE):
        _count_download()
    return dst


def _count_download() -> None:
    """Fetch the repo's config.json once per download: it is the file Hugging Face counts
    downloads on. Best effort and silent; never blocks or fails enhancement."""
    import urllib.request  # noqa: PLC0415

    try:
        req = urllib.request.Request(f"{RELEASE}/config.json", method="HEAD",
                                     headers={"User-Agent": "resenh"})
        urllib.request.urlopen(req, timeout=5).close()
    except Exception:  # noqa: BLE001
        pass


def available(name: str) -> bool:
    try:
        resolve(name, download=False)
        return True
    except (FileNotFoundError, KeyError):
        return False
