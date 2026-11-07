"""Convenience helpers: folders of images, environment report, weight pre-download."""

from __future__ import annotations

import platform
import sys
from pathlib import Path

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}


def enhance_folder(src, dst, suffix: str = "_resenh", ext: str = ".png", progress: bool = True,
                   **kwargs) -> list[Path]:
    """Enhance every image in folder ``src`` and write the results to folder ``dst``.

    ``kwargs`` go to :func:`resenh.enhance` (``mode``, ``device``, ...). An image that cannot
    be read is skipped with a warning. Returns the paths that were written.
    """
    import warnings  # noqa: PLC0415

    from resenh.io import ImageInputError  # noqa: PLC0415
    from resenh.pipeline import enhance  # noqa: PLC0415

    src, dst = Path(src), Path(dst)
    files = sorted(p for p in src.iterdir() if p.suffix.lower() in IMAGE_EXTS)
    if not files:
        raise FileNotFoundError(f"no images found in {src}")
    it = files
    if progress:
        try:
            from tqdm import tqdm  # noqa: PLC0415

            it = tqdm(files, desc="resenh", unit="img")
        except ImportError:
            pass
    written = []
    for f in it:
        try:
            res = enhance(f, **kwargs)
        except ImageInputError as e:
            warnings.warn(f"skipped {f.name}: {e}", stacklevel=2)
            continue
        written.append(res.save(dst / f"{f.stem}{suffix}{ext}"))
    return written


def download_weights(names: list[str] | None = None) -> dict[str, Path]:
    """Fetch (and checksum) model weights now, e.g. before going offline. Returns their paths.

    By default this downloads the models that ``resenh.enhance`` uses.
    """
    from resenh.models.weights import DEFAULT_MODEL, FALLBACK_MODEL, REGISTRY, resolve  # noqa: PLC0415

    if names is None:
        names = [n for n in (DEFAULT_MODEL, FALLBACK_MODEL) if REGISTRY[n].get("sha256")]
    return {n: resolve(n) for n in names}


def info(print_it: bool = True) -> dict:
    """Report versions, the compute device and cached weights (handy for bug reports)."""
    import numpy as np  # noqa: PLC0415

    import resenh  # noqa: PLC0415
    from resenh.models.weights import REGISTRY, available, cache_dir  # noqa: PLC0415

    d = {"resenh": resenh.__version__, "python": sys.version.split()[0],
         "platform": platform.platform(), "numpy": np.__version__}
    try:
        import cv2  # noqa: PLC0415

        d["opencv"] = cv2.__version__
    except ImportError:
        d["opencv"] = None
    try:
        import torch  # noqa: PLC0415

        from resenh.models.runner import pick_device  # noqa: PLC0415

        d["torch"] = torch.__version__
        d["device"] = pick_device("auto")
        if d["device"] == "cuda":
            d["gpu"] = torch.cuda.get_device_name(0)
    except ImportError:
        d["torch"] = None
        d["device"] = "cpu (classical only: pip install 'resenh[dl]' for the network)"
    d["weights_cache"] = str(cache_dir())
    d["models_available"] = [n for n in REGISTRY if available(n)]
    if print_it:
        w = max(len(k) for k in d)
        for k, v in d.items():
            print(f"{k:<{w}}  {v}")
    return d
