"""Illumination estimation and Retinex-style decomposition.

``estimate_illumination`` follows LIME (Guo et al., TIP 2017): the initial map is the per-pixel
max over RGB, refined with a structure-aware smoother (here a guided filter, which keeps strong
edges and flattens texture). ``retinex_enhance`` divides the image by the gamma-adjusted
illumination, so dark regions are lifted more than bright ones (non-uniform lighting handled
per pixel).
"""

from __future__ import annotations

import numpy as np

from resenh.classical.filters import fast_guided_filter


def estimate_illumination(img: np.ndarray, radius: int | None = None,
                          eps: float = 1e-3) -> np.ndarray:
    """Return a smooth HxW illumination map in (0, 1] for an RGB float image in [0, 1]."""
    t0 = img.max(axis=2)
    h, w = t0.shape
    r = radius or max(4, int(round(min(h, w) / 40)))
    t = fast_guided_filter(t0, t0, r, eps)
    return np.clip(t, 1e-3, 1.0).astype(np.float32)


def retinex_enhance(img: np.ndarray, illum: np.ndarray | None = None, gamma: float = 0.6,
                    floor: float = 0.04) -> tuple[np.ndarray, np.ndarray]:
    """LIME-style enhancement. Returns (enhanced RGB, illumination actually applied).

    ``gamma`` < 1 compresses the illumination map before division: smaller gamma = stronger
    lift. ``floor`` bounds the gain at 1/floor to stop noise blow-up in near-black regions.
    """
    if illum is None:
        illum = estimate_illumination(img)
    t = np.maximum(illum, floor) ** gamma
    out = img / t[..., None]
    return np.clip(out, 0.0, 1.0).astype(np.float32), t.astype(np.float32)


def reflectance(img: np.ndarray, illum: np.ndarray, floor: float = 0.04) -> np.ndarray:
    """Retinex reflectance R = I / L (clipped), for visualisation and priors."""
    return np.clip(img / np.maximum(illum, floor)[..., None], 0.0, 1.0).astype(np.float32)
