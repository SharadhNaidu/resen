"""Edge-preserving denoising: guided filter (self-guided, per channel) or bilateral."""

from __future__ import annotations

import cv2
import numpy as np

from resenh.classical.filters import guided_filter


def denoise(img: np.ndarray, strength: float, method: str = "guided") -> np.ndarray:
    """Denoise an RGB float image. ``strength`` in [0, 1] (0 returns the input unchanged).

    Guided filtering uses the luma as guide, so colour noise is smoothed along luma edges
    (edges preserved); bilateral is offered for comparison/ablation.
    """
    if strength <= 1e-3:
        return img
    if method == "bilateral":
        sigma_c = 0.02 + 0.10 * strength
        d = 5 if strength < 0.5 else 9
        return cv2.bilateralFilter(img.astype(np.float32), d, sigma_c, d).clip(0, 1)
    guide = img @ np.array([0.299, 0.587, 0.114], dtype=np.float32)
    r = 2 if strength < 0.5 else 3
    eps = (0.01 + 0.04 * strength) ** 2
    out = guided_filter(guide, img, r, eps)
    return np.clip(out, 0.0, 1.0).astype(np.float32)


def denoise_measured(img: np.ndarray, k: float = 2.5) -> np.ndarray:
    """Noise-adaptive guided filtering: eps tied to the *measured* noise variance.

    Regions whose local variance is well above (k*sigma)^2 are edges and are kept; flatter
    regions are averaged. Chroma gets a stronger pass than luma (colour noise is the most
    visible artefact after brightening dark images).
    """
    from resenh.analysis import noise_sigma  # noqa: PLC0415

    y = img @ np.array([0.299, 0.587, 0.114], dtype=np.float32)
    sigma = noise_sigma(y)
    if sigma < 0.004:
        return img
    eps = (k * sigma) ** 2
    r = 2 if sigma < 0.03 else 3
    y_d = guided_filter(y, y, r, eps)
    chroma = img - y[..., None]
    chroma_d = guided_filter(y_d, chroma, r + 2, eps)
    return np.clip(y_d[..., None] + chroma_d, 0.0, 1.0).astype(np.float32)
