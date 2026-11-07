"""Colour and tone stages: gray-world white balance, exposure normalisation, highlight protection."""

from __future__ import annotations

import numpy as np

from resenh.analysis import luma


def gray_world(img: np.ndarray, strength: float = 1.0) -> np.ndarray:
    """Partial gray-world white balance (``strength`` 0 = off, 1 = full)."""
    if strength <= 0:
        return img
    means = img.reshape(-1, 3).mean(axis=0) + 1e-6
    gains = means.mean() / means
    gains = 1.0 + strength * (gains - 1.0)
    return np.clip(img * gains, 0.0, 1.0).astype(np.float32)


def match_exposure(img: np.ndarray, target_mean: float, max_gain: float = 4.0) -> np.ndarray:
    """Scale so mean luma approaches ``target_mean`` with a soft shoulder (no hard clipping)."""
    m = float(luma(img).mean())
    if m <= 1e-4:
        return img
    g = float(np.clip(target_mean / m, 1.0 / max_gain, max_gain))
    x = img * g
    # filmic shoulder above 0.8 to avoid blown highlights
    knee = 0.8
    over = x > knee
    x = np.where(over, knee + (1 - knee) * np.tanh((x - knee) / (1 - knee)), x)
    return np.clip(x, 0.0, 1.0).astype(np.float32)


def protect_highlights(enhanced: np.ndarray, original: np.ndarray,
                       strength: float) -> np.ndarray:
    """Blend back the original where it was already bright (lamps, signs), to stop blow-out."""
    if strength <= 0:
        return enhanced
    y = luma(original)
    w = np.clip((y - 0.6) / 0.35, 0.0, 1.0)[..., None] * strength
    return (enhanced * (1 - w) + np.maximum(enhanced, original) * w).astype(np.float32)
