"""Classical priors fed to the hybrid network (fixed, non-adaptive so train == test)."""

from __future__ import annotations

import numpy as np

from resenh.classical.clahe import clahe_luma
from resenh.classical.illumination import estimate_illumination, retinex_enhance

PRIOR_CHANNELS = 7  # illumination (1) + Retinex/LIME output (3) + CLAHE output (3)


def compute_priors(img: np.ndarray) -> np.ndarray:
    """Return an HxWx7 float32 stack of classical priors for an RGB float image in [0, 1]."""
    illum = estimate_illumination(img)
    lime, _ = retinex_enhance(img, illum, gamma=0.6)
    clahe = clahe_luma(lime, clip_limit=2.0)
    return np.concatenate([illum[..., None], lime, clahe], axis=2).astype(np.float32)
