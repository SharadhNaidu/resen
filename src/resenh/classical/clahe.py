"""Contrast-limited adaptive histogram equalisation on the luma channel only (keeps hue)."""

from __future__ import annotations

import cv2
import numpy as np


def clahe_luma(img: np.ndarray, clip_limit: float = 2.0, tiles: int = 8) -> np.ndarray:
    """Apply CLAHE to the L channel of Lab; ``img`` is RGB float in [0, 1]."""
    u8 = (np.clip(img, 0, 1) * 255.0 + 0.5).astype(np.uint8)
    lab = cv2.cvtColor(u8, cv2.COLOR_RGB2LAB)
    clahe = cv2.createCLAHE(clipLimit=float(clip_limit), tileGridSize=(tiles, tiles))
    lab[..., 0] = clahe.apply(lab[..., 0])
    return cv2.cvtColor(lab, cv2.COLOR_LAB2RGB).astype(np.float32) / 255.0
