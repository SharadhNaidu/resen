"""Edge-preserving filters implemented with OpenCV box filters (no contrib dependency)."""

from __future__ import annotations

import cv2
import numpy as np


def box(x: np.ndarray, r: int) -> np.ndarray:
    return cv2.boxFilter(x, -1, (2 * r + 1, 2 * r + 1), borderType=cv2.BORDER_REFLECT)


def guided_filter(guide: np.ndarray, src: np.ndarray, r: int, eps: float) -> np.ndarray:
    """Gray-guide guided filter (He et al., TPAMI 2013).

    ``guide`` is HxW float32; ``src`` is HxW or HxWxC float32. Works per channel of ``src``.
    """
    guide = guide.astype(np.float32)
    src = src.astype(np.float32)
    mean_i = box(guide, r)
    var_i = box(guide * guide, r) - mean_i * mean_i
    if src.ndim == 3:
        mean_i3, var_i3, guide3 = mean_i[..., None], var_i[..., None], guide[..., None]
    else:
        mean_i3, var_i3, guide3 = mean_i, var_i, guide
    mean_p = box(src, r)
    cov_ip = box(guide3 * src if src.ndim == 3 else guide * src, r) - mean_i3 * mean_p
    a = cov_ip / (var_i3 + eps)
    b = mean_p - a * mean_i3
    return box(a, r) * guide3 + box(b, r)


def fast_guided_filter(guide: np.ndarray, src: np.ndarray, r: int, eps: float,
                       scale: int = 4) -> np.ndarray:
    """Subsampled guided filter (He & Sun 2015): same result shape, ~scale^2 faster."""
    if scale <= 1 or min(guide.shape[:2]) < 8 * scale:
        return guided_filter(guide, src, r, eps)
    h, w = guide.shape[:2]
    small = (max(1, w // scale), max(1, h // scale))
    g_s = cv2.resize(guide, small, interpolation=cv2.INTER_AREA)
    p_s = cv2.resize(src, small, interpolation=cv2.INTER_AREA)
    rs = max(1, r // scale)
    mean_i = box(g_s, rs)
    var_i = box(g_s * g_s, rs) - mean_i * mean_i
    if p_s.ndim == 3:
        mean_i, var_i, g3 = mean_i[..., None], var_i[..., None], g_s[..., None]
    else:
        g3 = g_s
    mean_p = box(p_s, rs)
    a = (box(g3 * p_s, rs) - mean_i * mean_p) / (var_i + eps)
    b = mean_p - a * mean_i
    a_up = cv2.resize(box(a, rs), (w, h), interpolation=cv2.INTER_LINEAR)
    b_up = cv2.resize(box(b, rs), (w, h), interpolation=cv2.INTER_LINEAR)
    if src.ndim == 3 and a_up.ndim == 2:
        a_up, b_up = a_up[..., None], b_up[..., None]
    g = guide[..., None] if src.ndim == 3 else guide
    return a_up * g + b_up
