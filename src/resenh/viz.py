"""Turn pipeline intermediates into displayable RGB panels."""

from __future__ import annotations

import cv2
import numpy as np

from resenh.analysis import luma

LABELS = {
    "denoised": "1. Denoised input (edge-preserving, before gain)",
    "illumination": "2. Illumination map (max-RGB + guided filter)",
    "reflectance": "3. Reflectance R = I / L",
    "gain_map": "4. Per-pixel gain (Retinex)",
    "retinex": "5. Retinex / LIME enhancement",
    "post_denoised": "6. Noise-adaptive post-denoise",
    "clahe": "7. CLAHE on luma",
    "classical": "8. Classical branch output",
    "network": "9. Network output (hybrid Retinexformer)",
}


def heatmap(x: np.ndarray, cmap=cv2.COLORMAP_INFERNO) -> np.ndarray:
    u8 = (np.clip(x, 0, 1) * 255).astype(np.uint8)
    return cv2.cvtColor(cv2.applyColorMap(u8, cmap), cv2.COLOR_BGR2RGB).astype(np.float32) / 255


def noise_map(img: np.ndarray) -> np.ndarray:
    """High-frequency residual magnitude (what a denoiser would remove), contrast-stretched."""
    y = luma(img)
    hp = np.abs(y - cv2.GaussianBlur(y, (0, 0), 1.5))
    return heatmap(hp / (np.percentile(hp, 99) + 1e-6), cv2.COLORMAP_VIRIDIS)


def panels(intermediates: dict, inp: np.ndarray, out: np.ndarray) -> list[tuple[np.ndarray, str]]:
    items = [(inp, "0. Input")]
    for key, label in LABELS.items():
        if key not in intermediates:
            continue
        v = intermediates[key]
        if v.ndim == 2:
            v = heatmap(v)
        items.append((np.clip(v, 0, 1), label))
    items.append((noise_map(inp), "Noise map (input high-pass)"))
    items.append((out, "Final output"))
    return items
