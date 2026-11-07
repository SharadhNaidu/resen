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


def side_by_side(items: list[tuple[np.ndarray, str]], height: int = 480, columns: int = 0):
    """Labelled images in a row (or a grid with ``columns``), as a PIL image."""
    from PIL import Image  # noqa: PLC0415

    tiles = []
    for img, label in items:
        img = np.clip(img, 0, 1)
        s = height / img.shape[0]
        t = cv2.resize(img, (max(1, round(img.shape[1] * s)), height),
                       interpolation=cv2.INTER_AREA if s < 1 else cv2.INTER_LINEAR)
        t = (t * 255 + 0.5).astype(np.uint8)
        scale = max(0.4, height / 600)
        (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, scale, 1)
        cv2.rectangle(t, (0, 0), (tw + 12, th + 12), (0, 0, 0), -1)
        cv2.putText(t, label, (6, th + 6), cv2.FONT_HERSHEY_SIMPLEX, scale, (255, 255, 255), 1,
                    cv2.LINE_AA)
        tiles.append(t)
    cols = columns or len(tiles)
    w = max(t.shape[1] for t in tiles)
    pad = [np.pad(t, ((0, 0), (0, w - t.shape[1]), (0, 0))) if columns else t for t in tiles]
    rows = []
    for i in range(0, len(pad), cols):
        row = pad[i:i + cols]
        if columns and len(row) < cols:
            row += [np.zeros_like(pad[0])] * (cols - len(row))
        gap = np.full((height, 4, 3), 255, np.uint8)
        rows.append(np.concatenate([x for t in row for x in (t, gap)][:-1], 1))
    if len(rows) > 1:
        hgap = np.full((4, rows[0].shape[1], 3), 255, np.uint8)
        grid = np.concatenate([x for r in rows for x in (r, hgap)][:-1], 0)
    else:
        grid = rows[0]
    return Image.fromarray(grid)
