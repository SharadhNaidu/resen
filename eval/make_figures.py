"""README example grid: input | classical | DL only | hybrid, on unseen real photos (no GT).

    python eval/make_figures.py --model resenh-light-general
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
import resenh  # noqa: E402

EXAMPLES = ["lime_1", "dicm_3", "vv_5", "mef_2"]


def label(img: np.ndarray, text: str) -> np.ndarray:
    im = (np.clip(img, 0, 1) * 255).astype(np.uint8).copy()
    (w, h), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
    cv2.rectangle(im, (0, 0), (w + 14, h + 16), (0, 0, 0), -1)
    cv2.putText(im, text, (7, h + 7), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2,
                cv2.LINE_AA)
    return im


def fit(img: np.ndarray, h: int = 300) -> np.ndarray:
    s = h / img.shape[0]
    return cv2.resize(img, (int(img.shape[1] * s), h), interpolation=cv2.INTER_AREA)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default=None)
    ap.add_argument("--out", default="docs/figures/examples_grid.jpg")
    a = ap.parse_args()
    rows = []
    for name in EXAMPLES:
        x = fit(resenh.load_image(f"app/examples/{name}.jpg"))
        c = resenh.enhance(x, mode="classical").image
        d = resenh.enhance(x, mode="dl").image
        h = resenh.enhance(x, mode="hybrid", model=a.model).image
        rows.append(np.concatenate([label(x, "input"), label(c, "classical"),
                                    label(d, "DL only"), label(h, "resenh hybrid")], 1))
    w = min(r.shape[1] for r in rows)
    g = np.concatenate([r[:, :w] for r in rows], 0)
    cv2.imwrite(a.out, cv2.cvtColor(g, cv2.COLOR_RGB2BGR), [cv2.IMWRITE_JPEG_QUALITY, 88])
    print("wrote", a.out)


if __name__ == "__main__":
    main()
