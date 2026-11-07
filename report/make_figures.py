"""Photographic figures for the technical report (report/fig/*.jpg). Run from the repo root."""
import json
from pathlib import Path

import numpy as np
from PIL import Image

import resenh
from resenh.io import load_image, to_uint8

OUT = Path("report/fig")
OUT.mkdir(parents=True, exist_ok=True)
H = 360


def save(img, name, h=H):
    im = Image.fromarray(to_uint8(img) if img.dtype != np.uint8 else img)
    if im.height != h:
        im = im.resize((round(im.width * h / im.height), h), Image.LANCZOS)
    im.save(OUT / f"{name}.jpg", quality=90)


def gray3(x):
    x = np.asarray(x, np.float32)
    if x.ndim == 2:
        x = np.repeat(x[..., None], 3, 2)
    return np.clip(x / max(x.max(), 1e-6), 0, 1)


# 1) three-way comparison on a paired LOL-v2-Real test image (with ground truth)
m = json.load(open("data/manifest.json"))
pairs = m["lolv2_real"]["test"]
low, gt = pairs[7]
x = load_image(low)
save(x, "pair_input"); save(load_image(gt), "pair_gt")
for mode in ("classical", "dl", "hybrid"):
    save(resenh.enhance(x, mode=mode, preset="benchmark").image, f"pair_{mode}")

# 2) unseen real photos: extreme dark (LIME) and backlit / bright-sky (VV)
for ex in ("lime_1", "vv_5", "dicm_3"):
    x = load_image(f"site/public/examples/{ex}.jpg")
    save(x, f"{ex}_input")
    for mode in ("classical", "dl", "hybrid"):
        save(resenh.enhance(x, mode=mode).image, f"{ex}_{mode}")

# 3) intermediate states for one photo
r = resenh.enhance(load_image("site/public/examples/lime_1.jpg"), mode="hybrid")
for k, v in r.intermediates.items():
    print("stage", k, getattr(v, "shape", None))
    if isinstance(v, np.ndarray):
        save(v if v.ndim == 3 and v.shape[2] == 3 else gray3(v), f"stage_{k}")
print(json.dumps(r.decision, indent=1, default=float)[:800])
