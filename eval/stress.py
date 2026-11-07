"""Stress tests: harder variants of the LOL-v2-Real test inputs, paired with the original GT.

* extreme_dark: input × 0.35 (about 1.5 stops darker), plus matching shot noise
* nonuniform:   input × a radial falloff (1.0 centre → 0.25 corners), like a single light source
* noisy:        input + Gaussian read noise σ=0.02 (on [0,1]) + Poisson-like shot noise

The degradation is seeded per image, so every method sees identical inputs.

    python eval/stress.py --methods classical "dl:retinexformer-lolv2-real" \
        "hybrid:runs/real_light/last.pth"
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from run_eval import run_method  # noqa: E402

from resenh import metrics as M  # noqa: E402
from resenh.io import load_image  # noqa: E402


def degrade(x: np.ndarray, kind: str, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    if kind == "extreme_dark":
        y = x * 0.35
        y = y + rng.normal(0, 1, y.shape) * np.sqrt(np.maximum(y, 0) * 0.0005)
    elif kind == "nonuniform":
        h, w = x.shape[:2]
        yy, xx = np.mgrid[0:h, 0:w]
        r = np.sqrt(((yy - h / 2) / (h / 2)) ** 2 + ((xx - w / 2) / (w / 2)) ** 2) / np.sqrt(2)
        y = x * (1.0 - 0.75 * r)[..., None]
    elif kind == "noisy":
        y = x + rng.normal(0, 0.02, x.shape) + rng.normal(0, 1, x.shape) * np.sqrt(
            np.maximum(x, 0) * 0.002)
    else:
        raise ValueError(kind)
    return np.clip(y, 0, 1).astype(np.float32)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--methods", nargs="+", required=True)
    ap.add_argument("--bench", default="lolv2_real")
    ap.add_argument("--kinds", nargs="+", default=["extreme_dark", "nonuniform", "noisy"])
    ap.add_argument("--n", type=int, default=50)
    ap.add_argument("--out", default="outputs/stress.json")
    a = ap.parse_args()
    man = json.loads(Path("data/manifest.json").read_text())
    pairs = man[a.bench]["test"][: a.n]
    res = json.loads(Path(a.out).read_text()) if Path(a.out).exists() else {}
    for kind in a.kinds:
        for m in a.methods:
            key = f"{kind}|{m}"
            if key in res:
                continue
            ps, ss, nr, ep = [], [], [], []
            for i, (lo, hi) in enumerate(pairs):
                x = degrade(load_image(lo), kind, i)
                y = load_image(hi)
                p = run_method(m, x, a.bench, 0, False)
                ps.append(M.psnr(p, y)), ss.append(M.ssim(p, y))
                nr.append(M.noise_ratio(p, y)), ep.append(M.edge_preservation(p, y))
            res[key] = {"psnr": float(np.mean(ps)), "ssim": float(np.mean(ss)),
                        "noise_ratio": float(np.mean(nr)), "epi": float(np.mean(ep)), "n": len(ps)}
            print(f"{kind:13s} {m:45s} PSNR {res[key]['psnr']:.2f} SSIM {res[key]['ssim']:.4f} "
                  f"noise {res[key]['noise_ratio']:.2f} EPI {res[key]['epi']:.3f}", flush=True)
            Path(a.out).write_text(json.dumps(res, indent=1))


if __name__ == "__main__":
    main()
