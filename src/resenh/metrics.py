"""Evaluation metrics, implemented to match the conventions of the LLIE literature.

* PSNR: RGB, on the uint8-quantised output (what is actually saved), peak 1.0.
* SSIM: MATLAB-style (Wang et al. 2004) as in the Retinexformer / HVI-CIDNet eval scripts:
  11x11 Gaussian window, sigma 1.5, C1=(0.01*255)^2, C2=(0.03*255)^2, valid region,
  computed per RGB channel on [0,255] and averaged.
* LPIPS: AlexNet backbone (``lpips`` package), inputs scaled to [-1, 1].
* NIQE: ``pyiqa`` ``niqe_matlab`` (MATLAB-calibrated), Y channel.
* Noise ratio: Immerkaer sigma of the output divided by that of the reference (paired) or of
  the gain-adjusted input (unpaired). Lower = better noise suppression; ~1 matches reference.
* EPI (edge preservation index): correlation of high-pass (Laplacian) luma between output and
  reference. 1 = edges identical to reference.
"""

from __future__ import annotations

import time
from functools import lru_cache

import cv2
import numpy as np

from resenh.analysis import luma, noise_sigma


def _u8(img: np.ndarray) -> np.ndarray:
    return (np.clip(img, 0, 1) * 255.0 + 0.5).astype(np.uint8)


def psnr(pred: np.ndarray, gt: np.ndarray) -> float:
    p = _u8(pred).astype(np.float64) / 255.0
    g = _u8(gt).astype(np.float64) / 255.0
    mse = np.mean((p - g) ** 2)
    return float("inf") if mse == 0 else float(10 * np.log10(1.0 / mse))


def _ssim_channel(a: np.ndarray, b: np.ndarray) -> float:
    c1, c2 = (0.01 * 255) ** 2, (0.03 * 255) ** 2
    k = cv2.getGaussianKernel(11, 1.5)
    w = np.outer(k, k.transpose())
    mu1 = cv2.filter2D(a, -1, w)[5:-5, 5:-5]
    mu2 = cv2.filter2D(b, -1, w)[5:-5, 5:-5]
    s1 = cv2.filter2D(a ** 2, -1, w)[5:-5, 5:-5] - mu1 ** 2
    s2 = cv2.filter2D(b ** 2, -1, w)[5:-5, 5:-5] - mu2 ** 2
    s12 = cv2.filter2D(a * b, -1, w)[5:-5, 5:-5] - mu1 * mu2
    m = ((2 * mu1 * mu2 + c1) * (2 * s12 + c2)) / ((mu1 ** 2 + mu2 ** 2 + c1) * (s1 + s2 + c2))
    return float(m.mean())


def ssim(pred: np.ndarray, gt: np.ndarray) -> float:
    p = _u8(pred).astype(np.float64)
    g = _u8(gt).astype(np.float64)
    return float(np.mean([_ssim_channel(p[..., c], g[..., c]) for c in range(3)]))


@lru_cache(maxsize=2)
def _lpips_model(device: str):
    import lpips  # noqa: PLC0415

    return lpips.LPIPS(net="alex", verbose=False).to(device).eval()


def lpips_score(pred: np.ndarray, gt: np.ndarray, device: str | None = None) -> float:
    import torch  # noqa: PLC0415

    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    m = _lpips_model(device)
    t = lambda a: (torch.from_numpy(_u8(a).astype(np.float32) / 127.5 - 1.0)  # noqa: E731
                   .permute(2, 0, 1)[None].to(device))
    with torch.no_grad():
        return float(m(t(pred), t(gt)).item())


@lru_cache(maxsize=2)
def _niqe_model(device: str):
    import pyiqa  # noqa: PLC0415

    return pyiqa.create_metric("niqe_matlab", device=device)


def niqe(img: np.ndarray, device: str | None = None) -> float:
    import torch  # noqa: PLC0415

    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    t = torch.from_numpy(_u8(img).astype(np.float32) / 255.0).permute(2, 0, 1)[None].to(device)
    with torch.no_grad():
        return float(_niqe_model(device)(t).item())


def noise_ratio(pred: np.ndarray, ref: np.ndarray) -> float:
    """sigma(pred)/sigma(ref) on luma. For unpaired use, pass ref = gain-matched input."""
    return float(noise_sigma(luma(pred)) / max(noise_sigma(luma(ref)), 1e-6))


def edge_preservation(pred: np.ndarray, ref: np.ndarray) -> float:
    """EPI: Pearson correlation of Laplacian-filtered luma (Sattar et al. 1997 style)."""
    lp = cv2.Laplacian(luma(pred).astype(np.float32), cv2.CV_32F, ksize=3)
    lr = cv2.Laplacian(luma(ref).astype(np.float32), cv2.CV_32F, ksize=3)
    lp, lr = lp - lp.mean(), lr - lr.mean()
    den = np.sqrt((lp ** 2).sum() * (lr ** 2).sum())
    return float((lp * lr).sum() / den) if den > 0 else 0.0


def gain_matched(inp: np.ndarray, out: np.ndarray) -> np.ndarray:
    """Input scaled to the output's mean brightness (reference for unpaired noise/edge)."""
    g = luma(out).mean() / max(luma(inp).mean(), 1e-4)
    return np.clip(inp * g, 0, 1)


def bootstrap_ci(values, n: int = 2000, alpha: float = 0.05, seed: int = 0):
    v = np.asarray(values, dtype=np.float64)
    rng = np.random.default_rng(seed)
    means = v[rng.integers(0, len(v), (n, len(v)))].mean(axis=1)
    return float(np.quantile(means, alpha / 2)), float(np.quantile(means, 1 - alpha / 2))


class Timer:
    def __init__(self, cuda_sync: bool = False):
        self.cuda_sync = cuda_sync

    def __enter__(self):
        self._sync()
        self.t = time.perf_counter()
        return self

    def __exit__(self, *exc):
        self._sync()
        self.elapsed = time.perf_counter() - self.t

    def _sync(self):
        if self.cuda_sync:
            import torch  # noqa: PLC0415

            if torch.cuda.is_available():
                torch.cuda.synchronize()
