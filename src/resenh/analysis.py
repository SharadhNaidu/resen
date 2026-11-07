"""Image analyzer: measures the input and maps the measurements to enhancement parameters.

This is what makes the pipeline adaptive (no fixed parameters): every classical stage and the
controller read their settings from :class:`Params`, computed per image from :class:`Stats`.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass

import cv2
import numpy as np

LUMA = np.array([0.299, 0.587, 0.114], dtype=np.float32)


def luma(img: np.ndarray) -> np.ndarray:
    return img @ LUMA


def noise_sigma(gray: np.ndarray) -> float:
    """Immerkaer (1996) fast noise estimate on a [0,1] gray image (returns sigma in [0,1] units).

    Uses the Laplacian-difference kernel, which cancels image structure up to 2nd order.
    """
    h, w = gray.shape
    if h < 3 or w < 3:
        return 0.0
    k = np.array([[1, -2, 1], [-2, 4, -2], [1, -2, 1]], dtype=np.float32)
    conv = cv2.filter2D(gray.astype(np.float32), -1, k, borderType=cv2.BORDER_REFLECT)
    s = np.abs(conv[1:-1, 1:-1]).sum()
    return float(s * np.sqrt(0.5 * np.pi) / (6.0 * (w - 2) * (h - 2)))


@dataclass
class Stats:
    mean_luma: float          # global brightness
    median_luma: float
    dark_frac: float          # fraction of pixels with luma < 0.1
    highlight_frac: float     # fraction of pixels with luma > 0.95 (lamps, signs)
    noise_sigma: float        # Immerkaer estimate on the raw input
    rel_noise: float          # noise relative to brightness (what gets amplified)
    cast: tuple               # per-channel mean / gray mean (colour cast)
    cast_strength: float      # max |channel ratio - 1|
    nonuniformity: float      # std/mean of a heavily blurred luma map (uneven lighting)
    width: int
    height: int

    def to_dict(self):
        return asdict(self)


@dataclass
class Params:
    is_low_light: bool
    target_mean: float        # desired output mean luma
    retinex_gamma: float      # illumination compression; smaller = stronger lift
    gain_floor: float         # bounds max per-pixel gain (1 / floor)
    clahe_clip: float
    denoise_strength: float   # 0 = off, 1 = strong
    wb_strength: float        # 0 = no white balance, 1 = full gray-world
    highlight_protect: float  # 0..1 how strongly to keep bright regions
    dl_weight_hint: float     # prior preference for the network output in the controller

    def to_dict(self):
        return asdict(self)


def analyze(img: np.ndarray) -> Stats:
    y = luma(img)
    h, w = y.shape
    small = cv2.resize(y, (max(8, w // 8), max(8, h // 8)), interpolation=cv2.INTER_AREA)
    blur = cv2.GaussianBlur(small, (0, 0), sigmaX=max(2.0, min(small.shape) / 8))
    means = img.reshape(-1, 3).mean(axis=0)
    gray_mean = float(means.mean()) + 1e-6
    cast = tuple(float(m / gray_mean) for m in means)
    sigma = noise_sigma(y)
    mean_l = float(y.mean())
    return Stats(
        mean_luma=mean_l,
        median_luma=float(np.median(y)),
        dark_frac=float((y < 0.1).mean()),
        highlight_frac=float((y > 0.95).mean()),
        noise_sigma=sigma,
        rel_noise=float(sigma / max(mean_l, 1e-3)),
        cast=cast,
        cast_strength=float(max(abs(c - 1.0) for c in cast)),
        nonuniformity=float(blur.std() / (blur.mean() + 1e-3)),
        width=w,
        height=h,
    )


def _lerp(a, b, t):
    return a + (b - a) * float(np.clip(t, 0.0, 1.0))


def choose_params(s: Stats) -> Params:
    """Map measurements to parameters. Thresholds were set on LOL train/val, never on test."""
    is_low = s.mean_luma < 0.35 or s.dark_frac > 0.5
    darkness = np.clip((0.35 - s.mean_luma) / 0.33, 0.0, 1.0)  # 0 = normal, 1 = near black
    gamma = _lerp(0.85, 0.45, darkness) if is_low else 0.95
    gamma = min(0.95, gamma + 0.1 * np.clip(s.nonuniformity - 0.6, 0, 1))  # keep relief
    denoise = np.clip((s.rel_noise - 0.02) / 0.10, 0.0, 1.0)
    return Params(
        is_low_light=bool(is_low),
        target_mean=0.45 if is_low else max(s.mean_luma, 0.4),
        retinex_gamma=float(gamma),
        gain_floor=float(_lerp(0.08, 0.025, darkness)),
        clahe_clip=float(_lerp(2.5, 1.5, denoise)),  # less local contrast when noisy
        denoise_strength=float(denoise),
        wb_strength=float(np.clip((s.cast_strength - 0.05) / 0.25, 0.0, 0.8)),
        highlight_protect=float(np.clip(s.highlight_frac * 20, 0.0, 1.0)),
        dl_weight_hint=float(1.0 if is_low else 0.3),
    )
