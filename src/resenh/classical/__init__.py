"""Classical low-light enhancement branch (adaptive, parameters come from the analyzer)."""

from __future__ import annotations

import numpy as np

from resenh.analysis import Params, analyze, choose_params
from resenh.classical.clahe import clahe_luma
from resenh.classical.color import gray_world, match_exposure, protect_highlights
from resenh.classical.denoise import denoise
from resenh.classical.illumination import estimate_illumination, reflectance, retinex_enhance

__all__ = ["enhance_classical"]


def enhance_classical(img: np.ndarray, params: Params | None = None,
                      denoise_method: str = "guided",
                      disable: frozenset[str] = frozenset()) -> tuple[np.ndarray, dict]:
    """Run the classical branch. Returns (output, intermediates).

    Stage order: denoise (before any gain, so noise is not amplified) -> illumination
    estimation -> Retinex/LIME lift -> CLAHE on luma -> gray-world WB -> exposure match ->
    highlight protection. ``disable`` names stages to skip (for ablation): any of
    {"denoise", "retinex", "clahe", "wb", "exposure", "highlights"}.
    """
    if params is None:
        params = choose_params(analyze(img))
    inter: dict[str, np.ndarray] = {}
    x = img
    if "denoise" not in disable:
        x = denoise(x, params.denoise_strength, denoise_method)
    inter["denoised"] = x
    illum = estimate_illumination(x)
    inter["illumination"] = illum
    inter["reflectance"] = reflectance(x, illum)
    if "retinex" not in disable:
        x, applied = retinex_enhance(x, illum, gamma=params.retinex_gamma,
                                     floor=params.gain_floor)
        inter["retinex"] = x
        inter["gain_map"] = np.clip(1.0 / applied / 25.0, 0, 1)
    if "clahe" not in disable:
        c = clahe_luma(x, params.clahe_clip)
        x = (0.5 * x + 0.5 * c).astype(np.float32)  # half-strength: CLAHE alone over-sharpens noise
        inter["clahe"] = c
    if "wb" not in disable:
        x = gray_world(x, params.wb_strength)
    if "exposure" not in disable:
        x = match_exposure(x, params.target_mean)
    if "highlights" not in disable:
        x = protect_highlights(x, img, params.highlight_protect)
    return np.clip(x, 0.0, 1.0).astype(np.float32), inter
