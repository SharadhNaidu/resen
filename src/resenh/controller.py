"""Quality controller: scores candidate outputs with failure-mode guards and fuses them.

No-reference IQA scores (NIQE, MUSIQ) are deliberately *not* used to pick outputs: they were
shown to select the best enhancement 0% of the time (BlindPSNR, arXiv 2607.27628). Instead each
candidate is checked for the concrete failure modes the task names (over-exposure, colour
shift, detail loss, noise amplification) against the input, and weights favour candidates that
pass the guards. A per-pixel highlight guard is applied after fusion.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field

import cv2
import numpy as np

from resenh.analysis import Params, luma, noise_sigma
from resenh.metrics import edge_preservation, gain_matched


@dataclass
class Guards:
    clipped: float        # fraction of pixels blown to white that were not white in the input
    chroma_shift: float   # mean |hue-chroma| change vs. gain-matched input (Lab a,b units / 100)
    edge_keep: float      # EPI vs. gain-matched input (1 = all structure kept)
    noise_amp: float      # output noise / gain-matched-input noise (>1 = amplified)
    brightness: float     # output mean luma

    def penalty(self, target_mean: float) -> float:
        p = 6.0 * self.clipped
        p += 3.0 * max(0.0, self.chroma_shift - 0.03)
        p += 1.5 * max(0.0, 0.55 - self.edge_keep)
        p += 0.6 * max(0.0, self.noise_amp - 1.0)
        p += 2.0 * max(0.0, abs(self.brightness - target_mean) - 0.15)
        return float(p)


@dataclass
class Decision:
    weights: dict
    guards: dict
    penalties: dict
    chosen: str
    notes: list = field(default_factory=list)

    def to_dict(self):
        return asdict(self)


def _ab(img: np.ndarray) -> np.ndarray:
    lab = cv2.cvtColor(np.clip(img, 0, 1).astype(np.float32), cv2.COLOR_RGB2LAB)
    return lab[..., 1:]


def measure(candidate: np.ndarray, inp: np.ndarray) -> Guards:
    ref = gain_matched(inp, candidate)
    y_in, y_out = luma(inp), luma(candidate)
    clipped = float(((y_out > 0.98) & (y_in < 0.9)).mean())
    chroma = float(np.abs(_ab(candidate) - _ab(ref)).mean() / 100.0)
    s_ref = max(noise_sigma(luma(ref)), 1e-6)
    return Guards(
        clipped=clipped,
        chroma_shift=chroma,
        edge_keep=edge_preservation(candidate, ref),
        noise_amp=float(noise_sigma(y_out) / s_ref),
        brightness=float(y_out.mean()),
    )


def fuse(candidates: dict[str, np.ndarray], inp: np.ndarray, params: Params,
         prior: dict[str, float] | None = None, temperature: float = 0.25) -> tuple[np.ndarray, Decision]:
    """Fuse candidates with weights ~ prior * exp(-penalty / T). Returns (image, decision)."""
    names = list(candidates)
    guards = {n: measure(candidates[n], inp) for n in names}
    pens = {n: guards[n].penalty(params.target_mean) for n in names}
    prior = prior or {}
    logits = np.array([np.log(max(prior.get(n, 1.0), 1e-6)) - pens[n] / temperature
                       for n in names])
    w = np.exp(logits - logits.max())
    w /= w.sum()
    out = sum(wi * candidates[n] for wi, n in zip(w, names))
    notes = []
    if not params.is_low_light:
        notes.append("input is not low-light: gain capped")
    for n in names:
        g = guards[n]
        if g.clipped > 0.01:
            notes.append(f"{n}: {g.clipped:.1%} newly clipped pixels")
        if g.noise_amp > 1.5:
            notes.append(f"{n}: noise amplified x{g.noise_amp:.1f}")
    weights = {n: float(wi) for n, wi in zip(names, w)}
    return np.clip(out, 0, 1).astype(np.float32), Decision(
        weights=weights,
        guards={n: asdict(g) for n, g in guards.items()},
        penalties=pens,
        chosen=max(weights, key=weights.get),
        notes=notes,
    )
