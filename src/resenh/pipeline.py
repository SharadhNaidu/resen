"""Top-level enhancement API tying analyzer, classical branch, network and controller together."""

from __future__ import annotations

import time
import warnings
from dataclasses import dataclass, field

import numpy as np

from resenh.analysis import Params, Stats, analyze, choose_params
from resenh.classical import enhance_classical
from resenh.classical.color import protect_highlights
from resenh.io import load_image

MODES = ("auto", "classical", "dl", "hybrid")
PRESETS = ("robust", "benchmark")
# Adaptive pre-network denoising gate on the measured noise sigma (Immerkaer). Thresholds fitted
# on val_b of LOL-v1/v2 (clean + synthetic dark/non-uniform/noisy variants), never on test:
# +4.3 dB on heavily noisy inputs, -0.55 dB on clean LOL-like inputs. "benchmark" disables it.
NOISE_GATE = (0.0107, 0.0210)
# Bright-input cap: start blending back towards the input above this mean luma (val_b-fitted).
BRIGHT_CAP = (0.6, 0.3)


@dataclass
class EnhanceResult:
    image: np.ndarray                     # RGB float32 [0, 1]
    mode: str
    stats: Stats
    params: Params
    intermediates: dict = field(default_factory=dict)
    decision: dict | None = None
    timings: dict = field(default_factory=dict)
    model: str | None = None


def _torch_ok() -> bool:
    try:
        import torch  # noqa: F401, PLC0415

        return True
    except ImportError:
        return False


def _default_model(kind: str) -> str:
    from resenh.models.weights import DEFAULT_MODEL, FALLBACK_MODEL, available  # noqa: PLC0415

    if kind == "dl":
        return FALLBACK_MODEL
    return DEFAULT_MODEL if available(DEFAULT_MODEL) else FALLBACK_MODEL


def enhance(img, mode: str = "auto", model: str | None = None, device: str = "auto",
            max_side: int = 1280, tta: bool = False, preset: str = "robust",
            disable: frozenset[str] = frozenset()) -> EnhanceResult:
    """Enhance a low-light image.

    Args:
        img: path, bytes, PIL image or RGB ndarray (uint8/uint16/float).
        mode: ``"classical"`` (no deep learning), ``"dl"`` (network only, stock Retinexformer
            unless ``model`` is given), ``"hybrid"`` (classical + network fused by the quality
            controller) or ``"auto"`` (= hybrid, falling back to classical without torch).
        model: weight name from :mod:`resenh.models.weights` or a checkpoint path.
        max_side: network runs at most at this long side; gain is guided-upsampled beyond.
        tta: x8 flip/rotation self-ensemble (slower, slightly better).
        preset: ``"robust"`` (default) gates an edge-preserving pre-denoise on the measured
            noise level, which strongly helps noisy real photos; ``"benchmark"`` disables it
            (best on clean benchmark images such as LOL).
        disable: stage names to switch off (ablation): classical stages plus
            ``"controller"`` (use network output only), ``"highlights"`` and ``"noise_gate"``.
    """
    if preset not in PRESETS:
        raise ValueError(f"preset must be one of {PRESETS}")
    if mode not in MODES:
        raise ValueError(f"mode must be one of {MODES}")
    t0 = time.perf_counter()
    x = load_image(img)
    stats = analyze(x)
    params = choose_params(stats)
    timings = {"load_analyze": time.perf_counter() - t0}

    if mode in ("auto", "hybrid", "dl") and not _torch_ok():
        warnings.warn("PyTorch not installed: falling back to the classical pipeline "
                      "(pip install 'resenh[dl]' for the hybrid model)", stacklevel=2)
        mode = "classical"
    if mode == "auto":
        mode = "hybrid"

    if mode == "classical":
        t = time.perf_counter()
        out, inter = enhance_classical(x, params, disable=disable)
        timings["classical"] = time.perf_counter() - t
        timings["total"] = time.perf_counter() - t0
        return EnhanceResult(out, mode, stats, params, inter, None, timings)

    from resenh.models.runner import enhance_net  # noqa: PLC0415

    name = model or _default_model(mode)
    t = time.perf_counter()
    net_in, gate = x, 0.0
    if mode == "hybrid" and preset == "robust" and "noise_gate" not in disable:
        lo, hi = NOISE_GATE
        gate = float(np.clip((stats.noise_sigma - lo) / (hi - lo), 0.0, 1.0))
        if gate > 0:
            from resenh.classical.denoise import denoise  # noqa: PLC0415

            net_in = denoise(x, gate)
    timings["noise_gate"] = gate
    net = enhance_net(net_in, name, device=device, max_side=max_side, tta=tta)
    timings["network"] = time.perf_counter() - t
    if mode == "dl":
        timings["total"] = time.perf_counter() - t0
        return EnhanceResult(net, mode, stats, params, {"network": net}, None, timings, name)

    t = time.perf_counter()
    cls, inter = enhance_classical(x, params, disable=disable)
    timings["classical"] = time.perf_counter() - t
    inter["classical"], inter["network"] = cls, net

    from resenh.controller import fuse  # noqa: PLC0415

    t = time.perf_counter()
    if "controller" in disable:
        out, decision = net, None
    else:
        # prior strength and temperature fitted on val_b of LOL-v1/v2 (never on test)
        prior = {"network": 256.0 * params.dl_weight_hint, "classical": 1.0}
        out, dec = fuse({"network": net, "classical": cls}, x, params, prior=prior,
                        temperature=0.25)
        decision = dec.to_dict()
    if "highlights" not in disable:
        out = protect_highlights(out, x, params.highlight_protect)
    if stats.mean_luma > BRIGHT_CAP[0]:  # already-bright input: stay close to it
        a = float(np.clip((stats.mean_luma - BRIGHT_CAP[0]) / BRIGHT_CAP[1], 0.0, 0.8))
        out = (1 - a) * out + a * x
    timings["controller"] = time.perf_counter() - t
    timings["total"] = time.perf_counter() - t0
    return EnhanceResult(np.clip(out, 0, 1).astype(np.float32), mode, stats, params, inter,
                         decision, timings, name)
