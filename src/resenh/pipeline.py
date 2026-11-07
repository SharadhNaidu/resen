"""Top-level enhancement API tying analyzer, classical branch, network and controller together."""

from __future__ import annotations

import time
import warnings
from dataclasses import dataclass, field
from pathlib import Path

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


@dataclass(repr=False)
class EnhanceResult:
    """What :func:`enhance` returns.

    The main fields are ``image`` (the enhanced RGB float32 image in [0, 1]) and ``input``
    (the loaded original). Everything else explains how the result was made:

    * ``stats`` and ``params``: what the analyzer measured, and the parameters it chose
    * ``intermediates``: every stage (illumination map, reflectance, classical and network
      outputs, ...), also available as labelled images through :attr:`stages`
    * ``decision``: the quality controller's fusion weights, guard scores and notes
    * ``timings``: seconds per step

    In a notebook, putting the result on the last line of a cell shows a before/after image.
    """

    image: np.ndarray
    mode: str
    stats: Stats
    params: Params
    intermediates: dict = field(default_factory=dict)
    decision: dict | None = None
    timings: dict = field(default_factory=dict)
    model: str | None = None
    input: np.ndarray | None = None

    # ---------------------------------------------------------------- conversions
    def to_uint8(self) -> np.ndarray:
        """The result as an ``HxWx3`` uint8 array."""
        from resenh.io import to_uint8  # noqa: PLC0415

        return to_uint8(self.image)

    def to_pil(self):
        """The result as a PIL image."""
        from PIL import Image  # noqa: PLC0415

        return Image.fromarray(self.to_uint8())

    def save(self, path) -> Path:
        """Save the result (PNG, JPEG, ... from the extension). Returns the path."""
        from resenh.io import save_image  # noqa: PLC0415

        return save_image(self.image, path)

    # ---------------------------------------------------------------- inspection
    @property
    def stages(self) -> list[tuple[str, np.ndarray]]:
        """Every intermediate stage as ``(label, RGB image)`` pairs, input first."""
        from resenh.viz import panels  # noqa: PLC0415

        ref = self.input if self.input is not None else self.image
        return [(label, img) for img, label in panels(self.intermediates, ref, self.image)]

    def compare(self, height: int = 480):
        """A labelled before/after PIL image (input left, result right)."""
        from resenh.viz import side_by_side  # noqa: PLC0415

        ref = self.input if self.input is not None else self.image
        return side_by_side([(ref, "input"), (self.image, f"resenh · {self.mode}")], height)

    def show(self, stages: bool = False, height: int = 480):
        """Display before/after (and optionally every stage) in a notebook or image viewer."""
        from resenh.viz import side_by_side  # noqa: PLC0415

        img = self.compare(height)
        if stages:
            img = side_by_side([(im, lab) for lab, im in self.stages], 220, columns=4)
        try:
            from IPython.display import display  # noqa: PLC0415

            display(img)
        except ImportError:
            img.show()

    def summary(self) -> str:
        """A short human-readable report of what was measured, chosen and why."""
        s, p = self.stats, self.params
        lines = [
            f"resenh | mode {self.mode}" + (f" | model {self.model}" if self.model else "")
            + f" | {s.width}x{s.height} | {self.timings.get('total', 0):.2f}s",
            f"  input     brightness {s.mean_luma:.3f}, dark pixels {s.dark_frac:.0%}, "
            f"noise sigma {s.noise_sigma:.4f}, colour cast {s.cast_strength:.2f}, "
            f"uneven light {s.nonuniformity:.2f}",
            f"  chosen    low-light={p.is_low_light}, retinex gamma {p.retinex_gamma:.2f}, "
            f"denoise {p.denoise_strength:.2f}, CLAHE {p.clahe_clip:.1f}, "
            f"white balance {p.wb_strength:.2f}",
        ]
        if self.decision:
            w = ", ".join(f"{k} {v:.0%}" for k, v in self.decision["weights"].items())
            lines.append(f"  fused     {w}")
            lines += [f"  note      {n}" for n in self.decision.get("notes", [])]
        return "\n".join(lines)

    def __repr__(self) -> str:
        h, w = self.image.shape[:2]
        return (f"EnhanceResult(mode={self.mode!r}, model={self.model!r}, size={w}x{h}, "
                f"time={self.timings.get('total', 0):.2f}s)")

    def _repr_png_(self):  # Jupyter / Colab / Kaggle rich display
        import io as _io  # noqa: PLC0415

        buf = _io.BytesIO()
        self.compare().save(buf, format="PNG")
        return buf.getvalue()


def _torch_ok() -> bool:
    try:
        import torch  # noqa: F401, PLC0415

        return True
    except ImportError:
        return False


def _default_model(kind: str) -> str:
    from resenh.models.weights import DEFAULT_MODEL, FALLBACK_MODEL, available  # noqa: PLC0415

    from resenh.models.weights import REGISTRY  # noqa: PLC0415

    if kind == "dl":
        return FALLBACK_MODEL
    # the default is used once it is released (sha256 pinned) or present locally
    if REGISTRY[DEFAULT_MODEL].get("sha256") or available(DEFAULT_MODEL):
        return DEFAULT_MODEL
    return FALLBACK_MODEL


def enhance(img, mode: str = "auto", model: str | None = None, device: str = "auto",
            max_side: int = 1280, tta: bool = False, preset: str = "robust",
            disable: frozenset[str] = frozenset(), net_fn=None) -> EnhanceResult:
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
        net_fn: optional replacement for the PyTorch runner, called as
            ``net_fn(img, name, device=..., max_side=..., tta=...)`` and returning an RGB float
            image of the same size. Used by the in-browser build (ONNX Runtime Web).
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

    if mode in ("auto", "hybrid", "dl") and net_fn is None and not _torch_ok():
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
        return EnhanceResult(out, mode, stats, params, inter, None, timings, input=x)

    if net_fn is None:
        from resenh.models.runner import enhance_net as net_fn  # noqa: PLC0415

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
    net = net_fn(net_in, name, device=device, max_side=max_side, tta=tta)
    timings["network"] = time.perf_counter() - t
    if mode == "dl":
        timings["total"] = time.perf_counter() - t0
        return EnhanceResult(net, mode, stats, params, {"network": net}, None, timings, name,
                                input=x)

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
                         decision, timings, name, input=x)
