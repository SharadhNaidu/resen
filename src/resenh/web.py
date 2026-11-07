"""In-browser bridge (Pyodide + ONNX Runtime Web).

The browser cannot run PyTorch, and Python cannot block on the JavaScript ONNX session, so a
request runs in two calls that share the normal :func:`resenh.enhance` code path:

1. ``start(data, mode, preset, work_side)`` decodes the photo, resizes it to the working size and
   runs the pipeline up to the network. If a network is needed it returns the network input (and
   the classical priors) for JavaScript to run through ONNX.
2. ``finish(net_out)`` runs the same pipeline again with the network output plugged in, then
   returns the result PNG, the intermediate panels and a JSON report.

The re-run repeats only image loading, analysis and the optional pre-denoise (tens of ms), so the
browser runs exactly the same logic as ``pip install resenh``.
"""

from __future__ import annotations

import json

import cv2
import numpy as np

from resenh.classical.priors import compute_priors
from resenh.io import load_image
from resenh.metrics import edge_preservation, gain_matched, noise_ratio
from resenh.pipeline import enhance
from resenh.viz import panels

WEB_MODELS = {"hybrid": ("resenh-web-hybrid", True), "dl": ("resenh-web-dl", False)}
_state: dict = {}


def _bytes(data) -> bytes:
    """bytes from a Python buffer or a Pyodide JsProxy of a JS typed array."""
    return data.to_bytes() if hasattr(data, "to_bytes") else bytes(data)


class NeedNetwork(Exception):
    def __init__(self, img: np.ndarray):
        super().__init__("network output required")
        self.img = img


def _resize(x: np.ndarray, side: int) -> np.ndarray:
    h, w = x.shape[:2]
    if side and max(h, w) > side:
        s = side / max(h, w)
        x = cv2.resize(x, (max(8, round(w * s)), max(8, round(h * s))),
                       interpolation=cv2.INTER_AREA)
    # the network needs sides that are multiples of 8; crop at most 7 px
    h, w = x.shape[:2]
    return np.ascontiguousarray(x[: h // 8 * 8, : w // 8 * 8]).astype(np.float32)


def _png(img: np.ndarray, max_side: int = 0) -> bytes:
    u8 = (np.clip(img, 0, 1) * 255 + 0.5).astype(np.uint8)
    if max_side and max(u8.shape[:2]) > max_side:
        s = max_side / max(u8.shape[:2])
        u8 = cv2.resize(u8, None, fx=s, fy=s, interpolation=cv2.INTER_AREA)
    ok, buf = cv2.imencode(".png", cv2.cvtColor(u8, cv2.COLOR_RGB2BGR))
    return buf.tobytes()


def _run(net_out: np.ndarray | None):
    st = _state

    def net_fn(img, name, **_):
        if net_out is None:
            raise NeedNetwork(img)
        return net_out

    model = WEB_MODELS.get(st["mode"], (None,))[0]
    return enhance(st["x"], mode=st["mode"], model=model, preset=st["preset"], max_side=0,
                   net_fn=net_fn)


def start(data, mode: str = "hybrid", preset: str = "robust", work_side: int = 640) -> dict:
    """Begin a request. ``data`` is the encoded image (bytes / memoryview)."""
    x = _resize(load_image(_bytes(data)), int(work_side))
    _state.clear()
    _state.update(x=x, mode=mode, preset=preset)
    try:
        res = _run(None)
    except NeedNetwork as e:
        net_in = e.img
        _, priors = WEB_MODELS[mode]
        h, w = net_in.shape[:2]
        out = {"need_network": True, "model": WEB_MODELS[mode][0], "h": h, "w": w,
               "x": np.ascontiguousarray(net_in.transpose(2, 0, 1)).tobytes()}
        if priors:
            p = compute_priors(net_in)
            out["priors"] = np.ascontiguousarray(p.transpose(2, 0, 1)).tobytes()
            out["prior_channels"] = p.shape[2]
        return out
    return _finalize(res)


def finish(net_chw) -> dict:
    """Resume with the network output (float32 CHW buffer from ONNX Runtime Web)."""
    h, w = _state["x"].shape[:2]
    y = np.frombuffer(_bytes(net_chw), dtype=np.float32).reshape(3, h, w).transpose(1, 2, 0)
    return _finalize(_run(np.clip(y, 0, 1).astype(np.float32)))


def _finalize(res) -> dict:
    x, out = _state["x"], res.image
    ref = gain_matched(x, out)
    s, p = res.stats, res.params
    report = {
        "mode": res.mode, "model": res.model, "width": s.width, "height": s.height,
        "timings": {k: round(float(v), 4) for k, v in res.timings.items()},
        "analyzer": {"brightness": s.mean_luma, "dark_frac": s.dark_frac,
                     "highlight_frac": s.highlight_frac, "noise_sigma": s.noise_sigma,
                     "cast": s.cast_strength, "nonuniformity": s.nonuniformity},
        "params": {"low_light": bool(p.is_low_light), "retinex_gamma": p.retinex_gamma,
                   "denoise": p.denoise_strength, "clahe_clip": p.clahe_clip,
                   "wb": p.wb_strength, "highlight_protect": p.highlight_protect},
        "quality": {"noise_ratio": noise_ratio(out, ref), "epi": edge_preservation(out, ref)},
        "decision": res.decision,
    }
    gallery = [(_png(im, 512), label) for im, label in panels(res.intermediates, x, out)]
    return {"need_network": False, "image": _png(out), "input": _png(x),
            "report": json.dumps(report, default=float), "panels": gallery}
