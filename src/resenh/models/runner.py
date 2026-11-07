"""Run a (hybrid) Retinexformer on an RGB float image, at native or capped resolution."""

from __future__ import annotations

import cv2
import numpy as np

from resenh.classical.filters import fast_guided_filter
from resenh.classical.priors import PRIOR_CHANNELS, compute_priors
from resenh.models.weights import REGISTRY, resolve

_CACHE: dict = {}


def pick_device(device: str = "auto") -> str:
    import torch  # noqa: PLC0415

    if device != "auto":
        return device
    if torch.cuda.is_available():
        return "cuda"
    if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def load_model(name: str, device: str = "auto"):
    import torch  # noqa: PLC0415

    from resenh.models.retinexformer import RetinexFormer  # noqa: PLC0415

    device = pick_device(device)
    key = (name, device)
    if key in _CACHE:
        return _CACHE[key]
    path = resolve(name)
    ckpt = torch.load(path, map_location="cpu", weights_only=False)
    sd = ckpt.get("params", ckpt)
    has_priors = any(".prior_" in k for k in sd)
    if name in REGISTRY:
        has_priors = REGISTRY[name]["priors"] or has_priors
    model = RetinexFormer(prior_channels=PRIOR_CHANNELS if has_priors else 0)
    model.load_official(sd)
    model = model.to(device).eval()
    _CACHE[key] = (model, has_priors, device)
    return _CACHE[key]


def _infer(model, has_priors, device, img: np.ndarray) -> np.ndarray:
    import torch  # noqa: PLC0415
    import torch.nn.functional as F  # noqa: PLC0415

    h, w = img.shape[:2]
    x = torch.from_numpy(np.ascontiguousarray(img.transpose(2, 0, 1)))[None].to(device)
    ph, pw = (8 - h % 8) % 8, (8 - w % 8) % 8
    x = F.pad(x, (0, pw, 0, ph), mode="reflect")
    pr = None
    if has_priors:
        p = compute_priors(img)
        pr = torch.from_numpy(np.ascontiguousarray(p.transpose(2, 0, 1)))[None].to(device)
        pr = F.pad(pr, (0, pw, 0, ph), mode="reflect")
    with torch.inference_mode():
        out = model(x, pr) if has_priors else model(x)
    return out[0, :, :h, :w].clamp(0, 1).permute(1, 2, 0).float().cpu().numpy()


def enhance_net(img: np.ndarray, name: str, device: str = "auto", max_side: int = 0,
                tta: bool = False) -> np.ndarray:
    """Enhance with a network. If ``max_side`` > 0 and the image is larger, the network runs on
    a downscaled copy and the per-pixel gain (out/in) is transferred to full resolution with a
    guided filter (edges from the full-res input), keeping full detail at bounded cost.
    """
    model, has_priors, device = load_model(name, device)
    run = (lambda im: _tta(model, has_priors, device, im)) if tta else (
        lambda im: _infer(model, has_priors, device, im))
    h, w = img.shape[:2]
    if not max_side or max(h, w) <= max_side:
        return run(img)
    s = max_side / max(h, w)
    small = cv2.resize(img, (max(8, round(w * s)), max(8, round(h * s))),
                       interpolation=cv2.INTER_AREA)
    out_s = run(small)
    eps = 1.0 / 255
    ratio_s = (out_s + eps) / (small + eps)
    ratio = cv2.resize(ratio_s, (w, h), interpolation=cv2.INTER_LINEAR)
    guide = img @ np.array([0.299, 0.587, 0.114], dtype=np.float32)
    ratio = fast_guided_filter(guide, ratio.astype(np.float32), r=max(4, int(8 / s)), eps=1e-3)
    return np.clip((img + eps) * ratio - eps, 0, 1).astype(np.float32)


def _tta(model, has_priors, device, img):
    """x8 geometric self-ensemble (flips + 90-degree rotations)."""
    acc = np.zeros_like(img)
    for k in range(4):
        for flip in (False, True):
            a = np.rot90(img, k)
            a = a[:, ::-1] if flip else a
            o = _infer(model, has_priors, device, np.ascontiguousarray(a))
            o = o[:, ::-1] if flip else o
            acc += np.rot90(o, -k)
    return acc / 8.0
