"""Robust image loading: anything reasonable in, RGB float32 [0, 1] out."""

from __future__ import annotations

import io as _io
from pathlib import Path

import numpy as np
from PIL import Image, ImageOps, UnidentifiedImageError

MAX_PIXELS = 40_000_000  # 40 MP hard limit (decompression-bomb guard)
MIN_SIDE = 16


class ImageInputError(ValueError):
    """Raised with a user-facing message when an input image cannot be used."""


def _from_pil(im: Image.Image) -> np.ndarray:
    im = ImageOps.exif_transpose(im)
    if im.mode in ("I;16", "I;16B", "I;16L", "I"):
        arr = np.asarray(im, dtype=np.float32)
        arr = arr / (65535.0 if arr.max() > 255 else 255.0)
        return np.repeat(arr[..., None], 3, axis=2)
    if im.mode in ("RGBA", "LA", "PA") or (im.mode == "P" and "transparency" in im.info):
        im = im.convert("RGBA")
        bg = Image.new("RGBA", im.size, (0, 0, 0, 255))  # flatten onto black (low-light scene)
        im = Image.alpha_composite(bg, im)
    return np.asarray(im.convert("RGB"), dtype=np.float32) / 255.0


def _from_array(arr: np.ndarray) -> np.ndarray:
    a = np.asarray(arr)
    if a.ndim == 2:
        a = a[..., None]
    if a.ndim != 3 or a.shape[2] not in (1, 3, 4):
        raise ImageInputError(f"expected HxW, HxWx3 or HxWx4 array, got shape {a.shape}")
    if a.shape[2] == 4:
        alpha = a[..., 3:4].astype(np.float32)
        alpha = alpha / (alpha.max() if alpha.max() > 1 else 1.0)
        a = a[..., :3] * alpha
    if a.shape[2] == 1:
        a = np.repeat(a, 3, axis=2)
    if a.dtype == np.uint8:
        return a.astype(np.float32) / 255.0
    if a.dtype == np.uint16:
        return a.astype(np.float32) / 65535.0
    a = a.astype(np.float32)
    if not np.isfinite(a).all():
        raise ImageInputError("image contains NaN or infinite values")
    if a.max() > 1.0 + 1e-3:
        a = a / 255.0
    return np.clip(a, 0.0, 1.0)


def load_image(src) -> np.ndarray:
    """Load ``src`` (path, bytes, PIL image or ndarray; arrays are assumed RGB) as RGB float32.

    Applies EXIF orientation, converts gray / alpha / 16-bit inputs, and rejects corrupt,
    empty, tiny or oversized images with :class:`ImageInputError`.
    """
    if isinstance(src, np.ndarray):
        img = _from_array(src)
    else:
        try:
            if isinstance(src, Image.Image):
                im = src
            elif isinstance(src, (bytes, bytearray)):
                im = Image.open(_io.BytesIO(src))
            else:
                p = Path(src)
                if not p.exists():
                    raise ImageInputError(f"file not found: {p}")
                if p.suffix.lower() in (".heic", ".heif"):
                    raise ImageInputError("HEIC/HEIF is not supported; please upload JPEG or PNG")
                im = Image.open(p)
            if im.size[0] * im.size[1] > MAX_PIXELS:
                raise ImageInputError(
                    f"image too large ({im.size[0]}x{im.size[1]}); max {MAX_PIXELS // 10**6} MP")
            im.load()
        except ImageInputError:
            raise
        except (UnidentifiedImageError, OSError, SyntaxError) as e:
            raise ImageInputError(f"could not read image: {e}") from e
        img = _from_pil(im)
    h, w = img.shape[:2]
    if min(h, w) < MIN_SIDE:
        raise ImageInputError(f"image too small ({w}x{h}); minimum side is {MIN_SIDE}px")
    if h * w > MAX_PIXELS:
        raise ImageInputError(f"image too large ({w}x{h}); max {MAX_PIXELS // 10**6} MP")
    return np.ascontiguousarray(img, dtype=np.float32)


def to_uint8(img: np.ndarray) -> np.ndarray:
    return (np.clip(img, 0.0, 1.0) * 255.0 + 0.5).astype(np.uint8)


def save_image(img: np.ndarray, path) -> None:
    Image.fromarray(to_uint8(img)).save(path)
