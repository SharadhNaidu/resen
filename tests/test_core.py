import io

import numpy as np
import pytest
from PIL import Image

import resenh
from resenh.analysis import analyze, choose_params, noise_sigma
from resenh.classical import enhance_classical
from resenh.io import ImageInputError, load_image


def dark_image(h=120, w=160, level=0.08, seed=0):
    rng = np.random.default_rng(seed)
    yy, xx = np.mgrid[0:h, 0:w]
    base = np.stack([xx / w, yy / h, 0.5 * np.ones_like(xx)], axis=2).astype(np.float32)
    return np.clip(base * level + rng.normal(0, 0.01, base.shape), 0, 1).astype(np.float32)


def _png_bytes(arr, mode=None):
    buf = io.BytesIO()
    Image.fromarray(arr, mode=mode).save(buf, format="PNG")
    return buf.getvalue()


# ----------------------------------------------------------------- io
def test_load_uint8_and_float():
    a = (dark_image() * 255).astype(np.uint8)
    assert load_image(a).dtype == np.float32
    assert load_image(a.astype(np.float32) / 255).max() <= 1.0


def test_load_gray_alpha_16bit():
    g = load_image(np.zeros((32, 32), np.uint8))
    assert g.shape == (32, 32, 3)
    rgba = np.zeros((32, 32, 4), np.uint8)
    rgba[..., 3] = 255
    assert load_image(_png_bytes(rgba)).shape == (32, 32, 3)
    s16 = (np.ones((32, 32)) * 40000).astype(np.uint16)
    out = load_image(Image.fromarray(s16))
    assert out.shape == (32, 32, 3) and 0.5 < out.mean() < 0.7


def test_bad_inputs():
    with pytest.raises(ImageInputError):
        load_image(b"not an image")
    with pytest.raises(ImageInputError):
        load_image(np.zeros((4, 4, 3), np.uint8))
    with pytest.raises(ImageInputError):
        load_image("does_not_exist.png")
    with pytest.raises(ImageInputError):
        load_image(np.full((32, 32, 3), np.nan, np.float32))


# ----------------------------------------------------------------- analysis
def test_noise_estimate_tracks_sigma():
    rng = np.random.default_rng(1)
    flat = np.full((200, 200), 0.5, np.float32)
    low = noise_sigma(flat + rng.normal(0, 0.01, flat.shape).astype(np.float32))
    high = noise_sigma(flat + rng.normal(0, 0.05, flat.shape).astype(np.float32))
    assert 0.007 < low < 0.013 and 0.04 < high < 0.06


def test_params_adapt_to_darkness():
    p_dark = choose_params(analyze(dark_image(level=0.05)))
    p_mid = choose_params(analyze(dark_image(level=0.5)))
    assert p_dark.is_low_light
    assert p_dark.retinex_gamma < p_mid.retinex_gamma  # stronger lift when darker


# ----------------------------------------------------------------- classical
def test_classical_brightens_and_keeps_range():
    x = dark_image()
    out, inter = enhance_classical(x)
    assert out.shape == x.shape and out.dtype == np.float32
    assert 0 <= out.min() and out.max() <= 1
    assert out.mean() > 3 * x.mean()
    assert {"illumination", "reflectance", "denoised"} <= set(inter)


def test_classical_ablation_flags():
    x = dark_image()
    full, _ = enhance_classical(x)
    no_ret, _ = enhance_classical(x, disable=frozenset({"retinex"}))
    assert not np.allclose(full, no_ret)


def test_api_classical_mode_deterministic():
    x = dark_image()
    a = resenh.enhance(x, mode="classical").image
    b = resenh.enhance(x, mode="classical").image
    assert np.array_equal(a, b)


def test_bright_input_not_blown_out():
    x = np.clip(dark_image(level=1.0) + 0.3, 0, 1)
    res = resenh.enhance(x, mode="classical")
    assert not res.params.is_low_light
    assert abs(res.image.mean() - x.mean()) < 0.2
