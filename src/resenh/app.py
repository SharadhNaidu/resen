"""Gradio web app: upload a low-light photo, see the enhancement and every intermediate stage."""

from __future__ import annotations

import os
from pathlib import Path

import numpy as np

import resenh
from resenh.metrics import edge_preservation, gain_matched, noise_ratio
from resenh.viz import panels

try:  # Hugging Face ZeroGPU: allocate a GPU only for the duration of a call
    import spaces

    _gpu = spaces.GPU(duration=20)
except Exception:  # noqa: BLE001 - running locally / CPU / Docker
    def _gpu(fn):
        return fn

MODE_MAP = {
    "Hybrid (adaptive, recommended)": "hybrid",
    "Deep learning only": "dl",
    "Classical only": "classical",
}

EXAMPLES_DIR = Path(os.environ.get("RESENH_EXAMPLES",
                                   Path(__file__).resolve().parents[2] / "app" / "examples"))


@_gpu
def _run(img, mode, tta):
    return resenh.enhance(img, mode=mode, tta=tta,
                          max_side=int(os.environ.get("RESENH_MAX_SIDE", 1280)))


def process(img, mode_label, tta):
    import gradio as gr  # noqa: PLC0415

    if img is None:
        raise gr.Error("Please upload an image first.")
    try:
        x = resenh.load_image(img)
        res = _run(x, MODE_MAP[mode_label], bool(tta))
    except resenh.ImageInputError as e:
        raise gr.Error(str(e)) from e
    out = res.image
    ref = gain_matched(x, out)
    s, p = res.stats, res.params
    lines = [
        f"### Result: `{res.mode}` mode" + (f" · model `{res.model}`" if res.model else ""),
        f"**Time:** {res.timings['total']:.2f}s  ·  **Size:** {s.width}×{s.height}",
        "",
        "**Analyzer (input measurements → adaptive parameters)**",
        f"- brightness {s.mean_luma:.3f}, dark pixels {s.dark_frac:.0%}, "
        f"highlights {s.highlight_frac:.1%}",
        f"- noise σ {s.noise_sigma:.4f} (relative {s.rel_noise:.3f}), "
        f"colour cast {s.cast_strength:.2f}, uneven lighting {s.nonuniformity:.2f}",
        f"- low-light: **{p.is_low_light}** → Retinex γ {p.retinex_gamma:.2f}, "
        f"denoise {p.denoise_strength:.2f}, CLAHE clip {p.clahe_clip:.1f}, "
        f"WB {p.wb_strength:.2f}, highlight protect {p.highlight_protect:.2f}",
        "",
        "**Output quality (no reference available)**",
        f"- noise vs. brightened input: ×{noise_ratio(out, ref):.2f} (lower = cleaner)",
        f"- edge preservation (EPI): {edge_preservation(out, ref):.3f} (1 = all structure kept)",
    ]
    if res.decision:
        d = res.decision
        lines += ["", "**Quality controller**",
                  "- weights: " + ", ".join(f"{k} {v:.2f}" for k, v in d["weights"].items()),
                  "- guard penalties: " + ", ".join(f"{k} {v:.2f}"
                                                   for k, v in d["penalties"].items())]
        lines += [f"- ⚠ {n}" for n in d["notes"]]
    gallery = [((np.clip(im, 0, 1) * 255).astype(np.uint8), lab)
               for im, lab in panels(res.intermediates, x, out)]
    out_u8 = (np.clip(out, 0, 1) * 255 + 0.5).astype(np.uint8)
    in_u8 = (np.clip(x, 0, 1) * 255 + 0.5).astype(np.uint8)
    return (in_u8, out_u8), out_u8, gallery, "\n".join(lines)


def build():
    import gradio as gr  # noqa: PLC0415

    examples = sorted(str(p) for p in EXAMPLES_DIR.glob("*")) if EXAMPLES_DIR.exists() else []
    with gr.Blocks(title="resenh: low-light enhancement") as demo:
        gr.Markdown(
            "# resenh: robust low-light image enhancement\n"
            "Adaptive hybrid of classical Retinex/CLAHE/edge-preserving denoising and a "
            "fine-tuned Retinexformer, fused by a quality controller. "
            "Upload a dark photo (JPEG/PNG)."
        )
        with gr.Row():
            with gr.Column(scale=1):
                inp = gr.Image(label="Low-light input", type="numpy", image_mode="RGB",
                               sources=["upload", "clipboard", "webcam"])
                mode = gr.Radio(list(MODE_MAP), value=list(MODE_MAP)[0], label="Method")
                tta = gr.Checkbox(label="High quality (×8 self-ensemble, slower)", value=False)
                btn = gr.Button("Enhance", variant="primary")
                if examples:
                    gr.Examples(examples, inputs=inp, label="Examples (unseen test images)")
            with gr.Column(scale=2):
                slider = gr.ImageSlider(label="Before / after", type="numpy")
                out_img = gr.Image(label="Enhanced (download)", type="numpy", format="png")
        report = gr.Markdown()
        gallery = gr.Gallery(label="Intermediate stages", columns=4, height="auto")
        btn.click(process, [inp, mode, tta], [slider, out_img, gallery, report],
                  concurrency_limit=2)
    return demo


if __name__ == "__main__":
    build().launch()
