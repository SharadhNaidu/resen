---
license: mit
library_name: resenh
pipeline_tag: image-to-image
tags:
  - low-light-image-enhancement
  - image-restoration
  - retinex
  - retinexformer
  - onnx
datasets:
  - LOL
  - LOL-v2
  - LSRW
metrics:
  - psnr
  - ssim
  - lpips
  - niqe
---

<p align="center">
  <img src="https://huggingface.co/SharadhNaiduTrains/resenh/resolve/main/assets/banner.png" alt="resenh: photos from the dark, brought back to life" width="100%">
</p>

# resenh weights

These are the weights for **[resenh](https://github.com/SharadhNaidu/resen)**, a library for
robust low-light image restoration and enhancement. An adaptive classical pipeline (Retinex,
CLAHE, edge-aware denoising) and a Retinexformer network run side by side, and a quality
controller blends the two photo by photo.

**Try it in your browser:** [resenh.sharadh.blog](https://resenh.sharadh.blog) ·
**Install:** `pip install "resenh[dl]"` ·
**Report:** [technical report (PDF)](https://github.com/SharadhNaidu/resen/blob/main/report/resenh_report_T46.pdf)

```python
import resenh

res = resenh.enhance("night.jpg")   # downloads and verifies the weights below on first use
res.save("night_restored.png")
print(res.summary())
```

<p align="center">
  <img src="https://huggingface.co/SharadhNaiduTrains/resenh/resolve/main/assets/how-it-works.png" alt="Measure, restore twice, judge and fuse" width="100%">
</p>

## Files

| File | What it is | Used by |
|---|---|---|
| `resenh_light_general.pth` | Retinexformer with classical-prior inputs, fine-tuned on LOL + LSRW (de-duplicated) | **package default** (`mode="hybrid"`) |
| `resenh_light_lolv1.pth`, `resenh_light_lolv2_real.pth`, `resenh_light_lolv2_syn.pth` | the same architecture fine-tuned per benchmark | `model=` argument, evaluation |
| `LOL_v1.pth`, `LOL_v2_real.pth`, `LOL_v2_synthetic.pth` | official Retinexformer checkpoints (Cai et al., ICCV 2023, MIT) | `mode="dl"`, fallback |
| `web/resenh-web-hybrid.onnx`, `web/resenh-web-dl.onnx` | ONNX exports (opset 17, dynamic size) for ONNX Runtime Web | the in-browser demo |

The package pins every file's SHA-256 (`src/resenh/models/weights.py`) and refuses a mismatch.

## Results

Plain RGB PSNR on the saved 8-bit image. There is no ground-truth brightness correction, no
per-image knobs, and no checkpoint picked on the test set. Our harness reproduces the official
Retinexformer checkpoints exactly. Full tables with 95% confidence intervals:
[`eval_P1_T46.md`](https://github.com/SharadhNaidu/resen/blob/main/eval_P1_T46.md).

| PSNR (dB) ↑ | LOL-v1 (15) | LOL-v2-Real (100) | LOL-v2-Syn (100) | LSRW test (50) |
|---|---|---|---|---|
| Classical branch | 16.75 | 17.44 | 16.83 | 16.15 |
| Retinexformer, network only | 25.15 | 22.79 | 25.67 | 17.19 |
| **resenh hybrid** | **25.13** | **22.82** | **25.35** | **19.26** |

| LOL-v2-Real test | PSNR ↑ | SSIM ↑ | LPIPS ↓ | NIQE ↓ |
|---|---|---|---|---|
| Classical branch | 17.44 | 0.724 | 0.417 | 4.83 |
| Retinexformer, network only | 22.79 | 0.840 | 0.171 | 3.59 |
| **resenh hybrid** | **22.82** | **0.840** | **0.171** | **3.59** |

- **LOL hybrid rows:** the official per-benchmark backbone with the `benchmark` preset.
- **LSRW hybrid row:** `resenh_light_general.pth`, which is in-domain for LSRW cameras (it was
  trained on LSRW train). It was chosen on LSRW validation before the test set was scored.
- **Fine-tuned per-benchmark weights:** trained with the GT-mean loss. They improve SSIM and
  LPIPS (LOL-v2-Real: 0.852 / 0.139) but score lower in plain PSNR (20.82 dB). See the
  [ablation](https://github.com/SharadhNaidu/resen/blob/main/docs/ablation.md).
- **Latency:** a 1 MP photo takes 0.49 s for the network and 1.9 s for the full hybrid on an
  RTX 4070 laptop GPU, and about 6 s for the hybrid in a laptop browser with WebGPU.

## Training

- **Architecture:** Retinexformer, 1.6 M parameters, with a zero-initialised stem extension for
  the classical priors (illumination map, Retinex output).
- **Loss:** Charbonnier + 0.2·(1−SSIM) + 0.02·VGG-19 perceptual, wrapped in the GT-mean loss.
- **Recipe:** progressive patches (128→256→384), EMA 0.999, gamma and noise augmentation, and
  scene-grouped validation splits.

## Team

Team V-Max (T46): Sharadh Naidu, Shivani, S. V. Swamy, Sumukha K. MIT licence. Retinexformer
weights © their authors (MIT).
