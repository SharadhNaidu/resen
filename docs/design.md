# resenh: design document

Problem P1, *Robust Low-Light Image Restoration and Enhancement* (Computer Vision Hackathon 2026).

## 1. Problem and goals
Low-light photos are dark, noisy, colour-shifted and often lit unevenly (a lamp in a dark
street). Brightening alone amplifies noise, blows out highlights and shifts colour. The goals:

1. Restore brightness, colour and detail, and suppress noise, without over-exposure, halos or
   hallucinated colour.
2. Adapt to each image instead of applying fixed parameters.
3. Compare classical, deep and hybrid approaches under one **honest** protocol.
4. Ship it as a usable product: a pip package, a CLI and a web app with intermediate
   visualisations.

## 2. Architecture
See `docs/architecture.svg` (source: `docs/architecture.mmd`).

```
input → sanitize → Analyzer → adaptive Params
                     │
       ┌─────────────┴──────────────┐
 Classical branch               Hybrid network
 (denoise → illumination →      (Retinexformer + classical priors:
  Retinex → post-denoise →       illumination map, LIME output,
  CLAHE → WB → exposure)         CLAHE output as extra inputs)
       └─────────────┬──────────────┘
           Quality controller (guards + fitted fusion)
                     │
           highlight protection → bright-input cap → output + intermediates
```

| Module | File | Role |
|---|---|---|
| Sanitizer | `src/resenh/io.py` | EXIF rotation, gray/alpha/16-bit → RGB, size limits, corrupt/HEIC errors |
| Analyzer | `src/resenh/analysis.py` | brightness, dark %, highlight %, Immerkaer noise σ, colour cast, non-uniformity → `Params` |
| Classical branch | `src/resenh/classical/` | guided-filter denoise, LIME illumination (max-RGB + guided filter), Retinex lift, noise-adaptive post-denoise, CLAHE (Lab L), gray-world WB, exposure with filmic shoulder |
| Network | `src/resenh/models/retinexformer.py`, `runner.py` | Retinexformer (ICCV'23) with zero-initialised prior inputs; guided-filter gain upsampling for large images; ×8 TTA |
| Controller | `src/resenh/controller.py` | per-candidate guards (new clipping, chroma shift, edge loss, noise amplification, exposure error) → softmax fusion |
| Metrics | `src/resenh/metrics.py` | PSNR, SSIM, LPIPS, NIQE, noise ratio, EPI, latency, bootstrap CI |
| App | `src/resenh/app.py` | Gradio UI: before/after slider, 12-panel intermediate gallery, analyzer/controller report |

## 3. Classical computer vision (requirement A)
- **Illumination estimation and decomposition.** Following LIME, the initial illumination is the
  per-pixel max over RGB. It is refined with a guided filter, which keeps strong edges (so no
  halos) and flattens texture. Reflectance is R = I / L.
- **Retinex enhancement.** I / L^γ, with γ chosen by the analyzer: darker images get a smaller γ
  (a stronger lift). The gain is capped at 1/floor so near-black pixels don't explode.
  Non-uniform lighting is handled per pixel: dark regions get a large gain, lit regions a small one.
- **CLAHE.** Applied to the Lab L channel only, so hue is preserved. The clip limit drops as
  measured noise rises, because CLAHE amplifies noise. It is blended at 50%.
- **Edge-preserving denoising.**
  - A guided filter runs before the gain, with strength set by the measured relative noise.
  - A second guided filter runs after the gain, with ε tied to the re-measured noise variance;
    chroma gets a stronger pass than luma. On LOL-v2-Real val_b this cut output noise
    amplification from ×5.5 to ×1.5.
  - A bilateral filter is available for comparison.
- **Contrast and colour correction.** Partial gray-world white balance (strength grows with the
  measured cast). Exposure is normalised to a target mean, calibrated on *training* GT
  brightness, with a filmic shoulder.
- **Adaptive parameters.** Every parameter above comes from `choose_params(analyze(img))`.
  Nothing is fixed per dataset.

## 4. Deep learning (requirement B)
- **Architecture.** Retinexformer (Cai et al., ICCV 2023): one-stage Retinex-guided transformer
  with illumination-guided attention, 1.6 M params. Chosen because the audit arXiv 2610.38121
  found it the only method whose score does not depend on test-time knobs.
- **Fine-tuning.**
  - Starts from the official checkpoints for each benchmark.
  - Loss: GT-mean loss (Liao et al., ICCV 2025) over Charbonnier + 0.2·(1−SSIM) + 0.02·VGG.
  - Training details: progressive patches 128→256, EMA 0.999, AdamW, cosine schedule, bf16.
- **Generalisation.** Evaluated on three benchmarks plus five unpaired real-world sets
  (DICM, LIME, MEF, NPE, VV) with no-reference metrics.
- **Image-quality-aware strategy.** The analyzer conditions the controller's prior: less trust in
  the network for inputs that are not low-light. The controller guards measure each output's
  quality without GT.

## 5. Hybrid intelligence (requirement C)
1. **Classical illumination feeds the network.** The illumination map, the LIME output and the
   CLAHE output (7 channels) are extra inputs to Retinexformer: added to the illumination-
   estimator features and to the denoiser embedding. They are zero-initialised, so the hybrid
   starts exactly at the pretrained model and learns how to use them.
2. **Adaptive selection and fusion.**
   - The controller scores each candidate with guards against the input: newly clipped pixels,
     Lab chroma shift, EPI edge loss, noise amplification and exposure error.
   - Candidates are fused with weights ∝ prior·exp(−penalty/T). Prior and T were fitted on
     **val_b** (never on test).
   - On in-distribution LOL images the fit gives the network about 99% weight. Classical
     contributes when the guards fire (out-of-distribution inputs).
3. **Highlight protection.** A dilated soft mask of already-bright sources (lamps, signs) blends
   the original back in. This fixes a failure where LOL-trained networks paint clipped lamps
   red (`docs/figures/failure_lamp_fix.jpg`).
4. **Extreme dark and non-uniform input** is covered by the analyzer-driven γ and gain floor,
   the per-pixel Retinex gain, and guided-filter gain upsampling for large images.

## 6. Evaluation protocol
- **Benchmarks:** LOL-v1 (15 test), LOL-v2-Real (100), LOL-v2-Syn (100). Each model is trained and
  tested on the same benchmark.
- **Validation splits:** val_a and val_b are carved from train, grouped by scene. Test is never
  used for any decision.
- **Checkpoints:** the last EMA checkpoint after a fixed budget; see
  `docs/devlog/protocol-decision.md` for why val_a cannot select checkpoints for models
  initialised from official weights.
- **Metric conventions** (match the Retinexformer and CIDNet scripts; verified by reproducing
  the official checkpoints to within 0.01 dB):
  - PSNR: RGB on uint8 outputs.
  - SSIM: MATLAB-style, per channel.
  - LPIPS: AlexNet.
  - NIQE: pyiqa `niqe_matlab`.
- **No GT-mean correction, no test-time knobs, bootstrap 95% CIs.**
- **Leakage:** pHash dedup found 101 LOL-v1 training images that duplicate LOL test GTs. They
  are excluded from the generalist model, which is never scored on LOL test sets.

## 7. Results, ablation and failure analysis
See `eval_P1_<TeamID>.md`, `docs/ablation.md` and `docs/failures.md`, all generated by the
scripts in `eval/`.

## 8. Limitations and ethics
- **Training data.** LOL is mostly indoor scenes and small. Phone night shots with strong point
  lights or heavy sensor noise are out of distribution; the guards and highlight protection
  reduce but don't remove artefacts.
- **The network can hallucinate.** It may invent colour or texture in near-black regions. Do
  not use enhanced images as evidence (forensics, surveillance identification) without the
  original.
- **Benchmark caveats.** LOL-v1 has only 15 test images (95% CIs about ±1.5 dB). LOL-v2-Real
  test labels are darker than its training labels, so honest scores there are capped.
- **Licences.** Retinexformer code and weights are MIT. The example images come from the public
  DICM, LIME, MEF, NPE and VV benchmark sets and are used for research demonstration.
