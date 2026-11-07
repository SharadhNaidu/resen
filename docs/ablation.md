# Ablation study

Benchmark: LOL-v2-Synthetic test (100 pairs). Training ablations are 3,000-iteration fine-tunes from the same official initialisation, with identical schedule and seed, reported at their last checkpoint. Only the named component differs.

## Training-time components

| Method | Approach | PSNR ↑ (95% CI) | SSIM ↑ | LPIPS ↓ | NIQE ↓ | Latency (ms/MP, GPU) ↓ | Noise ratio (vs GT) | Edge preservation EPI ↑ |
|---|---|---|---|---|---|---|---|---|
| Full hybrid (priors + GT-mean loss) |  | **25.60** (24.53 to 26.67) | 0.9365 | 0.0511 | 3.97 | 1043 | 1.00 | 0.963 |
| − classical priors |  | **25.55** (24.48 to 26.59) | 0.9362 | 0.0508 | 3.98 | 936 | 1.00 | 0.963 |
| − GT-mean loss (plain loss) |  | **25.92** (24.85 to 26.97) | 0.9349 | 0.0521 | 3.94 | 1162 | 0.99 | 0.962 |
| No fine-tuning (official checkpoint) |  | **25.67** (24.65 to 26.64) | 0.9295 | 0.0592 | 3.94 | 972 | 0.98 | 0.960 |

## Inference-time components (full 8k model)

| Method | Approach | PSNR ↑ (95% CI) | SSIM ↑ | LPIPS ↓ | NIQE ↓ | Latency (ms/MP, GPU) ↓ | Noise ratio (vs GT) | Edge preservation EPI ↑ |
|---|---|---|---|---|---|---|---|---|
| Full pipeline |  | **24.57** (23.51 to 25.64) | 0.9050 | 0.0917 | 3.91 | 1816 | 0.90 | 0.945 |
| − quality controller (network only) |  | **25.69** (24.55 to 26.82) | 0.9364 | 0.0499 | 3.97 | 970 | 1.00 | 0.964 |
| Classical branch alone |  | **16.83** (16.20 to 17.48) | 0.7580 | 0.3061 | 4.14 | 723 | 0.95 | 0.851 |

## Inference-time components on LOL-v1 (15 test pairs)

| Method | Approach | PSNR ↑ (95% CI) | SSIM ↑ | LPIPS ↓ | NIQE ↓ | Latency (ms/MP, GPU) ↓ | Noise ratio (vs GT) | Edge preservation EPI ↑ |
|---|---|---|---|---|---|---|---|---|
| Full pipeline |  | **23.33** (21.97 to 24.67) | 0.8502 | 0.1221 | 3.07 | 1960 | 0.59 | 0.792 |
| − quality controller (network only) |  | **23.35** (21.97 to 24.69) | 0.8503 | 0.1219 | 3.07 | 1089 | 0.59 | 0.793 |
| Classical branch alone |  | **16.75** (15.32 to 18.12) | 0.7211 | 0.3554 | 4.22 | 517 | 0.73 | 0.696 |

## Inference-time components on LOL-v2-Real (100 test pairs)

| Method | Approach | PSNR ↑ (95% CI) | SSIM ↑ | LPIPS ↓ | NIQE ↓ | Latency (ms/MP, GPU) ↓ | Noise ratio (vs GT) | Edge preservation EPI ↑ |
|---|---|---|---|---|---|---|---|---|
| Full pipeline |  | **20.66** (19.63 to 21.64) | 0.8477 | 0.1454 | 3.58 | 1805 | 0.57 | 0.824 |
| − quality controller (network only) |  | **20.82** (19.84 to 21.78) | 0.8522 | 0.1391 | 3.59 | 1006 | 0.58 | 0.826 |
| Classical branch alone |  | **17.44** (16.96 to 17.91) | 0.7237 | 0.4172 | 4.83 | 550 | 0.69 | 0.714 |

## Training-loss ablation: GT-mean loss vs plain loss

The three Syn ablations were trained for 3k iterations each from the same official initialisation,
with one seed and the same schedule. They were scored on LOL-v2-Syn test with plain RGB PSNR.

| Variant | PSNR, test | PSNR, val_b |
|---|---|---|
| Full recipe (classical priors + GT-mean loss) | 25.60 | 29.04 |
| Without the classical priors | 25.55 | 29.42 |
| Plain loss (priors kept, no GT-mean term) | **25.92** | **30.05** |

- **The priors' effect is within noise:** +0.06 dB on test, −0.38 dB on val_b.
- **The GT-mean loss consistently costs plain PSNR:** 0.32 dB on test and 1.0 dB on val_b. It trains the network to get structure right up to a
  global brightness scale. GT-mean PSNR, which rescales the output to the reference mean, rewards this.
  Plain PSNR, which we report, penalises it.
- **The same effect at full scale:**
  - the fine-tuned LOL-v1 and LOL-v2-Real models, both trained with the GT-mean loss, lose 1.8 and 2.0 dB
    of PSNR against the official checkpoints;
  - they *improve* SSIM (0.850 vs 0.846 on v1, 0.852 vs 0.840 on Real) and LPIPS (0.122 vs 0.131 on v1,
    0.139 vs 0.171 on Real).
- **What we did about it:** we report those rows unchanged, because they were pre-registered (see
  `docs/devlog/protocol-decision.md`, incident 2). The shipped configuration uses the official backbone
  inside the hybrid pipeline.
- **Next step:** retraining with the plain loss, which this ablation predicts will help.

