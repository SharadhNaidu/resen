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
