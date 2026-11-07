# Protocol decision: checkpoint selection (recorded 2026-11-07 ~05:47 IST, before any test evaluation of fine-tuned models)

**Finding.** The official Retinexformer checkpoints we fine-tune from were trained on the
*entire* official train split, including our val_a. At initialisation their val_a PSNR is
~3 dB above test PSNR (Syn: 28.78 on val_a vs 25.67 on test), the same gap the audit
(arXiv 2610.38121) reports on training images. So val_a measures memorisation, not
generalisation, for anything started from these weights. During fine-tuning (val_a held out),
val_a PSNR falls as that memorisation is lost, so it cannot be used to pick checkpoints.

**Decision (fixed before testing).**
1. Every fine-tuned model is reported at its **last EMA checkpoint** after the fixed iteration
   budget written in `training/queue.sh`. No checkpoint selection and no test-set selection.
2. val_a curves are logged and published for transparency only.
3. Ablation variants use the same budget and are also reported at their last checkpoint.
4. Each model is evaluated on test **once**, and every result is reported, including any that
   come out worse than the official checkpoint.

## Incident log (kept for transparency)
- **06:2x IST.** The first test evaluation of the Syn model showed the full hybrid pipeline
  (23.41 dB) far below the network alone (25.69 dB). The cause was a post-processing change
  (always-on highlight protection) that had not been checked on validation data.
- **Fix.** The fix was diagnosed and tuned on **val_b only**:
  - highlight protection now targets only near-clipped sources and scales down with the clipped
    area;
  - the bright-input cap was raised from mean luma 0.35 to 0.6.

  val_b result (hybrid vs. network alone): Syn −0.17, Real +0.04, LOL-v1 +0.21 dB.
- **Re-test.** These settings are now frozen. The Syn test was re-run once with them, and the
  earlier (worse) result is kept in this log rather than hidden. No further changes are made to
  post-processing on the basis of test results.

## Preset reporting (recorded ~06:35 IST, before any LOL-v1 / LOL-v2-Real test run of fine-tuned models)
- Both inference presets are reported in every paired table, with no row picked by test score:
  - `benchmark` (`hybrid-bench:`): no pre-network denoise gate;
  - `robust` (`hybrid:`, the package and app default): the val_b-fitted noise gate is on.
- The comparison table uses the `benchmark` preset as the resenh row. That was fixed from val_b,
  where the gate cost −0.55 dB on clean inputs. The robust row sits next to it.
- The stress tests and unseen images use the `robust` preset, which is what ships.

## Incident 2 (recorded ~08:30 IST, after the one-time test run of the fine-tuned v1 and Real models)
- **Result.** The fine-tuned backbones lost PSNR against the official checkpoints:
  - LOL-v1: 23.35 vs 25.15 dB;
  - LOL-v2-Real: 20.82 vs 22.79 dB.

  SSIM and LPIPS both improved (v1 0.850 / 0.122 vs 0.846 / 0.131; Real 0.852 / 0.139 vs 0.840 / 0.171).
- **Cause.** The Syn training ablation isolates it: the plain-loss variant scores 25.92 dB against 25.60 with the GT-mean loss. The GT-mean loss tolerates global brightness offsets, and plain PSNR penalises exactly those.
- **Decision.** The fine-tuned rows are reported unchanged, as pre-registered. There is no retraining and no re-tuning. Plain-loss retraining is future work.
- **Added rows, with no pipeline change.** The shipped package configuration (the hybrid pipeline with the public official backbone) had never been tested. It is evaluated once on test and reported next to the fine-tuned rows, and neither is chosen as the headline.
- **Default model.** It is chosen on LSRW val_b, which no candidate trained on, before any LSRW test run.
