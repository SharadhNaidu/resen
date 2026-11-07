# Protocol decision: checkpoint selection (recorded 2026-11-07 ~06:05 IST, before any test evaluation of fine-tuned models)

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
