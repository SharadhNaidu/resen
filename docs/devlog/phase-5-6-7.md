# Phases 5–7: training, experiments, release (session 2, 2026-11-07, 06:30–10:15 IST)

## Training (`training/queue.sh`, `training/queue2.sh`)
- Finished: syn_light 8k, v1_light 6k, real_light 8k (re-run after a DataLoader pickling crash, fixed with
  `getattr(self, "noise_aug", False)`), and 3 Syn ablations at 3k (full / no priors / plain loss).
  general_light 8k: LOL + LSRW, deduped, with gamma and noise augmentation.
- Test results (pre-registered last.pth, run once): see `eval_P1_T01.md`.
  - **The fine-tuned backbones lost about 2 dB PSNR on v1 and Real** (SSIM and LPIPS improved).
  - The cause is the GT-mean loss: the Syn ablation scores 25.92 with plain loss against 25.60 with it.
  - Incident 2 in `protocol-decision.md`.
  - The shipped configuration (hybrid pipeline + official backbone) is reported next to the fine-tuned one.
- Future work: retrain v1 and Real with the plain loss (`--plain-loss`); the ablation predicts it helps.

## Experiments
- `docs/failures.md`: 4 cases (clip-guard over-ride on bright daylight scenes, dusk GT ambiguity, lamp colour
  hallucination (fixed), noise amplification).
- CPU latency, while training was running: classical 0.47 s, DL 3.3 s, hybrid 4.5 s at 1 MP (`outputs/bench.json`).

## Release
- **PyPI:** `resenh` 0.1.0 and 0.1.1 published (account owning the token in `Desktop/synapse/.env`).
  - 0.1.1 adds `res.save/show/compare/summary/stages`, notebook display, URL and file-like inputs,
    `enhance_folder`, `download_weights`, `info`, `resenh --env`, folder CLI input and `python -m resenh`.
- **Weights:** HF model repo `SharadhNaiduTrains/resenh` (also the GitHub release `weights-v1`); the package
  downloads from HF. The web ONNX models are under `web/` in the same repo.
- **Web demo:** https://resenh.sharadh.blog
  - Cloudflare Worker with static assets (`wrangler.jsonc`, assets in `site/dist`).
  - React + Vite + Tailwind 4 + shadcn (base-nova), Sarvam-inspired gradients with Material 3 Expressive shapes.
  - The engine (`site/public/worker.js`) runs the real `resenh` wheel in Pyodide and the network in ONNX
    Runtime Web (WebGPU, falling back to threaded WASM; COOP/COEP credentialless headers).
  - Rebuild: `bash deploy/build_web.sh && cd site && npm run build && cd .. && npx wrangler deploy`.
  - Hugging Face Spaces needs PRO, and Render free has 512 MB (it OOMs), so both were dropped. A Render
    service `srv-dbn7vu6gekts739b1f80` exists (free, unused). Delete it, and the user should revoke the Render key.
- **Notebook:** `notebooks/quickstart.ipynb`. It runs clean locally (CPU venv) and on Kaggle
  (`iamsharadh7/resenh-quickstart`, private).
- **CI:** `.github/workflows/tests.yml` and `publish.yml`. Trusted publishing still has to be configured on pypi.org.

## Open
- Team ID and USNs (eval file line 1); the deadline; the organisers' ranking dataset.
