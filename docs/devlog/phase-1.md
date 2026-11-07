# Phase 1: setup and data (session 1, 2026-11-07)

## Done
- `.gitignore`:
  - secrets, data, weights and runs are excluded;
  - the old `*.MD` rule was removed, because on Windows it also hid `eval_*.md` and the docs.
- `pyproject.toml`:
  - distribution `resenh`, `requires-python>=3.10`;
  - core dependencies: numpy, opencv-headless, pillow;
  - extras: `[dl] [full] [eval] [app] [train] [dev]`;
  - console script `resenh`.
- Data downloaded and indexed with `training/data_prep.py`:
  - counts verified: LOL-v1 485/15, v2-Real 689/100, v2-Syn 900/100;
  - splits after carving val:

| Benchmark | train | val_a | val_b | test |
|---|---|---|---|---|
| lolv1 | 435 | 30 | 20 | 15 |
| v2-Real | 609 | 50 | 30 | 100 |
| v2-Syn | 810 | 54 | 36 | 100 |

- Dedup with pHash (Hamming distance ≤ 12) against every LOL test GT: 101 hits in lolv1 train, 28 in v2-Real train, 0 in v2-Syn train. This confirms the overlap the audit reported.
- Official Retinexformer weights fetched for v1, v2-Real, v2-Syn and FiveK.

## Decisions
- **Scene grouping for v2-Real:** blocks of 10 consecutive ids. This is conservative: it may merge two scenes but never splits one.
- **val_a selects checkpoints; val_b fits the controller and ensemble.**
- **Known caveat:** the official checkpoints we fine-tune from saw the whole official train split, val_a included. So val_a is optimistic for the *init* model, which biases selection toward the init (a conservative bias). This is stated in the eval file.

## Next
Phase 2/3 code is already written ahead of time (see STATUS). Remaining: the stock baseline eval, then the app.
