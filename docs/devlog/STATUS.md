# resenh: build status (rolling handoff)

Read this first in every new session. Then read the latest `phase-N.md`.
Plan: `~/.claude/plans/so-check-out-the-pure-wozniak.md` (7 phases).

| Phase | State |
|---|---|
| 1 Setup and data | done |
| 2 Classical branch and analyzer | code written, tests green (finishing) |
| 3 Eval harness, DL branch and controller | code written, baseline eval pending |
| 4 App, docs, MVP | todo |
| 5 Train models | running (`training/queue.sh`, logs in `runs/*/train.log`) |
| 6 Experiments | todo |
| 7 Release | todo (PyPI publish needs user confirmation) |

## Environment
- **Python 3.14** system interpreter (torch 2.12.1+cu126, RTX 4070 8 GB). A 3.12 venv was abandoned because the CUDA torch download was too slow.
- `pip install -e . --no-deps` has been done.
- Extra deps installed: einops, opencv-python-headless, imagehash, lpips, pyiqa, pytest.
- Kaggle CLI: `python -m kaggle` (`kaggle.json` lives in `~/.kaggle` and the repo root; it is gitignored).

## Data (gitignored, under `data/`)
- `data/raw/lol`: Kaggle `ohmahler91/lolv1-and-lolv2`.
- `data/raw/unpaired`: DICM 69, LIME 10, MEF 17, NPE 8, VV 24.
- `data/raw/lsrw`: Kaggle `bjzh123/lsrw123`. It was still downloading in session 1; rerun `training/data_prep.py` once it lands.
- `data/manifest.json`: train / val_a / val_b / test per benchmark. val is carved from train, grouped by scene; test is never used for selection.
- `data/dedup.json`: train images whose GT is a near-duplicate (pHash) of any LOL test GT. They are excluded from the generalist model.

## Weights (gitignored, under `weights/`)
- `weights/retinexformer/{LOL_v1,LOL_v2_real,LOL_v2_synthetic,FiveK}.pth`: official checkpoints from Google Drive via gdown. sha256 values are in `src/resenh/models/weights.py`.
