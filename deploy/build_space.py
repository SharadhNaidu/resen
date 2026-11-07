"""Stage and (optionally) upload the Hugging Face ZeroGPU Space.

    python deploy/build_space.py                      # stage into outputs/space
    python deploy/build_space.py --push USER/resenh   # stage + upload (needs `hf auth login`)

The Space bundles the package source, example images and the release weights, so it never
needs network access to GitHub at start-up. No tokens are written anywhere.
"""

from __future__ import annotations

import argparse
import shutil
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "outputs" / "space"

README = """---
title: resenh
emoji: 🌙
colorFrom: indigo
colorTo: yellow
sdk: gradio
sdk_version: {gradio}
app_file: app.py
pinned: true
license: mit
short_description: Robust low-light image restoration (classical + DL hybrid)
---

# resenh: robust low-light image restoration

Upload a dark photo to see the enhanced result and every intermediate stage: illumination map,
reflectance, denoising and the quality controller's decision.
Code: https://github.com/SharadhNaidu/resen
"""

APP = '''import os
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE / "src"))
os.environ.setdefault("RESENH_WEIGHTS", str(HERE / "weights"))
os.environ.setdefault("RESENH_EXAMPLES", str(HERE / "examples"))

from resenh.app import build  # noqa: E402

demo = build()

if __name__ == "__main__":
    demo.launch()
'''

REQS = """torch
einops
numpy
opencv-python-headless
pillow
huggingface_hub
spaces
"""


def stage() -> Path:
    import gradio

    if OUT.exists():
        shutil.rmtree(OUT)
    (OUT / "src").mkdir(parents=True)
    shutil.copytree(ROOT / "src" / "resenh", OUT / "src" / "resenh",
                    ignore=shutil.ignore_patterns("__pycache__", "*.pyc"))
    shutil.copytree(ROOT / "app" / "examples", OUT / "examples")
    shutil.copytree(ROOT / "weights" / "release", OUT / "weights")
    (OUT / "README.md").write_text(README.format(gradio=gradio.__version__), encoding="utf-8")
    (OUT / "app.py").write_text(APP, encoding="utf-8")
    (OUT / "requirements.txt").write_text(REQS, encoding="utf-8")
    return OUT


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--push", metavar="REPO_ID")
    a = ap.parse_args()
    out = stage()
    print("staged", out)
    if a.push:
        from huggingface_hub import HfApi

        api = HfApi()
        api.create_repo(a.push, repo_type="space", space_sdk="gradio", exist_ok=True)
        api.upload_folder(folder_path=str(out), repo_id=a.push, repo_type="space",
                          commit_message="update space")
        print(f"https://huggingface.co/spaces/{a.push}")


if __name__ == "__main__":
    main()
