"""Export the web models to ONNX (dynamic H/W) and check parity with PyTorch.

    python deploy/export_onnx.py --hybrid runs/general_light/last.pth --dl retinexformer-lolv2-real
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from resenh.models.runner import load_model  # noqa: E402


def export(name: str, out: Path) -> None:
    model, has_priors, _ = load_model(name, "cpu")
    x = torch.rand(1, 3, 128, 160)
    args, names = (x,), ["x"]
    axes = {"x": {2: "h", 3: "w"}, "y": {2: "h", 3: "w"}}
    if has_priors:
        args += (torch.rand(1, model.prior_channels if hasattr(model, "prior_channels") else 7,
                            128, 160),)
        names.append("priors")
        axes["priors"] = {2: "h", 3: "w"}
    torch.onnx.export(model, args, str(out), input_names=names, output_names=["y"],
                      dynamic_axes=axes, opset_version=17, dynamo=False)
    import onnxruntime as ort

    sess = ort.InferenceSession(str(out), providers=["CPUExecutionProvider"])
    t = [torch.rand(1, a.shape[1], 200, 296) for a in args]
    with torch.inference_mode():
        ref = model(*t).numpy()
    got = sess.run(None, {n: v.numpy() for n, v in zip(names, t)})[0]
    print(f"{out.name}: priors={has_priors} max|diff|={np.abs(ref - got).max():.2e} "
          f"size={out.stat().st_size / 1e6:.1f} MB")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--hybrid", default="runs/general_light/last.pth")
    ap.add_argument("--dl", default="retinexformer-lolv2-real")
    ap.add_argument("--out", default="outputs/web")
    a = ap.parse_args()
    Path(a.out).mkdir(parents=True, exist_ok=True)
    export(a.hybrid, Path(a.out) / "resenh-web-hybrid.onnx")
    export(a.dl, Path(a.out) / "resenh-web-dl.onnx")
