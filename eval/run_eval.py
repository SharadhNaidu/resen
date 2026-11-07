"""Evaluate methods on the LOL test splits (and unpaired sets) under one honest protocol.

Methods are named ``<kind>:<model>`` where kind is classical | dl | hybrid | net (a raw network
checkpoint, no controller) and model a registry name or checkpoint path. Results are cached per
(method, benchmark) in ``outputs/eval/*.json`` so tables can be rebuilt without re-running.

    python eval/run_eval.py --bench lolv1 lolv2_real lolv2_syn \
        --methods classical dl:retinexformer-lolv1 ...

Use ``--paired-model`` to map each benchmark to its own checkpoint (``{bench}`` placeholder):
    --methods "net:retinexformer-{bench}"   with bench aliases lolv1, lolv2-real, lolv2-syn.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
import resenh  # noqa: E402
from resenh import metrics as M  # noqa: E402
from resenh.io import load_image  # noqa: E402

BENCH_ALIAS = {"lolv1": "lolv1", "lolv2_real": "lolv2-real", "lolv2_syn": "lolv2-syn"}


def run_method(method: str, img: np.ndarray, bench: str, max_side: int, tta: bool):
    kind, _, model = method.partition(":")
    model = model.replace("{bench}", BENCH_ALIAS.get(bench, bench)) or None
    if kind == "net":
        from resenh.models.runner import enhance_net

        return enhance_net(img, model, max_side=0, tta=tta)
    preset = "benchmark" if kind.endswith("-bench") else "robust"
    res = resenh.enhance(img, mode=kind.removesuffix("-bench"), model=model, max_side=max_side,
                         tta=tta, preset=preset)
    return res.image


def eval_paired(method, pairs, bench, args, out_dir):
    rows = []
    # warm-up (model load, cudnn autotune) excluded from latency
    run_method(method, load_image(pairs[0][0]), bench, args.max_side, args.tta)
    for lo, hi in pairs:
        x, y = load_image(lo), load_image(hi)
        with M.Timer(cuda_sync=True) as t:
            p = run_method(method, x, bench, args.max_side, args.tta)
        r = {
            "image": Path(lo).name,
            "psnr": M.psnr(p, y),
            "ssim": M.ssim(p, y),
            "lpips": M.lpips_score(p, y),
            "niqe": M.niqe(p) if args.niqe else float("nan"),
            "latency_s": t.elapsed,
            "mp": x.shape[0] * x.shape[1] / 1e6,
            "noise_ratio": M.noise_ratio(p, y),
            "epi": M.edge_preservation(p, y),
        }
        rows.append(r)
        if args.save_images:
            d = out_dir / "images" / method.replace(":", "_").replace("/", "_") / bench
            d.mkdir(parents=True, exist_ok=True)
            resenh.save_image(p, d / (Path(lo).stem + ".png"))
    return rows


def eval_unpaired(method, paths, bench, args, out_dir):
    """No ground truth: NIQE plus noise and edge measured against the brightness-matched input."""
    rows = []
    run_method(method, load_image(paths[0]), bench, args.max_side, args.tta)
    for path in paths:
        x = load_image(path)
        with M.Timer(cuda_sync=True) as t:
            p = run_method(method, x, bench, args.max_side, args.tta)
        ref = M.gain_matched(x, p)
        rows.append({"image": Path(path).name, "psnr": float("nan"), "ssim": float("nan"),
                     "lpips": float("nan"), "niqe": M.niqe(p), "latency_s": t.elapsed,
                     "mp": x.shape[0] * x.shape[1] / 1e6,
                     "noise_ratio": M.noise_ratio(p, ref), "epi": M.edge_preservation(p, ref)})
        if args.save_images:
            d = out_dir / "images" / method.replace(":", "_").replace("/", "_") / bench
            d.mkdir(parents=True, exist_ok=True)
            resenh.save_image(p, d / (Path(path).stem + ".png"))
    return rows


def summarize(rows):
    keys = ["psnr", "ssim", "lpips", "niqe", "noise_ratio", "epi"]
    s = {k: float(np.nanmean([r[k] for r in rows])) for k in keys}
    ps = [r["psnr"] for r in rows if np.isfinite(r["psnr"])]
    s["psnr_ci"] = M.bootstrap_ci(ps) if ps else (float("nan"), float("nan"))
    s["ms_per_mp"] = float(np.mean([r["latency_s"] / r["mp"] for r in rows]) * 1000)
    s["n"] = len(rows)
    return s


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", default="data/manifest.json")
    ap.add_argument("--bench", nargs="+", default=["lolv1", "lolv2_real", "lolv2_syn"])
    ap.add_argument("--split", default="test", help="test | val_a | val_b")
    ap.add_argument("--methods", nargs="+", required=True)
    ap.add_argument("--out", default="outputs/eval")
    ap.add_argument("--max-side", type=int, default=0, help="0 = native resolution")
    ap.add_argument("--tta", action="store_true")
    ap.add_argument("--niqe", action="store_true")
    ap.add_argument("--save-images", action="store_true")
    ap.add_argument("--force", action="store_true")
    args = ap.parse_args()

    man = json.loads(Path(args.manifest).read_text())
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    for bench in args.bench:
        unpaired = bench in man.get("unpaired", {})
        pairs = man["unpaired"][bench] if unpaired else man[bench][args.split]
        for method in args.methods:
            tag = f"{method}{'+tta' if args.tta else ''}|{bench}|{args.split}"
            f = out / (tag.replace(":", "_").replace("|", "__").replace("/", "_")
                       .replace("{", "").replace("}", "") + ".json")
            if f.exists() and not args.force:
                s = json.loads(f.read_text())["summary"]
            else:
                t0 = time.time()
                rows = (eval_unpaired if unpaired else eval_paired)(method, pairs, bench, args, out)
                s = summarize(rows)
                f.write_text(json.dumps({"method": method, "bench": bench, "split": args.split,
                                         "tta": args.tta, "summary": s, "rows": rows}, indent=1))
                print(f"  ({time.time() - t0:.0f}s)")
            print(f"{bench:11s} {tag.split('|')[0]:45s} PSNR {s['psnr']:.3f} "
                  f"[{s['psnr_ci'][0]:.2f},{s['psnr_ci'][1]:.2f}] SSIM {s['ssim']:.4f} "
                  f"LPIPS {s['lpips']:.4f} NIQE {s['niqe']:.2f} {s['ms_per_mp']:.0f}ms/MP "
                  f"noise {s['noise_ratio']:.2f} EPI {s['epi']:.3f}", flush=True)


if __name__ == "__main__":
    main()
