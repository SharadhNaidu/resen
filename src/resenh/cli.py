"""resenh command-line interface.

    resenh enhance night.jpg -o out.png [--mode hybrid] [--save-intermediates DIR]
    resenh bench [--device cpu] [--size 1024]
    resenh app [--port 7860]
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

from resenh import __version__


def _cmd_enhance(a) -> int:
    import resenh  # noqa: PLC0415
    from resenh.viz import panels  # noqa: PLC0415

    srcs = [Path(s) for s in a.inputs]
    for src in srcs:
        try:
            res = resenh.enhance(src, mode=a.mode, model=a.model, device=a.device,
                                 max_side=a.max_side, tta=a.tta)
        except resenh.ImageInputError as e:
            print(f"error: {src}: {e}", file=sys.stderr)
            return 2
        dst = Path(a.output) if (a.output and len(srcs) == 1) else (
            Path(a.output or ".") / f"{src.stem}_resenh.png")
        dst.parent.mkdir(parents=True, exist_ok=True)
        resenh.save_image(res.image, dst)
        print(f"{src} -> {dst}  mode={res.mode} model={res.model} "
              f"time={res.timings['total']:.2f}s")
        if a.save_intermediates:
            d = Path(a.save_intermediates) / src.stem
            d.mkdir(parents=True, exist_ok=True)
            for i, (img, label) in enumerate(panels(res.intermediates, resenh.load_image(src),
                                                    res.image)):
                resenh.save_image(img, d / f"{i:02d}_{label.split(' ', 1)[-1][:40]}.png"
                                  .replace("/", "-").replace(" ", "_"))
            (d / "report.json").write_text(json.dumps({
                "stats": res.stats.to_dict(), "params": res.params.to_dict(),
                "decision": res.decision, "timings": res.timings}, indent=1, default=float))
    return 0


def _cmd_bench(a) -> int:
    import numpy as np  # noqa: PLC0415

    import resenh  # noqa: PLC0415

    rng = np.random.default_rng(0)
    h = a.size * 3 // 4
    img = (rng.random((h, a.size, 3)) * 0.15).astype(np.float32)
    rows = []
    for mode in a.modes:
        resenh.enhance(img, mode=mode, device=a.device, max_side=a.max_side)  # warm-up
        ts = []
        for _ in range(a.repeat):
            t = time.perf_counter()
            resenh.enhance(img, mode=mode, device=a.device, max_side=a.max_side)
            ts.append(time.perf_counter() - t)
        mp = img.shape[0] * img.shape[1] / 1e6
        print(f"{mode:10s} {a.size}x{h} ({mp:.2f} MP) device={a.device}: "
              f"{np.median(ts):.3f}s median, {np.median(ts) / mp * 1000:.0f} ms/MP")
        from resenh.models.runner import pick_device  # noqa: PLC0415

        rows.append({"mode": mode, "device": pick_device(a.device) if mode != "classical" else "cpu",
                     "size": f"{a.size}x{h}", "mean_s": float(np.median(ts)),
                     "ms_per_mp": float(np.median(ts) / mp * 1000)})
    if a.json:
        import json  # noqa: PLC0415
        from pathlib import Path  # noqa: PLC0415

        f = Path(a.json)
        old = json.loads(f.read_text()) if f.exists() else []
        keep = {(r["mode"], r["device"], r["size"]) for r in rows}
        old = [r for r in old if (r["mode"], r["device"], r["size"]) not in keep]
        f.write_text(json.dumps(old + rows, indent=1))
    return 0


def _cmd_app(a) -> int:
    from resenh.app import build  # noqa: PLC0415

    build().launch(server_name=a.host, server_port=a.port, share=a.share)
    return 0


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(prog="resenh", description="Robust low-light image enhancement")
    ap.add_argument("--version", action="version", version=f"resenh {__version__}")
    sub = ap.add_subparsers(dest="cmd")

    e = sub.add_parser("enhance", help="enhance one or more images")
    e.add_argument("inputs", nargs="+")
    e.add_argument("-o", "--output", help="output file (single input) or directory")
    e.add_argument("--mode", default="auto", choices=["auto", "classical", "dl", "hybrid"])
    e.add_argument("--model", default=None)
    e.add_argument("--device", default="auto")
    e.add_argument("--max-side", type=int, default=1280)
    e.add_argument("--tta", action="store_true")
    e.add_argument("--save-intermediates", metavar="DIR")
    e.set_defaults(func=_cmd_enhance)

    b = sub.add_parser("bench", help="measure latency")
    b.add_argument("--modes", nargs="+", default=["classical", "dl", "hybrid"])
    b.add_argument("--device", default="auto")
    b.add_argument("--size", type=int, default=1155, help="width; height = 3/4 (default ~1 MP)")
    b.add_argument("--max-side", type=int, default=1280)
    b.add_argument("--repeat", type=int, default=3)
    b.add_argument("--json", metavar="FILE", help="append results to a JSON file")
    b.set_defaults(func=_cmd_bench)

    p = sub.add_parser("app", help="launch the Gradio web app")
    p.add_argument("--host", default="0.0.0.0")
    p.add_argument("--port", type=int, default=7860)
    p.add_argument("--share", action="store_true")
    p.set_defaults(func=_cmd_app)

    a = ap.parse_args(argv)
    if not getattr(a, "func", None):
        ap.print_help()
        return 0
    return a.func(a)


if __name__ == "__main__":
    raise SystemExit(main())
