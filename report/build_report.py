"""Build the technical report (report/report.html -> report/resenh_report_T46.pdf).

Every number is read from the cached evaluation results in outputs/eval/*.json and
outputs/bench.json, the same files eval/make_report.py turns into eval_P1_T46.md.
Run from the repo root:  python report/build_report.py
"""

from __future__ import annotations

import json
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
EV = ROOT / "outputs" / "eval"
REPO = "https://github.com/SharadhNaidu/resen"

INK, MUTED, SOFT = "#1E2033", "#5D6380", "#8A90A8"
RULE = "rgba(30,32,51,.12)"
PERI, PERI_T, SUN, SUN_T = "#6A88E2", "#EEF2FF", "#F7862A", "#FFF1E4"
SANS = "'Google Sans Flex','Google Sans',system-ui,sans-serif"
MONO = "'Google Sans Code',ui-monospace,monospace"


def res(method: str, bench: str, split: str = "test") -> dict | None:
    key = method.replace(":", "_").replace("/", "_")
    f = EV / f"{key}__{bench}__{split}.json"
    if not f.exists():
        return None
    d = json.loads(f.read_text())
    s = d.get("summary", d)
    return {k: (v["mean"] if isinstance(v, dict) and "mean" in v else v) for k, v in s.items()}


BENCH = [("lolv1", "LOL-v1"), ("lolv2_real", "LOL-v2-Real"), ("lolv2_syn", "LOL-v2-Syn")]
APPROACH = [("Classical", "classical"), ("Deep learning", "net:retinexformer-bench"),
            ("Hybrid (ours)", "hybrid-bench:retinexformer-bench")]


def esc(s: str) -> str:
    return s.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")


# --------------------------------------------------------------------------- diagrams
def arch_svg() -> str:
    """System architecture: hand-laid orthogonal layout on a 4px grid."""

    def node(x, y, w, h, name, sub, tag, focal=False, kind="step"):
        fill, stroke = ("#FFFFFF", INK)
        if focal:
            fill, stroke = SUN_T, SUN
        elif kind == "io":
            fill, stroke = PERI_T, PERI
        cx = x + w / 2
        return f"""
  <rect x="{x}" y="{y}" width="{w}" height="{h}" rx="8" fill="{fill}" stroke="{stroke}" stroke-width="1"/>
  <text x="{x + 12}" y="{y + 18}" font-family="{MONO}" font-size="8" letter-spacing=".08em" fill="{stroke}">{tag}</text>
  <text x="{cx}" y="{y + h / 2 + 6}" text-anchor="middle" font-family="{SANS}" font-size="13" font-weight="600" fill="{INK}">{name}</text>
  <text x="{cx}" y="{y + h / 2 + 22}" text-anchor="middle" font-family="{MONO}" font-size="9" fill="{MUTED}">{sub}</text>"""

    def label(x, y, text, w):
        return (f'<rect x="{x - w / 2}" y="{y - 9}" width="{w}" height="12" rx="2" fill="#FFFFFF"/>'
                f'<text x="{x}" y="{y}" text-anchor="middle" font-family="{MONO}" font-size="8" '
                f'letter-spacing=".06em" fill="{SOFT}">{text}</text>')

    a = f'stroke="{MUTED}" stroke-width="1" fill="none" marker-end="url(#ah)"'
    r = 8
    paths = [
        # input -> analyzer
        f'<path d="M120 164 H152" {a}/>',
        # analyzer -> classical (up)
        f'<path d="M272 152 H{288 - r} Q288 152 288 {152 - r} V{80 + r} Q288 80 {288 + r} 80 H328" {a}/>',
        # analyzer -> network (down)
        f'<path d="M272 176 H{304 - r} Q304 176 304 {176 + r} V{248 - r} Q304 248 {304 + r} 248 H328" {a}/>',
        # classical -> controller
        f'<path d="M488 80 H{512 - r} Q512 80 512 {80 + r} V{152 - r} Q512 152 {512 + r} 152 H536" {a}/>',
        # network -> controller
        f'<path d="M488 248 H{512 - r} Q512 248 512 {248 - r} V{176 + r} Q512 176 {512 + r} 176 H536" {a}/>',
        # controller -> output
        f'<path d="M664 164 H696" {a}/>',
        # priors (dashed)
        f'<path d="M408 112 V216" stroke="{PERI}" stroke-width="1" stroke-dasharray="4,3" fill="none" marker-end="url(#ahp)"/>',
    ]
    labels = [
              f'<text x="420" y="160" font-family="{MONO}" font-size="8" letter-spacing=".06em" fill="{PERI}">PRIORS</text>',
              f'<text x="420" y="172" font-family="{MONO}" font-size="8" letter-spacing=".06em" fill="{PERI}">(FINE-TUNED)</text>']
    nodes = [
        node(24, 136, 96, 56, "Photo", "any size", "INPUT", kind="io"),
        node(152, 136, 120, 56, "Analyzer", "stats → params", "MEASURE"),
        node(328, 48, 160, 64, "Classical branch", "Retinex, CLAHE, denoise", "CLASSICAL"),
        node(328, 216, 160, 64, "Network branch", "Retinexformer, 1.6 M", "DEEP"),
        node(536, 132, 128, 64, "Quality controller", "guards → fusion", "HYBRID", focal=True),
        node(696, 136, 96, 56, "Result", "lamps protected", "OUTPUT", kind="io"),
    ]
    return f"""<svg viewBox="0 0 816 304" role="img" aria-labelledby="arch-title arch-desc" xmlns="http://www.w3.org/2000/svg">
  <title id="arch-title">resenh system architecture</title>
  <desc id="arch-desc">A photo is measured by the analyzer, restored by a classical branch and a network branch in parallel, and the quality controller fuses the two into the result.</desc>
  <defs>
    <marker id="ah" markerWidth="8" markerHeight="6" refX="7" refY="3" orient="auto"><polygon points="0 0, 8 3, 0 6" fill="{MUTED}"/></marker>
    <marker id="ahp" markerWidth="8" markerHeight="6" refX="7" refY="3" orient="auto"><polygon points="0 0, 8 3, 0 6" fill="{PERI}"/></marker>
  </defs>
  {''.join(paths)}
  {''.join(labels)}
  {''.join(nodes)}
</svg>"""


def bar_chart() -> str:
    """Grouped PSNR bars: 3 benchmarks x 3 approaches."""
    W, H, L, B, T = 640, 248, 48, 208, 24
    vmax = 30
    gw = (W - L - 16) / 3
    bw = 40
    colors = ["#C9CEDD", PERI, SUN]
    out = [f'<svg viewBox="0 0 {W} {H + 32}" role="img" aria-labelledby="bars-title bars-desc" xmlns="http://www.w3.org/2000/svg">',
           '<title id="bars-title">PSNR by approach and benchmark</title>',
           '<desc id="bars-desc">On all three LOL benchmarks the classical branch scores 16.8 to 17.4 dB, while deep learning and the hybrid both score 22.8 to 25.7 dB.</desc>']
    for v in range(0, vmax + 1, 10):
        y = B - (B - T) * v / vmax
        out.append(f'<line x1="{L}" x2="{W - 8}" y1="{y}" y2="{y}" stroke="{RULE}" stroke-width=".8"/>')
        out.append(f'<text x="{L - 8}" y="{y + 3}" text-anchor="end" font-family="{MONO}" font-size="9" fill="{SOFT}">{v}</text>')
    out.append(f'<text x="12" y="{T - 8}" font-family="{MONO}" font-size="9" fill="{SOFT}">PSNR, dB</text>')
    for gi, (bk, bname) in enumerate(BENCH):
        gx = L + gi * gw + (gw - 3 * bw - 2 * 8) / 2
        for ai, (aname, m) in enumerate(APPROACH):
            r_ = res(m, bk)
            v = r_["psnr"] if r_ else 0
            h = (B - T) * v / vmax
            x = gx + ai * (bw + 8)
            out.append(f'<rect x="{x:.1f}" y="{B - h:.1f}" width="{bw}" height="{h:.1f}" rx="4" fill="{colors[ai]}"/>')
            out.append(f'<text x="{x + bw / 2:.1f}" y="{B - h - 6:.1f}" text-anchor="middle" font-family="{SANS}" '
                       f'font-size="11" font-weight="{600 if ai == 2 else 500}" fill="{INK}">{v:.2f}</text>')
        out.append(f'<text x="{L + gi * gw + gw / 2:.1f}" y="{B + 20}" text-anchor="middle" font-family="{SANS}" '
                   f'font-size="12" font-weight="500" fill="{INK}">{bname}</text>')
    out.append(f'<line x1="{L}" x2="{W - 8}" y1="{B}" y2="{B}" stroke="{INK}" stroke-width=".8"/>')
    lx = L
    for ai, (aname, _) in enumerate(APPROACH):
        out.append(f'<rect x="{lx}" y="{H + 12}" width="10" height="10" rx="2" fill="{colors[ai]}"/>'
                   f'<text x="{lx + 16}" y="{H + 21}" font-family="{SANS}" font-size="11" fill="{MUTED}">{aname}</text>')
        lx += 140
    out.append("</svg>")
    return "".join(out)


def delta_chart(rows: list[tuple[str, float, str]]) -> str:
    """Horizontal diverging bars of PSNR change (dB) for the ablation."""
    W, rowh = 680, 36
    H = rowh * len(rows) + 28
    zero = 330
    scale = 64  # px per dB
    out = [f'<svg viewBox="0 0 {W} {H}" role="img" aria-labelledby="abl-title abl-desc" xmlns="http://www.w3.org/2000/svg">',
           '<title id="abl-title">Ablation: PSNR change per component</title>',
           '<desc id="abl-desc">Change in PSNR when each component is switched on, in decibels.</desc>',
           f'<line x1="{zero}" x2="{zero}" y1="4" y2="{H - 24}" stroke="{INK}" stroke-width=".8"/>']
    for i, (name, d, note) in enumerate(rows):
        y = 8 + i * rowh
        w = min(abs(d) * scale, 280)
        x = zero if d >= 0 else zero - w
        col = SUN if d > 0.2 else (PERI if d < -0.2 else "#C9CEDD")
        out.append(f'<text x="0" y="{y + 13}" font-family="{SANS}" font-size="11.5" fill="{INK}">{esc(name)}</text>')
        out.append(f'<text x="0" y="{y + 26}" font-family="{MONO}" font-size="8.5" fill="{SOFT}">{esc(note)}</text>')
        out.append(f'<rect x="{x:.1f}" y="{y + 4}" width="{max(w, 2):.1f}" height="16" rx="3" fill="{col}"/>')
        tx = (x + w + 6) if d >= 0 else (x - 6)
        anc = "start" if d >= 0 else "end"
        out.append(f'<text x="{tx:.1f}" y="{y + 16}" text-anchor="{anc}" font-family="{SANS}" font-size="11" '
                   f'font-weight="600" fill="{INK}">{d:+.2f} dB</text>')
    out.append(f'<text x="{zero}" y="{H - 8}" text-anchor="middle" font-family="{MONO}" font-size="8.5" fill="{SOFT}">0 dB</text>')
    out.append("</svg>")
    return "".join(out)


def guard_chart() -> str:
    """Controller guard scores on the church photo (dicm_3): network vs classical."""
    import numpy as np  # noqa: PLC0415

    import resenh  # noqa: PLC0415

    r = resenh.enhance(str(ROOT / "site/public/examples/dicm_3.jpg"), mode="hybrid",
                       model="retinexformer-lolv2-real")
    g = r.decision["guards"]
    w = r.decision["weights"]
    items = [("Newly clipped highlights", "clipped", "{:.1%}"), ("Colour shift vs input", "chroma_shift", "{:.2f}"),
             ("Edges kept", "edge_keep", "{:.2f}"), ("Noise amplification", "noise_amp", "×{:.1f}")]
    rows = "".join(
        f"<tr><td>{n}</td><td class='num'>{f.format(g['network'][k])}</td><td class='num'>{f.format(g['classical'][k])}</td></tr>"
        for n, k, f in items)
    rows += (f"<tr class='total'><td>Fusion weight</td><td class='num'>{w['network']:.0%}</td>"
             f"<td class='num'>{w['classical']:.0%}</td></tr>")
    _ = np
    return rows


# --------------------------------------------------------------------------- page
def fmt(v, n=2):
    return "–" if v is None or v != v else f"{v:.{n}f}"


def main():
    commit = subprocess.run(["git", "rev-parse", "--short", "HEAD"], capture_output=True, text=True,
                            cwd=ROOT).stdout.strip()
    bench = {b["mode"]: b for b in json.loads((ROOT / "outputs/bench.json").read_text())}
    gpu = {b["mode"]: b for b in json.loads((ROOT / "outputs/bench_gpu.json").read_text())} \
        if (ROOT / "outputs/bench_gpu.json").exists() else {}

    # main table on LOL-v2-Real (100 pairs) + averages over the three benchmarks
    def row(name, m, mode, cls=""):
        cells = []
        for bk, _ in BENCH:
            r_ = res(m, bk)
            cells.append(r_)
        real = cells[1]
        lat_cpu = bench.get(mode, {}).get("ms_per_mp")
        lat_gpu = gpu.get(mode, {}).get("ms_per_mp")
        lat = (f"{lat_gpu:.0f} <span class='dim'>GPU</span><br>{lat_cpu:.0f} <span class='dim'>CPU</span>"
               if lat_gpu else f"{lat_cpu:.0f} <span class='dim'>CPU</span>")
        return (f"<tr class='{cls}'><td>{name}</td>"
                f"<td class='num'>{fmt(real['psnr'])}</td><td class='num'>{fmt(real['ssim'], 3)}</td>"
                f"<td class='num'>{fmt(real['lpips'], 3)} / {fmt(real['niqe'])}</td><td class='num'>{lat}</td></tr>")

    main_rows = (row("Classical baseline", "classical", "classical")
                 + row("Deep learning only", "net:retinexformer-bench", "dl")
                 + row("Proposed hybrid system", "hybrid-bench:retinexformer-bench", "hybrid", "ours"))

    def b(m, bk):
        r_ = res(m, bk)
        return r_["psnr"] if r_ else None

    abl = [
        ("Plain loss instead of GT-mean loss", b("net:runs/syn_abl_plainloss/last.pth", "lolv2_syn")
         - b("net:runs/syn_abl_full3k/last.pth", "lolv2_syn"), "LOL-v2-Syn test, 3k-iteration fine-tunes"),
        ("Classical priors as network inputs", b("net:runs/syn_abl_full3k/last.pth", "lolv2_syn")
         - b("net:runs/syn_abl_nopriors/last.pth", "lolv2_syn"), "LOL-v2-Syn test, 3k-iteration fine-tunes"),
        ("Quality controller vs network alone", b("hybrid-bench:retinexformer-bench", "lolv2_real")
         - b("net:retinexformer-bench", "lolv2_real"), "LOL-v2-Real test, clean benchmark images"),
        ("Adaptive noise gate, noisy photos", 4.3, "validation images with added noise"),
        ("Adaptive noise gate, clean photos", b("hybrid:retinexformer-bench", "lolv2_syn")
         - b("hybrid-bench:retinexformer-bench", "lolv2_syn"), "LOL-v2-Syn test, clean images"),
    ]

    lsrw = [(n, res(m, "lsrw")) for n, m in [
        ("Classical", "classical"), ("Deep learning only", "net:retinexformer-lolv2-real"),
        ("Hybrid, official backbone", "hybrid:retinexformer-lolv2-real"),
        ("Hybrid, general model (default)", "hybrid:runs/general_light/last.pth")]]
    lsrw_rows = "".join(f"<tr><td>{n}</td><td class='num'>{fmt(r_ and r_['psnr'])}</td>"
                        f"<td class='num'>{fmt(r_ and r_['ssim'], 3)}</td><td class='num'>{fmt(r_ and r_['lpips'], 3)}</td></tr>"
                        for n, r_ in lsrw if r_)

    unp = []
    for d in ("DICM", "LIME", "MEF", "NPE"):
        c, n = res("classical", d), res("dl:retinexformer-lolv2-real", d)
        if c and n:
            unp.append((d, n["niqe"], c["niqe"]))

    html = TEMPLATE
    subs = {
        "{{ARCH}}": arch_svg(), "{{BARS}}": bar_chart(), "{{ABL}}": delta_chart(abl),
        "{{MAIN_ROWS}}": main_rows, "{{GUARDS}}": guard_chart(), "{{LSRW_ROWS}}": lsrw_rows,
        "{{COMMIT}}": commit, "{{REPO}}": REPO,
        "{{UNP}}": "".join(f"<tr><td>{d}</td><td class='num'>{a:.2f}</td><td class='num'>{c:.2f}</td></tr>" for d, a, c in unp),
        "{{CPU_H}}": f"{bench['hybrid']['mean_s']:.1f}", "{{GPU_C}}": f"{gpu['classical']['mean_s']:.2f}", "{{GPU_D}}": f"{gpu['dl']['mean_s']:.2f}", "{{GPU_H}}": f"{gpu['hybrid']['mean_s']:.1f}", "{{CPU_C}}": f"{bench['classical']['mean_s']:.2f}",
    }
    for k, v in subs.items():
        html = html.replace(k, v)
    out = ROOT / "report" / "report.html"
    out.write_text(html, encoding="utf-8")
    print("wrote", out)

    from playwright.sync_api import sync_playwright  # noqa: PLC0415

    with sync_playwright() as p:
        br = p.chromium.launch()
        pg = br.new_page()
        pg.goto(out.as_uri(), wait_until="networkidle")
        pg.wait_for_timeout(800)
        pdf = ROOT / "report" / "resenh_report_T46.pdf"
        pg.pdf(path=str(pdf), format="A4", print_background=True, prefer_css_page_size=True)
        br.close()
    print("wrote", pdf)


TEMPLATE = (Path(__file__).parent / "template.html").read_text(encoding="utf-8")

if __name__ == "__main__":
    main()
