"""Render the brand assets (logo, banner, illustrations) used by the README and social preview.

Original artwork in resenh's own design system: Material 3 Expressive shapes generated as polar
curves (same maths as site/src/lib/shapes.ts), the blue-hour palette and Google Sans Flex.
Run from the repo root:  python assets/build_assets.py
"""

from __future__ import annotations

import base64
import math
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "assets"

SPEC = {"flower": (8, .16, .55, 0), "cookie": (9, .07, 1, 0), "clover": (4, .22, .7, math.pi / 4),
        "sunny": (12, .06, .35, 0), "burst": (10, .10, .5, 0)}


def shape(name: str, steps: int = 240) -> str:
    lobes, depth, sharp, rot = SPEC[name]
    pts = []
    for i in range(steps):
        t = i / steps * 2 * math.pi
        c = math.cos(lobes * (t + rot))
        wave = math.copysign(abs(c) ** sharp, c)
        r = 50 * (1 - depth) + 50 * depth * wave
        pts.append(f"{50 + r * math.cos(t):.2f},{50 + r * math.sin(t):.2f}")
    return "M" + "L".join(pts) + "Z"


def clip(name: str) -> str:
    pts = shape(name, 120)[1:-1].split("L")
    return "polygon(" + ",".join(" ".join(f"{v}%" for v in p.split(",")) for p in pts) + ")"


def img64(rel: str) -> str:
    p = ROOT / rel
    mime = "image/webp" if p.suffix == ".webp" else "image/jpeg"
    return f"data:{mime};base64," + base64.b64encode(p.read_bytes()).decode()


def blob(name, pal, size, x, y, rot=0, op=1.0, blur=0):
    a, b, c = {"dusk": ("#FDC58A", "#F7862A", "#A5BBFC"), "mist": ("#E6ECFF", "#A5BBFC", "#6A88E2"),
               "night": ("#6A88E2", "#3A3F5C", "#1E2033")}[pal]
    gid = f"g{name}{pal}{x}{y}"
    return (f'<svg viewBox="0 0 100 100" style="position:absolute;left:{x}px;top:{y}px;width:{size}px;height:{size}px;'
            f'transform:rotate({rot}deg);opacity:{op};filter:blur({blur}px)"><defs><radialGradient id="{gid}" cx="35%" cy="30%" r="80%">'
            f'<stop offset="0" stop-color="{a}"/><stop offset=".55" stop-color="{b}"/><stop offset="1" stop-color="{c}"/>'
            f'</radialGradient></defs><path d="{shape(name)}" fill="url(#{gid})"/></svg>')


HEAD = """<!doctype html><html><head><meta charset="utf-8">
<link href="https://fonts.googleapis.com/css2?family=Google+Sans+Flex:opsz,wght@6..144,300..700&family=Google+Sans+Code:wght@400;500&display=swap" rel="stylesheet">
<style>*{margin:0;box-sizing:border-box}body{font-family:'Google Sans Flex',sans-serif;color:#1E2033;-webkit-font-smoothing:antialiased}
.stage{position:relative;overflow:hidden}.mono{font-family:'Google Sans Code',monospace}</style></head><body>"""


def mark(size: int) -> str:
    return blob("flower", "dusk", size, 0, 0).replace("position:absolute;left:0px;top:0px;", "display:block;")


def logo(dark: bool) -> str:
    ink = "#FFFFFF" if dark else "#1E2033"
    return HEAD + f"""<div class="stage" style="width:560px;height:160px;display:flex;align-items:center;gap:22px;padding:0 24px;background:transparent">
{mark(104)}<div style="font-size:92px;font-weight:540;letter-spacing:-.045em;color:{ink};font-variation-settings:'opsz' 120;line-height:1">resenh</div></div></body></html>"""


def banner() -> str:
    a, b = img64("site/public/hero/lime_1_in.webp"), img64("site/public/hero/lime_1_out.webp")
    return HEAD + f"""<div class="stage" style="width:1280px;height:640px;background:#FAFBFF">
<div style="position:absolute;inset:0;background:radial-gradient(520px 380px at 88% 18%,rgba(247,134,42,.28),transparent 70%),radial-gradient(560px 420px at 70% 75%,rgba(106,136,226,.30),transparent 70%),radial-gradient(420px 300px at 8% 100%,rgba(165,187,252,.35),transparent 70%)"></div>
{blob('sunny','mist',150,-40,-50,0,.55)}
{blob('clover','dusk',120,560,500,15,.9)}
{blob('cookie','mist',70,1180,560,0,.8)}
<div style="position:absolute;left:84px;top:96px;display:flex;align-items:center;gap:16px">{mark(56)}
<span style="font-size:44px;font-weight:560;letter-spacing:-.04em">resenh</span></div>
<div style="position:absolute;left:84px;top:208px;width:560px;font-size:64px;line-height:1.02;font-weight:430;letter-spacing:-.035em;font-variation-settings:'opsz' 144">Photos from the dark, brought back to life.</div>
<div style="position:absolute;left:86px;top:378px;width:470px;font-size:22px;line-height:1.45;color:#4A5072">Classical restoration and a transformer network, judged photo by photo. In your browser or in Python.</div>
<div style="position:absolute;left:84px;top:488px;display:flex;gap:12px;align-items:center">
 <span class="mono" style="font-size:19px;background:#fff;border:1px solid rgba(30,32,51,.12);border-radius:999px;padding:12px 22px">pip install resenh</span>
 <span style="font-size:19px;font-weight:500;color:#fff;border-radius:999px;padding:13px 24px;background:linear-gradient(180deg,#3A3F5C,#1E2033)">resenh.sharadh.blog</span></div>
<div style="position:absolute;left:676px;top:92px;width:330px;height:330px;clip-path:{clip('cookie')};background:url({a}) center/cover"></div>
<div style="position:absolute;left:862px;top:212px;width:360px;height:360px;clip-path:{clip('flower')};background:url({b}) center/cover"></div>
<div style="position:absolute;left:744px;top:430px;font-size:15px;color:#7A8099">before</div>
<div style="position:absolute;left:1146px;top:584px;font-size:15px;color:#1E2033;font-weight:500">after</div>
</div></body></html>"""


def triptych() -> str:
    """How it works, illustrated: measure, two restorers, a judge."""
    src = img64("site/public/hero/dicm_20_in.webp")
    cards = [
        ("sunny", "Measure", "Brightness, noise, colour cast, uneven light and highlights set every parameter.",
         "", img64("site/public/hero/lime_1_in.webp")),
        ("clover", "Restore twice", "A classical Retinex pipeline and a Retinexformer network work side by side.",
         "", img64("site/public/hero/npe_0_out.webp")),
        ("flower", "Judge and fuse", "A quality controller scores both and keeps what each does well.",
         "", img64("site/public/hero/lime_1_out.webp")),
    ]
    _ = src
    items = "".join(f"""<div style="flex:1;position:relative;background:#fff;border-radius:28px;padding:36px 32px 34px;box-shadow:0 0 0 1px rgba(30,32,51,.07)">
<div style="width:150px;height:150px;clip-path:{clip(s)};background:url({im}) center/cover;{f}"></div>
<div class="mono" style="margin-top:26px;font-size:14px;color:#6A88E2">{i + 1:02d}</div>
<div style="margin-top:6px;font-size:28px;font-weight:520;letter-spacing:-.02em">{t}</div>
<div style="margin-top:8px;font-size:17px;line-height:1.5;color:#4A5072">{d}</div></div>""" for i, (s, t, d, f, im) in enumerate(cards))
    return HEAD + f"""<div class="stage" style="width:1280px;height:470px;background:#FAFBFF;padding:40px 44px">
<div style="position:absolute;inset:0;background:radial-gradient(500px 260px at 100% 0%,rgba(247,134,42,.18),transparent 70%),radial-gradient(500px 300px at 0% 100%,rgba(106,136,226,.20),transparent 70%)"></div>
<div style="position:relative;display:flex;gap:20px">{items}</div></div></body></html>"""


def render(html: str, name: str, w: int, h: int, transparent=False):
    from playwright.sync_api import sync_playwright  # noqa: PLC0415

    f = OUT / "src" / f"{name}.html"
    f.parent.mkdir(parents=True, exist_ok=True)
    f.write_text(html, encoding="utf-8")
    with sync_playwright() as p:
        b = p.chromium.launch()
        pg = b.new_page(viewport={"width": w, "height": h}, device_scale_factor=2)
        pg.goto(f.as_uri(), wait_until="networkidle")
        pg.wait_for_timeout(600)
        pg.locator(".stage").first.screenshot(path=str(OUT / f"{name}.png"), omit_background=transparent)
        b.close()
    print("wrote", OUT / f"{name}.png")


if __name__ == "__main__":
    (OUT / "logo-mark.svg").write_text(
        '<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 100 100"><defs><radialGradient id="g" cx="35%" cy="30%" r="80%">'
        '<stop offset="0" stop-color="#FDC58A"/><stop offset=".55" stop-color="#F7862A"/><stop offset="1" stop-color="#A5BBFC"/>'
        f'</radialGradient></defs><path d="{shape("flower")}" fill="url(#g)"/></svg>', encoding="utf-8")
    render(logo(False), "logo-light", 560, 160, transparent=True)
    render(logo(True), "logo-dark", 560, 160, transparent=True)
    render(banner(), "banner", 1280, 640)
    render(triptych(), "how-it-works", 1280, 470)
