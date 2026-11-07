"""Fine-tune Retinexformer (+ classical priors) with the GT-mean loss.

Honest protocol: checkpoints are selected on ``val_a`` (carved from the official train split by
``data_prep.py``); the test split is never touched here.

Example::

    python training/train.py --bench lolv2_syn --init weights/retinexformer/LOL_v2_synthetic.pth \
        --out runs/syn_light --iters 30000

Use ``--no-priors`` / ``--plain-loss`` for ablations.
"""

from __future__ import annotations

import argparse
import copy
import json
import math
import random
import sys
import time
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from resenh.classical.priors import PRIOR_CHANNELS, compute_priors  # noqa: E402
from resenh.models.retinexformer import RetinexFormer  # noqa: E402


def read_rgb(path: str) -> np.ndarray:
    img = cv2.imread(path, cv2.IMREAD_COLOR)
    if img is None:
        raise FileNotFoundError(path)
    return cv2.cvtColor(img, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0


class PairDataset(Dataset):
    def __init__(self, pairs, patch: int | None, priors: bool, gamma_aug: bool = False,
                 cache: bool = True, noise_aug: bool = False):
        self.pairs, self.patch, self.priors, self.gamma_aug = pairs, patch, priors, gamma_aug
        self.noise_aug = noise_aug
        self.cache: dict[int, tuple] = {} if cache else None

    def __len__(self):
        return len(self.pairs)

    def _load(self, i):
        if self.cache is not None and i in self.cache:
            return self.cache[i]
        lo, hi = read_rgb(self.pairs[i][0]), read_rgb(self.pairs[i][1])
        item = (lo, hi)
        if self.cache is not None:
            self.cache[i] = item
        return item

    def __getitem__(self, i):
        lo, hi = self._load(i)
        if self.gamma_aug and self.patch:
            g = random.uniform(0.8, 1.25)
            lo = np.clip(lo, 0, 1) ** g
        if getattr(self, "noise_aug", False) and self.patch and random.random() < 0.5:
            # Poisson-Gaussian sensor noise (read + shot), as in real phone night shots
            read, shot = random.uniform(0.0, 0.03), random.uniform(0.0, 0.004)
            lo = lo + np.random.normal(0, 1, lo.shape).astype(np.float32) * np.sqrt(
                read ** 2 + shot * np.maximum(lo, 0))
            lo = np.clip(lo, 0, 1).astype(np.float32)
        pr = compute_priors(lo) if self.priors else None
        if self.patch:
            h, w = lo.shape[:2]
            p = min(self.patch, h, w)
            y, x = random.randint(0, h - p), random.randint(0, w - p)
            sl = (slice(y, y + p), slice(x, x + p))
            lo, hi = lo[sl], hi[sl]
            pr = pr[sl] if pr is not None else None
            k, flip = random.randint(0, 3), random.random() < 0.5
            def aug(a):
                a = np.rot90(a, k)
                return a[:, ::-1] if flip else a
            lo, hi = aug(lo), aug(hi)
            pr = aug(pr) if pr is not None else None
        t = lambda a: torch.from_numpy(np.ascontiguousarray(a.transpose(2, 0, 1)))  # noqa: E731
        out = {"lq": t(lo), "gt": t(hi)}
        if pr is not None:
            out["pr"] = t(pr)
        return out


# ---------------------------------------------------------------- losses
def charbonnier(x, y, eps=1e-3):
    return torch.sqrt((x - y) ** 2 + eps * eps).mean()


def _gauss_window(c, size=11, sigma=1.5, device=None):
    g = torch.arange(size, dtype=torch.float32, device=device) - size // 2
    g = torch.exp(-(g ** 2) / (2 * sigma ** 2))
    g = (g / g.sum())[:, None] @ (g / g.sum())[None, :]
    return g.expand(c, 1, size, size).contiguous()


def ssim_loss(x, y):
    c = x.shape[1]
    w = _gauss_window(c, device=x.device).to(x.dtype)
    mu_x, mu_y = F.conv2d(x, w, groups=c), F.conv2d(y, w, groups=c)
    sxx = F.conv2d(x * x, w, groups=c) - mu_x ** 2
    syy = F.conv2d(y * y, w, groups=c) - mu_y ** 2
    sxy = F.conv2d(x * y, w, groups=c) - mu_x * mu_y
    c1, c2 = 0.01 ** 2, 0.03 ** 2
    s = ((2 * mu_x * mu_y + c1) * (2 * sxy + c2)) / ((mu_x ** 2 + mu_y ** 2 + c1) * (sxx + syy + c2))
    return 1 - s.mean()


class VGGPerceptual(torch.nn.Module):
    def __init__(self):
        super().__init__()
        from torchvision.models import VGG19_Weights, vgg19

        feats = vgg19(weights=VGG19_Weights.IMAGENET1K_V1).features[:27].eval()
        for p in feats.parameters():
            p.requires_grad_(False)
        self.feats = feats
        self.register_buffer("mean", torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1))
        self.register_buffer("std", torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1))

    def forward(self, x, y):
        x, y = (x - self.mean) / self.std, (y - self.mean) / self.std
        loss, layers = 0.0, {3, 8, 17, 26}
        for i, m in enumerate(self.feats):
            x, y = m(x), m(y)
            if i in layers:
                loss = loss + F.l1_loss(x, y)
        return loss


class BaseLoss(torch.nn.Module):
    def __init__(self, w_ssim=0.2, w_vgg=0.02):
        super().__init__()
        self.w_ssim, self.w_vgg = w_ssim, w_vgg
        self.vgg = VGGPerceptual() if w_vgg > 0 else None

    def forward(self, x, y):
        loss = charbonnier(x, y) + self.w_ssim * ssim_loss(x, y)
        if self.vgg is not None:
            loss = loss + self.w_vgg * self.vgg(x, y)
        return loss


class GTMeanLoss(torch.nn.Module):
    """GT-mean loss (Liao et al., ICCV 2025, arXiv 2507.20148).

    L = W * L(f, y) + (1 - W) * L(m * f, y),  m = E[y] / E[f]  (per image, gray means)
    W = clip(D_B, 0, 1), D_B = Bhattacharyya distance between N(E[f], (s E[f])^2) and
    N(E[y], (s E[y])^2), s = 0.1. Training-only: inference uses the raw output.
    """

    def __init__(self, base: torch.nn.Module, sigma=0.1):
        super().__init__()
        self.base, self.sigma = base, sigma

    def forward(self, x, y):
        mx = x.mean(dim=(1, 2, 3)).clamp_min(1e-4)
        my = y.mean(dim=(1, 2, 3)).clamp_min(1e-4)
        v1, v2 = (self.sigma * mx) ** 2, (self.sigma * my) ** 2
        db = 0.25 * (mx - my) ** 2 / (v1 + v2) + 0.5 * torch.log((v1 + v2) / (2 * torch.sqrt(v1 * v2)))
        w = db.clamp(0, 1).detach().view(-1, 1, 1, 1)
        scale = (my / mx).detach().view(-1, 1, 1, 1)
        # weighted per-sample by splitting the batch-mean loss into the two terms
        l_raw = self.base(x, y)
        l_adj = self.base((x * scale).clamp(0, 1), y)
        wm = w.mean()
        return wm * l_raw + (1 - wm) * l_adj


# ---------------------------------------------------------------- eval
def pad_to(x, m=8):
    h, w = x.shape[-2:]
    ph, pw = (m - h % m) % m, (m - w % m) % m
    return F.pad(x, (0, pw, 0, ph), mode="reflect"), h, w


@torch.no_grad()
def evaluate(model, loader, device, use_priors):
    model.eval()
    psnrs = []
    for b in loader:
        lq, gt = b["lq"].to(device), b["gt"].to(device)
        x, h, w = pad_to(lq)
        pr = pad_to(b["pr"].to(device))[0] if use_priors else None
        with torch.autocast("cuda", dtype=torch.bfloat16, enabled=device.type == "cuda"):
            out = model(x, pr) if use_priors else model(x)
        out = out.float()[..., :h, :w].clamp(0, 1)
        out = torch.round(out * 255) / 255  # score the uint8 image that would be saved
        mse = ((out - gt) ** 2).mean(dim=(1, 2, 3))
        psnrs += (10 * torch.log10(1 / mse.clamp_min(1e-10))).tolist()
    model.train()
    return float(np.mean(psnrs))


# ---------------------------------------------------------------- main
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--bench", required=True, help="lolv1 | lolv2_real | lolv2_syn | generalist")
    ap.add_argument("--manifest", default="data/manifest.json")
    ap.add_argument("--init", default=None, help="official Retinexformer checkpoint")
    ap.add_argument("--out", required=True)
    ap.add_argument("--iters", type=int, default=30000)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--schedule", default="128:8:0.4,256:4:0.35,384:2:0.25",
                    help="patch:batch:fraction stages (progressive patches)")
    ap.add_argument("--val-every", type=int, default=1000)
    ap.add_argument("--ema", type=float, default=0.999)
    ap.add_argument("--no-priors", action="store_true")
    ap.add_argument("--plain-loss", action="store_true", help="disable GT-mean loss")
    ap.add_argument("--w-vgg", type=float, default=0.02)
    ap.add_argument("--w-ssim", type=float, default=0.2)
    ap.add_argument("--gamma-aug", action="store_true")
    ap.add_argument("--noise-aug", action="store_true")
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    random.seed(args.seed), np.random.seed(args.seed), torch.manual_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    use_priors = not args.no_priors
    man = json.loads(Path(args.manifest).read_text())
    if args.bench == "generalist":
        dedup = set(sum(json.loads(Path("data/dedup.json").read_text()).values(), []))
        train = [p for b in ("lolv1", "lolv2_real", "lolv2_syn", "lsrw") if b in man
                 for p in man[b]["train"] if p[0] not in dedup]
        val = [p for b in ("lolv2_real", "lolv2_syn", "lsrw") if b in man for p in man[b]["val_a"]]
    else:
        train, val = man[args.bench]["train"], man[args.bench]["val_a"]
    print(f"train {len(train)} val {len(val)} priors={use_priors} gtmean={not args.plain_loss}")

    model = RetinexFormer(prior_channels=PRIOR_CHANNELS if use_priors else 0).to(device)
    if args.init:
        model.load_official(torch.load(args.init, map_location="cpu", weights_only=False))
    ema = copy.deepcopy(model).eval()
    for p in ema.parameters():
        p.requires_grad_(False)

    base = BaseLoss(args.w_ssim, args.w_vgg).to(device)
    crit = base if args.plain_loss else GTMeanLoss(base)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, betas=(0.9, 0.999), weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.LambdaLR(
        opt, lambda it: min(1.0, (it + 1) / 500) * 0.5 * (1 + math.cos(math.pi * it / args.iters)))

    val_loader = DataLoader(PairDataset(val, None, use_priors), batch_size=1, num_workers=2)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    (out / "args.json").write_text(json.dumps(vars(args), indent=1))
    base_psnr = evaluate(ema, val_loader, device, use_priors)
    print(f"[init] val_a PSNR {base_psnr:.3f}")
    best, log = base_psnr, [{"iter": 0, "val_psnr": base_psnr}]
    torch.save({"params": ema.state_dict(), "iter": 0, "val_psnr": best}, out / "best.pth")

    stages = []
    for s in args.schedule.split(","):
        p, b, f = s.split(":")
        stages.append((int(p), int(b), float(f)))
    it, t0 = 0, time.time()
    for patch, bs, frac in stages:
        n_it = int(round(frac * args.iters)) if (patch, bs, frac) != stages[-1] else args.iters - it
        ds = PairDataset(train, patch, use_priors, gamma_aug=args.gamma_aug, cache=False,
                         noise_aug=args.noise_aug)
        dl = DataLoader(ds, batch_size=bs, shuffle=True, num_workers=args.workers, drop_last=True,
                        persistent_workers=args.workers > 0, pin_memory=True)
        end = it + n_it
        while it < end:
            for b in dl:
                lq, gt = b["lq"].to(device, non_blocking=True), b["gt"].to(device, non_blocking=True)
                pr = b["pr"].to(device, non_blocking=True) if use_priors else None
                with torch.autocast("cuda", dtype=torch.bfloat16, enabled=device.type == "cuda"):
                    pred = model(lq, pr) if use_priors else model(lq)
                loss = crit(pred.float(), gt)
                opt.zero_grad(set_to_none=True)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 0.5)
                opt.step()
                sched.step()
                with torch.no_grad():
                    for pe, pm in zip(ema.parameters(), model.parameters()):
                        pe.lerp_(pm, 1 - args.ema)
                it += 1
                if it % 100 == 0:
                    print(f"it {it} patch {patch} loss {loss.item():.4f} "
                          f"lr {sched.get_last_lr()[0]:.2e} {time.time() - t0:.0f}s", flush=True)
                if it % args.val_every == 0 or it == args.iters:
                    v = evaluate(ema, val_loader, device, use_priors)
                    log.append({"iter": it, "val_psnr": v})
                    flag = ""
                    if v > best:
                        best, flag = v, " *best*"
                        torch.save({"params": ema.state_dict(), "iter": it, "val_psnr": v},
                                   out / "best.pth")
                    print(f"[val] it {it} val_a PSNR {v:.3f} (best {best:.3f}){flag}", flush=True)
                    (out / "log.json").write_text(json.dumps(log, indent=1))
                if it >= end:
                    break
    torch.save({"params": ema.state_dict(), "iter": it}, out / "last.pth")
    print(f"done: best val_a PSNR {best:.3f} (init {base_psnr:.3f})")


if __name__ == "__main__":
    main()
