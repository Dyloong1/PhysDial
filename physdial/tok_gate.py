#!/usr/bin/env python
"""Reconstruction gate of the tokenizer (train_tok2d.py runs it after decoder fine-tuning): sampled frames of a split are normalised,
encoded and decoded channel by channel, and measured on the physical field:
  rel_l2            ||x' - x|| / ||x||
  band retention    energy of x' / energy of x in bands of the normalised wavenumber kappa = |(k_x / (N_x/2), k_y / (N_y/2))|:
                    (0, 1/8], (1/8, 1/4], (1/4, 1/2], (1/2, 1] (top band = upper half of the resolved wavenumbers, the dissipation range
                    of the turbulent families; for N = 256 it is 64 < k <= 128). Non-periodic axes get a Hann window
                    (same window on x and x').
  grad ratio        <|grad x'|^2> / <|grad x|^2> (2D vorticity: palinstrophy ratio)
Normalisation: each frame and channel on its own, x_n = (x - mean) / max|x - mean| (in [-1, 1]; fieldnorm, as encode_corpus and the
evaluation path), so amplitudes that differ by decades between families, conditions or times (e.g. decaying turbulence) all reach the
tokenizer at full scale.
Check (run by train_tok2d.py with the fine-tuned decoder, 24 frames of every evaluation split): every channel of every split
must reach rel_l2 <= 0.05 and grad ratio in [0.9, 1.1]. Band retention is reported alongside."""
import os, sys
HERE = os.path.dirname(os.path.abspath(__file__)); PROJ = os.path.dirname(HERE); sys.path.insert(0, PROJ); sys.path.insert(0, HERE)
import numpy as np, torch
import fieldnorm

EDGES = [0, 1 / 8, 1 / 4, 1 / 2, 1.0 + 1e-9]; BANDS = ["(0,1/8]", "(1/8,1/4]", "(1/4,1/2]", "(1/2,1]"]


def window(H, W, periodic):
    w = np.ones((H, W))
    if not periodic[0]: w *= np.hanning(H)[:, None]
    if not periodic[1]: w *= np.hanning(W)[None, :]
    return w


def band_energy(x, win):                         # x [N, H, W] -> [N, 4]
    H, W = x.shape[-2:]; X = np.fft.fft2(x * win) / (H * W)
    kx = np.fft.fftfreq(H) * 2; ky = np.fft.fftfreq(W) * 2                      # k / (N/2) in [-1, 1)
    kap = np.sqrt(kx[:, None] ** 2 + ky[None, :] ** 2); P = np.abs(X) ** 2
    return np.stack([P[:, (kap > lo) & (kap <= hi)].sum(-1) for lo, hi in zip(EDGES[:-1], EDGES[1:])], -1)


def grad2(x, periodic):
    if periodic[0]: gx = (np.roll(x, -1, -2) - np.roll(x, 1, -2)) / 2
    else: gx = np.gradient(x, axis=-2)
    if periodic[1]: gy = (np.roll(x, -1, -1) - np.roll(x, 1, -1)) / 2
    else: gy = np.gradient(x, axis=-1)
    return (gx ** 2 + gy ** 2).mean()


class Acc:
    """pooled metrics of one channel over frames of possibly different grid sizes (kappa bands are grid-independent)."""
    def __init__(s): s.e2 = s.x2 = s.xm2 = s.gr = s.gt = 0.0; s.n = 0; s.Et = np.zeros(4); s.Er = np.zeros(4); s.mx = []
    def add(s, x, r, periodic):                    # x, r [N, H, W] float64 (same grid)
        win = window(*x.shape[-2:], periodic); s.Et += band_energy(x, win).sum(0); s.Er += band_energy(r, win).sum(0)
        s.e2 += ((r - x) ** 2).sum(); s.x2 += (x ** 2).sum(); s.xm2 += ((x - x.mean((-1, -2), keepdims=True)) ** 2).sum()
        s.gr += grad2(r, periodic) * len(x); s.gt += grad2(x, periodic) * len(x); s.n += len(x)
        mx = np.abs(x - x.mean((-1, -2), keepdims=True)).max((-1, -2)); mr = np.abs(r - r.mean((-1, -2), keepdims=True)).max((-1, -2))
        s.mx += (mr / np.maximum(mx, 1e-30)).tolist()   # extreme-value fidelity per frame
    def result(s):
        return {"rel_l2": float(np.sqrt(s.e2 / max(s.x2, 1e-30))), "rel_l2_fluct": float(np.sqrt(s.e2 / max(s.xm2, 1e-30))),
                "band_retention": {b: float(s.Er[i] / max(s.Et[i], 1e-30)) for i, b in enumerate(BANDS)},
                "top_fraction": {"true": float(s.Et[-1] / max(s.Et.sum(), 1e-30)), "recon": float(s.Er[-1] / max(s.Er.sum(), 1e-30))},
                "grad_ratio": float(s.gr / max(s.gt, 1e-30)), "n_frames": s.n,
                "max_ratio_median": float(np.median(s.mx)) if s.mx else None, "max_ratio_mean": float(np.mean(s.mx)) if s.mx else None}


def passes(m):
    """the reconstruction check: rel_l2 <= 0.05 and gradient-energy ratio in [0.9, 1.1]"""
    return bool(m["rel_l2"] <= 0.05 and 0.9 <= m["grad_ratio"] <= 1.1)


def sample(fam, split, n, rng):
    """n frames of a split, grouped by grid size: {(H, W): [k, C, H, W]}."""
    its = fam.items(split)
    if not its: return None
    G = {}
    for j in rng.choice(len(its), n, replace=len(its) < n):
        it = its[j]; t = int(rng.integers(fam.t_start, it.n_frames)); x = fam.load(it, t, t + 1)[0]; G.setdefault(x.shape[-2:], []).append(x)
    return {k: np.stack(v) for k, v in G.items()}


@torch.no_grad()
def roundtrip(tok, X, bs):
    """X [n, C, H, W] physical -> reconstruction, channel by channel, each frame normalised on its own (fieldnorm), and the latent shape."""
    mean, sc = fieldnorm.scale_np(X)
    Xn = (X - mean) / sc; R = np.empty_like(Xn); lat = None
    for c in range(X.shape[1]):
        for i in range(0, len(X), bs):
            x = torch.from_numpy(Xn[i:i + bs, c:c + 1]).float().cuda()
            with torch.autocast("cuda", dtype=torch.bfloat16):
                z = tok.encode(x).float(); lat = tuple(z.shape[1:]); R[i:i + bs, c:c + 1] = tok.decode(z).float().cpu().numpy()
    return R * sc + mean, lat
