#!/usr/bin/env python
"""Small specialist baselines: one small model per family, trained from scratch and scored on the same corpus items, native grids and
evaluation windows as the main model: train_gen.eval_windows (EVAL_SEED, EVAL_HIST 6, 16 frames at the family stride), the per-window /
frame / channel relative L2 of eval_metrics with apply_degenerate, and train_gen's json fields (rel_l2_by_frame = mean over windows and
channels, rel_l2@1 / @16, rel_l2_pooled_by_frame, persistence@1 / @16).

Models (1 GPU):
  unet : a conditioned U-Net in the style of PDEArena (Gupta & Brandstetter 2022, modules/conditioned): residual blocks with GroupNorm and GELU,
         FiLM scale / shift in every block from an MLP over a Fourier embedding of the condition scalars, 4 levels (96-192-192-384 channels),
         circular padding on periodic axes; condition fields as extra input channels. ~28M parameters.
  fno  : FNO-2D (Li et al. 2021; non-periodic axes zero domain-padded by 1/8): lifting, 4 Fourier layers (width 96, --modes per axis, capped at
         the grid; 16: ~19M parameters, 32: ~76M), 1x1 skip, projection; the condition scalars (values + presence mask) as constant input
         channels, condition fields as input channels.
Input: the last --k_in (4) frames at the family stride, each channel normalised with the training mean / std of the family; condition fields =
the family's cond_model (a fixed per-family transform); condition vector = the family's slots of
corpus.effective_params (as the main model's numeric input), standardised over the training items, plus the presence mask. Output: the next
frame as a residual on the last input frame (normalised units). Loss MSE; Gaussian input noise --noise (normalised units; 0.01 in every run) on
the history frames against roll-out drift.
Training: AdamW (U-Net 3e-4, FNO 1e-3, weight decay 1e-4), 2 % warm-up then cosine to 0 on the wall-clock budget --budget_gpu_h (one GPU),
grad clip 1, bf16 autocast for the U-Net (FNO in fp32: complex FFTs), EMA 0.999 of the weights for evaluation. Resumable: ckpt/<name>.state.pt
every 20 minutes (elapsed time, step, optimizer, EMA), so an interrupted run continues.
Data: the family's train split (+ the Kolmogorov amplitude-anchor trajectories, which every other model gets: kolmo_extra anchor,drag);
--reference adds the family's ref_* splits (the in-range reference); --switch_train adds the Kolmogorov switch training groups (switch_train via switch_data):
windows whose target lies after the switch (history frames may be before it, as in the main model's switch training), first start
OFF = 10 - (k_in - 1) stride, starts up to --H model steps later, labelled with the post-switch condition. Every trajectory is drawn with
equal probability (within a grid group, grid groups by trajectory count), the window start uniformly in its range.
Evaluation (end of training, or --eval_only <run>): every evaluation column of the family (or --splits), --frames steps; --dump_main also
saves the prediction and truth fields of the first --dump_pred windows of every evaluated split to physdial/baselines/results/pred/<name>/
(spectra); --switch_eval evaluates every Kolmogorov switch view switch_<group>_<view> (views as switch_corpus) instead and writes per-window
predictions for switch/analyze_switch.py; the JSON goes to physdial/baselines/results and, for such a run, also to switch/results."""
import argparse, json, math, os, sys, time
HERE = os.path.dirname(os.path.abspath(__file__)); PKG = os.path.dirname(HERE); PROJ = os.path.dirname(PKG); SWITCH_DATA = os.path.join(PROJ, "switch")
sys.path.insert(0, PROJ); sys.path.insert(0, PKG); sys.path.insert(0, SWITCH_DATA)
import numpy as np, torch, torch.nn as nn, torch.nn.functional as F
import corpus
from eval_metrics import rel_l2, rel_l2_pooled, truth_stats, apply_degenerate, nonfinite_to_inf, physics_metrics, new_acc, accumulate_last_frame

SWITCH_SHARE = 0.21   # upper bound of the share of batches drawn from switch windows (--switch_train)


# ------------------------------------------------------------------------------------------------------------------ models
def pad_mode(periodic):
    periodic = tuple(bool(x) for x in periodic)
    if len(set(periodic)) == 1: return "circular" if periodic[0] else "zeros"
    return periodic


class MixConv(nn.Conv2d):
    """3x3 conv with per-axis padding: circular on the periodic axis, zeros on the other (periodic = (axis -2, axis -1), as eval_metrics)."""
    def __init__(s, ci, co, k, stride=1, padding=1, periodic=(True, False)):
        super().__init__(ci, co, k, stride=stride, padding=0); s.p = padding; s.per = periodic
    def forward(s, x):
        p = s.p
        x = F.pad(x, (0, 0, p, p), mode="circular" if s.per[0] else "constant")
        x = F.pad(x, (p, p, 0, 0), mode="circular" if s.per[1] else "constant")
        return super().forward(x)


def conv3(ci, co, pm, stride=1):
    return nn.Conv2d(ci, co, 3, stride=stride, padding=1, padding_mode=pm) if isinstance(pm, str) else MixConv(ci, co, 3, stride, 1, pm)


class FourierEmb(nn.Module):
    """condition scalars [B, P] -> [B, D]: [c, sin(2^k c), cos(2^k c)], k = 0..3, through a 2-layer MLP (PDEArena embeds its scalar parameters with
    sinusoidal features + MLP)."""
    def __init__(s, P, D):
        super().__init__(); s.P = P; s.mlp = nn.Sequential(nn.Linear(P * 9, D), nn.GELU(), nn.Linear(D, D))
    def forward(s, c):
        f = torch.cat([c] + [g(c * 2.0 ** k) for k in range(4) for g in (torch.sin, torch.cos)], -1); return s.mlp(f)


class ResBlock(nn.Module):
    def __init__(s, ci, co, D, pm):
        super().__init__()
        s.n1, s.c1 = nn.GroupNorm(8, ci), conv3(ci, co, pm)
        s.n2, s.c2 = nn.GroupNorm(8, co), conv3(co, co, pm)
        s.film = nn.Linear(D, 2 * co); s.skip = nn.Conv2d(ci, co, 1) if ci != co else nn.Identity()
        nn.init.zeros_(s.film.weight); nn.init.zeros_(s.film.bias)
    def forward(s, x, e):
        h = s.c1(F.gelu(s.n1(x))); sc, sh = s.film(e)[:, :, None, None].chunk(2, 1)
        h = s.n2(h) * (1 + sc) + sh; return s.c2(F.gelu(h)) + s.skip(x)


class UNet(nn.Module):
    def __init__(s, cin, cout, P, periodic, chs=(96, 192, 192, 384), D=256):
        super().__init__(); pm = pad_mode(periodic); s.emb = FourierEmb(P, D)
        s.inp = conv3(cin, chs[0], pm)
        s.down, s.pool = nn.ModuleList(), nn.ModuleList(); c = chs[0]; skips = []
        for i, ch in enumerate(chs):
            s.down.append(nn.ModuleList([ResBlock(c, ch, D, pm), ResBlock(ch, ch, D, pm)])); c = ch; skips.append(ch)
            s.pool.append(conv3(ch, ch, pm, stride=2) if i < len(chs) - 1 else nn.Identity())
        s.mid = nn.ModuleList([ResBlock(c, c, D, pm), ResBlock(c, c, D, pm)])
        s.up, s.ups = nn.ModuleList(), nn.ModuleList()
        for i, ch in reversed(list(enumerate(chs))):
            s.up.append(nn.ModuleList([ResBlock(c + skips[i], ch, D, pm), ResBlock(ch, ch, D, pm)])); c = ch
            s.ups.append(conv3(ch, chs[i - 1], pm) if i > 0 else nn.Identity()); c = chs[i - 1] if i > 0 else ch
        s.out = nn.Sequential(nn.GroupNorm(8, c), nn.GELU(), conv3(c, cout, pm))
        nn.init.zeros_(s.out[-1].weight); nn.init.zeros_(s.out[-1].bias)
    def forward(s, x, c):
        e = s.emb(c); h = s.inp(x); hs = []
        for (b1, b2), p in zip(s.down, s.pool):
            h = b2(b1(h, e), e); hs.append(h); h = p(h)
        for b in s.mid: h = b(h, e)
        for (b1, b2), u in zip(s.up, s.ups):
            h = b2(b1(torch.cat([h, hs.pop()], 1), e), e)
            if not isinstance(u, nn.Identity): h = u(F.interpolate(h, scale_factor=2, mode="nearest"))
        return s.out(h)


class SpectralConv(nn.Module):
    def __init__(s, w, m):
        super().__init__(); s.m = m; sc = 1.0 / (w * w)
        s.w1 = nn.Parameter(sc * torch.randn(w, w, m, m, dtype=torch.cfloat)); s.w2 = nn.Parameter(sc * torch.randn(w, w, m, m, dtype=torch.cfloat))
    def forward(s, x):
        B, C, H, W = x.shape; m1, m2 = min(s.m, H // 2), min(s.m, W // 2 + 1)
        xf = torch.fft.rfft2(x.float()); o = torch.zeros(B, C, H, W // 2 + 1, dtype=torch.cfloat, device=x.device)
        o[:, :, :m1, :m2] = torch.einsum("bixy,ioxy->boxy", xf[:, :, :m1, :m2], s.w1[:, :, :m1, :m2])
        o[:, :, -m1:, :m2] = torch.einsum("bixy,ioxy->boxy", xf[:, :, -m1:, :m2], s.w2[:, :, :m1, :m2])
        return torch.fft.irfft2(o, s=(H, W))


class FNO(nn.Module):
    def __init__(s, cin, cout, P, periodic, width=96, modes=16, layers=4):
        super().__init__()
        s.dpad = tuple(0 if per else 8 for per in periodic); s.dfrac = 8
        s.lift = nn.Conv2d(cin + 2 * P, width, 1)
        s.spec = nn.ModuleList([SpectralConv(width, modes) for _ in range(layers)]); s.loc = nn.ModuleList([nn.Conv2d(width, width, 1) for _ in range(layers)])
        s.proj = nn.Sequential(nn.Conv2d(width, 128, 1), nn.GELU(), nn.Conv2d(128, cout, 1))
        nn.init.zeros_(s.proj[-1].weight); nn.init.zeros_(s.proj[-1].bias)
    def forward(s, x, c):
        B, _, H, W = x.shape; h = s.lift(torch.cat([x, c[:, :, None, None].expand(B, c.shape[1], H, W)], 1))
        ph, pw = (max(s.dpad[0], H // s.dfrac) if s.dpad[0] else 0), (max(s.dpad[1], W // s.dfrac) if s.dpad[1] else 0)
        if ph or pw: h = F.pad(h, (0, pw, 0, ph))
        for i, (sp, lc) in enumerate(zip(s.spec, s.loc)):
            h = sp(h) + lc(h); h = F.gelu(h) if i < len(s.spec) - 1 else h
        return s.proj(h[:, :, :H, :W])


# ------------------------------------------------------------------------------------------------------------------ data
class Fam:
    """everything family-specific: items, normalisation, condition vector, condition fields, window ranges."""
    def __init__(s, key, a, stats=None):
        s.key = key; s.f = corpus.get(key); s.st = s.f.stride; s.K = a.k_in; s.C = len(s.f.channels)
        tr = corpus.cached_items(key, "train") + (corpus.cached_items(key, "anchor") if key == "kolmo" and "anchor" in corpus.kolmo_extra_splits() else [])
        if a.reference: tr += [it for sp in s.f.splits() if sp.startswith("ref_") for it in corpus.cached_items(key, sp)]
        s.train = [(it, s.f.t_start, it.n_frames - s.K * s.st - 1) for it in tr]
        s.sw = []
        if a.switch_train:
            off = 10 - (s.K - 1) * s.st; assert off >= 0
            for it in corpus.cached_items(key, "switch_train"): s.sw.append((it, off, min(off + a.H * s.st, it.n_frames - s.K * s.st - 1)))
        s.plain = [w for w in s.train if w[2] >= w[1]]; s.sw = [w for w in s.sw if w[2] >= w[1]]; s.train = s.plain + s.sw; assert s.plain, f"{key}: no training windows"
        s.sw_share = min(len(s.sw) / len(s.train), SWITCH_SHARE) if s.sw else 0.0
        if stats is None:
            v = [corpus.param_vector(corpus.effective_params(key, it.params))[1] for it, _, _ in s.train]
            s.slots = [sl for i, sl in enumerate(corpus.SLOTS) if max(m[i] for m in v) > 0]
            V = np.array([s.vec_raw(it)[0] for it, _, _ in s.train]); mu, sd = V.mean(0), V.std(0); sd[sd < 1e-6] = 1.0
            rng = np.random.default_rng(0); W_ = [s.train[i] for i in rng.choice(len(s.train), min(64, len(s.train)), replace=False)]
            fr_ = [s.f.load(it, t, t + 1)[0].astype(np.float64) for it, t in ((it, int(rng.integers(lo, hi + 1))) for it, lo, hi in W_)]   # uniformly in the window range
            m1 = np.mean([f.mean((1, 2)) for f in fr_], 0); m2 = np.mean([(f ** 2).mean((1, 2)) for f in fr_], 0)   # per frame, so grids may differ (Kolmogorov 256^2 / 512^2)
            stats = {"slots": s.slots, "p_mean": mu.tolist(), "p_std": sd.tolist(), "x_mean": m1.tolist(), "x_std": np.maximum(np.sqrt(np.maximum(m2 - m1 ** 2, 0)), 1e-8).tolist()}
        s.stats = stats; s.slots = stats["slots"]
        s.pm, s.ps = np.array(stats["p_mean"], np.float32), np.array(stats["p_std"], np.float32)
        s.xm, s.xs = np.array(stats["x_mean"], np.float32)[:, None, None], np.array(stats["x_std"], np.float32)[:, None, None]
        s.nf = len(s.fields(s.train[0][0]))

    def vec_raw(s, it):
        return corpus.param_vector(corpus.effective_params(s.key, it.params), s.slots)

    def vec(s, it):
        v, m = s.vec_raw(it); return np.concatenate([(v - s.pm) / s.ps * m, m]).astype(np.float32)

    def fields(s, it):
        return [np.asarray(f, np.float32) for f in s.f.cond_model(it)]

    def norm(s, x): return (x - s.xm) / s.xs


class Stream(torch.utils.data.IterableDataset):
    """endless batches of one grid shape: x [B, K*C + nf, H, W], c [B, 2P], y [B, C, H, W] (normalised)."""
    def __init__(s, fam, bs, seed): s.fam, s.bs, s.seed = fam, bs, seed

    def __iter__(s):
        wi = torch.utils.data.get_worker_info(); rng = np.random.default_rng([s.seed, wi.id if wi else 0])   # seed = --seed + the step the run (re)starts at
        fm = s.fam; pools = []
        for ws_ in (fm.plain, fm.sw):
            by = {}
            for w in ws_: by.setdefault(tuple(w[0].shape), []).append(w)
            shapes = sorted(by); p = np.array([len(by[k]) for k in shapes], float); pools.append((by, shapes, p / max(p.sum(), 1)))
        while True:
            by, shapes, p = pools[1] if fm.sw and rng.random() < fm.sw_share else pools[0]   # a batch is all switch windows with probability sw_share
            ws = by[shapes[rng.choice(len(shapes), p=p)]]; xs, cs, ys = [], [], []
            for _ in range(s.bs):
                it, lo, hi = ws[rng.integers(len(ws))]; t = int(rng.integers(lo, hi + 1))
                X = fm.norm(fm.f.load(it, t, t + fm.K * fm.st + 1)[::fm.st])                  # [K+1, C, H, W]
                xs.append(np.concatenate([X[:fm.K].reshape(-1, *X.shape[2:])] + [f[None] for f in fm.fields(it)])); ys.append(X[fm.K]); cs.append(fm.vec(it))
            yield torch.from_numpy(np.stack(xs)), torch.from_numpy(np.stack(cs)), torch.from_numpy(np.stack(ys))


def build(kind, fam, modes=16):
    cin = fam.K * fam.C + fam.nf; P = len(fam.slots)
    return UNet(cin, fam.C, 2 * P, fam.f.periodic) if kind == "unet" else FNO(cin, fam.C, P, fam.f.periodic, modes=modes)


def step(model, fam, x, c, noise=0.0):
    """x [B, K*C + nf, H, W] normalised -> next frame (normalised): last frame + net."""
    if noise > 0: x = torch.cat([x[:, :fam.K * fam.C] + noise * torch.randn_like(x[:, :fam.K * fam.C]), x[:, fam.K * fam.C:]], 1)
    return x[:, (fam.K - 1) * fam.C:fam.K * fam.C] + model(x, c).float()


# ------------------------------------------------------------------------------------------------------------------ evaluation
@torch.no_grad()
def evaluate(model, fam, a, dev, splits, dump_dir=None):
    import train_gen as TG
    out = {}; Hh = TG.EVAL_HIST; frames = a.frames
    s = fam.st; need = (Hh + frames - 1) * s + 1; assert fam.K <= Hh
    for sp in splits:
        its = corpus.cached_items(fam.key, sp); win = TG.eval_windows(fam.f, its, a.eval_n, TG.EVAL_SEED, frames)
        if not win: out[f"{fam.key}/{sp}"] = {"skipped": f"no trajectory with >= {need} frames"}; continue
        E, EP, PE, PEP, R, meta, dumps, TS = [], [], [], [], [], [], {}, ([], [], []); GT, GR, BT, BR = new_acc(fam.C); order = []
        for i0 in range(0, len(win), a.eval_bs):
            gw = win[i0:i0 + a.eval_bs]; shp = {tuple(it.shape) for it, _ in gw}
            for sh in shp:
                g = [(it, t) for it, t in gw if tuple(it.shape) == sh]
                X = torch.from_numpy(np.stack([fam.f.load(it, t, t + need)[::s] for it, t in g])).to(dev)          # [B, Hh+frames, C, H, W] physical
                xm, xs = torch.from_numpy(fam.xm).to(dev), torch.from_numpy(fam.xs).to(dev)
                h = (X[:, Hh - fam.K:Hh] - xm) / xs; cf = torch.from_numpy(np.stack([np.stack(fam.fields(it)) if fam.nf else np.zeros((0, *sh), np.float32) for it, _ in g])).to(dev)
                cv = torch.from_numpy(np.stack([fam.vec(it) for it, _ in g])).to(dev); P = []
                for _ in range(frames):
                    with torch.autocast("cuda", dtype=torch.bfloat16, enabled=a.model == "unet"):
                        y = step(model, fam, torch.cat([h.flatten(1, 2), cf], 1), cv)
                    P.append(y); h = torch.cat([h[:, 1:], y[:, None]], 1)
                P = torch.stack(P, 1) * xs + xm; T = X[:, Hh:]
                e_, ep_, nf_ = nonfinite_to_inf(P, rel_l2(P - T, T), rel_l2_pooled(P - T, T))
                E.append(e_.cpu()); EP.append(ep_.cpu()); PE.append(rel_l2(X[:, Hh - 1:Hh] - T, T).cpu()); PEP.append(rel_l2_pooled(X[:, Hh - 1:Hh] - T, T).cpu()); R.append(nf_)
                for l_, v_ in zip(TS, truth_stats(T)): l_.append(v_.cpu())   # train_gen: the degenerate-truth masks use the whole split's statistics
                accumulate_last_frame(T, P, fam.f.periodic, GT, GR, BT, BR); order += g
                for j, (it, t) in enumerate(g):
                    meta.append({"path": it.path, "traj": it.traj, "first_frame": int(t)})
                    if dump_dir is not None and len(dumps) // 2 < a.dump_pred:
                        k = len(dumps) // 2; dumps[f"P{k}"] = P[j].half().cpu().numpy(); dumps[f"T{k}"] = T[j].half().cpu().numpy()
        d = {"E": torch.cat(E), "PE": torch.cat(PE), "EP": torch.cat(EP), "PEP": torch.cat(PEP)}
        d = apply_degenerate(d, *[torch.cat(l_) for l_ in TS]); E = d["E"]; fr = torch.nanmean(E, (0, 2))
        r = physics_metrics(E, d["PE"], None, GT, GR, BT, BR, fam.f.channels, 4, EP=d["EP"], PEP=d["PEP"], groups=[fam.f.ci_group(it) for it, _ in order], degenerate=d["degenerate"])   # F = n_out 4: ends {1, 4, 16} as ours
        r["nonfinite_pred_windows"] = int(sum(R)); out[f"{fam.key}/{sp}"] = r
        if dump_dir is not None:
            os.makedirs(dump_dir, exist_ok=True); n = len(dumps) // 2
            np.savez_compressed(os.path.join(dump_dir, f"{fam.key}__{sp}.npz"), meta=json.dumps({"stride": s, "windows": meta[:n]}), **dumps)
        print(f"[eval] {fam.key}/{sp}: n {E.shape[0]} rel_l2@1 {float(fr[0]):.4f} @16 {float(fr[-1]):.4f}", flush=True)
    return out


def eval_splits(fam):
    return [sp for sp in fam.f.splits() if sp not in ("train", "val") + tuple(corpus.TRAIN_ONLY_SPLITS) and not sp.startswith("ref_") and not sp.startswith("switch_")]


# ------------------------------------------------------------------------------------------------------------------ main
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--name", required=True); ap.add_argument("--family", required=True); ap.add_argument("--model", choices=["unet", "fno"], default=None)
    ap.add_argument("--k_in", type=int, default=4); ap.add_argument("--bs", type=int, default=16); ap.add_argument("--lr", type=float, default=0.0)
    ap.add_argument("--noise", type=float, default=0.01); ap.add_argument("--budget_gpu_h", type=float, default=4.0)
    ap.add_argument("--reference", action="store_true"); ap.add_argument("--switch_train", action="store_true"); ap.add_argument("--H", type=int, default=16)
    ap.add_argument("--frames", type=int, default=16, help="evaluation rollout steps (64 = the long-rollout windows)")
    ap.add_argument("--eval_n", type=int, default=32); ap.add_argument("--eval_bs", type=int, default=8); ap.add_argument("--eval_only", default="")
    ap.add_argument("--switch_eval", action="store_true", help="evaluate the switch views of the family instead of its main columns (with --eval_only)")
    ap.add_argument("--dump_pred", type=int, default=32, help="windows per split saved with --switch_eval / --dump_main")
    ap.add_argument("--dump_main", action="store_true", help="also save the fields of the evaluated main splits (physdial/baselines/results/pred/<name>/)"); ap.add_argument("--workers", type=int, default=10)
    ap.add_argument("--seed", type=int, default=0); ap.add_argument("--modes", type=int, default=16, help="FNO Fourier modes per axis (an --eval_only run takes the trained run's)")
    ap.add_argument("--splits", default="", help="evaluate these splits instead of the evaluation columns (e.g. test for the 64-step rollouts)")
    a = ap.parse_args(); assert a.eval_only or a.model, "--model is required for training (--eval_only takes the trained run's)"
    assert not a.switch_eval or a.eval_only, "--switch_eval: an evaluation-only run (its JSON goes to switch/results)"
    rdir = os.path.join(HERE, "results"); out = os.path.join(rdir, a.name + ".json"); os.makedirs(rdir, exist_ok=True)
    if os.path.exists(out): print("exists", out); return
    corpus.set_kolmo_extra("anchor,drag")
    if a.family == "kolmo" and (a.switch_train or a.switch_eval):
        import switch_corpus, switch_data
        if a.switch_train: switch_data.install_train(["switch_train"])
        if a.switch_eval: xs = switch_corpus.install(os.path.join(os.environ["PHYSDIAL_DATA_ROOT"], "switch", "kolmo"), 16)
    else: assert not (a.switch_train or a.switch_eval), "--switch_train / --switch_eval: Kolmogorov only"
    torch.manual_seed(a.seed); dev = "cuda"; ck = os.path.join(PKG, "ckpt", (a.eval_only or a.name) + ".pt")
    if a.eval_only:
        st = torch.load(ck, map_location="cpu", weights_only=False); ra = st["args"]
        a.model, a.k_in = ra["model"], ra["k_in"]; assert ra["family"] == a.family, (ra["family"], a.family)
        fam = Fam(a.family, argparse.Namespace(**{**ra, "switch_train": False, "reference": False}), st["stats"]); a.modes = ra.get("modes", 16); model = build(a.model, fam, a.modes).to(dev); model.load_state_dict(st["ema"]); info = {"run": a.eval_only, "step": st["step"]}
    else:
        sp_ = ck.replace(".pt", ".state.pt")   # a resumed run keeps its normalisation / condition statistics
        fam = Fam(a.family, a, torch.load(sp_, map_location="cpu", weights_only=False)["stats"] if os.path.exists(sp_) else None)
        model = build(a.model, fam, a.modes).to(dev); info = train(model, fam, a, dev, ck)
    model.eval()
    if a.switch_eval: splits = sorted(xs); dd = os.path.join(SWITCH_DATA, "results", "pred", a.name)   # every group, views B, A, N, T
    else: splits = a.splits.split(",") if a.splits else eval_splits(fam); dd = os.path.join(HERE, "results", "pred", a.name) if a.dump_main else None
    t0 = time.time(); res = evaluate(model, fam, a, dev, splits, dd)
    js = {"name": a.name, "args": {**vars(a), "eval_only": a.eval_only or a.name}, "family": a.family, "model": a.model, "n_params": sum(p.numel() for p in model.parameters()),
          "train": info, "eval": res, "eval_min": (time.time() - t0) / 60, "stats": fam.stats}
    if a.switch_eval:   # analyze_switch reads switch/results/<name>.json (+ the dumps written there above)
        xo = os.path.join(SWITCH_DATA, "results", a.name + ".json"); json.dump(js, open(xo + ".tmp", "w"), indent=1); os.replace(xo + ".tmp", xo)
    json.dump(js, open(out + ".tmp", "w"), indent=1); os.replace(out + ".tmp", out); print("saved", out, flush=True)


def train(model, fam, a, dev, ck):
    lr = a.lr or (3e-4 if a.model == "unet" else 1e-3)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4); ema = {k: v.detach().clone() for k, v in model.state_dict().items()}
    sp = ck.replace(".pt", ".state.pt"); step_, el = 0, 0.0
    if os.path.exists(sp):
        s = torch.load(sp, map_location="cpu", weights_only=False); assert s["args"]["family"] == a.family and s["args"]["model"] == a.model and s["args"]["switch_train"] == a.switch_train and s["args"]["reference"] == a.reference and s["args"].get("modes", 16) == a.modes
        model.load_state_dict(s["model"]); opt.load_state_dict(s["opt"]); ema = {k: v.to(dev) for k, v in s["ema"].items()}; step_, el = s["step"], s["elapsed"]; assert s["stats"] == fam.stats
        print(f"[{a.name}] resumed at step {step_}, {el / 3600:.2f} h", flush=True)
    budget = a.budget_gpu_h * 3600; t_comp = 0.0; el0 = el; dl = torch.utils.data.DataLoader(Stream(fam, a.bs, a.seed + step_), batch_size=None, num_workers=a.workers, prefetch_factor=4, persistent_workers=True)
    t_last, t_save, losses, wait = time.time(), time.time(), [], 0.0
    for x, c, y in dl:
        now = time.time(); el += now - t_last; wait += now - t_last - (t_comp if losses else 0.0); t_last = now   # wait = time not spent in the step (data)
        frac = el / budget
        if frac >= 1: break
        g = lr * (frac / 0.02 if frac < 0.02 else 0.5 * (1 + math.cos(math.pi * (frac - 0.02) / 0.98)))
        for pg in opt.param_groups: pg["lr"] = g
        x, c, y = x.to(dev, non_blocking=True), c.to(dev, non_blocking=True), y.to(dev, non_blocking=True)
        with torch.autocast("cuda", dtype=torch.bfloat16, enabled=a.model == "unet"):
            p = step(model, fam, x, c, a.noise)
        loss = F.mse_loss(p.float(), y); opt.zero_grad(set_to_none=True); loss.backward(); torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0); opt.step(); step_ += 1
        torch.cuda.synchronize(); t_comp = time.time() - now
        with torch.no_grad():
            dcy = 0.999 if frac > 0.02 else 0.0
            for k, v in model.state_dict().items(): ema[k].mul_(dcy).add_(v.detach(), alpha=1 - dcy) if v.dtype.is_floating_point or v.dtype.is_complex else ema[k].copy_(v)
        losses.append(float(loss))
        if not math.isfinite(losses[-1]): raise SystemExit(f"[{a.name}] non-finite loss at step {step_}")
        if step_ % 200 == 0: print(f"[{a.name}] step {step_} {el / 3600:.2f} h lr {g:.2e} loss {np.mean(losses[-200:]):.5f} data-wait {wait / max(el - el0, 1e-9):.2f}", flush=True)
        if time.time() - t_save > 1200:
            torch.save({"model": model.state_dict(), "opt": opt.state_dict(), "ema": ema, "step": step_, "elapsed": el, "stats": fam.stats, "args": vars(a)}, sp + ".tmp"); os.replace(sp + ".tmp", sp); t_save = time.time()
    model.load_state_dict(ema)
    torch.save({"ema": ema, "step": step_, "stats": fam.stats, "args": vars(a)}, ck); print(f"[{a.name}] trained {step_} steps, {el / 3600:.2f} h -> {ck}", flush=True)
    return {"steps": step_, "hours": el / 3600, "lr": lr, "final_loss": float(np.mean(losses[-200:])) if losses else None,
            "n_train_windows": len(fam.train), "n_switch_trajectories": len(fam.sw), "switch_share": fam.sw_share, "data_wait_fraction_this_segment": wait / max(el - el0, 1e-9), "noise": a.noise}


if __name__ == "__main__":
    main()
