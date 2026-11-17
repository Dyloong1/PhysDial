#!/usr/bin/env python
"""Linear probes: how well each Kolmogorov condition can be read from the 6 history frames of the evaluation windows, (a) from hand-made
physical statistics of the frames (information that is in the data and linearly readable), (b) from Walrus's internal representation of
the same windows (what the model has encoded).

Windows: every Kolmogorov evaluation directory written by physdial/walrus_data.py ($PHYSDIAL_DATA_ROOT/walrus/eval/kolmo__<split>__<HxW>/;
export with --families kolmo --kolmo_extra drag,beta --eval_n 32 so the drag and beta columns are included), history = the first
EVAL_HIST = 6 frames of each window (velocity, turned into vorticity for (a)). Targets per window, from the source trajectory
(corpus.Kolmo._item): log10 nu, k_f (multi-mode trajectories excluded: no single k_f), amplitude, drag alpha (0.1 outside the drag columns),
beta (0 outside the beta column), and log10 of the frame interval at the model stride (dt). Nothing is fitted to the forecast targets; no
window enters a model's training.

(a) physics features (numpy, per window; domain [0, 2 pi)^2; only the modes |k_x|, |k_y|, |k| <= KC = 85 that both grids resolve):
    log energy spectrum of the last frame in 12 log-spaced |k| bins; log energy, enstrophy, palinstrophy; share of energy in the k_x = 0
    and in the k_y = 0 modes (zonal / meridional jets) per band (|k| <= 8, 8-32, > 32); the log energy share of the modes (0, +-k) and
    (+-k, 0) for k = 1..8 (the Kolmogorov forcing shape sin(k_f y)); per model step (from linear fits over the K frames): the rates of log
    energy, log enstrophy and log band energies; the energy-weighted phase speed of the modes |k| <= 6 against k_x / |k|^2 (a Rossby wave
    has omega = -beta k_x / |k|^2) and the mean |omega|; the relative frame-to-frame change; skewness and flatness of the last frame.
    Variants: K = 2 (last two frames) and K = 6, each on the raw history and (suffix _wnorm) on the history as Walrus's encoder sees it
    (each velocity component divided by its RMS over the 6 frames and space), which has no absolute level. r2_256: the same probes on the
    256^2 windows only (the grid follows nu).
(b) Walrus features (1 GPU, Walrus environment): the evaluator's model build (baselines/walrus/walrus_eval.build: --run / --ckpt, the run's
    normalisation) and one forward step on the 6-frame history (walrus_eval.rollout with steps = 1, the per-split seed of walrus_eval), with
    forward hooks on the processor: the input of block 0 (= the encoder output) and the outputs of blocks --blocks; each feature = mean and
    standard deviation over space at the last time position (causal: it has seen the whole history), 2 x 1408 numbers. --no_walrus skips (b).
Probe: ridge regression per target and feature set; features standardised inside each fold; 5 outer folds grouped by source trajectory
(a trajectory never spans folds); the ridge strength is chosen per outer fold by an inner grouped 4-fold CV on the training part. Reported:
pooled out-of-fold R^2 per (feature set, target); per split the mean true / predicted value and the mean absolute error in units of the
target's std over all windows. Linear probes give a lower bound of the information present.
Output: results/<name>.json, features results/feat/<name>.npz.

  python3 probe_conditions.py --run walrus-ft --ckpt last --name probe-walrus-ft
  python3 probe_conditions.py --ckpt coalesced --name probe-walrus-released
  python3 probe_conditions.py --no_walrus --name probe-physics"""
import argparse, glob, json, os, sys, time, zlib
from types import SimpleNamespace as NS

HERE = os.path.dirname(os.path.abspath(__file__)); REPO = os.path.dirname(HERE); PKG = os.path.join(REPO, "physdial"); WDIR = os.path.join(PKG, "baselines", "walrus")
sys.path.insert(0, REPO); sys.path.insert(0, PKG); sys.path.insert(0, WDIR)
import numpy as np, h5py   # noqa: E402
import corpus   # noqa: E402

EVAL_HIST = 6
EVAL_DIR = "eval"   # = walrus_data.eval_dir(16): the 16-frame evaluation windows
TARGETS = ("log10_nu", "kf", "amp", "alpha", "beta", "log10_dt")
KC = 85   # 2/3 dealiasing cut of the 256^2 solver (kolmo_solver: N // 3); only modes |k_x|, |k_y|, |k| <= KC enter the features, so a 256^2
          # and a 512^2 window are read on the same mode set (the grid follows nu: a band beyond 85 would identify the grid, not nu)
BINS = np.array([1, 2, 3, 4, 6, 8, 12, 16, 24, 32, 48, 64, 86])
EPS_W = 1e-5   # Walrus RMSSamplewiseRevNormalization epsilon (the fine-tuning config and the released config)


def vort(u, v):   # = walrus_data.vel_to_vort (x = first grid axis)
    N, M = u.shape[-2:]; kx = np.fft.fftfreq(N, 1.0 / N)[:, None]; ky = np.fft.fftfreq(M, 1.0 / M)[None, :]
    return np.real(np.fft.ifft2(1j * kx * np.fft.fft2(v) - 1j * ky * np.fft.fft2(u)))


def walrus_view(vel):
    """vel [T, H, W, 2] -> each velocity component divided by its RMS over time and space (floor EPS_W): Walrus's RMSSamplewiseRevNormalization."""
    return vel / np.maximum(np.sqrt((vel ** 2).mean((0, 1, 2))), EPS_W)


def on_grid(Wf, N, M, NF=512):
    """the field of the resolved modes (|k_x|, |k_y| <= KC; Wf = fft2 / (N M)) sampled on an NF^2 grid (zero-padding up): moments of a field
    band-limited to KC need NF >= 4 KC to be free of aliasing, so skewness / flatness do not depend on the window's own grid."""
    out = np.zeros((NF, NF), complex); ix = np.rint(np.fft.fftfreq(N, 1.0 / N)).astype(int); iy = np.rint(np.fft.fftfreq(M, 1.0 / M)).astype(int)
    sx, sy = np.abs(ix) <= KC, np.abs(iy) <= KC
    out[np.ix_(ix[sx] % NF, iy[sy] % NF)] = Wf[np.ix_(sx, sy)]
    return np.real(np.fft.ifft2(out * (NF * NF)))


def phys_features(w):
    """w [K, N, M] vorticity at the model stride (K >= 2) -> 1-D feature vector (see the module doc)."""
    K, N, M = w.shape; kx = np.fft.fftfreq(N, 1.0 / N)[:, None] * np.ones((1, M)); ky = np.ones((N, 1)) * np.fft.fftfreq(M, 1.0 / M)[None, :]
    k2 = kx ** 2 + ky ** 2; k = np.sqrt(k2); k2s = np.where(k2 == 0, 1.0, k2)
    W = np.fft.fft2(w) / (N * M) * ((np.abs(kx) <= KC) & (np.abs(ky) <= KC) & (k <= KC))   # the common resolved modes of both grids
    E = 0.5 * np.abs(W) ** 2 / k2s; E[:, 0, 0] = 0.0   # energy density per mode, every frame
    Z = 0.5 * np.abs(W) ** 2; Z[:, 0, 0] = 0.0; P = Z * k2
    tiny = 1e-30; f = []
    El = E[-1]; Et = E.sum((1, 2)); Zt = Z.sum((1, 2)); Pt = P.sum((1, 2))
    for lo, hi in zip(BINS[:-1], BINS[1:]): f.append(np.log10(El[(k >= lo) & (k < hi)].sum() + tiny))
    f += [np.log10(Et[-1] + tiny), np.log10(Zt[-1] + tiny), np.log10(Pt[-1] + tiny)]
    bands = [(k > 0) & (k <= 8), (k > 8) & (k <= 32), (k > 32) & (k <= KC)]
    for b in bands:
        tot = El[b].sum() + tiny; f += [El[b & (kx == 0)].sum() / tot, El[b & (ky == 0)].sum() / tot]
    for kk in range(1, 9):
        f += [np.log10(El[(kx == 0) & (np.abs(ky) == kk)].sum() / (Et[-1] + tiny) + tiny), np.log10(El[(ky == 0) & (np.abs(kx) == kk)].sum() / (Et[-1] + tiny) + tiny)]
    t = np.arange(K, dtype=np.float64)
    slope = lambda y: float(np.polyfit(t, y, 1)[0])
    f += [slope(np.log10(Et + tiny)), slope(np.log10(Zt + tiny))] + [slope(np.log10(E[:, b].sum(1) + tiny)) for b in bands]
    low = (k > 0) & (k <= 6); ph = np.angle(W[1:] * np.conj(W[:-1]))   # per-step phase change of every mode, [K-1, N, M]
    wgt = (El * low).ravel(); xr = (kx / k2s)[None].repeat(K - 1, 0)[:, low].ravel(); yr = ph[:, low].ravel(); ww = np.tile(wgt[low.ravel()], K - 1)
    sw = ww.sum() + tiny; xm = (ww * xr).sum() / sw; ym = (ww * yr).sum() / sw
    f += [float((ww * (xr - xm) * (yr - ym)).sum() / ((ww * (xr - xm) ** 2).sum() + tiny)), float((ww * np.abs(yr)).sum() / sw)]   # phase speed vs k_x/|k|^2; mean |omega|
    f.append(float(np.mean([np.linalg.norm(W[i + 1] - W[i]) / (np.linalg.norm(W[i]) + tiny) for i in range(K - 1)])))   # Parseval, resolved modes only
    x = on_grid(W[-1], N, M); x = x - x.mean(); s = x.std() + tiny; f += [float((x ** 3).mean() / s ** 3), float((x ** 4).mean() / s ** 4)]   # the resolved field, sampled on 512^2 for every grid
    return np.array(f, np.float64)


def windows(root):
    """[(split, grid dir, window record, file)] of every Kolmogorov evaluation directory, in file order."""
    out = []
    for ed in sorted(glob.glob(os.path.join(root, "kolmo__*"))):
        sp = os.path.basename(ed).split("__")[1]; recs = {r["window"]: r for r in json.load(open(os.path.join(ed, "windows.json")))}
        for fn in sorted(os.listdir(os.path.join(ed, "data", "test"))):
            if fn.startswith("win_") and fn.endswith(".hdf5"): out.append((sp, ed, recs[int(fn[4:-5])], os.path.join(ed, "data", "test", fn)))
    return out


def targets(fam, rec):
    it = fam._item(rec["path"], str(rec["traj"])); p, raw = it.params, it.raw
    return {"log10_nu": p["log10_nu"], "kf": np.nan if raw.get("multimode") else p["kf"], "amp": p["amp"], "alpha": raw.get("alpha", corpus.KOLMO_ALPHA),
            "beta": raw.get("beta", 0.0), "log10_dt": p["log10_dt"] + np.log10(rec.get("stride", fam.stride))}


def folds_of(groups, k, salt):
    return np.array([zlib.crc32(f"{salt}|{g}".encode()) % k for g in groups])


def ridge_fit_predict(Xtr, ytr, Xte, lams):
    mu, sd = Xtr.mean(0), Xtr.std(0); sd[sd < 1e-12] = 1.0; A = (Xtr - mu) / sd; B = (Xte - mu) / sd; ym = ytr.mean()
    U, s, Vt = np.linalg.svd(A, full_matrices=False); uy = U.T @ (ytr - ym)
    return [ym + B @ (Vt.T @ ((s / (s ** 2 + lam)) * uy)) for lam in lams]


def probe(X, y, groups, lams=tuple(10.0 ** np.arange(-2, 6))):
    """out-of-fold predictions (5 outer folds by trajectory, lambda by inner 4-fold CV); NaN targets are left out."""
    ok = np.isfinite(y); X, y, g = X[ok], y[ok], np.asarray(groups)[ok]; pred = np.full(len(y), np.nan)
    fo = folds_of(g, 5, "outer")
    for i in range(5):
        tr, te = fo != i, fo == i
        if not te.any() or tr.sum() < 10: continue
        fi = folds_of(g[tr], 4, f"inner{i}"); sse = np.zeros(len(lams))
        for j in range(4):
            a_, b_ = fi != j, fi == j
            if not b_.any() or a_.sum() < 5: continue
            for li, pp in enumerate(ridge_fit_predict(X[tr][a_], y[tr][a_], X[tr][b_], lams)): sse[li] += ((pp - y[tr][b_]) ** 2).sum()
        best = lams[int(np.argmin(sse))]; pred[te] = ridge_fit_predict(X[tr], y[tr], X[te], [best])[0]
    full = np.full(len(ok), np.nan); full[ok] = pred; return full


def walrus_features(a, wins, dev):
    import torch, walrus_eval as WE
    ea = NS(run=a.run, ckpt=a.ckpt, eval_bs=a.bs, frames=16, workers=2)   # frames = the files' 16 targets: one sample per window file (walrus_eval asserts the same); rollout runs 1 step
    model, revin, fmap, cfg, info = WE.build(ea, dev); eps = info["epsilon"]
    from walrus.data.well_to_multi_transformer import ChannelsFirstWithTimeFormatter   # importable once build() has put the Walrus source on the path
    fmt = ChannelsFirstWithTimeFormatter()
    blocks = [int(b) for b in a.blocks.split(",")]; cap = {}
    def pool(x):   # [T, B, C, H, W, D] -> [B, 2C]: mean / std over space at the last time position
        z = x[-1].float(); return torch.cat([z.mean((2, 3, 4)), z.std((2, 3, 4))], 1).cpu()
    hs = [model.blocks[0].register_forward_pre_hook(lambda m, inp: cap.__setitem__("pre", pool(inp[0])))]
    for b in blocks: hs.append(model.blocks[b].register_forward_hook(lambda m, inp, out, b=b: cap.__setitem__(f"b{b}", pool(out[0] if isinstance(out, (tuple, list)) else out))))
    feats = {}; order = {}
    for ed in sorted({w[1] for w in wins}):
        sp = os.path.basename(ed).split("__")[1]; sd = a.seed + int.from_bytes(f"kolmo/{sp}".encode(), "little") % 100003   # walrus_eval's per-split seed
        torch.manual_seed(sd); np.random.seed(sd % 2 ** 32)
        ds = [f[:-5] for f in os.listdir(ed) if f.endswith(".yaml") and f != "stats.yaml"][0]
        dm = WE.eval_datamodule(cfg, ed, ds, fmap, ea); dset = dm.test_dataloaders(replicas=1, rank=0, full=True)[0].dataset
        files = dset.sub_dsets[0].files_paths; assert len(dset) == len(files), (ed, len(dset), len(files))
        loader = torch.utils.data.DataLoader(dset, sampler=torch.utils.data.BatchSampler(torch.utils.data.SequentialSampler(dset), a.bs, drop_last=False), batch_size=None, num_workers=2)
        k = 0
        for batch in loader:
            cap.clear(); WE.rollout(model, revin, fmt, batch, 1, eps, dev)
            n = next(iter(cap.values())).shape[0]
            for i in range(n):
                fp = os.path.realpath(files[k + i]); order[fp] = True
                for key, v in cap.items(): feats.setdefault(key, {})[fp] = v[i].numpy()
            k += n
        print(f"[{a.name}] walrus features {os.path.basename(ed)}: {k} windows", flush=True)
    for h in hs: h.remove()
    keys = ["pre"] + [f"b{b}" for b in blocks]
    return {f"walrus_{key}": np.stack([feats[key][os.path.realpath(w[3])] for w in wins]) for key in keys}, info


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--name", required=True); ap.add_argument("--run", default="walrus-ft", help="walrus_ft run name (ignored with --ckpt coalesced)")
    ap.add_argument("--ckpt", default="last", help="last | best | coalesced (released Walrus, zero-shot)"); ap.add_argument("--blocks", default="4,9,19,29,39"); ap.add_argument("--bs", type=int, default=2)
    ap.add_argument("--seed", type=int, default=0); ap.add_argument("--no_walrus", action="store_true")
    a = ap.parse_args()
    out = os.path.join(HERE, "results", a.name + ".json"); os.makedirs(os.path.join(HERE, "results", "feat"), exist_ok=True)
    if os.path.exists(out): print("exists", out); return
    corpus.set_kolmo_extra("drag,beta"); fam = corpus.get("kolmo"); t0 = time.time()
    root = os.path.join(os.environ.get("PHYSDIAL_DATA_ROOT") or os.path.join(REPO, "data"), "walrus", EVAL_DIR); wins = windows(root); assert wins, f"no Kolmogorov windows under {root}"
    Y = {t: [] for t in TARGETS}; groups, splits, F2, F6, F2n, F6n, grids = [], [], [], [], [], [], []
    for sp, ed, rec, fp in wins:
        for t, v in targets(fam, rec).items(): Y[t].append(v)
        groups.append(f"{rec['path']}|{rec['traj']}"); splits.append(sp)
        with h5py.File(fp, "r") as h: vel = np.asarray(h["t1_fields"]["velocity"][0, :EVAL_HIST], np.float64)   # [T, H, W, 2]
        w = vort(vel[..., 0], vel[..., 1]); F2.append(phys_features(w[-2:])); F6.append(phys_features(w)); grids.append(int(vel.shape[1]))
        # the history as Walrus's encoder sees it (sample-wise RMS normalisation: each velocity component divided by its RMS over the
        # 6 frames and space, floor EPS_W): absolute levels and the u / v ratio are gone
        vn = walrus_view(vel); wn = vort(vn[..., 0], vn[..., 1])
        F2n.append(phys_features(wn[-2:])); F6n.append(phys_features(wn))
    Y = {t: np.array(v, np.float64) for t, v in Y.items()}
    FS = {"phys_K2": np.stack(F2), "phys_K6": np.stack(F6), "phys_K2_wnorm": np.stack(F2n), "phys_K6_wnorm": np.stack(F6n)}
    print(f"[{a.name}] {len(wins)} windows, {len(set(splits))} splits, physics features {FS['phys_K6'].shape[1]} ({time.time() - t0:.0f}s)", flush=True)
    info = None
    if not a.no_walrus:
        import torch
        dev = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        if dev.type == "cuda": torch.cuda.set_device(dev)
        wf, info = walrus_features(a, wins, dev); FS.update(wf)
    res = {"name": a.name, "run": None if a.ckpt == "coalesced" else a.run, "ckpt": a.ckpt, "walrus_info": info, "n_windows": len(wins),
           "splits": sorted(set(splits)), "targets": list(TARGETS), "feature_dims": {k: int(v.shape[1]) for k, v in FS.items()}, "r2": {}, "per_split": {},
           "r2_256": {}, "n_256": int(sum(g == 256 for g in grids))}   # r2_256: the same probe on the 256^2 windows only (grid control)
    S = np.array(splits); G = np.array(grids)
    for fs, X in FS.items():
        X = np.where(np.isfinite(X), X, 0.0); res["r2"][fs] = {}; res["per_split"][fs] = {}; res["r2_256"][fs] = {}
        for t in TARGETS:
            y = Y[t]; ok = np.isfinite(y)
            if np.nanstd(y) < 1e-12: res["r2"][fs][t] = None; continue
            p = probe(X, y, groups); m = ok & np.isfinite(p)
            res["r2"][fs][t] = float(1 - ((p[m] - y[m]) ** 2).sum() / ((y[m] - y[m].mean()) ** 2).sum()); sdv = float(np.std(y[m]))
            res["per_split"][fs][t] = {s: {"n": int((m & (S == s)).sum()), "true": float(y[m & (S == s)].mean()), "pred": float(p[m & (S == s)].mean()),
                                           "mae_over_std": float(np.abs(p - y)[m & (S == s)].mean() / sdv)} for s in sorted(set(splits)) if (m & (S == s)).any()}
            sel = G == 256; y2 = np.where(sel, y, np.nan)
            if np.nanstd(y2) > 1e-12:
                p2 = probe(X[sel], y[sel], [g_ for g_, s_ in zip(groups, sel) if s_]); m2 = np.isfinite(y[sel]) & np.isfinite(p2)
                res["r2_256"][fs][t] = float(1 - ((p2[m2] - y[sel][m2]) ** 2).sum() / ((y[sel][m2] - y[sel][m2].mean()) ** 2).sum())
            else: res["r2_256"][fs][t] = None
        print(f"[{a.name}] {fs}: " + " ".join(f"{t} {res['r2'][fs][t]:.3f}" if res["r2"][fs][t] is not None else f"{t} -" for t in TARGETS), flush=True)
    np.savez_compressed(os.path.join(HERE, "results", "feat", a.name + ".npz"), splits=S, groups=np.array(groups), grids=G, **{f"y_{t}": v for t, v in Y.items()},
                        **{f"X_{k}": v.astype(np.float32) for k, v in FS.items()})
    import runmeta
    res.update(seconds=time.time() - t0, meta=runmeta.run_meta())
    json.dump(res, open(out + ".tmp", "w"), indent=1); os.replace(out + ".tmp", out); print("saved", out, flush=True)


if __name__ == "__main__":
    main()
