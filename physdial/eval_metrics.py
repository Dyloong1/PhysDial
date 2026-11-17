"""Evaluation metrics shared by PhysDial (train_gen.evaluate_split), the Walrus / DPOT-H baselines (walrus_eval.py) and the small
specialists (baselines/small_expert.py): per-window relative L2 (per channel and channel-pooled), VRMSE, persistence and linear
extrapolation baselines, rules for undefined (numerically constant) truth frames, trajectory-clustered bootstrap intervals, and
last-frame physics statistics (gradient-energy ratio, spectral band retention)."""
import numpy as np, torch

EDGES = [0, 1 / 8, 1 / 4, 1 / 2, 1.0 + 1e-9]; BANDS = ["(0,1/8]", "(1/8,1/4]", "(1/4,1/2]", "(1/2,1]"]   # radial wavenumber bands (fraction of Nyquist); = tok_gate


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


def grad2_t(x, periodic):
    gx = (torch.roll(x, -1, -2) - torch.roll(x, 1, -2)) / 2 if periodic[0] else torch.gradient(x, dim=-2)[0]
    gy = (torch.roll(x, -1, -1) - torch.roll(x, 1, -1)) / 2 if periodic[1] else torch.gradient(x, dim=-1)[0]
    return (gx ** 2 + gy ** 2).mean((-1, -2))


# ---------------------------------------------------------------- per-window errors and baselines --------------------------------
def rel_l2(d, t):
    """relative L2 over the spatial axes: ||d|| / ||t|| for every leading index (e.g. [B, frames, C])."""
    return (d ** 2).sum((-1, -2)).sqrt() / (t ** 2).sum((-1, -2)).sqrt().clamp_min(1e-12)


def rel_l2_demeaned(P, T):
    """relative L2 of the fields with their spatial means removed, ||(P - <P>) - (T - <T>)|| / ||T - <T>||, per leading index."""
    pm = P.mean((-1, -2), keepdim=True); tm = T.mean((-1, -2), keepdim=True)
    return rel_l2((P - pm) - (T - tm), T - tm)


def shape_amp(P, T):
    """shape error sqrt(1 - c^2) (c = pattern correlation of the de-meaned fields), log amplitude ratio and c, per leading index."""
    P, T = P.double(), T.double()   # 1 - c^2 cancels: in float32 a pure rescaling would show a spurious shape error
    p = P - P.mean((-1, -2), keepdim=True); t = T - T.mean((-1, -2), keepdim=True)
    pn = p.pow(2).sum((-1, -2)).sqrt(); tn = t.pow(2).sum((-1, -2)).sqrt()
    c = ((p * t).sum((-1, -2)) / (pn * tn).clamp_min(1e-300)).clamp(-1, 1)
    return (1 - c.clamp_min(0) ** 2).clamp_min(0).sqrt().float(), (torch.log(pn) - torch.log(tn.clamp_min(1e-300))).float(), c.float()


def shape_amp_summary(SH, AM, CR):
    """SH, AM, CR [N, frames, C] (NaN = undefined) -> shape error, amplitude log-ratio (signed mean and mean |.|) and pattern correlation at frames
    1 / 4 / last and by frame."""
    n = SH.shape[1]; fv = lambda x: None if torch.isnan(x) else float(x)
    fs, fa, fb, fc = torch.nanmean(SH, (0, 2)), torch.nanmean(AM, (0, 2)), torch.nanmean(AM.abs(), (0, 2)), torch.nanmean(CR, (0, 2))
    J = sorted({1, min(4, n), n})
    return {**{f"rel_l2_shape@{j}": fv(fs[j - 1]) for j in J}, **{f"log_amp_ratio@{j}": fv(fa[j - 1]) for j in J}, **{f"abs_log_amp_ratio@{j}": fv(fb[j - 1]) for j in J},
            **{f"pattern_corr@{j}": fv(fc[j - 1]) for j in J},
            "rel_l2_shape_by_frame": [fv(x) for x in fs], "log_amp_ratio_by_frame": [fv(x) for x in fa], "pattern_corr_by_frame": [fv(x) for x in fc]}


def demeaned_summary(DM):
    """DM [N, frames, C] (NaN = undefined) -> rel_l2_demeaned at frames 1 / 4 / last and by frame (mean over windows and channels)."""
    n = DM.shape[1]; fv = lambda x: None if torch.isnan(x) else float(x); fr = torch.nanmean(DM, (0, 2))
    return {**{f"rel_l2_demeaned@{j}": fv(fr[j - 1]) for j in sorted({1, min(4, n), n})}, "rel_l2_demeaned_by_frame": [fv(x) for x in fr]}


def rel_l2_pooled(d, t):
    """relative L2 with the channel axis pooled into the norm (d, t [..., C, H, W] -> [...]): one near-zero channel cannot dominate the mean."""
    return (d ** 2).sum((-1, -2, -3)).sqrt() / (t ** 2).sum((-1, -2, -3)).sqrt().clamp_min(1e-12)


def vrmse(P, T):
    """VRMSE of The Well / Walrus: RMSE over space divided by the spatial standard deviation of the truth, per leading index; > 1 = worse than
    predicting the spatial mean."""
    return ((P - T) ** 2).mean((-1, -2)).sqrt() / T.var((-1, -2), unbiased=False).sqrt().clamp_min(1e-12)


def linear_extrap(x_prev, x_last, frames):
    """two-frame linear extrapolation baseline: x_last + j * (x_last - x_prev) for j = 1..frames; x_* [B, C, H, W] -> [B, frames, C, H, W]."""
    j = torch.arange(1, frames + 1, device=x_last.device, dtype=x_last.dtype)[None, :, None, None, None]
    return x_last[:, None] + j * (x_last - x_prev)[:, None]


def crps_rel(M, T):
    """CRPS of an ensemble per pixel, E|X - y| - E|X - X'| / 2 (the pairwise term unbiased over the m members), averaged over space and divided
    by the spatial mean |y|. M [m, B, frames, C, H, W] members, T [B, frames, C, H, W] truth -> [B, frames, C]."""
    m = M.shape[0]; e1 = (M - T[None]).abs().mean(0)
    e2 = sum((M[i] - M[j]).abs() for i in range(m) for j in range(i + 1, m)) * (2.0 / (m * (m - 1))) if m > 1 else torch.zeros_like(e1)
    return (e1 - 0.5 * e2).mean((-1, -2)) / T.abs().mean((-1, -2)).clamp_min(1e-12)


DEGEN_REL = 1e-5   # = train_gen.FLOOR_REL (the training definition of a numerically constant frame)


def truth_stats(T):
    """T [..., C, H, W] -> spatial rms, std and mean |T| [..., C]: the denominators of rel_l2 / spread, vrmse and crps_rel."""
    return T.pow(2).mean((-1, -2)).sqrt(), T.var((-1, -2), unbiased=False).sqrt(), T.abs().mean((-1, -2))


def degenerate_masks(rms, std, mabs, rel=DEGEN_REL):
    """window-frame-channels whose truth is numerically constant (denominator below rel x the split's median spatial std of the channel)
    -> masks for the rms, std and mean-|.| denominators; such entries are undefined (NaN) instead of dominating the means."""
    C = std.shape[-1]; s = std.reshape(-1, C).double(); thr = torch.empty(C, dtype=torch.float64)
    for c in range(C):
        pos = s[:, c][s[:, c] > 0]; thr[c] = rel * pos.median() if pos.numel() else float("inf")
    return rms.double() < thr, std.double() < thr, mabs.double() < thr


def _nan(x, m):
    return None if x is None else x.masked_fill(m.to(x.device), float("nan"))


def apply_degenerate(d, rms, std, mabs, ens_rows=None):
    """d: dict of per-window arrays (E, PE, LE, TP [N, frames, C]; V [N, frames, C]; FL [N, C]; EP, PEP [N, frames]; optional ens dict of
    [n, frames, C] on the rows ens_rows) -> the same with undefined entries NaN, plus the masks' counts."""
    Mr, Ms, Mm = degenerate_masks(rms, std, mabs); o = dict(d)
    for k in ("E", "PE", "LE", "TP"): o[k] = _nan(d.get(k), Mr)
    o["V"] = _nan(d.get("V"), Ms); o["FL"] = _nan(d.get("FL"), Mr[:, 0]); o["DM"] = _nan(d.get("DM"), Ms)   # DM: rel_l2_demeaned (std denominator, as VRMSE)
    o["SH"] = _nan(d.get("SH"), Ms); o["AM"] = _nan(d.get("AM"), Ms); o["CR"] = _nan(d.get("CR"), Ms)   # shape_amp (std denominator, as DM)
    o["EQ"] = _nan(d.get("EQ"), Mr)   # equivariance error (rms denominator, as E)
    for k in ("EP", "PEP"): o[k] = _nan(d.get(k), Mr.all(-1))   # pooled over channels: undefined only when every channel is
    for k in ("PA", "MO", "ANG", "PPA"):
        if k in d: o[k] = _nan(d[k], Mr.all(-1))
    if "FLPA" in d: o["FLPA"] = _nan(d["FLPA"], Mr[:, 0].all(-1))
    if d.get("ens") is not None:
        r_ = Mr[ens_rows]; o["ens"] = {"mean": _nan(d["ens"]["mean"], r_), "single": _nan(d["ens"]["single"], r_), "spread": _nan(d["ens"]["spread"], r_),
                                       "crps": _nan(d["ens"]["crps"], Mm[ens_rows])}
    o["degenerate"] = {"rule": f"truth denominator < {DEGEN_REL:g} x the split's median spatial std of the channel", "rel_l2_entries@1": int(Mr[:, 0].sum()),
                       "rel_l2_windows_any_channel@1": int(Mr[:, 0].any(-1).sum()), "rel_l2_entries": int(Mr.sum()), "vrmse_entries": int(Ms.sum()), "crps_entries": int(Mm.sum())}
    return o


def bootstrap_ci(v, n_boot=1000, seed=0, groups=None, counts=None):
    """95 % bootstrap interval [lo, hi] of the mean of a per-window vector v [N]. groups [N]: the trajectory of every window; whole
    trajectories are resampled (splits with fewer trajectories than windows take several windows per trajectory, which are not independent).
    NaN windows are left out. counts [N]: v holds per-window SUMS over counts defined
    entries and the statistic is sum / count over the resampled windows, i.e. the entry mean that the point estimate reports (a window with
    one channel undefined weighs less); windows with no defined entry are left out. counts None = every window weighs 1."""
    x = np.asarray(v, dtype=np.float64).reshape(-1); c = np.ones_like(x) if counts is None else np.asarray(counts, dtype=np.float64).reshape(-1)
    ok = ~np.isnan(x) & (c > 0)
    if groups is not None: groups = [g for g, o in zip(groups, ok) if o]
    x, c = x[ok], c[ok]; N = len(x)
    if N == 0: return [None, None]
    if N < 2: return [float(x.sum() / c.sum())] * 2
    rng = np.random.default_rng(seed)
    if groups is not None and len(set(groups)) < N:
        ids = {g: i for i, g in enumerate(dict.fromkeys(groups))}; gi = np.array([ids[g] for g in groups]); G = len(ids)
        if G < 2: return [float(x.sum() / c.sum())] * 2
        S = np.bincount(gi, x, G); Cn = np.bincount(gi, c, G); pick = rng.integers(0, G, (n_boot, G))
        m = S[pick].sum(1) / Cn[pick].sum(1)
        return [float(np.percentile(m, 2.5)), float(np.percentile(m, 97.5))]
    pick = rng.integers(0, N, (n_boot, N)); m = x[pick].sum(1) / c[pick].sum(1)
    return [float(np.percentile(m, 2.5)), float(np.percentile(m, 97.5))]


def _ci_entries(Ej, groups):
    """bootstrap interval of the entry mean of Ej [N, C] (one frame; NaN = undefined entry), consistent with nanmean(Ej)."""
    ok = ~torch.isnan(Ej); return bootstrap_ci(torch.where(ok, Ej, torch.zeros_like(Ej)).sum(1), groups=groups, counts=ok.sum(1))


# ---------------------------------------------------------------- last-frame physics statistics -------------------------------------
def new_acc(C):
    """per-channel accumulators of last-frame gradient energies (true / predicted) and band energies (true / predicted)."""
    return tuple({ci: [] for ci in range(C)} for _ in range(4))


def accumulate_last_frame(T, P, periodic, GT, GR, BT, BR):
    """T, P [B, frames, C, H, W] true / predicted physical frames: add the last frame's statistics per channel."""
    for ci in range(T.shape[2]):
        pt, pr = T[:, -1, ci], P[:, -1, ci]; GT[ci].append(grad2_t(pt, periodic).cpu()); GR[ci].append(grad2_t(pr, periodic).cpu())
        win = window(*pt.shape[-2:], periodic); BT[ci].append(band_energy(pt.double().cpu().numpy(), win)); BR[ci].append(band_energy(pr.double().cpu().numpy(), win))


def accumulate_pred_last_frame(R, periodic, G, Bd):
    """R [B, C, H, W] one frame per window (e.g. the encode-decode reconstruction of the true last frame): add its statistics per channel."""
    for ci in range(R.shape[1]):
        x = R[:, ci]; G[ci].append(grad2_t(x, periodic).cpu()); Bd[ci].append(band_energy(x.double().cpu().numpy(), window(*x.shape[-2:], periodic)))


def _f(x):
    x = float(x); return None if np.isnan(x) else x


def physics_metrics(E, PE, FL, GT, GR, BT, BR, channels, F, LE=None, V=None, GC=None, BC=None, ens=None, members=1, EP=None, PEP=None, TP=None, groups=None, degenerate=None):
    """metric dict: E, PE [N, frames, C] relative L2 of the prediction and of persistence per window / frame / channel, FL [N] tokenizer floor
    (None for a model without tokenizer), GT / GR / BT / BR from new_acc / accumulate_last_frame, F = frames per generation chunk.
    Optional: LE [N, frames, C] linear-extrapolation baseline, V [N, frames, C] VRMSE, GC / BC statistics of the reconstruction of the
    true last frame (tokenizer floor of the statistics), ens = dict of [n, frames, C] tensors mean / single / spread / crps of `members`
    samples on the first n windows; TP [N, frames, C] the tokenizer-passed persistence (decode(encode(last history frame)) as the prediction
    of every frame)."""
    nm = torch.nanmean
    fr = nm(E, (0, 2)); n = E.shape[1]; ends = sorted({1, F, n}); C = len(channels)
    gt = {ci: float(torch.cat(GT[ci]).mean()) for ci in range(C)}; gr = {ci: float(torch.cat(GR[ci]).mean()) for ci in range(C)}
    bt = {ci: np.concatenate(BT[ci]).sum(0) for ci in range(C)}; br = {ci: np.concatenate(BR[ci]).sum(0) for ci in range(C)}
    bts = sum(bt.values()); brs = sum(br.values())
    r = {"n_windows": int(E.shape[0]), "frames": int(n), "rel_l2_by_frame": [_f(v) for v in fr],
         **{f"rel_l2@{j}": _f(fr[j - 1]) for j in ends}, "rel_l2@1_by_channel": {c: _f(nm(E[:, 0, i])) for i, c in enumerate(channels)},
         "rel_l2@1_ci95": _ci_entries(E[:, 0], groups), f"rel_l2@{n}_ci95": _ci_entries(E[:, -1], groups),   # 95 % bootstrap over windows of the entry mean
         "persistence@1": _f(nm(PE[:, 0])), f"persistence@{n}": _f(nm(PE[:, -1])), "tokenizer_floor@1": None if FL is None else _f(nm(FL)),
         "last_frame_grad_ratio": float(sum(gr.values()) / max(sum(gt.values()), 1e-30)),   # pooled over channels (energy-weighted)
         "last_frame_grad_ratio_by_channel": {c: float(gr[i] / max(gt[i], 1e-30)) for i, c in enumerate(channels)},
         "last_frame_band_retention": {b: float(brs[i] / max(bts[i], 1e-30)) for i, b in enumerate(BANDS)},
         "last_frame_top_retention_by_channel": {c: float(br[i][-1] / max(bt[i][-1], 1e-30)) for i, c in enumerate(channels)},
         "last_frame_top_fraction_true": float(bts[-1] / max(bts.sum(), 1e-30)),
         "last_frame_top_fraction_true_by_channel": {c: float(bt[i][-1] / max(bt[i].sum(), 1e-30)) for i, c in enumerate(channels)}}
    if degenerate is not None: r["degenerate"] = degenerate
    if EP is not None:   # channels pooled into the norm (EP, PEP [N, frames])
        fp = nm(EP, 0); fpp = nm(PEP, 0); r.update({"rel_l2_pooled@1": _f(fp[0]), f"rel_l2_pooled@{n}": _f(fp[-1]), "rel_l2_pooled_by_frame": [_f(v) for v in fp],
                                                   "rel_l2_pooled@1_ci95": _ci_entries(EP[:, 0:1], groups), f"rel_l2_pooled@{n}_ci95": _ci_entries(EP[:, -1:], groups),
                                                   "persistence_pooled@1": _f(fpp[0]), f"persistence_pooled@{n}": _f(fpp[-1])})
    if TP is not None:   # persistence through the tokenizer: model / tok_persistence reads the dynamics learned on top of the floor
        ft = nm(TP, (0, 2)); r.update({"tok_persistence@1": _f(ft[0]), f"tok_persistence@{n}": _f(ft[-1]), "tok_persistence_by_frame": [_f(v) for v in ft]})
    if LE is not None:   # two-frame linear extrapolation baseline
        fl = nm(LE, (0, 2)); r.update({"linear@1": _f(fl[0]), f"linear@{n}": _f(fl[-1]), "linear_by_frame": [_f(v) for v in fl]})
    if V is not None:    # VRMSE and The Well's time windows (1; 6-12; 13-end)
        fv = nm(V, (0, 2)); r.update({"vrmse@1": _f(fv[0]), f"vrmse@{n}": _f(fv[-1]), "vrmse_by_frame": [_f(v) for v in fv],
                                       "vrmse_6_12": _f(nm(fv[5:12])) if n >= 6 else None, "vrmse_13_end": _f(nm(fv[12:])) if n >= 13 else None})
    if GC is not None and BC is not None and all(len(GC[ci]) for ci in range(C)):   # statistics of decode(encode(true last frame))
        gc = {ci: float(torch.cat(GC[ci]).mean()) for ci in range(C)}; bc = {ci: np.concatenate(BC[ci]).sum(0) for ci in range(C)}; bcs = sum(bc.values())
        r.update({"recon_last_frame_grad_ratio": float(sum(gc.values()) / max(sum(gt.values()), 1e-30)),
                  "recon_last_frame_band_retention": {b: float(bcs[i] / max(bts[i], 1e-30)) for i, b in enumerate(BANDS)},
                  "last_frame_grad_ratio_vs_recon": float(sum(gr.values()) / max(sum(gc.values()), 1e-30)),
                  "last_frame_grad_ratio_by_channel_vs_recon": {c: float(gr[i] / max(gc[i], 1e-30)) for i, c in enumerate(channels)},
                  "last_frame_band_retention_vs_recon": {b: float(brs[i] / max(bcs[i], 1e-30)) for i, b in enumerate(BANDS)},
                  "last_frame_top_retention_by_channel_vs_recon": {c: float(br[i][-1] / max(bc[i][-1], 1e-30)) for i, c in enumerate(channels)}})
    if ens is not None:
        em, es, esp, ec = ens["mean"], ens["single"], ens["spread"], ens["crps"]; js = sorted({1, n})
        r["ensemble"] = {"members": int(members), "windows": int(em.shape[0]),
                         **{f"mean_rel_l2@{j}": _f(nm(em[:, j - 1])) for j in js}, **{f"single_rel_l2@{j}": _f(nm(es[:, j - 1])) for j in js},
                         **{f"spread@{j}": _f(nm(esp[:, j - 1])) for j in js}, **{f"crps@{j}": _f(nm(ec[:, j - 1])) for j in js},
                         **{f"spread_calibrated@{j}": _f(nm(esp[:, j - 1]) * np.sqrt((members + 1) / members)) for j in js},
                         # ensemble has E|mean - truth|^2 = (1 + 1/m) E[spread^2]; compare mean_rel_l2 with this, not with spread (m = 8: factor 1.06)
                         "mean_rel_l2_by_frame": [_f(v) for v in nm(em, (0, 2))], "spread_by_frame": [_f(v) for v in nm(esp, (0, 2))],
                         "crps_by_frame": [_f(v) for v in nm(ec, (0, 2))]}
    return r


def nonfinite_to_inf(P, *errs):
    """a non-finite prediction must not vanish from nanmean-based means as an undefined entry: errors of such entries are set to +inf.
    P [B, F, C, H, W]; returns the errors and the number of windows with a non-finite prediction."""
    bad = ~torch.isfinite(P).flatten(P.dim() - 2).all(-1)   # [B, F, C]; P may also be an ensemble [m, B, F, C, H, W]
    if bad.dim() == 4: bad = bad.any(0)
    out = [e.masked_fill(bad if e.dim() == 3 else bad.any(-1), float("inf")) for e in errs]
    return (*out, int(bad.flatten(1).any(1).sum()))
