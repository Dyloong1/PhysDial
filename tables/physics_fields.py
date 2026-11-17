#!/usr/bin/env python
"""Physics report on dumped rollouts: physical statistics of the rollouts of three models (ours, Walrus, DPOT-H) on the same windows,
against the solver truth, behind data-path checks window_match-truth_match. CPU only.

Sources:
  * main readings A1 and A2 come from the evaluation JSONs (every window of a column): A1 = rel_l2_by_frame (dev / test halves from
    per_window at step 1 and the last step); A2 = last_frame_band_retention (pooled over windows and channels), bands whose truth share
    >= 1e-4; compressor floor = the same distance of ours' recon_last_frame_band_retention.
  * the dumps (--dump_pred 32: every window of the column, paired by (path, traj, first_frame)) give the per-step A2 curve (per channel),
    A3, A4, B, C and the persistence baseline.
Dump files: one npz per (family, split) with H<i> (2 history frames), P<i>, T<i> [frames, C, H, W] fp16 physical units, meta json:
  ours physdial/results/pred/<run>/, walrus physdial/baselines/walrus/results/pred/<run>/, dpot physdial/baselines/dpot/results/pred/<run>/;
  JSONs: physdial/results/<run>.json ("eval"), physdial/baselines/walrus/results/<run>.json, physdial/baselines/dpot/results/<run>.json.
Default runs: --ours physdial-s0-eval-64step, --walrus walrus-ft-eval-64step,
--dpot dpot-ft-eval-64step (64-step rollouts of kolmo and nsu).
A missing model / file / window is skipped and listed, never substituted.

Checks (a failed check removes that model's numbers of that family from the report and the result JSON; the persistence baseline is
built from ours' dump, so it is gated by ours' checks):
  window_match  every dumped window is found in the JSON per_window and its relative L2 recomputed from the dump at step 1 and the last step
      matches (eval_metrics.rel_l2, channel nanmean) within tol = 1e-4 + FP16_SAFETY x the fp16 storage term of that window, capped at
      1e-2, floored at TOL_FLOOR; NaN fails; a window non-finite on both sides matches.
  persistence_match ours only: the last history frame of the dump as persistence reproduces the JSON per_window persistence@1.
  band_match  when the dump holds the whole column: last-step band energies pooled over windows and channels / truth = the JSON
      last_frame_band_retention, bands with truth share >= 1e-3, tolerance 1e-3.
  truth_match  walrus / dpot: their dumped truth equals ours on the shared windows, per channel ||T_m - T_ours|| / ||T_ours|| <= 1e-3.
Dump metrics per step t: A2c spectral distance per channel (|log| of band ratios clipped to [1e-6, 1e6]), A2top pooled top-band ratio;
A3 log(Q_P / Q_T) signed and |.| (Q = kinetic energy and enstrophy); A4 exact 1-D W1 / std(T); B family quantity (NSU energy decay rate
ratio); C |mean| / rms(T) of the zero-mean channels.
Family values: geometric mean over the test / interp / ext columns, all windows and the dev / test halves.
Outputs: physdial/results/<name>.json, physdial/results/physics/<name>/report.md, spectra.json, spectra_<family>__test.png."""
import argparse, json, math, os, sys, time
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__)); PKG = os.path.join(os.path.dirname(HERE), "physdial")
sys.path.insert(0, PKG); sys.path.insert(0, HERE)
from eval_metrics import band_energy, window, BANDS   # the evaluation's own spectral definition

MODELS = ("ours", "walrus", "dpot")
PRED = {"ours": os.path.join(PKG, "results", "pred"), "walrus": os.path.join(PKG, "baselines", "walrus", "results", "pred"), "dpot": os.path.join(PKG, "baselines", "dpot", "results", "pred")}
JSON = {"ours": os.path.join(PKG, "results"), "walrus": os.path.join(PKG, "baselines", "walrus", "results"), "dpot": os.path.join(PKG, "baselines", "dpot", "results")}
SHOW = (1, 4, 16, 32, 48, 64)
SHARE = 1e-4
SHARE_BAND_MATCH = 1e-3   # band_match: fp16 dumps are too coarse for the lowest-share bands
CLIP = 1e6
TOL_CAP = 1e-2
FP16_SAFETY = 4.0   # window_match: margin on the analytic fp16 storage term
TOL_FLOOR = 1e-3   # window_match: lower bound of the tolerance (readings are printed to <= 3 significant digits)
FAMILIES = ("kolmo", "nsu")
PERIODIC = (True, True)   # every dumped family
QMAIN = {"kolmo": "kinetic_energy", "nsu": "kinetic_energy"}
ZERO_MEAN = {"kolmo": ["vorticity"], "nsu": ["vorticity"]}


# ------------------------------------------------------------------------------------------------ field statistics
def kgrid(H, W):
    kx = np.fft.fftfreq(H, 1.0 / H); ky = np.fft.fftfreq(W, 1.0 / W); return kx, ky, kx[:, None] ** 2 + ky[None, :] ** 2


def bands(x):
    """band energies of fields x [N, H, W] (eval_metrics.band_energy with the periodic window) -> [N, 4]."""
    return band_energy(np.asarray(x, np.float64), window(*x.shape[-2:], PERIODIC))


def ratio_distance(r, include):
    """mean over the included bands of |log r|, r clipped to [1 / CLIP, CLIP] (nan if no band)."""
    xs = [abs(math.log(min(max(r_ if r_ is not None and math.isfinite(r_) else 0.0, 1 / CLIP), CLIP))) for r_, i_ in zip(r, include) if i_]
    return float(np.mean(xs)) if xs else float("nan")


def spec_distance(Ep, Et):
    """Ep, Et [4] band energies -> ratio_distance over the bands whose truth share >= SHARE."""
    tot = Et.sum(); inc = [bool(e >= SHARE * tot and e > 0) for e in Et]
    return ratio_distance([Ep[i] / Et[i] if Et[i] > 0 else None for i in range(len(Et))], inc)


def w1(p, t):
    """exact 1-D Wasserstein-1 distance of two equal-size empirical distributions / std(t)."""
    p = np.sort(np.ravel(p)); t = np.sort(np.ravel(t)); s = t.std()
    return float(np.abs(p - t).mean() / s) if s > 0 else float("nan")


def quadratic(fam, x, chans):
    H, W = x.shape[-2:]; _, _, k2 = kgrid(H, W); c = {n: i for i, n in enumerate(chans)}; q = {}
    if fam in ("kolmo", "nsu"):
        w = x[:, c["vorticity"]]; wk = np.fft.fft2(w) / (H * W); inv = np.where(k2 > 0, 1.0 / np.where(k2 > 0, k2, 1.0), 0.0)
        q["kinetic_energy"] = 0.5 * (np.abs(wk) ** 2 * inv).sum((-2, -1)); q["enstrophy"] = 0.5 * ((w - w.mean((-2, -1), keepdims=True)) ** 2).mean((-2, -1))
    else:
        for n, i in c.items(): q[f"var_{n}"] = ((x[:, i] - x[:, i].mean((-2, -1), keepdims=True)) ** 2).mean((-2, -1))
    return q


def family_quantity(fam, x, chans):
    """B: x [F, C, H, W] -> {name: [F]}; kolmo / nsu have no extra per-frame quantity (NSU's decay rate is derived in window_stats)."""
    return {}


def rel_l2_np(P, T):
    """eval_metrics.rel_l2 on numpy: [F, C, H, W] -> [F, C]."""
    return np.sqrt(((P - T) ** 2).sum((-2, -1))) / np.maximum(np.sqrt((T ** 2).sum((-2, -1))), 1e-12)


def fp16_tol(P16, T16, idx):
    """window_match tolerance of one window and step: 1e-4 + the fp16 storage term (module doc), max over channels, capped at TOL_CAP."""
    tol = 0.0
    for ci in range(P16.shape[1]):
        p = P16[idx, ci]; t = T16[idx, ci]; d = p.astype(np.float64) - t.astype(np.float64)
        sq = math.sqrt((np.spacing(np.abs(p)).astype(np.float64) ** 2 + np.spacing(np.abs(t)).astype(np.float64) ** 2).mean() / 12)
        sd = math.sqrt((d ** 2).mean()) or 1e-30; tol = max(tol, FP16_SAFETY * (0.5 * (sq / sd) ** 2 + 3 * sq / (sd * math.sqrt(d.size))))
    return min(max(TOL_FLOOR, 1e-4 + tol), TOL_CAP)


def window_stats(fam, P, T, chans):
    """per-window, per-step statistics of a prediction P vs truth T [F, C, H, W] (float64)."""
    F = T.shape[0]; r = {"bands_p": np.stack([bands(P[:, ci]) for ci in range(P.shape[1])], 1), "bands_t": np.stack([bands(T[:, ci]) for ci in range(T.shape[1])], 1)}
    qp, qt = quadratic(fam, P, chans), quadratic(fam, T, chans)
    with np.errstate(divide="ignore", invalid="ignore"):
        for k in qt: r[f"A3_{k}"] = np.log(qp[k] / qt[k])
    r["A4_w1"] = np.array([np.mean([w1(P[f, ci], T[f, ci]) for ci in range(P.shape[1])]) for f in range(F)])
    bp, bt = family_quantity(fam, P, chans), family_quantity(fam, T, chans)
    for k in bt: r[f"B_{k}"] = np.abs(bp[k] - bt[k]) / np.where(np.abs(bt[k]) > 0, np.abs(bt[k]), np.nan)
    if fam == "nsu" and F >= 2:
        s = lambda ke: np.polyfit(np.arange(F), np.log(np.maximum(ke, 1e-300)), 1)[0]
        r["B_decay_rate_ratio"] = np.full(F, s(qp["kinetic_energy"]) / s(qt["kinetic_energy"]))
    if fam in ZERO_MEAN:
        ci = [chans.index(n) for n in ZERO_MEAN[fam] if n in chans]; rms = np.sqrt((T[:, ci] ** 2).mean((-2, -1)))
        r["C_mean_pred"] = (np.abs(P[:, ci].mean((-2, -1))) / rms).mean(-1); r["C_mean_true"] = (np.abs(T[:, ci].mean((-2, -1))) / rms).mean(-1)
    return r


def pool(stats, F):
    """per-window stats -> per-step summaries (band energies pooled over windows; window means)."""
    if not stats: return {}
    bp = np.sum([s["bands_p"] for s in stats], 0); bt = np.sum([s["bands_t"] for s in stats], 0)   # [F, C, 4]
    out = {"A2c_dist": [float(np.nanmean([spec_distance(bp[f, ci], bt[f, ci]) for ci in range(bt.shape[1])])) for f in range(F)]}
    tp, tt = bp.sum(1), bt.sum(1)   # pooled over channels (as last_frame_band_retention)
    out["A2top"] = [float(tp[f, -1] / tt[f, -1]) if tt[f, -1] >= SHARE * tt[f].sum() and tt[f, -1] > 0 else None for f in range(F)]
    for k in stats[0]:
        if k.startswith(("A3_", "A4_", "B_", "C_")):
            v = np.stack([s[k] for s in stats]); out[k] = [float(x) for x in np.nanmean(v, 0)]
            if k.startswith("A3_"): out["abs" + k] = [float(x) for x in np.nanmean(np.abs(v), 0)]
    return out


def json_a2(colj, truth_share=None):
    if not colj or not colj.get("last_frame_band_retention"): return None
    top = colj.get("last_frame_top_fraction_true")
    inc = [True if truth_share is None else bool(truth_share[i] >= SHARE) for i in range(3)] + [bool(top is not None and top >= SHARE)]
    r = [colj["last_frame_band_retention"][b] for b in BANDS]; rc = colj.get("recon_last_frame_band_retention")
    return {"dist": ratio_distance(r, inc), "top": r[-1] if inc[-1] else None, "included": inc,
            "floor_dist": ratio_distance([rc[b] for b in BANDS], inc) if rc else None}


# ------------------------------------------------------------------------------------------------ data access and checks
def load_dump(path):
    """{(path, traj, first_frame): (H, P, T) fp16} and the meta dict, or None."""
    if not os.path.exists(path): return None
    with np.load(path) as z:
        meta = json.loads(str(z["meta"])); d = {}
        for i, w in enumerate(meta["windows"]):
            d[(w["path"], str(w["traj"]), int(w["first_frame"]))] = (z[f"H{i}"], z[f"P{i}"], z[f"T{i}"])
    return d, meta


def load_json(m, run):
    p = os.path.join(JSON[m], run + ".json")
    if not os.path.exists(p): return None
    d = json.load(open(p)); return d["eval"] if isinstance(d.get("eval"), dict) else d


def _match(mine, js, tol):
    """one window_match comparison: (ok, relative difference); non-finite on both sides matches, NaN anywhere else fails."""
    if js is None or not math.isfinite(js): return (not math.isfinite(mine)), float("nan")
    d = abs(mine - js) / max(abs(js), 1e-12); return bool(d <= tol), d


def checks(m, dump, ours_dump, colj):
    """window_match / persistence_match / band_match / truth_match of one model and column -> dict of results with an 'ok' flag each (None = not applicable)."""
    out = {"window_match": None, "persistence_match": None, "band_match": None, "truth_match": None, "n_dump": len(dump)}
    if colj is None or not isinstance(colj.get("per_window"), list): out["window_match"] = {"ok": False, "why": "no JSON per_window", "n_compared": 0}; return out
    pw = {(w["path"], str(w["traj"]), int(w["first_frame"])): w for w in colj["per_window"]}
    nF = colj.get("frames") or len(colj.get("rel_l2_by_frame") or [])
    bad, worst, n_cmp, badp, n_p = [], 0.0, 0, [], 0
    for k, (H16, P16, T16) in dump.items():
        w = pw.get(k)
        if w is None: bad.append(f"{k[1]}@{k[2]}: not in the JSON"); continue
        n_cmp += 1; e = rel_l2_np(P16.astype(np.float64), T16.astype(np.float64))
        for F_, idx in ((1, 0), (nF, P16.shape[0] - 1)):
            tol = fp16_tol(P16, T16, idx); mine = float(np.nanmean(e[idx])) if np.isfinite(e[idx]).any() else float("nan")
            ok, d = _match(mine, w.get(f"rel_l2@{F_}"), tol)
            if ok and math.isfinite(d): worst = max(worst, d / tol)
            if not ok: bad.append(f"{k[1]}@{k[2]} step {F_}: dump {mine:.6g} vs JSON {w.get(f'rel_l2@{F_}')} (rel {d:.2e}, tol {tol:.2e})")
        if m == "ours" and "persistence@1" in w:   # persistence_match: history / truth alignment of our dump
            n_p += 1
            pe = rel_l2_np(H16[-1:].astype(np.float64), T16[:1].astype(np.float64))
            okp, dp = _match(float(np.nanmean(pe[0])), w["persistence@1"], fp16_tol(np.repeat(H16[-1:], 1, 0), T16[:1], 0))
            if not okp: badp.append(f"{k[1]}@{k[2]}: persistence {float(np.nanmean(pe[0])):.6g} vs JSON {w['persistence@1']} (rel {dp:.2e})")
    out["window_match"] = {"ok": (not bad) and n_cmp >= len(dump) and n_cmp > 0, "n_compared": n_cmp, "worst_over_tol": worst, "fail": bad[:5]}
    if m == "ours" and n_p: out["persistence_match"] = {"ok": not badp and n_p >= n_cmp, "n_compared": n_p, "fail": badp[:5]}
    if dump and len(dump) == colj.get("n_windows") and colj.get("last_frame_band_retention"):   # band_match: only when the dump holds the whole column
        bp = sum(bands(P16[-1].astype(np.float64)).sum(0) for _, P16, _ in dump.values()); bt = sum(bands(T16[-1].astype(np.float64)).sum(0) for _, _, T16 in dump.values())
        diffs = [abs(bp[i] / bt[i] - colj["last_frame_band_retention"][b]) / max(abs(colj["last_frame_band_retention"][b]), 1e-12)
                 for i, b in enumerate(BANDS) if bt[i] >= SHARE_BAND_MATCH * bt.sum() and bt[i] > 0]
        out["band_match"] = {"ok": bool(diffs) and bool(max(diffs) <= 1e-3), "worst": float(max(diffs)) if diffs else None}
    if m != "ours":
        v = [float(np.max(rel_l2_np(T16.astype(np.float64), ours_dump[k][2].astype(np.float64)))) for k, (_, _, T16) in dump.items() if k in ours_dump]
        out["truth_match"] = {"ok": bool(v) and max(v) <= 1e-3, "worst": max(v) if v else None}
    return out


def passed(c):
    """a check dict -> True when every applicable check passed."""
    return all((c.get(k) is None) or bool(c[k].get("ok")) for k in ("window_match", "persistence_match", "band_match", "truth_match")) and c.get("window_match") is not None


def gate_of(checks_):
    """{"m:fam__split": checks} -> {(m, fam): ok}; persistence follows ours (built from ours' dump)."""
    g = {}
    for ck, c in checks_.items():
        m, key = ck.split(":", 1); fam = key.split("__")[0]; g[(m, fam)] = g.get((m, fam), True) and passed(c)
    for (m, fam), ok in list(g.items()):
        if m == "ours": g[("persistence", fam)] = ok
    return g


def apply_gate(splits, gate):
    """remove every number of a (model, family) whose checks failed (report AND result JSON)."""
    for key, ent in splits.items():
        fam = key.split("__")[0]
        for m in list(ent["models"]):
            if not gate.get((m, fam), False): ent["models"][m] = "withheld (checks)"
        for m in list(ent.get("json", {})):
            if not gate.get((m, fam), False): ent["json"][m] = "withheld (checks)"


def gmean(xs):
    xs = [x for x in xs if x is not None and math.isfinite(x) and x > 0]; return math.exp(sum(map(math.log, xs)) / len(xs)) if xs else None


def family_values(splits, fam, steps=(16, 64)):
    fv = {}
    cols = [(k, e) for k, e in splits.items() if k.startswith(fam + "__") and e["class"] in ("test", "interp", "ext")]
    for m in MODELS + ("persistence",):
        for metric in ("A2c_dist", "absA3_" + QMAIN[fam], "A4_w1"):
            for h in ("all", "dev", "test"):
                for f in steps:
                    xs = [ (e["models"][m][h].get(metric) or [None] * f)[f - 1] for _, e in cols
                           if isinstance(e["models"].get(m), dict) and e["models"][m].get(h) and len(e["models"][m][h].get(metric) or []) >= f]
                    if xs: fv.setdefault(m, {}).setdefault(metric, {}).setdefault(h, {})[f"t{f}"] = gmean(xs)
        xs = [e["json"][m]["A2"]["dist"] for _, e in cols if isinstance(e.get("json", {}).get(m), dict) and e["json"][m].get("A2")]
        if xs: fv.setdefault(m, {})["A2_json_dist"] = gmean(xs)
    return fv


def column(fam, key, col, cls, runs, js, maps):
    """everything of one column: checks, JSON readings, dump statistics per model and half."""
    import devsplit
    od = load_dump(os.path.join(PRED["ours"], runs["ours"], key + ".npz"))
    ent = {"class": cls, "models": {}, "json": {}, "n": {}, "checks": {}}
    for m in MODELS:   # JSON readings (every window), available even without dumps
        colj = (js.get(m) or {}).get(col)
        if colj: ent["json"][m] = {"A1": colj.get("rel_l2_by_frame"), "n_windows": colj.get("n_windows"), "A2": None}
    if od is None: return ent, None
    ours, meta = od; chans = meta["channels"]; F = min(v[1].shape[0] for v in ours.values())
    half = {k: maps.get(col, {}).get(devsplit.unit_of(k[0], k[1]), "x") for k in ours}
    tshare = np.sum([bands(v[2][-1].astype(np.float64)).sum(0) for v in ours.values()], 0); tshare = tshare / tshare.sum()
    spec = {}
    for m in MODELS:
        colj = (js.get(m) or {}).get(col)
        if m in ent["json"]: ent["json"][m]["A2"] = json_a2(colj, tshare)
        d = ours if m == "ours" else (load_dump(os.path.join(PRED[m], runs[m], key + ".npz")) or (None, None))[0]
        if d is None: continue
        d = {k: v for k, v in d.items() if k in ours}
        ent["checks"][m] = checks(m, d, ours, colj)
        if not d: continue
        st = {k: window_stats(fam, v[1][:F].astype(np.float64), v[2][:F].astype(np.float64), chans) for k, v in d.items()}
        ent["n"][m] = len(st); ent["models"][m] = {h: pool([s for k, s in st.items() if h == "all" or half[k] == h], F) for h in ("all", "dev", "test")}
        spec[m] = {f: {"pred": np.sum([s["bands_p"][f - 1] for s in st.values()], 0).sum(0).tolist(), "true": np.sum([s["bands_t"][f - 1] for s in st.values()], 0).sum(0).tolist()} for f in SHOW if f <= F}
        if m == "ours":
            pst = {k: window_stats(fam, np.repeat(v[0][-1:].astype(np.float64), F, 0), v[2][:F].astype(np.float64), chans) for k, v in d.items()}
            ent["models"]["persistence"] = {h: pool([s for k, s in pst.items() if h == "all" or half[k] == h], F) for h in ("all", "dev", "test")}
            ent["floor"] = {"A1_tokenizer_floor@1": (colj or {}).get("tokenizer_floor@1")}
    ent["frames"] = F; return ent, spec


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--name", required=True)
    ap.add_argument("--ours", default="physdial-s0-eval-64step")
    ap.add_argument("--walrus", default="walrus-ft-eval-64step"); ap.add_argument("--dpot", default="dpot-ft-eval-64step")
    ap.add_argument("--families", default=",".join(FAMILIES), help="comma list out of kolmo, nsu")
    a = ap.parse_args(); t0 = time.time(); runs = {"ours": a.ours, "walrus": a.walrus, "dpot": a.dpot}; fams = [f for f in a.families.split(",") if f]
    assert set(fams) <= set(FAMILIES), fams
    import devsplit, main_table
    js = {m: load_json(m, r) for m, r in runs.items()}
    if js["ours"] is None: raise SystemExit(f"no JSON of ours: {a.ours}")
    maps = devsplit.split_maps({c: v for c, v in js["ours"].items() if isinstance(v, dict) and isinstance(v.get("per_window"), list)})
    res = {"runs": runs, "protocol": "physics report on dumped rollouts", "dump_windows_per_column": "the dumped windows (--dump_pred)", "splits": {}, "checks": {}, "family": {}}; specs = {}
    for fam in fams:
        for cls, sps in main_table.FAMS.get(fam, {}).items():
            for sp in sps:
                key = f"{fam}__{sp}"; ent, spec = column(fam, key, f"{fam}/{sp}", cls, runs, js, maps)
                for m, c in ent.pop("checks").items(): res["checks"][f"{m}:{key}"] = c
                res["splits"][key] = ent
                if spec: specs[key] = spec
                print(f"[physfields] {key}: windows {ent['n']}", flush=True)
    gate = gate_of(res["checks"]); res["gate"] = {f"{m}:{f}": v for (m, f), v in gate.items()}
    apply_gate(res["splits"], gate); specs = {k: {m: v for m, v in s.items() if gate.get((m, k.split('__')[0]), False)} for k, s in specs.items()}
    for fam in fams: res["family"][fam] = family_values(res["splits"], fam)
    res["seconds"] = time.time() - t0
    out = os.path.join(PKG, "results", a.name + ".json"); json.dump(res, open(out + ".tmp", "w"), indent=1, default=str); os.replace(out + ".tmp", out)
    odir = os.path.join(PKG, "results", "physics", a.name); os.makedirs(odir, exist_ok=True); json.dump(specs, open(os.path.join(odir, "spectra.json"), "w"))
    write_report(res, odir, a.name); plot(specs, odir); print("saved", out, flush=True)


def _ck(c):
    if not isinstance(c, dict): return str(c)
    f = lambda k: "n/a" if c.get(k) is None else ("pass" if c[k].get("ok") else "FAIL")
    s = f"window_match {f('window_match')} ({(c['window_match'] or {}).get('n_compared')}/{c.get('n_dump')} windows)"
    if c.get("persistence_match") is not None: s += f"; persistence_match {f('persistence_match')}"
    s += f"; band_match {f('band_match')}" + (f" ({c['band_match']['worst']:.1e})" if c.get("band_match") and c["band_match"].get("worst") is not None else "")
    if c.get("truth_match") is not None: s += f"; truth_match {f('truth_match')} ({c['truth_match'].get('worst')})"
    fails = [x for k in ("window_match", "persistence_match") for x in ((c.get(k) or {}).get("fail") or [])]
    return s + (f" — {fails[:3]}" if fails else "")


def write_report(res, odir, name):
    g3 = lambda x: "-" if x is None or (isinstance(x, float) and not math.isfinite(x)) else f"{x:.3g}"
    L = [f"# Physics report on dumped rollouts ({name})", "", "A1 / A2 main readings: evaluation JSONs (every window). Dump statistics (A2c curve, A3, A4, B, C, persistence): the dumped windows of each column.", "",
         "## Checks", ""]
    L += [f"- {ck}: {_ck(c)}" for ck, c in sorted(res["checks"].items())]
    L += ["", "Gate: " + ", ".join(f"{k} {'ok' if v else 'WITHHELD'}" for k, v in sorted(res["gate"].items())), "", "## Family values (geometric mean over test / interp / ext columns; all [dev / test])", ""]
    for fam, fv in res["family"].items():
        for m, mv in fv.items():
            for metric, hv in mv.items():
                if metric == "A2_json_dist": L.append(f"- {fam} {m} A2 (JSON, every window): {g3(hv)}"); continue
                L.append(f"- {fam} {m} {metric}: " + ", ".join(f"{t} {g3(hv['all'].get(t))} [{g3((hv.get('dev') or {}).get(t))} / {g3((hv.get('test') or {}).get(t))}]" for t in ("t16", "t64") if t in hv.get("all", {})))
    L += ["", "## Per column", ""]
    for key, ent in res["splits"].items():
        F = ent.get("frames") or 0; st = [f for f in SHOW if f <= F]
        L += [f"### {key} ({ent['class']}; dump windows {ent['n']}; floor {ent.get('floor')})", ""]
        for m, j in ent["json"].items():
            if not isinstance(j, dict): L.append(f"- JSON {m}: {j}"); continue
            a1 = j.get("A1") or []; a2 = j.get("A2") or {}
            L.append(f"- JSON {m} (n {j.get('n_windows')}): A1 " + ", ".join(f"t{f} {g3(a1[f - 1])}" for f in SHOW if f <= len(a1))
                     + f"; A2 dist {g3(a2.get('dist'))}, top {g3(a2.get('top'))}, floor {g3(a2.get('floor_dist'))}")
        if st:
            L += ["", "| metric | model | " + " | ".join(f"t={f}" for f in st) + " |", "|---|---|" + "---|" * len(st)]
            for n in sorted({n for mm in ent["models"].values() if isinstance(mm, dict) for n in mm.get("all", {})}):
                for m in MODELS + ("persistence",):
                    mm = ent["models"].get(m)
                    if not isinstance(mm, dict): continue
                    v = mm.get("all", {}).get(n)
                    if v: L.append(f"| {n} | {m} | " + " | ".join(g3(v[f - 1]) if f <= len(v) else "-" for f in st) + " |")
        L.append("")
    open(os.path.join(odir, "report.md"), "w").write("\n".join(L) + "\n")


def plot(specs, odir):
    try:   # band-energy spectra of the test columns (truth dashed)
        import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
        for key, d in specs.items():
            if not key.endswith("__test") or not d: continue
            fs = [f for f in (1, 16, 64) if any(f in dm for dm in d.values())] or [1]
            fig, axs = plt.subplots(1, len(fs), figsize=(4 * len(fs), 3.6), squeeze=False)
            for ax, f in zip(axs[0], fs):
                tr = None
                for m, dm in d.items():
                    if f in dm: tr = dm[f]["true"]; ax.semilogy(range(4), dm[f]["pred"], "o-", label=m)
                if tr is not None: ax.semilogy(range(4), tr, "k--", label="truth")
                ax.set_xticks(range(4)); ax.set_xticklabels(BANDS, fontsize=7); ax.set_title(f"{key} step {f}"); ax.legend(fontsize=7)
            fig.tight_layout(); fig.savefig(os.path.join(odir, f"spectra_{key}.png"), dpi=110); plt.close(fig)
    except Exception as e: print(f"[physfields] plots skipped: {type(e).__name__} {e}", flush=True)


if __name__ == "__main__":
    main()
