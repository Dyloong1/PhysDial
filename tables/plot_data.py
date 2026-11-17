#!/usr/bin/env python
"""Plot data of the paper's figures: CSVs exported from finished evaluations (CPU; reads the prediction dumps, the switch group files and
the evaluation JSONs). Writes physdial/results/plot_data/<name>/.

  1. switch_curves_<switch|control>_<group>.csv: steps 1-16 after a single condition change (Kolmogorov, 32 windows per group), from the
     prediction dumps of physdial-switch-s0-eval-switch / physdial-switch-control-s0-eval-switch and the group files; the definitions of
     switch/case_study.py: distances to the switched truth S (B) and to the control truth C (A), both / ||S||, for the model told the new
     condition (view B) and the old one (view A); response r = <P - C, S - C> / ||S - C||^2 (window mean, left empty where the mean effect
     < 0.05, as switch/analyze_switch.py); effect = ||S - C|| / ||S||.
  2. switch_study.csv: the controlled switch study at step 16 (study-switch-s0 / -s1 and study-control-s0 / -s1, from their
     switch/analyze_switch.py read-outs *-response): per group the error under the new condition, the response r and the effect.
  3a. long_rollout_<family>_<split>.csv: steps 1-64, window mean of the relative L2 and its 95 % trajectory-bootstrap interval (2000
     resamples, seed 0), PhysDial / Walrus / DPOT-H 64-step evaluations (per-window records).
  3b. spectrum_<family>_<split>_step<k>[_N<grid>].csv: radial energy spectrum E(k) of the step-16 and step-64 frames, integer shells,
     summed over channels, window mean, from the 64-step dumps (all 32 windows) of PhysDial, Walrus and DPOT-H, the U-Net / FNO
     specialists (small_expert.py --dump_main 64-step evaluations, the same windows) and the autoencoder (encode-decode of the true frames,
     R<i> of the PhysDial dump); a column is empty where its fields are absent. The file without suffix holds the grid with most windows.
checks.json: per column and model the window counts and step-16 / step-64 readings recomputed from the dumps next to the evaluation JSONs'
values, and the band ratios (eval_metrics bands, pooled over windows and channels) from the fields next to each evaluation's all-window
last_frame_band_retention (step 64: the 64-step evaluation; step 16: the 16-step main evaluation, whose windows differ); band ratios at
step 4 as well.
Usage: python3 tables/plot_data.py --name <name> [--only curves,study,long,spectra]."""
import argparse, csv, glob, json, os, sys
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__)); REPO = os.path.dirname(HERE); PKG = os.path.join(REPO, "physdial"); SWITCH = os.path.join(REPO, "switch")
sys.path.insert(0, PKG); sys.path.insert(0, HERE); sys.path.insert(0, SWITCH)

R = os.path.join(PKG, "results"); WR = os.path.join(PKG, "baselines", "walrus", "results"); DR = os.path.join(PKG, "baselines", "dpot", "results")
BR = os.path.join(PKG, "baselines", "results"); SR = os.path.join(SWITCH, "results")
SW_EVAL = {"switch": "physdial-switch-s0-eval-switch", "control": "physdial-switch-control-s0-eval-switch"}   # read-outs: <run>-response
STUDY = {("switch", 0): "study-switch-s0-response", ("switch", 1): "study-switch-s1-response",
         ("control", 0): "study-control-s0-response", ("control", 1): "study-control-s1-response"}
F64 = {"physdial": (R, "physdial-s0-eval-64step"), "walrus": (WR, "walrus-ft-eval-64step"), "dpot": (DR, "dpot-ft-eval-64step")}
REF16 = {"physdial": (R, "physdial-s0-eval"), "walrus": (WR, "walrus-ft-eval"), "dpot": (DR, "dpot-ft-eval")}   # the 16-step main evaluations (band check)
EXPERTS = {"unet": (BR, "unet-{fam}-s0-eval-64step", "unet-{fam}-s0"), "fno": (BR, "fno-{fam}-s0-eval-64step", "fno-{fam}-s0")}   # (dir, 64-step, 16-step)
LONG_COLS = ("kolmo/test", "nsu/test", "kolmo/nu_dense_interp", "nsu/interp_test")
SPEC_COLS = ("kolmo/test", "nsu/test"); SPEC_STEPS = (16, 64)
SPEC_MODELS = ["physdial", "walrus", "dpot", "unet", "fno", "autoencoder"]
MIN_EFFECT = 0.05; NBOOT = 2000; SEED = 0


def g6(x):
    return "" if x is None or (isinstance(x, float) and not np.isfinite(x)) else f"{x:.6g}"


def write_csv(path, cols, rows):
    with open(path, "w", newline="") as f:
        w = csv.writer(f); w.writerow(cols)
        for r in rows: w.writerow([r[0]] + [g6(x) for x in r[1:]])


def nrm(x): return float(np.sqrt((x.astype(np.float64) ** 2).sum()))


def load_eval(d, run):
    """{column: column dict} of a result JSON (train_gen / specialist: key "eval"; walrus / dpot: top level), or None."""
    p = os.path.join(d, run + ".json")
    if not os.path.exists(p): return None
    j = json.load(open(p)); return j["eval"] if isinstance(j.get("eval"), dict) else j


# ------------------------------------------------------------------------------------------------ 1. switch curves
def switch_curves(od, chk):
    import case_study as cs
    for model, ev in SW_EVAL.items():
        pdir = os.path.join(SR, "pred", ev)
        groups = sorted({os.path.basename(f)[len("kolmo__switch_"):-len("_B.npz")] for f in glob.glob(os.path.join(pdir, "kolmo__switch_*_B.npz"))})
        for g in groups:
            meta, PBs, TBs = cs.load_view(pdir, "kolmo", g, "B"); metaA, PAs, _ = cs.load_view(pdir, "kolmo", g, "A")
            assert [(w["path"], w["traj"], w["first_frame"]) for w in meta["windows"]] == [(w["path"], w["traj"], w["first_frame"]) for w in metaA["windows"]], g
            st = meta["stride"]; n = PBs[0].shape[0]; M = {k: [] for k in ("nB", "nA", "oB", "oA", "rn", "ro", "e")}
            for i, w in enumerate(meta["windows"]):
                _, S, C, *_ = cs.frames(w, st, n)
                assert np.abs(S - TBs[i]).max() <= 1e-2 * max(np.abs(S).max(), 1e-6), f"{g} {i}: dumped truth != file frames"
                cur = {k: [] for k in M}
                for m in range(n):
                    s, c = S[m], C[m]; ns = nrm(s); d = (s - c).astype(np.float64); dd = float((d ** 2).sum())
                    for tag, P in (("n", PBs[i]), ("o", PAs[i])):
                        cur[tag + "B"].append(nrm(P[m] - s) / ns); cur[tag + "A"].append(nrm(P[m] - c) / ns)
                        cur["r" + tag].append(float(((P[m] - c).astype(np.float64) * d).sum()) / dd if dd > 0 else np.nan)
                    cur["e"].append(np.sqrt(dd) / ns)
                for k in M: M[k].append(cur[k])
            A = {k: np.array(v) for k, v in M.items()}   # [windows, steps]
            rows = []
            for m in range(n):
                q = lambda x: (float(np.median(x[:, m])), float(np.percentile(x[:, m], 25)), float(np.percentile(x[:, m], 75)))
                ok = float(np.mean(A["e"][:, m])) >= MIN_EFFECT
                rows.append([m + 1, *q(A["nB"]), *q(A["nA"]), *q(A["oB"]), *q(A["oA"]),
                             float(np.nanmean(A["rn"][:, m])) if ok else None, float(np.nanmean(A["ro"][:, m])) if ok else None, float(np.median(A["e"][:, m]))])
            write_csv(os.path.join(od, f"switch_curves_{model}_{g}.csv"),
                      ["step", "new_to_B_median", "new_to_B_q25", "new_to_B_q75", "new_to_A_median", "new_to_A_q25", "new_to_A_q75",
                       "old_to_B_median", "old_to_B_q25", "old_to_B_q75", "old_to_A_median", "old_to_A_q25", "old_to_A_q75",
                       "r_new_mean", "r_old_mean", "effect_median"], rows)
            chk.setdefault("switch_curves", {})[f"{model}/{g}"] = {"n_windows": int(A["e"].shape[0]), "steps": n,
                "step16_new_to_B_mean": float(A["nB"][:, -1].mean()), "step16_new_to_A_mean": float(A["nA"][:, -1].mean()),
                "step16_old_to_B_mean": float(A["oB"][:, -1].mean()), "step16_r_new_mean": float(np.nanmean(A["rn"][:, -1])), "step16_effect_mean": float(A["e"][:, -1].mean())}
            print(f"[plot_data] curves {model}/{g}: {A['e'].shape[0]} windows", flush=True)
        ana = os.path.join(SR, ev[:-len("-eval-switch")] + "-response.json")   # analyze_switch.py read-out of the same evaluation: window means at step 16
        if os.path.exists(ana):
            for g, x in json.load(open(ana))["groups"].items():
                cb = (x.get("cross_B") or {}).get("16")
                if cb and f"{model}/{g}" in chk.get("switch_curves", {}):
                    chk["switch_curves"][f"{model}/{g}"].update(readout_step16_err_S_B=cb["err_S"], readout_step16_effect=cb["effect"])


# ------------------------------------------------------------------------------------------------ 2. controlled study
def switch_study(od, chk):
    rows = []
    for (model, seed), nm in STUDY.items():
        G = json.load(open(os.path.join(SR, nm + ".json")))["groups"]
        for g in sorted(G):
            x = G[g]; cb = x["cross_B"]["16"]
            rows.append([g, model, seed, x["rel_l2_B"]["16"], cb["response_mean"], cb["effect"]])
    with open(os.path.join(od, "switch_study.csv"), "w", newline="") as f:
        w = csv.writer(f); w.writerow(["group", "model", "seed", "err_B", "r16", "effect16"])
        for r in rows: w.writerow(r[:3] + [g6(v) for v in r[3:]])
    chk["switch_study"] = {"rows": len(rows), "sources": {f"{m}/s{s}": n for (m, s), n in STUDY.items()}}


# ------------------------------------------------------------------------------------------------ 3a. long rollouts
def boot(groups, rng):
    keys = list(groups); vals = []
    for _ in range(NBOOT):
        pick = rng.integers(0, len(keys), len(keys)); vals.append(np.mean(np.concatenate([groups[keys[k]] for k in pick]), 0))
    v = np.array(vals); return np.percentile(v, 2.5, 0), np.percentile(v, 97.5, 0)


def long_rollout(od, chk):
    E = {m: load_eval(*v) for m, v in F64.items()}; rng = np.random.default_rng(SEED)
    for col in LONG_COLS:
        per = {}
        for m in F64:
            c = (E[m] or {}).get(col)
            if not c or not c.get("per_window"): continue
            per[m] = {(w["path"], str(w["traj"]), int(w["first_frame"])): np.array(w["rel_l2_by_frame"], np.float64) for w in c["per_window"]}
        if not per: continue
        keys = [set(v) for v in per.values()]; same = all(k == keys[0] for k in keys)
        F = min(len(next(iter(v.values()))) for v in per.values()); out = {}
        for m, d in per.items():
            X = np.stack([v[:F] for v in d.values()]); grp = {}
            for (p, t, _), v in d.items(): grp.setdefault((p, t), []).append(v[:F])
            lo, hi = boot({k: np.stack(v) for k, v in grp.items()}, rng); out[m] = (X.mean(0), lo, hi)
            jm = E[m][col]["rel_l2_by_frame"]
            chk.setdefault("long_rollout", {}).setdefault(col, {})[m] = {"n_windows": int(X.shape[0]), "n_traj": len(grp),
                "mean@16": float(X.mean(0)[15]), "mean@32": float(X.mean(0)[31]) if F >= 32 else None, "mean@64": float(X.mean(0)[F - 1]),
                "json_rel_l2_by_frame@16": jm[15], "json@64": jm[F - 1]}
        chk["long_rollout"][col]["same_windows_all_models"] = same
        rows = [[f + 1] + [x for m in F64 for x in ((out[m][0][f], out[m][1][f], out[m][2][f]) if m in out else (None, None, None))] for f in range(F)]
        fam, sp = col.split("/")
        write_csv(os.path.join(od, f"long_rollout_{fam}_{sp}.csv"), ["step"] + [f"{m}_{s}" for m in F64 for s in ("mean", "lo", "hi")], rows)
        print(f"[plot_data] long {col}: {[(m, chk['long_rollout'][col][m]['n_windows']) for m in out]} same windows {same}", flush=True)


# ------------------------------------------------------------------------------------------------ 3b. spectra
def radial(x):
    """x [C, H, W] -> E(k) for integer shells k = 0 .. N/2 (sum over channels; fft / (H W), as eval_metrics.band_energy, periodic window)."""
    C, H, W = x.shape; X = np.fft.fft2(x.astype(np.float64)) / (H * W); P = (np.abs(X) ** 2).sum(0)
    kx = np.fft.fftfreq(H, 1.0 / H); ky = np.fft.fftfreq(W, 1.0 / W); kr = np.rint(np.sqrt(kx[:, None] ** 2 + ky[None, :] ** 2)).astype(int)
    kmax = min(H, W) // 2; return np.bincount(kr.ravel(), P.ravel(), minlength=kmax + 1)[:kmax + 1]


def band_ratios(Ps, Ts):
    """pooled band ratios (sum over windows and channels of the predicted band energy / the true one), eval_metrics bands, periodic window"""
    import physics_fields as pf
    bp = np.sum([pf.bands(p.astype(np.float64)).sum(0) for p in Ps], 0); bt = np.sum([pf.bands(t.astype(np.float64)).sum(0) for t in Ts], 0)
    return [float(x) for x in bp / np.maximum(bt, 1e-30)]


def spectra(od, chk):
    """E(k) at steps 16 and 64 on the windows common to PhysDial / Walrus / DPOT-H, the U-Net / FNO specialists and the autoencoder where their
    fields exist; band-ratio checks against the evaluation JSONs."""
    import physics_fields as pf
    from eval_metrics import BANDS
    for col in SPEC_COLS:
        fam_, sp_ = col.split("/"); key = col.replace("/", "__"); D = {}; RC = {}; rs = []
        for m, (d, run) in F64.items():
            r = pf.load_dump(os.path.join(d, "pred", run, key + ".npz"))
            if not r: continue
            D[m] = r[0]
            if m == "physdial":
                rs = r[1].get("recon_steps") or []
                if rs:
                    with np.load(os.path.join(d, "pred", run, key + ".npz")) as z:
                        RC = {(w["path"], str(w["traj"]), int(w["first_frame"])): z[f"R{i}"] for i, w in enumerate(r[1]["windows"]) if f"R{i}" in z}
        if "physdial" not in D: print(f"[plot_data] spectra {col}: no PhysDial dump, skipped", flush=True); continue
        EX = {}
        for m, (d, pat, _) in EXPERTS.items():
            f = os.path.join(d, "pred", pat.format(fam=fam_), key + ".npz")
            if os.path.exists(f):
                with np.load(f) as z:
                    meta = json.loads(str(z["meta"])); EX[m] = {(w["path"], str(w["traj"]), int(w["first_frame"])): (z[f"P{i}"], z[f"T{i}"]) for i, w in enumerate(meta["windows"]) if f"P{i}" in z}
        common = [k for k in D["physdial"] if all(k in D[m] for m in D)]
        info = {"windows_physdial": len(D["physdial"]), "windows_common": len(common), "models": sorted(D),
                "experts": {m: {"windows": len(v), "cover_common": all(k in v for k in common)} for m, v in EX.items()}, "autoencoder_windows": len(RC)}
        src64 = dict(F64, **{m: (d, pat.format(fam=fam_)) for m, (d, pat, _) in EXPERTS.items()}, autoencoder=F64["physdial"])
        src16 = dict(REF16, **{m: (d, p16.format(fam=fam_)) for m, (d, _, p16) in EXPERTS.items()}, autoencoder=REF16["physdial"])
        js = {64: {m: load_eval(*v) for m, v in src64.items()}, 16: {m: load_eval(*v) for m, v in src16.items()}}
        for step in SPEC_STEPS:
            byN = {}
            for k in common:
                T = D["physdial"][k][2][step - 1].astype(np.float32); N = T.shape[-1]
                e = byN.setdefault(N, {"solver": [], "_T": [], **{m: [] for m in SPEC_MODELS}})
                e["solver"].append(radial(T)); e["_T"].append(T)
                for m in D:
                    Tm = D[m][k][2][step - 1].astype(np.float32)
                    assert np.abs(Tm - T).max() <= 1e-2 * max(np.abs(T).max(), 1e-6), f"{col} {m} {k}: truth differs between dumps"
                    e[m].append(D[m][k][1][step - 1].astype(np.float32))
                for m, v in EX.items():
                    if info["experts"][m]["cover_common"] and v[k][0].shape[0] >= step:
                        assert np.abs(v[k][1][step - 1].astype(np.float32) - T).max() <= 1e-2 * max(np.abs(T).max(), 1e-6), f"{col} {m} {k}: specialist truth differs"
                        e[m].append(v[k][0][step - 1].astype(np.float32))
                if k in RC and step in rs: e["autoencoder"].append(RC[k][rs.index(step)].astype(np.float32))
            if not byN: continue
            Nmain = max(byN, key=lambda N_: len(byN[N_]["solver"]))
            allg = {}
            for e in byN.values():
                for m in ["_T"] + SPEC_MODELS: allg.setdefault(m, []).extend(e[m])
            jv_ = lambda m: (((js[step].get(m) or {}).get(col) or {}).get("recon_last_frame_band_retention" if m == "autoencoder" else "last_frame_band_retention") or {})
            info[f"step{step}_all_grids"] = {"n_windows": len(allg["_T"]), "grids": {str(N_): len(e_["solver"]) for N_, e_ in byN.items()}, "bands": BANDS, "band_ratios": {
                m: {"from_fields": band_ratios(allg[m], allg["_T"]), "json_all_windows": [jv_(m).get(b) for b in BANDS] if jv_(m) else None,
                    "json_source": (src64 if step == 64 else src16)[m][1]} for m in SPEC_MODELS if len(allg[m]) == len(allg["_T"])}}
            for N, e in byN.items():
                kmax = N // 2; n = len(e["solver"])
                spec_ = {m: (np.mean([radial(x) for x in e[m]], 0) if len(e[m]) == n else None) for m in SPEC_MODELS}
                sol = np.mean(e["solver"], 0)
                rows = [[k, k / kmax, sol[k]] + [None if spec_[m] is None else spec_[m][k] for m in SPEC_MODELS] for k in range(1, kmax + 1)]
                write_csv(os.path.join(od, f"spectrum_{fam_}_{sp_}_step{step}{'' if N == Nmain else f'_N{N}'}.csv"), ["k", "k_over_kmax", "solver"] + SPEC_MODELS, rows)
                info[f"step{step}_N{N}"] = {"n_windows": n, "bands": BANDS, "band_ratios": {m: {"from_fields": band_ratios(e[m], e["_T"])} for m in SPEC_MODELS if len(e[m]) == n}}
        Ts = [D["physdial"][k][2][3] for k in common]; bc = {}   # band ratios at step 4 (checks only), same windows and pooling
        for m in list(D) + list(EX):
            if m in D: Ps = [D[m][k][1][3] for k in common]
            elif info["experts"][m]["cover_common"]: Ps = [EX[m][k][0][3] for k in common]
            else: continue
            bc[m] = {"from_fields": band_ratios(Ps, Ts)}
        info["step4"] = {"n_windows": len(common), "bands": BANDS, "band_ratios": bc}
        chk.setdefault("spectra", {})[col] = info
        print(f"[plot_data] spectra {col}: " + json.dumps(info, default=float)[:1500], flush=True)


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--name", required=True); ap.add_argument("--only", default="", help="comma list of curves, study, long, spectra")
    a = ap.parse_args(); od = os.path.join(R, "plot_data", a.name); os.makedirs(od, exist_ok=True)
    parts = [x for x in a.only.split(",") if x] or ["curves", "study", "long", "spectra"]; chk = {"name": a.name}
    for p, fn in (("curves", switch_curves), ("study", switch_study), ("long", long_rollout), ("spectra", spectra)):
        if p in parts: fn(od, chk); json.dump(chk, open(os.path.join(od, "checks.json"), "w"), indent=1, default=float)
    print("saved", od, flush=True)


if __name__ == "__main__":
    main()
