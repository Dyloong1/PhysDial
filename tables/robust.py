#!/usr/bin/env python
"""Robust per-column statistics of the main comparison: for every column of the families not seen in Walrus pretraining (kolmo,
nsu) and of the six seen ones (reported only), for every model with per-window records (ours, Walrus, DPOT-H), at frames 4 and 16:
  mean, standard deviation and median of the per-window relative L2, and a 95 % bootstrap interval of the mean that resamples trajectories;
  paired comparison ours vs Walrus / DPOT-H on the same windows (identity checked by (path, traj, first_frame)): share of windows where
  ours is lower, geometric mean of the per-window ratio ours / other with its trajectory-bootstrap 95 % interval, and the two-sided
  Wilcoxon signed-rank p-value of the per-window log ratios.
Per-window values: `rel_l2_by_frame` (every frame) else the stored `rel_l2@4` / `rel_l2@16`; a frame without per-window values is left '-'.
Inputs (result JSONs by run name):
  ours     physdial/results/physdial-s0-eval
  Walrus   physdial/baselines/walrus/results/walrus-ft-eval
  DPOT-H   physdial/baselines/dpot/results/dpot-ft-eval
Usage: python tables/robust.py --out <md> [--json <json>]"""
import argparse, json, math, os, sys
import numpy as np
HERE = os.path.dirname(os.path.abspath(__file__)); PKG = os.path.join(os.path.dirname(HERE), "physdial"); sys.path.insert(0, HERE); sys.path.insert(0, PKG)
import main_table as M

FRAMES = (4, 16); NBOOT = 2000; SEED = 0
RUNS = {"ours": "physdial-s0-eval", "walrus": "baselines/walrus/results/walrus-ft-eval", "dpot": "baselines/dpot/results/dpot-ft-eval"}
SEEN = ("gs", "rb", "sf", "trl", "euler", "acoustic")


def path(stem):
    return os.path.join(PKG, stem + ".json") if "/" in stem else os.path.join(PKG, "results", stem + ".json")


def load(key):
    p = path(RUNS[key])
    return (M.load(p) or {}), os.path.basename(p)


def windows(colv, F):
    """[(key, traj_id, value)] of a column at frame F, or None when no per-window value exists for F."""
    pw = colv.get("per_window") if isinstance(colv, dict) else None
    if not isinstance(pw, list): return None
    out = []
    for w in pw:
        x = None
        if isinstance(w.get("rel_l2_by_frame"), list) and len(w["rel_l2_by_frame"]) >= F: x = w["rel_l2_by_frame"][F - 1]
        elif f"rel_l2@{F}" in w: x = w[f"rel_l2@{F}"]
        else: return None
        out.append(((w["path"], str(w["traj"]), int(w["first_frame"])), f"{w['path']}:{w['traj']}", x))
    return out


def boot_groups(groups, stat, rng):
    """trajectory bootstrap: groups = {traj: [values...]}; stat(list of values) -> float; returns (lo, hi) 95 %."""
    keys = list(groups); vals = []
    for _ in range(NBOOT):
        pick = rng.integers(0, len(keys), len(keys)); xs = [v for k in pick for v in groups[keys[k]]]
        vals.append(stat(xs))
    return float(np.percentile(vals, 2.5)), float(np.percentile(vals, 97.5))


def wilcoxon_p(d):
    """two-sided Wilcoxon signed-rank p-value (scipy; zero differences dropped); None below 6 pairs."""
    from scipy.stats import wilcoxon
    d = np.asarray([x for x in d if x != 0.0])
    return float(wilcoxon(d).pvalue) if len(d) >= 6 else None


def describe(ws, rng):
    v = np.array([x for _, _, x in ws if x is not None and math.isfinite(x)])
    if len(v) == 0: return None
    g = {}
    for _, t, x in ws:
        if x is not None and math.isfinite(x): g.setdefault(t, []).append(x)
    lo, hi = boot_groups(g, lambda xs: float(np.mean(xs)), rng)
    return {"n": int(len(v)), "n_traj": len(g), "nonfinite": int(len(ws) - len(v)), "mean": float(v.mean()), "std": float(v.std(ddof=1)) if len(v) > 1 else 0.0,
            "median": float(np.median(v)), "ci95": [lo, hi]}


def paired(wa, wb, rng):
    A = {k: (t, x) for k, t, x in wa}; B = {k: x for k, _, x in wb}
    if set(A) != set(B): return {"status": "window_mismatch"}
    keys_ = [k for k in A if A[k][1] is not None and B[k] is not None and math.isfinite(A[k][1]) and math.isfinite(B[k]) and A[k][1] > 0 and B[k] > 0]
    if len(keys_) < 4: return {"status": "too_few"}
    lr = np.array([math.log(A[k][1] / B[k]) for k in keys_]); g = {}
    for k, x in zip(keys_, lr): g.setdefault(A[k][0], []).append(x)
    lo, hi = boot_groups(g, lambda xs: float(np.mean(xs)), rng)
    return {"status": "ok", "n": len(keys_), "win_rate": float(np.mean(lr < 0)), "gm_ratio": float(math.exp(lr.mean())), "ci95": [math.exp(lo), math.exp(hi)],
            "wilcoxon_p": wilcoxon_p(lr), "excluded_nonfinite": len(A) - len(keys_)}


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--out", required=True); ap.add_argument("--json", default=""); a = ap.parse_args()
    rng = np.random.default_rng(SEED)
    D = {k: load(k) for k in RUNS}; ours = dict(D["ours"][0])
    src = {"ours": ours, "walrus": D["walrus"][0], "dpot": D["dpot"][0]}
    fams = list(M.FAMS) + list(SEEN); res = {"sources": {k: v[1] for k, v in D.items()}, "frames": FRAMES, "columns": {}, "flags": []}
    for col in sorted(ours):
        fam = col.split("/")[0]
        if fam not in fams or "@" in col: continue
        if fam in M.FAMS and col.split("/", 1)[1] not in sum(M.FAMS[fam].values(), []): continue
        ent = {}
        for F in FRAMES:
            e = {}; W = {m: windows(src[m].get(col), F) for m in src}
            for m, ws in W.items():
                if ws is None: continue
                d = describe(ws, rng)
                if d is None: continue
                st = (src[m].get(col) or {}).get(f"rel_l2@{F}")
                if isinstance(st, (int, float)) and math.isfinite(st) and d["nonfinite"] == 0 and abs(d["mean"] - st) > 1e-5 * max(abs(st), 1e-12) + 1e-7:
                    res["flags"].append(f"{m} {col} F{F}: mean of per-window {d['mean']:.6g} != stored {st:.6g}")
                e[m] = d
            for other in ("walrus", "dpot"):
                if W.get("ours") is not None and W.get(other) is not None: e["ours_vs_" + other] = paired(W["ours"], W[other], rng)
            ent[F] = e
        res["columns"][col] = ent
    L = ["# Robust statistics of the main comparison (frame 4 = round 1, frame 16 = round 4)\n",
         "Per column and model: window mean +- standard deviation and the 95% interval of the mean (trajectory bootstrap, 2000 resamples); "
         "paired on the same windows: geometric mean of the per-window ratio ours / other with its 95% interval (trajectory bootstrap), "
         "share of windows where ours is lower, Wilcoxon signed-rank p-value. `-` = no per-window record at that frame or not covered.",
         f"Sources: {res['sources']}." + (f" **{len(res['flags'])} check mismatches, listed at the end.**" if res["flags"] else " Per-window means agree with the stored column values.") + "\n"]
    for F in FRAMES:
        L += [f"## Frame {F}\n", "| Column | Ours mean+-std [95%] | Walrus mean+-std [95%] | DPOT mean+-std | Ours / Walrus geo-mean [95%] . win rate . p | Ours / DPOT geo-mean [95%] . win rate . p |",
              "|---|---|---|---|---|---|"]
        for col, ent in res["columns"].items():
            e = ent.get(F, {})
            def m(k, ci=True):
                d = e.get(k)
                return "-" if not d else f"{d['mean']:.4f}+-{d['std']:.4f}" + (f" [{d['ci95'][0]:.4f}, {d['ci95'][1]:.4f}]" if ci else "") + (f" ({d['nonfinite']} non-finite)" if d["nonfinite"] else "")
            def p(k):
                d = e.get(k)
                if not d: return "-"
                if d["status"] != "ok": return d["status"]
                pv = "-" if d["wilcoxon_p"] is None else (f"{d['wilcoxon_p']:.1e}" if d["wilcoxon_p"] < 1e-3 else f"{d['wilcoxon_p']:.3f}")
                return f"{d['gm_ratio']:.2f} [{d['ci95'][0]:.2f}, {d['ci95'][1]:.2f}] . {d['win_rate']:.0%} . {pv}"
            L.append(f"| {col} | {m('ours')} | {m('walrus')} | {m('dpot', False)} | {p('ours_vs_walrus')} | {p('ours_vs_dpot')} |")
        L.append("")
    if res["flags"]: L += ["## Check mismatches", ""] + [f"- {x}" for x in res["flags"]]
    open(a.out, "w").write("\n".join(L) + "\n")
    if a.json: json.dump(res, open(a.json, "w"), indent=1)
    print("wrote", a.out, "flags", len(res["flags"]))


if __name__ == "__main__":
    main()
