#!/usr/bin/env python
"""Physics-property report (reads finished result JSONs only; CPU).

Models (same evaluation windows required per column, checked as in main_table.py: (path, traj, first_frame) of every window):
  ours    physdial/results/physdial-s0-eval
  walrus  physdial/baselines/walrus/results/walrus-ft-eval
  dpot    physdial/baselines/dpot/results/dpot-ft-eval
  64-step: physdial-s0-eval-64step / walrus-ft-eval-64step when they exist ("-" otherwise).

Metrics (only where every compared model stores them; the last frame = step 16, or 64 in the 64-step section):
  band      last_frame_band_retention: predicted / true spectral energy in 4 radial bands of the Nyquist wavenumber
            ((0,1/8], (1/8,1/4], (1/4,1/2], (1/2,1]); 1 = right energy. "spectral distance" = mean over the 4 bands of |log ratio|.
  grad      last_frame_grad_ratio: predicted / true gradient energy (1 = right; > 1 = rougher, < 1 = smoother).
  top       last_frame_top_retention_by_channel: highest band per channel.
  demeaned  rel_l2_demeaned@{1,4,16}; vrmse@{1,16}; error growth rel_l2_by_frame at steps 1/4/16 (32/48/64 in the 64-step files);
            persistence ratio = model / persistence (copy the last frame) at step 16; nonfinite_pred_windows.
  ours-only log_amp_ratio, pattern_corr, rel_l2_shape (the other models' evaluators do not compute them).
Families: the 2 Walrus has not seen in pretraining (kolmo, nsu) compared; the 6 it has (gs, rb, sf, trl, euler, acoustic) reported only.
Usage: python tables/physics_report.py [--out DIR]   (default physdial/results/tables/)"""
import argparse, json, math, os, sys

HERE = os.path.dirname(os.path.abspath(__file__)); PKG = os.path.join(os.path.dirname(HERE), "physdial")
sys.path.insert(0, HERE)
import devsplit   # window_key: the same (path, traj, first_frame) key main_table.py compares
R = os.path.join(PKG, "results")
FILES = {
    "ours": os.path.join(R, "physdial-s0-eval.json"),
    "walrus": os.path.join(PKG, "baselines", "walrus", "results", "walrus-ft-eval.json"),
    "dpot": os.path.join(PKG, "baselines", "dpot", "results", "dpot-ft-eval.json"),
    "ours64": os.path.join(R, "physdial-s0-eval-64step.json"),
    "walrus64": os.path.join(PKG, "baselines", "walrus", "results", "walrus-ft-eval-64step.json"),
}
UNSEEN = ["kolmo", "nsu"]
SEEN = ["gs", "rb", "sf", "trl", "euler", "acoustic"]
NAME = {"kolmo": "Kolmogorov", "nsu": "NSU", "gs": "GS", "rb": "RB",
        "sf": "Shear flow", "trl": "TRL", "euler": "Euler", "acoustic": "Acoustic scattering"}
BANDS = ["(0,1/8]", "(1/8,1/4]", "(1/4,1/2]", "(1/2,1]"]
MODELS = ["ours", "walrus", "dpot"]


def load(path):
    if not os.path.exists(path): return None
    d = json.load(open(path)); e = d["eval"] if isinstance(d.get("eval"), dict) else d
    return {c: v for c, v in e.items() if "/" in c and "@" not in c and not c.startswith("_") and isinstance(v, dict)}


def fin(x): return isinstance(x, (int, float)) and not isinstance(x, bool) and math.isfinite(x)


def num(x, d=2): return f"{x:.{d}f}" if fin(x) else "-"


def wkeys(col):
    pw = col.get("per_window")
    if not isinstance(pw, list): return None
    return {devsplit.window_key(w) for w in pw}   # a set: Walrus / DPOT list the windows in another order (largest grids first)


def specdist(band):
    """mean over the 4 bands of |log(pred / true)|; None if a band is missing or non-positive."""
    if not isinstance(band, dict): return None
    v = [band.get(b) for b in BANDS]
    if not all(fin(x) and x > 0 for x in v): return None
    return sum(abs(math.log(x)) for x in v) / len(v)


def at_frame(col, F):
    bf = col.get("rel_l2_by_frame")
    return bf[F - 1] if isinstance(bf, list) and len(bf) >= F and fin(bf[F - 1]) else None


def column_rows(srcs, fam, frames):
    """per column of the family: {model: metrics} for the models whose windows equal ours'; plus mismatch list."""
    base = srcs["ours"]; out = {}; mism = []
    for col in sorted(c for c in base if c.split("/")[0] == fam):
        ko = wkeys(base[col]); row = {}
        for m, src in srcs.items():
            if src is None or col not in src: continue
            c = src[col]; kc = wkeys(c)
            if m != "ours" and ko is not None and kc is not None and kc != ko:
                mism.append(f"{col}: {m} windows differ from ours"); continue
            band = c.get("last_frame_band_retention"); last = frames
            r = {"band": band if isinstance(band, dict) else None, "spec": specdist(band), "grad": c.get("last_frame_grad_ratio"),
                 "top": c.get("last_frame_top_retention_by_channel"), "nonfinite": c.get("nonfinite_pred_windows"),
                 "pers": (c.get(f"rel_l2@{last}") / c.get(f"persistence@{last}")) if fin(c.get(f"rel_l2@{last}")) and fin(c.get(f"persistence@{last}")) and c.get(f"persistence@{last}") > 0 else None,
                 "err": {F: at_frame(c, F) for F in (1, 4, 16, 32, 48, 64) if F <= last},
                 "demeaned": {F: c.get(f"rel_l2_demeaned@{F}") for F in (1, 4, 16)}, "vrmse": {F: c.get(f"vrmse@{F}") for F in (1, 16)}}
            if m == "ours":
                r["ours_only"] = {"log_amp_ratio@16": c.get("log_amp_ratio@16"), "pattern_corr@16": c.get("pattern_corr@16"),
                                  "rel_l2_shape@16": c.get("rel_l2_shape@16"), "pattern_corr@1": c.get("pattern_corr@1")}
            row[m] = r
        out[col] = row
    return out, mism


def family_summary(rows, models):
    s = {}
    for m in models:
        sp = [r[m]["spec"] for r in rows.values() if m in r and fin(r[m]["spec"])]
        gd = [abs(math.log(r[m]["grad"])) for r in rows.values() if m in r and fin(r[m]["grad"]) and r[m]["grad"] > 0]
        top = [r[m]["band"]["(1/2,1]"] for r in rows.values() if m in r and r[m]["band"] and fin(r[m]["band"].get("(1/2,1]"))]
        nf = sum(int(r[m]["nonfinite"] or 0) for r in rows.values() if m in r)
        s[m] = {"spec": sum(sp) / len(sp) if sp else None, "n": len(sp), "grad_dev": sum(gd) / len(gd) if gd else None,
                "top_min": min(top) if top else None, "top_max": max(top) if top else None, "top_over15": sum(t > 1.5 for t in top),
                "top_n": len(top), "nonfinite": nf}
    return s


def section(L, J, title, fams, srcs, frames, models, compare=True):
    L.append(f"## {title}\n")
    L.append("Family value: last-frame (step %d) spectral distance = mean over 4 bands of |log(predicted energy / true)| (0 = exact), averaged "
             "over the family's columns; gradient deviation = mean |log gradient-energy ratio|; top band = energy above half Nyquist / true "
             "(columns > 1.5 / columns, range). Where the truth has almost no energy in the top band (share < 1e-4) that ratio is "
             "unstable and ours also carries the compressor's reconstruction bias (last column); read the spectral distance and the gradient "
             "deviation there.\n" % frames)
    L.append("| Family | " + " | ".join(f"Spectral dist. {m}" for m in models) + " | " + " | ".join(f"Grad. dev. {m}" for m in models) + " | "
             + " | ".join(f"Top band {m}" for m in models) + " | Closest spectrum | Truth top-band energy share (max over columns) | Ours reconstruction top-band ratio |")
    L.append("|---|" + "---|" * (3 * len(models) + 3))
    J[title] = {}
    for f in fams:
        rows, mism = column_rows(srcs, f, frames); s = family_summary(rows, models); J[title][f] = {"summary": s, "columns": rows, "window_mismatch": mism}
        best = [m for m in models if fin(s[m]["spec"]) and s[m]["n"] == s["ours"]["n"]]
        fr = [srcs["ours"][c].get("last_frame_top_fraction_true") for c in rows if fin(srcs["ours"][c].get("last_frame_top_fraction_true"))]
        rc = [srcs["ours"][c].get("recon_last_frame_band_retention", {}).get("(1/2,1]") for c in rows if isinstance(srcs["ours"][c].get("recon_last_frame_band_retention"), dict)]
        rc = [x for x in rc if fin(x)]; s["truth_top_share"] = max(fr) if fr else None; s["recon_top"] = (min(rc), max(rc)) if rc else None
        bm = min(best, key=lambda m: s[m]["spec"]) if compare and len(best) > 1 else ("reported only" if not compare else "-")
        top = lambda m: "-" if not s[m]["top_n"] else f"{s[m]['top_over15']}/{s[m]['top_n']}({num(s[m]['top_min'])}–{num(s[m]['top_max'])})"
        L.append(f"| {NAME[f]} | " + " | ".join(num(s[m]["spec"]) for m in models) + " | " + " | ".join(num(s[m]["grad_dev"]) for m in models)
                 + " | " + " | ".join(top(m) for m in models) + f" | {bm} | {('%.0e' % s['truth_top_share']) if fin(s['truth_top_share']) else '-'} | "
                 + (f"{num(s['recon_top'][0])}–{num(s['recon_top'][1])}" if s["recon_top"] else "-") + " |")
    L.append("")
    L.append(f"### Columns (last frame of step {frames}; bands = predicted / true in 4 bands; grad = gradient-energy ratio; persistence ratio = model error / error of copying the last frame)\n")
    L.append("| Column | Model | Bands (0,1/8] / (1/8,1/4] / (1/4,1/2] / (1/2,1] | Grad | Persistence ratio | Error at step " + " / ".join(str(F) for F in (1, 4, 16, 32, 48, 64) if F <= frames)
             + " | De-meaned error 1 / 4 / 16 | VRMSE 1 / 16 | Non-finite windows |")
    L.append("|---|---|---|---|---|---|---|---|---|")
    for f in fams:
        rows = J[title][f]["columns"]
        for col, row in rows.items():
            for m in models:
                if m not in row: continue
                r = row[m]; b = r["band"] or {}
                L.append(f"| {col} | {m} | " + " / ".join(num(b.get(k)) for k in BANDS) + f" | {num(r['grad'])} | {num(r['pers'])} | "
                         + " / ".join(num(r["err"].get(F), 3) for F in (1, 4, 16, 32, 48, 64) if F <= frames) + " | "
                         + " / ".join(num(r["demeaned"].get(F), 3) for F in (1, 4, 16)) + " | " + " / ".join(num(r["vrmse"].get(F), 3) for F in (1, 16))
                         + f" | {r['nonfinite'] if r['nonfinite'] is not None else '-'} |")
        for x in J[title][f]["window_mismatch"]: L.append(f"| {x} | not compared | | | | | | | |")
    L.append("")


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--out", default="", help="output directory (default physdial/results/tables)"); a = ap.parse_args()
    a.out = a.out or os.path.join(R, "tables")
    raw = {k: load(p) for k, p in FILES.items()}
    if raw["ours"] is None: raise SystemExit(f"ours missing: {FILES['ours']}")
    ours = raw["ours"]; srcs = {"ours": ours, "walrus": raw["walrus"], "dpot": raw["dpot"]}
    L = ["# Physics-property report (reads finished results only; generated by physics_report.py)\n",
         "Ours = PhysDial; Walrus = fine-tuned Walrus; DPOT-H = fine-tuned DPOT-H. A column is compared only "
         "when the windows are identical. `-` = pending / not covered / not computed by that model's evaluator.\n"]
    J = {"files": {k: (p if raw[k] is not None else None) for k, p in FILES.items()}}
    section(L, J, f"Step 16 . {len(UNSEEN)} families not seen by Walrus", UNSEEN, srcs, 16, MODELS)
    section(L, J, "Step 16 . 6 families seen by Walrus (reported only)", SEEN, srcs, 16, MODELS, compare=False)
    if raw["ours64"] is not None or raw["walrus64"] is not None:
        if raw["ours64"] is None or raw["walrus64"] is None:
            L.append("## Step 64\n\nOnly one side is available (" + ", ".join(k for k in ("ours64", "walrus64") if raw[k] is None) + " pending); not compared.\n")
        else:
            section(L, J, "Step 64 . long rollout (Kolmogorov, NSU)", ["kolmo", "nsu"], {"ours": raw["ours64"], "walrus": raw["walrus64"]}, 64, ["ours", "walrus"])
    else:
        L.append("## Step 64\n\n- (both pending)\n")
    L.append("## Quantities only our evaluator computes (step 16)\n")
    L.append("| Column | log amplitude ratio | Pattern correlation (step 1 / 16) | Shape error |\n|---|---|---|---|")
    for f in UNSEEN + SEEN:
        for col in sorted(c for c in ours if c.split("/")[0] == f):
            c = ours[col]
            L.append(f"| {col} | {num(c.get('log_amp_ratio@16'), 3)} | {num(c.get('pattern_corr@1'), 3)} / {num(c.get('pattern_corr@16'), 3)} | {num(c.get('rel_l2_shape@16'), 3)} |")
    L.append("")
    os.makedirs(a.out, exist_ok=True)
    open(os.path.join(a.out, "physics_report.md"), "w").write("\n".join(L) + "\n")
    json.dump(J, open(os.path.join(a.out, "physics_report.json"), "w"), indent=1, ensure_ascii=False, default=str)
    print("wrote", os.path.join(a.out, "physics_report.md"))


if __name__ == "__main__":
    main()
