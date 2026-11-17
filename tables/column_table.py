#!/usr/bin/env python3
"""Per-column table of the standard prediction (the paper's main table) and the extrapolation cost. CPU, standard library only; reads finished
result JSONs. No averaging across columns or steps: every number is the mean over the column's evaluation windows (32 for most columns) of
the relative L2 at exactly that step (rel_l2@F, else rel_l2_by_frame[F - 1]). One round = 4 steps: step 4 = round 1, step 16 = round 4.

A. Errors per family and column, steps 4 and 16, of PhysDial, Walrus, DPOT-H, the U-Net and the FNO specialist, with the second seeds of
   PhysDial and of the two specialists; Kolmogorov flow and decaying turbulence also at step 64 (the 64-step evaluations; the specialists'
   cover the test column).
B. Extrapolation cost of every column outside the training range: error of the model / error of its in-range reference (same architecture
   and budget, trained with the held-out values; same windows), steps 4 and 16; 1 = no cost.
C. Cost of the two fixed-direction columns of Kolmogorov flow without and with anchors (the anchor table): amplitude 0.5 (extrap_amp) and
   beta = 4 (extrap_beta), with amplitude 0.5 together with k_f 3 / 6 (extrap_joint) and k_f 6 alone (extrap_kf) on the amplitude
   pairs; per seed, steps 4 and 16.

Inputs (result JSONs by run name):
  PhysDial      physdial/results/physdial-s0-eval, physdial-s1-eval; physdial-s0-eval-64step
  Walrus        physdial/baselines/walrus/results/walrus-ft-eval, walrus-ft-eval-64step
  DPOT-H        physdial/baselines/dpot/results/dpot-ft-eval, dpot-ft-eval-64step
  U-Net / FNO   physdial/baselines/results/unet-<fam>-s0, unet-<fam>-s1, unet-<fam>-s0-eval-64step (fno-... the same)
  costs         PhysDial physdial-s0-eval-shift3 and physdial-s1-eval-shift3 / physdial-ref-s0-eval-shift3 (sampling shift 3 on both sides);
                Walrus (Kolmogorov) walrus-ft-eval-kolmo / walrus-ft-ref-beta-eval-kolmo, and with the beta anchors
                walrus-ft-beta-eval-kolmo / walrus-ft-ref-beta-eval-kolmo (one reference for both: continued with the amplitude and beta
                anchors, the same training stages as the models); U-Net unet-<fam>-s0 / unet-<fam>-ref-s0 and FNO fno-<fam>-s0 / fno-<fam>-ref-s0 (Kolmogorov flow, decaying
                turbulence).
  anchor table  amplitude, joint and k_f columns (evaluation at the end of training, pretrained decoder on both sides): stage2-noanchor-s0 and -s1 /
                stage2-noanchor-ref-s0 (without anchors), stage2-noperiodic-s0 and -s1 / stage2-noperiodic-ref-s0 (with the amplitude anchors);
                beta (fine-tuned decoder on both sides): stage2-noperiodic-s0-eval-beta and -s1-eval-beta / beta-anchor-ref-s0-eval-beta
                (without beta anchors), beta-anchor-s0-eval and beta-anchor-s1-eval / beta-anchor-ref-s0-eval-beta (with them);
                Walrus on beta (one run): walrus-ft-eval-kolmo (without) and walrus-ft-beta-eval-kolmo (with) / walrus-ft-ref-beta-eval-kolmo.
A missing file prints "pending", a file without the column "-", a non-finite value "diverged". A column whose window count differs from
PhysDial's (n_windows) is listed in the notes.

Usage: python3 tables/column_table.py [--out DIR]  (default physdial/results/tables/: column_table.md, column_table.json)."""
import argparse
import json
import math
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__)); PKG = os.path.join(os.path.dirname(HERE), "physdial")
R = os.path.join(PKG, "results"); WR = os.path.join(PKG, "baselines", "walrus", "results"); DR = os.path.join(PKG, "baselines", "dpot", "results")
BR = os.path.join(PKG, "baselines", "results")
FAMILIES = ("kolmo", "nsu", "gs", "rb", "sf", "trl", "euler", "acoustic")
NAME = {"kolmo": "Kolmogorov flow", "nsu": "Decaying turbulence", "gs": "Gray-Scott", "rb": "Rayleigh-Benard", "sf": "Shear flow",
        "trl": "Radiative layer", "euler": "Euler, multi-quadrant", "acoustic": "Acoustic scattering"}
LONG = ("kolmo", "nsu")   # families with 64-step evaluations
STEPS = (4, 16)
# model -> (label, {step group: (directory, run name pattern)}); "{fam}" = the family key; "64" = the 64-step evaluation
MODELS = {
    "physdial": ("PhysDial", {"main": (R, "physdial-s0-eval"), "64": (R, "physdial-s0-eval-64step")}),
    "physdial_s1": ("PhysDial seed 1", {"main": (R, "physdial-s1-eval")}),
    "walrus": ("Walrus", {"main": (WR, "walrus-ft-eval"), "64": (WR, "walrus-ft-eval-64step")}),
    "dpot": ("DPOT-H", {"main": (DR, "dpot-ft-eval"), "64": (DR, "dpot-ft-eval-64step")}),
    "unet": ("U-Net", {"main": (BR, "unet-{fam}-s0"), "64": (BR, "unet-{fam}-s0-eval-64step")}),
    "unet_s1": ("U-Net seed 1", {"main": (BR, "unet-{fam}-s1")}),
    "fno": ("FNO", {"main": (BR, "fno-{fam}-s0"), "64": (BR, "fno-{fam}-s0-eval-64step")}),
    "fno_s1": ("FNO seed 1", {"main": (BR, "fno-{fam}-s1")}),
}
SEEDS1 = {"unet_s1", "fno_s1"}   # second seeds of the specialists: the two turbulence families only
# extrapolation cost: label -> ((directory, model run), (directory, reference run), families or None = every family)
COSTS = {
    "PhysDial": ((R, "physdial-s0-eval-shift3"), (R, "physdial-ref-s0-eval-shift3"), None),
    "PhysDial seed 1": ((R, "physdial-s1-eval-shift3"), (R, "physdial-ref-s0-eval-shift3"), None),
    "Walrus": ((WR, "walrus-ft-eval-kolmo"), (WR, "walrus-ft-ref-beta-eval-kolmo"), ("kolmo",)),
    "Walrus + beta anchors": ((WR, "walrus-ft-beta-eval-kolmo"), (WR, "walrus-ft-ref-beta-eval-kolmo"), ("kolmo",)),
    "U-Net": ((BR, "unet-{fam}-s0"), (BR, "unet-{fam}-ref-s0"), LONG),
    "FNO": ((BR, "fno-{fam}-s0"), (BR, "fno-{fam}-ref-s0"), LONG),
}
# anchor table: (column, anchors, seed) -> (model run, reference run), in physdial/results unless the key's seed is "Walrus"
ANCHOR_COSTS = {
    ("kolmo/extrap_amp", "without", 0): ("stage2-noanchor-s0", "stage2-noanchor-ref-s0"),
    ("kolmo/extrap_amp", "without", 1): ("stage2-noanchor-s1", "stage2-noanchor-ref-s0"),
    ("kolmo/extrap_amp", "with", 0): ("stage2-noperiodic-s0", "stage2-noperiodic-ref-s0"),
    ("kolmo/extrap_amp", "with", 1): ("stage2-noperiodic-s1", "stage2-noperiodic-ref-s0"),
    ("kolmo/extrap_joint", "without", 0): ("stage2-noanchor-s0", "stage2-noanchor-ref-s0"),
    ("kolmo/extrap_joint", "without", 1): ("stage2-noanchor-s1", "stage2-noanchor-ref-s0"),
    ("kolmo/extrap_joint", "with", 0): ("stage2-noperiodic-s0", "stage2-noperiodic-ref-s0"),
    ("kolmo/extrap_joint", "with", 1): ("stage2-noperiodic-s1", "stage2-noperiodic-ref-s0"),
    ("kolmo/extrap_kf", "without", 0): ("stage2-noanchor-s0", "stage2-noanchor-ref-s0"),
    ("kolmo/extrap_kf", "without", 1): ("stage2-noanchor-s1", "stage2-noanchor-ref-s0"),
    ("kolmo/extrap_kf", "with", 0): ("stage2-noperiodic-s0", "stage2-noperiodic-ref-s0"),
    ("kolmo/extrap_kf", "with", 1): ("stage2-noperiodic-s1", "stage2-noperiodic-ref-s0"),
    ("kolmo/extrap_beta", "without", 0): ("stage2-noperiodic-s0-eval-beta", "beta-anchor-ref-s0-eval-beta"),
    ("kolmo/extrap_beta", "without", 1): ("stage2-noperiodic-s1-eval-beta", "beta-anchor-ref-s0-eval-beta"),
    ("kolmo/extrap_beta", "with", 0): ("beta-anchor-s0-eval", "beta-anchor-ref-s0-eval-beta"),
    ("kolmo/extrap_beta", "with", 1): ("beta-anchor-s1-eval", "beta-anchor-ref-s0-eval-beta"),
    ("kolmo/extrap_beta", "without", "Walrus"): ("walrus-ft-eval-kolmo", "walrus-ft-ref-beta-eval-kolmo"),
    ("kolmo/extrap_beta", "with", "Walrus"): ("walrus-ft-beta-eval-kolmo", "walrus-ft-ref-beta-eval-kolmo"),
}
PENDING = "pending"
_CACHE = {}


def load(d, run):
    """{column: column dict} of a result JSON (train_gen / specialist: key "eval"; walrus / dpot: top level, '_' keys are metadata,
    '@HxW' slices dropped), or None when the file does not exist yet."""
    p = os.path.join(d, run + ".json")
    if p not in _CACHE:
        if not os.path.exists(p): _CACHE[p] = None
        else:
            j = json.load(open(p)); e = j["eval"] if isinstance(j.get("eval"), dict) else j
            _CACHE[p] = {c: v for c, v in e.items() if "/" in c and "@" not in c and not c.startswith("_") and isinstance(v, dict)}
    return _CACHE[p]


def value(cols, col, F):
    """the column's window mean of rel_l2 at step F; PENDING (no file), None (no column / step), or a float (inf = non-finite)."""
    if cols is None: return PENDING
    v = cols.get(col)
    if not isinstance(v, dict) or "skipped" in v: return None
    x = v.get(f"rel_l2@{F}")
    if x is None and isinstance(v.get("rel_l2_by_frame"), list) and len(v["rel_l2_by_frame"]) >= F: x = v["rel_l2_by_frame"][F - 1]
    if x is None: return None
    return float(x) if math.isfinite(float(x)) else math.inf


def outside(sp):
    """a column outside the training range (extrapolation): the held-out value lies beyond the training values"""
    return "extrap" in sp or sp.endswith("ends")


def order(fam, cols):
    """test first, then the in-range held-out columns, then the columns outside the range (each group in name order)"""
    sps = sorted(c.split("/", 1)[1] for c in cols if c.split("/", 1)[0] == fam)
    return [f"{fam}/{sp}" for sp in sorted(sps, key=lambda s: (s != "test", outside(s), s))]


def ratio(cn, cd, col, F):
    """model / reference at step F of one column; PENDING, None (not covered), inf (non-finite) or a float"""
    x, y = value(cn, col, F), value(cd, col, F)
    if PENDING in (x, y): return PENDING
    if x is None or y is None: return None
    return math.inf if not math.isfinite(x) or not math.isfinite(y) or y == 0 else x / y


def runfile(spec, fam):
    d, pat = spec; return d, pat.format(fam=fam)


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--out", default="", help="output directory (default physdial/results/tables)")
    a = ap.parse_args(); a.out = a.out or os.path.join(R, "tables"); os.makedirs(a.out, exist_ok=True)
    ours = load(*MODELS["physdial"][1]["main"])
    if ours is None: sys.exit(f"PhysDial's evaluation missing: {os.path.join(R, 'physdial-s0-eval.json')}")
    out = {"statistic": "rel_l2 at step F, mean over the column's windows", "families": {}, "cost": {}, "notes": []}
    for fam in FAMILIES:
        fo = {}
        for col in order(fam, ours):
            sp = col.split("/", 1)[1]; n0 = ours[col].get("n_windows"); co = {"outside": outside(sp), "n_windows": n0, "values": {}}
            for m, (_, src) in MODELS.items():
                if m in SEEDS1 and fam not in LONG: continue
                cols = load(*runfile(src["main"], fam)); vals = {F: value(cols, col, F) for F in STEPS}
                if "64" in src and fam in LONG: vals[64] = value(load(*runfile(src["64"], fam)), col, 64)
                co["values"][m] = vals
                n = (cols or {}).get(col, {}).get("n_windows") if isinstance((cols or {}).get(col), dict) else None
                if n is not None and n0 is not None and n != n0: out["notes"].append(f"{col}: {MODELS[m][0]} has {n} windows, PhysDial {n0}")
            fo[col] = co
            if co["outside"]:
                cc = {}
                for lab, (num, den, fams) in COSTS.items():
                    if fams is not None and fam not in fams: continue
                    cn, cd = load(*runfile(num, fam)), load(*runfile(den, fam)); cc[lab] = {F: ratio(cn, cd, col, F) for F in STEPS}
                out["cost"][col] = cc
        out["families"][fam] = fo
    out["anchor_cost"] = [{"column": col, "anchors": an, "seed": sd, "model": num, "reference": den,
                           "values": {F: ratio(load(WR if sd == "Walrus" else R, num), load(WR if sd == "Walrus" else R, den), col, F) for F in STEPS}}
                          for (col, an, sd), (num, den) in ANCHOR_COSTS.items()]
    json.dump(out, open(os.path.join(a.out, "column_table.json"), "w"), indent=1, default=str)
    open(os.path.join(a.out, "column_table.md"), "w").write(render(out)); print(render(out))


def cell(x, k=4):
    if x == PENDING: return "pending"
    if x is None: return "-"
    if not math.isfinite(x): return "diverged"
    return f"{x:.{k}f}"


def render(out):
    L = ["# Standard prediction per column (generated by column_table.py)", "",
         "Relative L2 at steps 4 and 16 (Kolmogorov flow and decaying turbulence also step 64): mean over the column's evaluation windows. "
         "`pending` = result file not there yet, `-` = not covered.", ""]
    for fam, fo in out["families"].items():
        if not fo: L += [f"## {NAME[fam]} (`{fam}`)", "", "not in PhysDial's evaluation", ""]; continue
        ms = [m for m in MODELS if any(m in co["values"] for co in fo.values())]
        hdr = ["Column", "Range"] + [f"{MODELS[m][0]} step {F}" for m in ms for F in (STEPS + ((64,) if fam in LONG and "64" in MODELS[m][1] else ()))]
        L += [f"## {NAME[fam]} (`{fam}`)", "", "| " + " | ".join(hdr) + " |", "|" + "---|" * len(hdr)]
        for col, co in fo.items():
            row = [col.split("/", 1)[1], "outside" if co["outside"] else "in range"]
            for m in ms:
                v = co["values"].get(m, {}); row += [cell(v.get(F)) for F in (STEPS + ((64,) if fam in LONG and "64" in MODELS[m][1] else ()))]
            L.append("| " + " | ".join(row) + " |")
        L.append("")
    labs = list(COSTS)
    L += ["## Extrapolation cost (model / its in-range reference; 1 = no cost)", "",
          "Columns outside the training range. PhysDial: both sides evaluated with sampling shift 3. Walrus: Kolmogorov flow, the runs continued "
          "with the amplitude and beta anchors.", "",
          "| Column | " + " | ".join(f"{lab} step {F}" for lab in labs for F in STEPS) + " |", "|---|" + "---|" * (len(labs) * len(STEPS))]
    for col, cc in out["cost"].items():
        L.append(f"| {col} | " + " | ".join(cell((cc.get(lab) or {}).get(F, None), 2) for lab in labs for F in STEPS) + " |")
    L += ["", "## Anchor table: cost of the fixed-direction columns of Kolmogorov flow without / with anchors", "",
          "Amplitude 0.5 (`extrap_amp`), joint (`extrap_joint`) and k_f 6 (`extrap_kf`): evaluation at the end of training, pretrained decoder. Beta = 4 (`extrap_beta`): fine-tuned decoder. "
          "Model / in-range reference, same windows.", "",
          "| Column | Anchors | Seed (or Walrus) | Model / reference | step 4 | step 16 |", "|---|---|---|---|---|---|"]
    for r in out["anchor_cost"]:
        L.append(f"| {r['column'].split('/', 1)[1]} | {r['anchors']} | {r['seed']} | {r['model']} / {r['reference']} | "
                 + " | ".join(cell(r["values"][F], 2) for F in STEPS) + " |")
    if out["notes"]: L += ["", "## Notes", ""] + [f"- {n}" for n in out["notes"]]
    return "\n".join(L) + "\n"


if __name__ == "__main__":
    main()
