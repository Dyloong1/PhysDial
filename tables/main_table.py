#!/usr/bin/env python3
"""Main comparison table on the families Walrus did not see in pretraining (kolmo, nsu). CPU, standard library only; reads
finished result JSONs.

Column classes per family:
  test    in-range test conditions                     interp  held-out values inside the training range
  ext     same-regime extrapolation (low / high side)  other   further Kolmogorov columns (k_f, amplitude, drag, band, joint)
Family value = geometric mean over the test, interp and ext columns of the per-column ratio ours / other; "other" columns are reported
one by one and never enter it.

Statistic: the standard per-window relative L2 "rel_l2@F", F = 16 (main) and 4, averaged over the column's windows; the all-window value
is checked against the stored column value (relative tolerance 1e-6). Windows are split by trajectory into dev / test halves with
devsplit.py; every ratio is given for all windows and for each half. Small specialists keep no per-window records (aggregate only).
Verdict per column: ratio < 0.92 in both halves -> ours better, > 1.08 in both -> other better, else tie; family value: 0.95 / 1.05.

Inputs (result JSONs by run name):
  ours        physdial/results/physdial-s0-eval.json
  reference   extrapolation cost of ours = physdial-s0-eval-shift3 / physdial-ref-s0-eval-shift3 (model and its
              in-range reference trained on the held-out values, same windows, both with sampling shift 3; flagged "shift 3")
  Walrus      physdial/baselines/walrus/results/walrus-ft-eval.json
  DPOT-H      physdial/baselines/dpot/results/dpot-ft-eval.json
  U-Net / FNO physdial/baselines/results/unet-<fam>-s0.json, unet-<fam>-ref-s0.json (U-Net reference), fno-<fam>-s0.json
A missing file prints "pending", a file without the column "-".

Usage: python3 tables/main_table.py [--out DIR]  (default physdial/results/tables/: main_table.md, main_table.json)."""
import argparse
import json
import math
import os
import sys
from statistics import fmean

HERE = os.path.dirname(os.path.abspath(__file__)); PKG = os.path.join(os.path.dirname(HERE), "physdial")
sys.path.insert(0, HERE)
import devsplit  # noqa: E402

FAMS = {
    "kolmo": dict(test=["test"], interp=["interp_nu"], ext=["extrap_nu_low", "extrap_nu_high"],
                  other=["interp_kf", "nu_dense_interp", "multif_test", "extrap_kf", "extrap_amp", "extrap_joint", "nu_dense_ends",
                         "extrap_kband", "extrap_drag_lo", "extrap_drag_hi"]),
    "nsu": dict(test=["test"], interp=["interp_test"], ext=["extrap_ends"]),
}
CLASSES = ("test", "interp", "ext", "other")
R = os.path.join(PKG, "results")
FILES = {
    "ours": os.path.join(R, "physdial-s0-eval.json"),
    "pair_ours": os.path.join(R, "physdial-s0-eval-shift3.json"),
    "pair_ref": os.path.join(R, "physdial-ref-s0-eval-shift3.json"),
    "walrus": os.path.join(PKG, "baselines", "walrus", "results", "walrus-ft-eval.json"),
    "dpot": os.path.join(PKG, "baselines", "dpot", "results", "dpot-ft-eval.json"),
}


SPECIALIST = {"unet": "unet-{}-s0", "unet_ref": "unet-{}-ref-s0", "fno": "fno-{}-s0"}
ROWS = ("ours", "ref", "walrus", "dpot", "unet", "unet_ref", "fno")   # "ref" = ours' in-range reference
LABEL = {"ours": "Ours", "ref": "Ours reference", "walrus": "Walrus", "dpot": "DPOT-H", "unet": "U-Net",
         "unet_ref": "U-Net reference", "fno": "FNO"}
PENDING, ABSENT = "pending", "absent"
STATUS = {PENDING: "pending", ABSENT: "- (not covered)", "mismatch": "window mismatch", "ours_div": "ours diverged", "other_div": "other diverged",
          "both_div": "both diverged", "model_div": "diverged (model)", "ref_div": "diverged (reference)", "ours_missing": "missing in ours"}
CHECK = []   # all-window mean vs stored rel_l2@F mismatches (tolerance 1e-6 relative)


def load(path):
    """{column: column dict} of a result file (train_gen / specialist: key "eval"; walrus / dpot: top level, '_' keys are metadata, '@HxW' slices
    dropped), or None when the file does not exist yet."""
    if not os.path.exists(path): return None
    d = json.load(open(path)); e = d["eval"] if isinstance(d.get("eval"), dict) else d
    return {c: v for c, v in e.items() if "/" in c and "@" not in c and not c.startswith("_") and isinstance(v, dict)}


def stat(colv, F, maps, col):
    """{'all', 'dev', 'test': mean of per-window rel_l2@F (None if no halves), 'nonfinite': windows, 'keys': window keys or None}."""
    pw = colv.get("per_window")
    if not isinstance(pw, list):   # small experts: stored aggregate only
        x = colv.get(f"rel_l2@{F}")
        nf = colv.get("nonfinite_pred_windows") or 0
        return {"all": None if x is None else float(x), "dev": None, "test": None, "nonfinite": int(nf), "keys": None, "n": colv.get("n_windows"),
                "persist": (colv.get("persistence@1"), colv.get("persistence@16"))}
    vals = {"dev": [], "test": []}; nf = 0; keys = set()
    for w in pw:
        k = devsplit.window_key(w); keys.add(k); x = w.get(f"rel_l2@{F}")
        if x is None and isinstance(w.get("rel_l2_by_frame"), list) and len(w["rel_l2_by_frame"]) >= F: x = w["rel_l2_by_frame"][F - 1]
        if x is None or math.isnan(x) or math.isinf(x): nf += 1; x = math.inf
        h = maps[col].get(devsplit.unit_of(w["path"], w["traj"]), "x")   # "x": a trajectory ours does not have (the window check withholds the pair)
        vals.setdefault(h, []).append(float(x))
    al = vals["dev"] + vals["test"] + vals.get("x", [])
    m = lambda xs: (fmean(xs) if xs and all(math.isfinite(v) for v in xs) else (math.inf if xs else None))
    sv = colv.get(f"rel_l2@{F}"); mv = m(al)
    if sv is not None and mv is not None:
        if math.isfinite(sv) and math.isfinite(mv):
            if abs(mv - sv) > 1e-6 * max(abs(sv), 1e-12): CHECK.append(f"{col} F{F}: window mean {mv:.7g} vs stored rel_l2@{F} {sv:.7g}")
        elif math.isfinite(sv) != math.isfinite(mv):
            CHECK.append(f"{col} F{F}: window mean {mv} ({nf} non-finite windows) vs stored rel_l2@{F} {sv}")
    return {"all": m(al), "dev": m(vals["dev"]), "test": m(vals["test"]), "nonfinite": nf, "keys": keys, "n": len(al),
            "persist": (colv.get("persistence@1"), colv.get("persistence@16"))}


def same_windows(s, o):
    """window-set check against ours: exact keys when both have per-window records; otherwise (small experts) the model-independent
    persistence errors at steps 1 and 16 (last history frame as the prediction: depends on the windows only) and the window count must agree
    (relative 1e-5) - a witness, not a proof."""
    if s["keys"] is not None and o["keys"] is not None: return s["keys"] == o["keys"]
    if s.get("n") != o.get("n"): return False
    for x, y in zip(s.get("persist") or (), o.get("persist") or ()):
        if x is None or y is None or abs(x - y) > 1e-5 * max(abs(y), 1e-12): return False
    return bool(s.get("persist"))


def why(o, s):
    """status of a pair whose ratio cannot be a number: which side diverged."""
    if o["nonfinite"] and s["nonfinite"]: return "both_div"
    if o["nonfinite"]: return "ours_div"
    if s["nonfinite"]: return "other_div"
    return None


def ratio(a, b):
    """a / b, None when either is missing or not finite (a diverged side is reported by its own mark, never as a number)."""
    if a is None or b is None or not math.isfinite(a) or not math.isfinite(b) or b == 0: return None
    return a / b


def gmean(xs):
    xs = [x for x in xs if x is not None]
    return math.exp(fmean(math.log(x) for x in xs)) if xs else None


def verdict(dev, test, lo, hi):
    if dev is None or test is None: return "no halves"
    if dev < lo and test < lo: return "ours better"
    if dev > hi and test > hi: return "other better"
    return "tie"


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--out", default="", help="output directory (default physdial/results/tables)")
    a = ap.parse_args(); a.out = a.out or os.path.join(R, "tables"); os.makedirs(a.out, exist_ok=True)
    raw = {k: load(p) for k, p in FILES.items()}
    if raw["ours"] is None: sys.exit(f"ours missing: {FILES['ours']}")
    specialist = {(r, f): load(os.path.join(PKG, "baselines", "results", t.format(f) + ".json")) for r, t in SPECIALIST.items() for f in FAMS}
    # dev / test maps from ours' windows (identical for every model by the window check below)
    maps = devsplit.split_maps({c: v for c, v in raw["ours"].items() if isinstance(v.get("per_window"), list)})
    out = {"protocol": "rel_l2@F (standard), mean over windows; dev/test halves by devsplit.py; ratios ours / other", "families": {}}
    notes = []
    for fam, cls in FAMS.items():
        fo = {"columns": {}}
        for c in CLASSES:
            for sp in cls.get(c, []):
                col = f"{fam}/{sp}"; co = {"class": c, "F": {}}
                if col not in raw["ours"]: co["missing_ours"] = True; fo["columns"][col] = co; notes.append(f"{col}: not in ours' evaluation"); continue
                pair = ("pair_ours", "pair_ref"); co["ref_used"] = "shift3"   # extrapolation cost of ours from the model / reference pair
                for F in (16, 4):
                    st = {}
                    src = {"ours": raw["ours"], "walrus": raw["walrus"], "dpot": raw["dpot"], "ref": raw["pair_ref"],
                           "unet": specialist[("unet", fam)], "unet_ref": specialist[("unet_ref", fam)], "fno": specialist[("fno", fam)]}
                    for r, d in src.items():
                        if d is ABSENT: st[r] = ABSENT
                        elif d is None: st[r] = PENDING
                        elif col not in d: st[r] = ABSENT
                        else: st[r] = stat(d[col], F, maps, col)
                    if pair:   # the model of the reference pair (cost = pair_ours / pair_ref, same training)
                        p0 = raw[pair[0]]
                        st["pair_ours"] = stat(p0[col], F, maps, col) if p0 and col in p0 else PENDING if p0 is None else ABSENT
                    # window check against ours (models with per-window records)
                    mism = []
                    for r, s in st.items():
                        if r == "ours" or not isinstance(s, dict): continue
                        if not same_windows(s, st["ours"]):
                            mism.append(r); how = (f"{len(s['keys'] ^ st['ours']['keys'])} not shared" if s["keys"] is not None
                                                   else f"persistence / window count differ: {s.get('persist')} n={s.get('n')} vs ours {st['ours'].get('persist')} n={st['ours'].get('n')}")
                            notes.append(f"{col} F{F}: windows of {r} differ from ours ({how}); its ratios withheld")
                    fd = {"values": {r: ({k: v for k, v in s.items() if k != "keys"} if isinstance(s, dict) else s) for r, s in st.items()},
                          "window_mismatch": mism, "vs": {}, "cost": {}}
                    o = st["ours"]
                    for r in ("walrus", "dpot", "unet", "fno"):
                        s = st.get(r)
                        if not isinstance(s, dict): fd["vs"][r] = {"status": s if s in (PENDING, ABSENT) else ABSENT}; continue
                        if r in mism: fd["vs"][r] = {"status": "mismatch"}; continue
                        w_ = why(o, s)
                        if w_: fd["vs"][r] = {"status": w_}; continue
                        fd["vs"][r] = {h: ratio(o[h], s[h]) for h in ("all", "dev", "test")}; fd["vs"][r]["status"] = "ok"
                        fd["vs"][r]["verdict"] = verdict(fd["vs"][r]["dev"], fd["vs"][r]["test"], 0.92, 1.08)
                    if c != "test":   # extrapolation cost; in-range test columns have no held-out value
                        for m, rr in (("ours", "ref"), ("unet", "unet_ref")):
                            nk = "pair_ours" if (m == "ours" and pair) else m; num, den = st.get(nk), st.get(rr)
                            if pair and m == "ours" and isinstance(num, dict) and isinstance(den, dict) and num["keys"] != den["keys"]:
                                fd["cost"][m] = ABSENT; notes.append(f"{col} F{F}: model / reference windows differ"); continue
                            if isinstance(num, dict) and isinstance(den, dict) and nk not in mism and (rr not in mism or (pair and m == "ours")):
                                w_ = why(num, den)
                                fd["cost"][m] = {"status": {"ours_div": "model_div", "other_div": "ref_div"}.get(w_, w_)} if w_ else \
                                    {**{h: ratio(num[h], den[h]) for h in ("all", "dev", "test")}, "status": "ok"}
                            elif isinstance(num, dict) and isinstance(den, dict):
                                fd["cost"][m] = {"status": "mismatch"}
                            else:
                                fd["cost"][m] = PENDING if PENDING in (num, den) or num is None or den is None else ABSENT
                    co["F"][F] = fd
                fo["columns"][col] = co
        # family values: geomean of per-column ratios over test + interp + ext, per half; a diverged column stays out
        fv = {}
        for r in ("walrus", "dpot", "unet", "fno"):
            per = {}
            for col, co in fo["columns"].items():
                if co.get("class") not in ("test", "interp", "ext"): continue
                per[col] = {"status": "ours_missing"} if co.get("missing_ours") else co["F"][16]["vs"][r]
            sts = {p["status"] for p in per.values()}
            if sts <= {PENDING}: fv[r] = {"status": PENDING}; continue
            if sts <= {ABSENT}: fv[r] = {"status": ABSENT}; continue
            if sts <= {"mismatch"}: fv[r] = {"status": "mismatch"}; continue
            use = {col: p for col, p in per.items() if p["status"] == "ok" and p["all"] is not None}
            g = {h: gmean([p[h] for p in use.values()]) if use and all(p[h] is not None for p in use.values()) else None for h in ("all", "dev", "test")}
            g["excluded"] = {col: per[col]["status"] for col in sorted(set(per) - set(use))}; g["n_cols"] = len(use); g["status"] = "ok"
            g["verdict"] = "contains divergence, no verdict" if any(s_ in ("ours_div", "both_div") for s_ in g["excluded"].values()) else verdict(g["dev"], g["test"], 0.95, 1.05)
            fv[r] = g
        fo["family_value"] = fv
        out["families"][fam] = fo
    out["notes"] = notes + CHECK; out["stored_check_mismatches"] = len(CHECK)
    json.dump(out, open(os.path.join(a.out, "main_table.json"), "w"), indent=1, ensure_ascii=False, default=str)
    open(os.path.join(a.out, "main_table.md"), "w").write(render(out))
    print(render(out))


def f3(x):
    return "—" if x is None else ("∞" if not math.isfinite(x) else f"{x:.3f}")


def cell(s):
    if s == PENDING: return "pending"
    if s == ABSENT or s is None: return "-"
    if s.get("nonfinite"): return f"diverged ({s['nonfinite']}/{s.get('n')})"
    return f3(s["all"])


def rcell(d):
    if d == PENDING: return "pending"
    if d in (ABSENT, None): return "-"
    if d.get("status", "ok") != "ok": return STATUS[d["status"]]
    x = d.get("all"); hv = "" if d.get("dev") is None else f" ({f3(d['dev'])}/{f3(d['test'])})"
    return f3(x) + hv


def render(out):
    L = ["# Main table: families not seen in Walrus pretraining (generated by main_table.py)", "",
         "Statistic: standard relative L2 (rel_l2), mean over windows; ratio = ours / other, brackets = dev / test halves. "
         "Extrapolation cost = model / its in-range reference on the same windows (ours: model / reference pair, sampling shift 3).", "",
         "## Family value (step 16; geometric mean of the per-column ratios over test + interp + ext)", "",
         "| Family | Ours / Walrus | Ours / DPOT-H | Ours / U-Net | Ours / FNO |", "|---|---|---|---|---|"]
    for fam, fo in out["families"].items():
        row = [fam]
        for r in ("walrus", "dpot", "unet", "fno"):
            g = fo["family_value"].get(r)
            if g["status"] != "ok": row.append(STATUS[g["status"]]); continue
            ex = ", ".join(f"{c.split('/')[1]}({STATUS[s_]})" for c, s_ in g["excluded"].items())
            row.append(f"{f3(g['all'])} ({f3(g['dev'])}/{f3(g['test'])}) {g['verdict']}" + (f"; excludes {ex}" if ex else ""))
        L.append("| " + " | ".join(row) + " |")
    for F in (16, 4):
        L += ["", f"## Columns, step {F} (absolute error; cost = model / its own reference)", ""]
        hdr = ["Column", "Class", *[LABEL[r] for r in ROWS], "Cost ours", "Cost U-Net", "Ours / Walrus"]
        L += ["| " + " | ".join(hdr) + " |", "|" + "---|" * len(hdr)]
        for fam, fo in out["families"].items():
            for col, co in fo["columns"].items():
                if F not in co.get("F", {}): L.append(f"| {col} | {co['class']} | not in ours' evaluation |"); continue
                fd = co["F"][F]; v = fd["values"]
                c_ours = rcell(fd["cost"].get("ours")) if co["class"] != "test" else ""
                if c_ours and co.get("ref_used") == "shift3" and c_ours not in ("pending", "-"): c_ours += " †"
                vals = [cell(v.get(r)) for r in ROWS]
                if co.get("ref_used") == "shift3" and vals[1] not in ("pending", "-"): vals[1] += " †"
                row = [col, co["class"], *vals, c_ours, rcell(fd["cost"].get("unet")) if co["class"] != "test" else "",
                       rcell(fd["vs"].get("walrus")) + ((" " + fd["vs"]["walrus"]["verdict"]) if fd["vs"].get("walrus", {}).get("status") == "ok" else "")]
                L.append("| " + " | ".join(row) + " |")
    if out["notes"]: L += ["", "## Notes", ""] + [f"- {n}" for n in out["notes"]]
    L += ["", "- †: the cost of ours is the ratio of PhysDial (physdial-s0-eval-shift3) to its in-range reference "
          "(physdial-ref-s0-eval-shift3), both evaluated with sample shift 3.",
          "- Walrus, DPOT-H and FNO have no in-range reference here.",
          "- Small specialists keep no per-window records: all-window values only, no dev / test halves, window identity checked through "
          "the persistence errors and the window count."]
    return "\n".join(L) + "\n"


if __name__ == "__main__":
    main()
