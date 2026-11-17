#!/usr/bin/env python
"""Evaluate a Walrus run with the Walrus evaluator (physdial/baselines/walrus/walrus_eval.py, imported, unchanged) on a switch evaluation root:
  --root cond  : $PHYSDIAL_DATA_ROOT/switch/walrus/eval[_fN]        Kolmogorov windows with the condition fields (walrus_cond_data.py), including
                 the switch views kolmo__switch_<group>_<view>__HxW, plus links to the evaluation windows of the other families -> conditional runs;
  --root plain : $PHYSDIAL_DATA_ROOT/switch/plain/walrus/eval[_fN]  switch windows without condition fields -> history-only runs
                 (views B, A and T are then the same input, N has the control truth).
In this process only: walrus_eval.HERE = this directory (a run's result JSON is read from ./results, else linked from the Walrus results),
PHYSDIAL_DATA_ROOT = the chosen root while main() runs, and the predictions / truths of the switch views are dumped to
results/pred/<name>/kolmo__<split>.npz for analyze_switch.py / case_study.py. The views of one group get the same sampling seed."""
import json, os, re, sys

HERE = os.path.dirname(os.path.abspath(__file__)); PROJ = os.path.dirname(HERE); PKG = os.path.join(PROJ, "physdial"); WDIR = os.path.join(PKG, "baselines", "walrus")
sys.path.insert(0, HERE); sys.path.insert(0, PROJ); sys.path.insert(0, PKG); sys.path.insert(0, WDIR)
import numpy as np, torch   # noqa: E402
import walrus_eval as WE   # noqa: E402
import switch_corpus   # noqa: E402

VIEW = re.compile(r"^(switch_.+)_[BANT]$")


def main():
    argv = sys.argv[1:]; root_kind = "cond"
    if "--root" in argv: i = argv.index("--root"); root_kind = argv[i + 1]; del argv[i:i + 2]
    assert root_kind in ("cond", "plain"), root_kind
    NAME[0] = argv[argv.index("--name") + 1]
    run = argv[argv.index("--run") + 1] if "--run" in argv else None
    os.makedirs(os.path.join(HERE, "results"), exist_ok=True)
    if run and not os.path.exists(os.path.join(HERE, "results", run + ".json")):   # a walrus_ft.py run: its JSON is linked from the Walrus results
        src = os.path.join(WDIR, "results", run + ".json"); assert os.path.exists(src), f"no result JSON for run {run}"
        if int(os.environ.get("RANK", 0)) == 0 and not os.path.lexists(os.path.join(HERE, "results", run + ".json")): os.symlink(src, os.path.join(HERE, "results", run + ".json"))
    WE.HERE = HERE
    acc0 = WE.accumulate_dir

    tp0 = WE.to_phys; DUMP = {"on": False, "calls": []}

    def to_phys(*x, **k):   # accumulate_dir calls it as P, T, L2 per batch (walrus_eval.py: 'P = to_phys(yp ..); T = to_phys(yr ..); L2 = ..')
        r = tp0(*x, **k)
        if DUMP["on"]: DUMP["calls"].append(r.detach().cpu())
        return r

    WE.to_phys = to_phys

    def accumulate_dir(ed, fam, *x, **k):
        parts = os.path.basename(ed).split("__"); m = VIEW.match(parts[1]) if len(parts) == 3 else None
        if not m: return acc0(ed, fam, *x, **k)
        sd = WE_seed + int(switch_corpus.GROUP_SEEDS[m.group(1)[len(switch_corpus.PREFIX):]]["walrus_seed"]); torch.manual_seed(sd); np.random.seed(sd % 2 ** 32)
        DUMP["on"], DUMP["calls"] = True, []
        try: ac = acc0(ed, fam, *x, **k)
        finally: DUMP["on"] = False
        c = DUMP["calls"]; assert len(c) % 3 == 0, len(c)
        P = torch.cat(c[0::3]); T = torch.cat(c[1::3]); n = len(ac["recs"]); assert P.ndim == 5 and len(P) >= n, (P.shape, n)
        meta = {"stride": int(ac["stride"]), "windows": [{"path": r_["path"], "traj": str(r_["traj"]), "first_frame": int(r_["first_frame"])} for r_ in ac["recs"]]}
        od = os.path.join(HERE, "results", "pred", NAME[0]); os.makedirs(od, exist_ok=True)
        np.savez(os.path.join(od, f"kolmo__{parts[1]}.npz"), meta=json.dumps(meta), **{f"P{i}": P[i].half().numpy() for i in range(n)}, **{f"T{i}": T[i].half().numpy() for i in range(n)})
        return ac

    WE.accumulate_dir = accumulate_dir
    global WE_seed
    WE_seed = int(argv[argv.index("--seed") + 1]) if "--seed" in argv else 0
    real = os.environ["PHYSDIAL_DATA_ROOT"]; xr = os.path.join(real, "switch")
    os.environ["PHYSDIAL_DATA_ROOT"] = xr if root_kind == "cond" else os.path.join(xr, "plain")
    sys.argv = [os.path.join(WDIR, "walrus_eval.py")] + argv
    try: WE.main()
    finally: os.environ["PHYSDIAL_DATA_ROOT"] = real


WE_seed = 0; NAME = [None]
if __name__ == "__main__":
    main()
