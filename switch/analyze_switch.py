"""Read-out of a switch evaluation (eval_switch.py with --dump_pred), CPU.

For every group and view (B: told the new condition; A: not told; N: no switch, condition A; T: as B with B's typical dt slot) from the
evaluation JSON: relative L2 at model steps 1, 4, 16 (32, 64 when evaluated). From the prediction dumps of views B, A and T, per window and
model step m, against the switched truth S and the control truth C (same history, condition A kept; trajectory 'c' + traj):
  effect    e = ||S - C|| / ||S||                       how much the switch changes the future
  response  r = <P - C, S - C> / ||S - C||^2           (0 = stays on the old condition's future, 1 = follows the switch)
  err_S, err_C = ||P - S|| / ||S||, ||P - C|| / ||S||
Window means and medians; r is reported only where the mean effect is >= --min_effect. Writes results/<name>.json."""
import argparse, json, os
import numpy as np, h5py

HERE = os.path.dirname(os.path.abspath(__file__))
STEPS = (1, 4, 16, 32, 64)


def nrm(x): return float(np.sqrt((x.astype(np.float64) ** 2).sum()))


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--name", required=True); ap.add_argument("--eval", required=True, help="the eval_switch.py result name (with --dump_pred)")
    ap.add_argument("--min_effect", type=float, default=0.05); ap.add_argument("--family", default="kolmo", choices=["kolmo"])
    a = ap.parse_args()
    out = os.path.join(HERE, "results", a.name + ".json"); os.makedirs(os.path.dirname(out), exist_ok=True)
    if os.path.exists(out): print("exists", out); return
    ev = json.load(open(os.path.join(HERE, "results", a.eval + ".json"))); E = ev.get("eval", ev)
    res = {"name": a.name, "eval": a.eval, "family": a.family, "run": (ev.get("args") or {}).get("eval_only"), "min_effect": a.min_effect, "groups": {}}
    groups = sorted({k.split("/")[1][len("switch_"):].rsplit("_", 1)[0] for k in E if k.startswith(f"{a.family}/switch_")})
    for g in groups:
        rg = {}
        for v in ("B", "A", "N", "T"):
            e = E.get(f"{a.family}/switch_{g}_{v}")
            if not e or "rel_l2_by_frame" not in e: continue
            bf = e["rel_l2_by_frame"]; rg[f"rel_l2_{v}"] = {m: bf[m - 1] for m in STEPS if m <= len(bf)}; rg[f"n_{v}"] = e.get("n_windows")
        for v in ("B", "A", "T"):
            f = os.path.join(HERE, "results", "pred", a.eval, f"{a.family}__switch_{g}_{v}.npz")
            if not os.path.exists(f): continue
            z = np.load(f, allow_pickle=False); meta = json.loads(str(z["meta"])); st = meta["stride"]; acc = {}
            for i, w in enumerate(meta["windows"]):
                P, S = z[f"P{i}"].astype(np.float32), z[f"T{i}"].astype(np.float32)   # [frames, C, H, W]
                t0 = w["first_frame"] + 6 * st; idx = [t0 + m * st for m in range(P.shape[0])]   # EVAL_HIST 6 history frames before the prediction start
                with h5py.File(w["path"], "r") as h:
                    C = np.moveaxis(np.asarray(h["c" + w["traj"]]["data"][idx[0]:idx[-1] + 1:st], dtype=np.float32), -1, 1)
                    S2 = np.moveaxis(np.asarray(h[w["traj"]]["data"][idx[0]:idx[-1] + 1:st], dtype=np.float32), -1, 1)
                assert np.abs(S2 - S).max() <= 1e-2 * max(np.abs(S).max(), 1e-6), f"{g}/{v} window {i}: dumped truth != file frames"   # fp16 both ways
                for m in STEPS:
                    if m > P.shape[0]: continue
                    p, s, c = P[m - 1], S[m - 1], C[m - 1]; ns, d = nrm(s), s - c; dd = float((d.astype(np.float64) ** 2).sum())
                    acc.setdefault(m, []).append((nrm(d) / ns, float(((p - c).astype(np.float64) * d).sum()) / dd if dd > 0 else np.nan, nrm(p - s) / ns, nrm(p - c) / ns))
            rv = {}
            for m, L in acc.items():
                L = np.array(L); eff = float(np.nanmean(L[:, 0]))
                rv[m] = {"effect": eff, "err_S": float(np.mean(L[:, 2])), "err_C": float(np.mean(L[:, 3])), "n": len(L),
                         "response_mean": float(np.nanmean(L[:, 1])) if eff >= a.min_effect else None, "response_median": float(np.nanmedian(L[:, 1])) if eff >= a.min_effect else None}
            rg[f"cross_{v}"] = rv
        if "rel_l2_B" in rg and "rel_l2_A" in rg: rg["told_over_untold"] = {m: rg["rel_l2_B"][m] / rg["rel_l2_A"][m] for m in rg["rel_l2_B"] if m in rg["rel_l2_A"]}   # paired noise (eval_switch)
        if "rel_l2_T" in rg and "rel_l2_B" in rg: rg["typicaldt_over_truedt"] = {m: rg["rel_l2_T"][m] / rg["rel_l2_B"][m] for m in rg["rel_l2_B"] if m in rg["rel_l2_T"]}
        res["groups"][g] = rg
        print(g, json.dumps({k: rg[k] for k in rg if k.startswith(("told", "cross_B"))}, default=float)[:400], flush=True)
    json.dump(res, open(out + ".tmp", "w"), indent=1, default=float); os.replace(out + ".tmp", out); print("saved", out)


if __name__ == "__main__":
    main()
