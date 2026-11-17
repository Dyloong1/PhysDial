"""Read-out of a multi-switch evaluation (eval_ms.py or walrus_ms_eval.py), CPU.

Inputs: the statistics of the predictions, results/ms/<eval>/kolmo__switch_<group>_<view>.npz (per window: ke, ens, proj on C0's and every
segment's forcing, spectrum, finite; per model step), and the group files (gen_ms.py). The truth statistics are computed here with the same
code (ms_stats.stats) from the switched trajectories and the controls at the predicted frames, cached in results/ms/truth/.
Per group, on the means over the windows of a statistic X (P_S0: told every segment's condition; P_U: told C0 throughout, the model's own
no-switch rollout from the same history with the same sampling noise; S: switched truth; C: control truth), D = S - C:
  tracking error (difference in differences; the model's own drift cancels):
    TE_dd = rms_m((P_S0 - P_U) - D) / rms_m(D)          0 = follows every switch, 1 = no response (history-only models: 1 by definition)
    r_s   = mean_seg(P_S0 - P_U) / mean_seg(D)          response on the last half of segment s
  secondary: TE_v = rms_m(P_v - S) / rms_m(D) per view, r_s,v = mean_seg(P_v - C) / mean_seg(D), and r_inc = within-segment increment of
  the response / of D (last half minus the previous segment's last 4 steps)
r_s is reported where the effect |mean_seg D| / |mean_seg S| >= --min_effect and |mean_seg D| >= 3 bootstrap standard errors. Statistics:
ke, ens (enstrophy), proj_cur (projection of the flow on the forcing in effect), proj_prev (on the previous condition's forcing); 'primary'
is the group's designated statistic (file attribute). Uncertainty: paired bootstrap over windows, 5-95% intervals. --vs <eval2>: the same
window resamples for a second model, giving intervals of TE_dd(eval2) - TE_dd(this) and of the r_s differences.
Checks of a switch-trained run and its same-compute control (SW_CTL), each reported as holds / fails / undecided from the point estimate
and its 90 % interval:
  M1  (switch-trained run, without --vs)  TE_dd at most M1_THR of the group;
  M2  (switch-trained run, without --vs)  in the later segments M2_SEGS the within-segment response increment r_inc is at most 0.2 below
                                          segment 0's;
  M3  (--eval switch-trained, --vs its control; ms_amp)          TE_dd of the control exceeds the switch-trained run's by at least 0.15;
  M4  (--eval switch-trained, --vs its control; ms_kf, ms_action)  the two TE_dd differ by less than 0.15.
Writes results/<name>.json (with the ensemble-mean series for figures)."""
import argparse, glob, hashlib, json, os, sys
import numpy as np, h5py

HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
import ms_stats   # noqa: E402

STRIDE, SWITCH = 2, 10
STATS_MD5 = ms_stats.MD5
KEYS = ("ke", "ens", "proj_cur", "proj_prev")


def truth_stats(path, trajs, frames):
    """{'S': stats of the switched trajectories, 'C': of the controls} at the predicted frames, cached per group file (path) and frame count."""
    cd = os.path.join(HERE, "results", "ms", "truth"); os.makedirs(cd, exist_ok=True)
    cf = os.path.join(cd, f"{os.path.basename(path)[:-3]}_{hashlib.md5(os.path.realpath(path).encode()).hexdigest()[:8]}_f{frames}.npz")
    if os.path.exists(cf):
        z = np.load(cf); meta = json.loads(str(z["meta"]))
        if meta["trajs"] == list(trajs) and meta["mtime"] == os.path.getmtime(path) and meta.get("stats_md5") == STATS_MD5: return {w: {k: z[f"{w}_{k}"] for k in ("ke", "ens", "proj", "spec")} for w in "SC"}
    with h5py.File(path, "r") as h: K = len(json.loads(h.attrs["seq"])) - 1
    idx = [SWITCH + STRIDE * m for m in range(1, frames + 1)]; out = {"S": {}, "C": {}}
    for t in trajs:
        Fr = ms_stats.forcings(path, t, K)[None]
        with h5py.File(path, "r") as h:
            for w, key in (("S", t), ("C", "c" + t)):
                x = np.asarray(h[key]["data"][idx[0]:idx[-1] + 1:STRIDE, ..., 0], np.float32)[None]   # [1, frames, H, W]
                assert x.shape[1] == frames, (path, key, x.shape)
                st = ms_stats.stats(x, Fr)
                for k, v in st.items(): out[w].setdefault(k, []).append(v[0])
    out = {w: {k: np.stack(v) for k, v in d.items()} for w, d in out.items()}
    tmp = f"{cf}.{os.getpid()}.tmp.npz"
    np.savez(tmp, meta=json.dumps({"trajs": list(trajs), "mtime": os.path.getmtime(path), "stats_md5": STATS_MD5}), **{f"{w}_{k}": v for w, d in out.items() for k, v in d.items()})
    os.replace(tmp, cf)
    return out


def per_window(st, key, K, ss):
    """per-window series [n, frames] of statistic key."""
    if key in ("ke", "ens"): return st[key]
    seg = np.minimum(np.arange(st["proj"].shape[1]) // ss, K - 1)   # model step m (1-based) = index m - 1 -> segment (m - 1) // ss
    j = 1 + seg if key == "proj_cur" else seg
    return np.stack([st["proj"][:, i, j[i]] for i in range(len(seg))], 1)


def ref_series(st, key, K, ss, q):
    """per-window series [n, frames] of statistic key read on segment q's reference: proj_cur on segment q's forcing (index 1 + q),
    proj_prev on the previous condition's (index q); ke / ens unchanged. Both ends of segment q's increment must be read on the same pattern."""
    if key in ("ke", "ens"): return st[key]
    return st["proj"][:, :, (1 + q) if key == "proj_cur" else q]


def readings(Pw, Sw, Cw, idx, K, ss, history_only=False, Pr=None, Sr=None, Cr=None):
    """the readings of one statistic on the windows idx (a bootstrap resample or all): TE per view, TE_dd, and per segment D, S, effect, r, and the
    within-segment increments: inc_X = mean over the segment's last half - mean over the previous segment's last 4 steps (0 before
    the first segment: prediction and truths start from the same state), both read on segment q's reference (Pr / Sr / Cr [q]), r_inc =
    inc(resp) / inc(D). Pw / Sw / Cw: the level series (per_window)."""
    s, c = Sw[idx].mean(0), Cw[idx].mean(0); d = s - c; den = max(float(np.sqrt(np.mean(d ** 2))), 1e-30); p = {v: w[idx].mean(0) for v, w in Pw.items()}
    out = {"TE": {v: float(np.sqrt(np.mean((pv - s) ** 2)) / den) for v, pv in p.items()}, "seg": []}
    resp = p["S0"] - p["U"] if "S0" in p and "U" in p else None
    out["TE_dd"] = float(np.sqrt(np.mean((resp - d) ** 2)) / den) if resp is not None else (1.0 if history_only else None)   # history-only (plain root): no response by definition
    for q in range(K):
        sl = slice(q * ss + ss // 2, (q + 1) * ss); Dq, Sq = float(d[sl].mean()), float(s[sl].mean())
        e = {"D": Dq, "S": Sq, "effect": abs(Dq) / max(abs(Sq), 1e-30), "r_dd": float(resp[sl].mean()) / Dq if resp is not None and Dq != 0 else None}
        dq = Sr[q][idx].mean(0) - Cr[q][idx].mean(0); rq = Pr[q]["S0"][idx].mean(0) - Pr[q]["U"][idx].mean(0) if resp is not None else None
        pv_ = slice(q * ss - 4, q * ss) if q else None   # the previous segment's last 4 steps, on segment q's reference
        iD = float(dq[sl].mean()) - (float(dq[pv_].mean()) if q else 0.0); e["inc_D"] = iD; e["inc_effect"] = abs(iD) / max(abs(Sq), 1e-30)
        e["r_inc"] = (float(rq[sl].mean()) - (float(rq[pv_].mean()) if q else 0.0)) / iD if rq is not None and iD != 0 else None
        e.update({f"r_{v}": (float((pv[sl] - c[sl]).mean()) / Dq if Dq != 0 else None) for v, pv in p.items()}); out["seg"].append(e)
    return out


M1_THR = {"ms_kf": 0.5, "ms_action": 0.5, "ms_amp": 0.6}      # M1: TE_dd threshold of a switch-trained run
M2_SEGS = {"ms_kf": [2], "ms_action": [1, 2, 3], "ms_amp": [1]}   # M2 (0-based): the segments compared with segment 0
SW_CTL = {"physdial-switch-s0": "physdial-switch-control-s0", "study-switch-s0": "study-control-s0", "study-switch-s1": "study-control-s1",
          "walrus-cond-switch": "walrus-cond"}   # switch-trained -> same-compute control
run_of = lambda pr: str((pr or {}).get("run") or "").split("@")[0]   # walrus_ms_eval records '<run>@<ckpt>'


def verdict(x, c, thr, mode):
    """holds if the point estimate passes and the 90% interval does not cross the threshold, fails if both are on the other
    side, else undecided. mode 'le' (x <= thr), 'ge' (x >= thr), 'band' (|x| < thr)."""
    if x is None or c is None: return "undecided"
    if mode == "band": return "holds" if abs(x) < thr and -thr < c[0] and c[1] < thr else ("fails" if abs(x) >= thr and (c[0] >= thr or c[1] <= -thr) else "undecided")
    ok, lo, hi = (x <= thr, c[1] <= thr, c[0] > thr) if mode == "le" else (x >= thr, c[0] >= thr, c[1] < thr)
    return "holds" if ok and lo else ("fails" if not ok and hi else "undecided")


def m2_combine(segments, qs):
    """M2 for one group: over the listed later segments qs, those whose increment passed the gate together with segment 0's."""
    cnt = {q: verdict(segments[q]["r_inc_minus_seg0"], segments[q]["r_inc_minus_seg0_ci"], -0.2, "ge") for q in qs if segments[q].get("r_inc_minus_seg0") is not None}
    return ("undecided" if not cnt else ("fails" if "fails" in cnt.values() else ("holds" if all(v == "holds" for v in cnt.values()) else "undecided"))), cnt


COMBINE = (("M1", ("ms_kf", "ms_action", "ms_amp")), ("M2", ("ms_kf", "ms_action", "ms_amp")), ("M3", ("ms_amp",)), ("M4", ("ms_kf", "ms_action")))


def combine(group_verdicts, applies):
    """combination over the groups of one model: {group: {check: verdict}} -> ({check: verdict}, inputs)."""
    comb, inp = {}, {}
    for m_, grps in COMBINE:
        if m_ not in applies: continue
        vs_ = {g_: (group_verdicts.get(g_) or {}).get(m_) for g_ in grps}
        vs_ = {g_: v or "undecided" for g_, v in vs_.items()}; inp[m_] = vs_
        comb[m_] = "fails" if "fails" in vs_.values() else ("holds" if all(v == "holds" for v in vs_.values()) else "undecided")
    return comb, inp


def ci(xs):
    xs = np.array([x for x in xs if x is not None and np.isfinite(x)], float)
    return [float(np.percentile(xs, 5)), float(np.percentile(xs, 95))] if len(xs) else None


def load_eval(ev, ms_dir):
    files = sorted(glob.glob(os.path.join(HERE, "results", "ms", ev, "kolmo__switch_ms_*.npz"))); assert files, f"no statistics under results/ms/{ev}"
    metas = {f: json.loads(str(np.load(f)["meta"])) for f in files}
    proto = {json.dumps({k: m_.get(k) for k in ("frames", "run", "root", "ode_steps", "sample_shift", "tok", "stats_md5")}, sort_keys=True) for m_ in metas.values()}
    assert len(proto) == 1, f"results/ms/{ev}: statistics of different protocols {proto}"; proto = json.loads(proto.pop())
    assert proto["stats_md5"] == STATS_MD5, f"results/ms/{ev}: predictions computed with another ms_stats.py ({proto['stats_md5']} != {STATS_MD5}); re-evaluate"
    G = {}
    for f, meta in metas.items():
        grp, view = os.path.basename(f)[len("kolmo__switch_"):-4].rsplit("_", 1); path = os.path.join(ms_dir, grp + ".h5")
        assert os.path.realpath(meta["data"]) == os.path.realpath(path), f"{f}: statistics of {meta['data']}, not of {path} (--subdir?)"
        assert meta["data_mtime"] == os.path.getmtime(path), f"{f}: {path} changed after this evaluation (regenerated data); re-evaluate"
        z = np.load(f); G.setdefault(grp, {})[view] = (meta, {k: z[k] for k in ("ke", "ens", "proj", "spec", "finite")})
    return proto, G


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--name", required=True); ap.add_argument("--eval", required=True, help="the eval_ms / walrus_ms_eval result name")
    ap.add_argument("--vs", default="", help="a second evaluation on the same windows (TE_dd and r differences with paired intervals)")
    ap.add_argument("--subdir", default="kolmo_ms"); ap.add_argument("--min_effect", type=float, default=0.1); ap.add_argument("--boot", type=int, default=2000)
    ap.add_argument("--min_windows", type=int, default=16, help="finite windows a group needs for a verdict; readings are written regardless")
    a = ap.parse_args()
    out = os.path.join(HERE, "results", a.name + ".json")
    if os.path.exists(out): print("exists", out); return
    ms_dir = os.path.join(os.environ["PHYSDIAL_DATA_ROOT"], "switch", a.subdir)
    proto, G = load_eval(a.eval, ms_dir); proto2, G2 = load_eval(a.vs, ms_dir) if a.vs else (None, {})
    assert not a.vs or set(G2) == set(G), f"--vs covers groups {sorted(G2)}, this evaluation {sorted(G)}"
    want_views = ["U"] if proto.get("root") == "plain" else ["S0", "U"]; ho, ho2 = proto.get("root") == "plain", bool(proto2) and proto2.get("root") == "plain"
    assert not proto2 or all(proto2.get(k) == proto.get(k) for k in ("frames", "ode_steps", "sample_shift", "tok", "root")), \
        f"--vs: protocol {[(k, proto.get(k), proto2.get(k)) for k in ('frames', 'ode_steps', 'sample_shift', 'tok', 'root')]} differs"
    res = {"name": a.name, "eval": a.eval, "vs": a.vs or None, "protocol": proto, "protocol_vs": proto2, "min_effect": a.min_effect, "boot": a.boot, "min_windows": a.min_windows, "groups": {}}
    last = lambda pr: "@" not in str((pr or {}).get("run") or "") or str(pr["run"]).endswith("@last")
    sw_run = run_of(proto) in SW_CTL and last(proto); pair = sw_run and bool(a.vs) and run_of(proto2) == SW_CTL[run_of(proto)] and last(proto2)
    assert not sw_run or a.min_windows >= 16, f"--min_windows {a.min_windows}: verdicts of a switch-trained run need >= 16 windows"
    for grp, V in sorted(G.items()):
        path = os.path.join(ms_dir, grp + ".h5")
        with h5py.File(path, "r") as h:
            seq = json.loads(h.attrs["seq"]); K = len(seq) - 1; ss = int(h.attrs["seg_steps"]); primary = str(h.attrs["primary"]); spec_ok = [bool(x) for x in h.attrs.get("spec_ok_seg_end", [])]
        assert proto["frames"] == K * ss, f"{grp}: evaluated {proto['frames']} frames, the group has {K} x {ss}"
        not_c0 = [seq[q + 1]["forcing"] == "multi" or json.dumps(seq[q + 1], sort_keys=True) != json.dumps(seq[0], sort_keys=True) for q in range(K)]
        V2 = G2.get(grp, {}); allv = {**V, **{"vs_" + v: x for v, x in V2.items()}}
        assert not a.vs or V2, f"{grp}: --vs {a.vs} has no statistics for this group"
        tsets = {v: sorted(m_["trajs"]) for v, (m_, _) in allv.items()}; same = len({json.dumps(t) for t in tsets.values()}) == 1
        trajs = sorted(set.intersection(*[set(t) for t in tsets.values()]))
        fin = np.ones(len(trajs), bool); W = {}
        for v, (m_, P) in allv.items():
            pos = {t: i for i, t in enumerate(m_["trajs"])}; sel = [pos[t] for t in trajs]; W[v] = {k: P[k][sel] for k in P}; fin &= W[v]["finite"].astype(bool)
            for k_ in ("ke", "ens", "proj"): fin &= np.isfinite(W[v][k_]).reshape(len(trajs), -1).all(1)
        frames = next(iter(W.values()))["ke"].shape[1]
        rg = {"primary": primary, "K": K, "seg_steps": ss, "seq": seq, "not_C0": not_c0, "views": sorted(V), "views_vs": sorted(V2),
              "missing_views": [v for v in want_views if v not in V], "same_windows": same, "n_windows": len(trajs), "n_nonfinite": int((~fin).sum()),
              "spec_ok_seg_end": spec_ok, "eval_bs": {v: m_.get("eval_bs") for v, (m_, _) in allv.items()}, "readings": {}}
        rg["paired_noise"] = all(len({m_.get("eval_bs") for v_, (m_, _) in allv.items() if v_.startswith("vs_") == is_vs}) <= 1 for is_vs in (False, True))
        assert not rg["missing_views"] and not (V2 and [v for v in (["U"] if ho2 else ["S0", "U"]) if v not in V2]), f"{grp}: missing views {rg['missing_views']} / vs {sorted(V2)}"
        assert rg["paired_noise"], f"{grp}: the views were evaluated with different batch sizes (an OOM retry), so their sampling noise is not paired: re-evaluate"
        if not same: print(f"WARNING {grp}: the views hold different windows; the common {len(trajs)} are read", flush=True)
        undecided = {m_: "undecided" for m_, ok_ in (("M1", sw_run and not a.vs and grp in M1_THR), ("M2", sw_run and not a.vs and grp in M2_SEGS),
                                                ("M3", pair and grp == "ms_amp"), ("M4", pair and grp in ("ms_kf", "ms_action"))) if ok_}
        if (run_of(proto).startswith("physdial-stage1") or run_of(proto2).startswith("physdial-stage1")) and grp in ("ms_amp", "ms_mixed"):
            rg["note_stage1"] = "the stage-1 model never trained on amplitudes 0.7 / 1.4: this group's numbers are extrapolation without an in-range reference"
        rg["n_nonfinite_per_view"] = {v: int((~(W[v]["finite"].astype(bool) & np.isfinite(W[v]["ke"]).all(1) & np.isfinite(W[v]["ens"]).all(1)
                                                 & np.isfinite(W[v]["proj"]).reshape(len(trajs), -1).all(1))).sum()) for v in W}
        if not fin.any(): rg["all_nonfinite"] = True; rg["verdicts"] = undecided; res["groups"][grp] = rg; print(f"{grp}: every window non-finite", flush=True); continue
        few = int(fin.sum()) < a.min_windows
        if (~fin).sum() > len(fin) / 4: few = True; rg["too_many_dropped"] = int((~fin).sum())
        keep = np.where(fin)[0]; n = len(keep); T = truth_stats(path, [trajs[i] for i in keep], frames)
        rng = np.random.default_rng(0); boots = [rng.integers(0, n, n) for _ in range(a.boot)]; allidx = np.arange(n)
        for key in KEYS:
            Sw, Cw = per_window(T["S"], key, K, ss), per_window(T["C"], key, K, ss)
            Pw = {v: per_window(W[v], key, K, ss)[keep] for v in V}; Pw2 = {v: per_window(W["vs_" + v], key, K, ss)[keep] for v in V2}
            Sr, Cr = [ref_series(T["S"], key, K, ss, q) for q in range(K)], [ref_series(T["C"], key, K, ss, q) for q in range(K)]
            Pr = [{v: ref_series(W[v], key, K, ss, q)[keep] for v in V} for q in range(K)]; Pr2 = [{v: ref_series(W["vs_" + v], key, K, ss, q)[keep] for v in V2} for q in range(K)]
            pt = readings(Pw, Sw, Cw, allidx, K, ss, ho, Pr, Sr, Cr); bs = [readings(Pw, Sw, Cw, b, K, ss, ho, Pr, Sr, Cr) for b in boots]
            pt2 = readings(Pw2, Sw, Cw, allidx, K, ss, ho2, Pr2, Sr, Cr) if Pw2 else None; bs2 = [readings(Pw2, Sw, Cw, b, K, ss, ho2, Pr2, Sr, Cr) for b in boots] if Pw2 else None
            rk = {"TE_dd": pt["TE_dd"], "TE_dd_ci": ci([b["TE_dd"] for b in bs]), "TE": pt["TE"], "TE_ci": {v: ci([b["TE"][v] for b in bs]) for v in pt["TE"]}, "segments": []}
            for q in range(K):
                se = float(np.std([b["seg"][q]["D"] for b in bs])); e = pt["seg"][q]; ok = e["effect"] >= a.min_effect and abs(e["D"]) >= 3 * se
                sq = {"segment": q, "not_C0": not_c0[q], "D": e["D"], "D_se": se, "effect": e["effect"], "reported": bool(ok),
                      "r_dd": e["r_dd"] if ok else None, "r_dd_ci": ci([b["seg"][q]["r_dd"] for b in bs]) if ok else None}
                sq.update({k_: (e[k_] if ok else None) for k_ in e if k_.startswith("r_") and k_ not in ("r_dd", "r_inc")})
                sei = float(np.std([b["seg"][q]["inc_D"] for b in bs])); oki = e["inc_effect"] >= a.min_effect and abs(e["inc_D"]) >= 3 * sei
                sq.update({"inc_D": e["inc_D"], "inc_D_se": sei, "inc_effect": e["inc_effect"], "inc_reported": bool(oki), "r_inc": e["r_inc"] if oki else None,
                           "r_inc_ci": ci([b["seg"][q]["r_inc"] for b in bs]) if oki else None})
                if q and oki and rk["segments"] and rk["segments"][0].get("inc_reported") and e["r_inc"] is not None and pt["seg"][0]["r_inc"] is not None:
                    sq["r_inc_minus_seg0"] = e["r_inc"] - pt["seg"][0]["r_inc"]
                    sq["r_inc_minus_seg0_ci"] = ci([(b["seg"][q]["r_inc"] - b["seg"][0]["r_inc"]) if b["seg"][q]["r_inc"] is not None and b["seg"][0]["r_inc"] is not None else None for b in bs])
                if pt2 is not None and ok:
                    sq["vs_r_dd"] = pt2["seg"][q]["r_dd"]
                    sq["vs_minus_this_r_dd_ci"] = ci([(b2["seg"][q]["r_dd"] - b1["seg"][q]["r_dd"]) if b1["seg"][q]["r_dd"] is not None and b2["seg"][q]["r_dd"] is not None else None
                                                      for b1, b2 in zip(bs, bs2)])
                rk["segments"].append(sq)
            if pt2 is not None:
                both = pt2["TE_dd"] is not None and pt["TE_dd"] is not None
                rk["vs_TE_dd"] = pt2["TE_dd"]; rk["vs_minus_this_TE_dd"] = (pt2["TE_dd"] - pt["TE_dd"]) if both else None
                rk["vs_minus_this_TE_dd_ci"] = ci([(b2["TE_dd"] - b1["TE_dd"]) if b1["TE_dd"] is not None and b2["TE_dd"] is not None else None for b1, b2 in zip(bs, bs2)])
            rk["series"] = {"switched": Sw.mean(0).tolist(), "control": Cw.mean(0).tolist(), **{f"pred_{v}": w.mean(0).tolist() for v, w in Pw.items()}}
            rg["readings"][key] = rk
        pk = {"ke": "ke", "ens": "ens", "proj": "proj_cur"}[primary]; rg["primary_key"] = pk; R = rg["readings"][pk]
        if not sw_run and ("switch" in run_of(proto) and "control" not in run_of(proto)):
            print(f"WARNING {grp}: run {run_of(proto)!r} looks switch-trained but is not in SW_CTL (or not @last), so no verdicts are written", flush=True)
        if a.vs and sw_run and not pair:
            print(f"WARNING {grp}: --vs {run_of(proto2)!r} is not the same-compute control of {run_of(proto)!r} (@last): no M3 / M4", flush=True)
        if a.vs and (run_of(proto2) in SW_CTL or run_of(proto) in SW_CTL.values()):
            print(f"WARNING {grp}: --eval {run_of(proto)!r} / --vs {run_of(proto2)!r}: M3 / M4 need --eval = the switch-trained run and --vs = its control", flush=True)
        V_ = {"applies": "M1, M2: a switch-trained --eval without --vs; M3, M4: --eval switch-trained with --vs its own same-compute control (SW_CTL)"}
        if grp in M2_SEGS and sw_run and not a.vs and not few:
            for q in M2_SEGS[grp]: R["segments"][q]["m2_listed"] = True; R["segments"][q]["m2_counted"] = R["segments"][q].get("r_inc_minus_seg0") is not None
        if few:
            if int(fin.sum()) < a.min_windows: rg["too_few_finite_windows"] = int(fin.sum())
            V_.update(undecided)
        elif sw_run and not a.vs:
            if grp in M1_THR: V_["M1"] = verdict(R["TE_dd"], R["TE_dd_ci"], M1_THR[grp], "le")
            if grp in M2_SEGS: V_["M2"], V_["M2_segments"] = m2_combine(R["segments"], M2_SEGS[grp])
        if pair and not few and grp == "ms_amp": V_["M3"] = verdict(R.get("vs_minus_this_TE_dd"), R.get("vs_minus_this_TE_dd_ci"), 0.15, "ge")
        if pair and not few and grp in ("ms_kf", "ms_action"): V_["M4"] = verdict(R.get("vs_minus_this_TE_dd"), R.get("vs_minus_this_TE_dd_ci"), 0.15, "band")
        rg["verdicts"] = V_
        print(f"{grp}: primary {pk} TE_dd {None if R['TE_dd'] is None else round(R['TE_dd'], 3)} {R['TE_dd_ci']}; r_dd per segment "
              f"{[None if x['r_dd'] is None else round(x['r_dd'], 2) for x in R['segments']]}" + (f"; vs - this TE_dd {R.get('vs_minus_this_TE_dd')} {R.get('vs_minus_this_TE_dd_ci')}" if a.vs else ""), flush=True)
        rg["note_vs"] = "with --vs, windows non-finite in either model are dropped from both; read M1 and M2 from runs without --vs" if a.vs else None
        res["groups"][grp] = rg
    applies = ({"M1", "M2"} if sw_run and not a.vs else set()) | ({"M3", "M4"} if pair else set())
    res["verdicts"], res["verdict_inputs"] = combine({g_: r_.get("verdicts") for g_, r_ in res["groups"].items()}, applies)
    print("verdicts", res["verdicts"], res["verdict_inputs"], flush=True)
    json.dump(res, open(out + ".tmp", "w"), indent=1); os.replace(out + ".tmp", out); print("saved", out)


if __name__ == "__main__":
    main()
