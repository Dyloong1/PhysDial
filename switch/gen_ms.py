"""Multi-switch data for Kolmogorov flow (groups_kolmo_ms.json).

Per group (K x seg_steps = 64 model steps), n trajectories: spin-up under C0 = seq[0] as gen_switch (same helpers), 11 stored frames (0..10)
under C0, then from frame 10 on the same state two continuations of K x 2 x seg_steps frames at the same frame interval:
  switched  ('0000'..): segment k (k = 0..K-1) runs under seq[k+1] for 2 x seg_steps stored frames
  control   ('c0000'..): C0 throughout
so the evaluation window (history = frames 0, 2, .., 10, stride 2) predicts model step m under segment (m - 1) // seg_steps.
Grid: every distinct condition of seq is tested on its own steady state (spectrum check and roll-off slope <= -1.5, doubled until it passes);
the trajectories run on the maximum of those grids and the group's N. Multi-mode conditions draw their own 4 modes per trajectory and segment.
File (one per group, $PHYSDIAL_DATA_ROOT/switch/<subdir>/<group>.h5): gen_switch's layout plus 'forcing_seg' [K, N, N] (every segment's forcing),
kf_slot_seg, modes_seg, and file attrs seq (JSON of the K + 1 conditions), seg_steps, ms 1. An interrupted run keeps finished groups."""
import argparse, json, math, os, sys, time
import numpy as np, torch, h5py

HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
import gen_switch as GI   # noqa: E402  (spin, solver, frame, forcing_field, kf_of, b_reference, STRIDE, SWITCH; it imports kolmo_solver read-only)
KSOL = GI.KSOL
import runmeta   # noqa: E402  (on the path through gen_switch)


import hashlib   # noqa: E402
CODE_MD5 = hashlib.md5(b"".join(open(f_, "rb").read() for f_ in (os.path.abspath(__file__), GI.__file__, KSOL.__file__))).hexdigest()   # generator code version


def same_cond(c, d): return json.dumps(c, sort_keys=True) == json.dumps(d, sort_keys=True)


def run_group(g, gi, a, out_dir, seg_steps):
    seq = g["seq"]; C0, segs = seq[0], seq[1:]; K = len(segs); n = a.n; alpha = KSOL.ALPHA; sf = GI.STRIDE * seg_steps   # stored frames per segment
    assert K >= 1 and all(not same_cond(seq[i], seq[i + 1]) or seq[i]["forcing"] == "multi" for i in range(K)), f"{g['name']}: a segment repeats its predecessor"
    seed = a.seed + 1000 * (1 + gi)   # unique per group (and different from gen_switch's evaluation / training seeds by --seed)
    gen = torch.Generator(device=KSOL.DEV); gen.manual_seed(seed); t0 = time.time()
    ph = torch.rand(n, device=KSOL.DEV, generator=gen) * 2 * math.pi
    modes = [KSOL.draw_modes(n, gen, 4, *c["kband"]) if c["forcing"] == "multi" else [{}] * n for c in seq]   # per condition (segment) and trajectory
    nr = min(8, n); grids, keref, teref = [], [], []
    for k, c in enumerate(seq):
        prev = next((j for j in range(k) if same_cond(seq[j], c) and c["forcing"] != "multi"), None)   # (gen_switch's B rule); repeated single-mode conditions once
        if prev is not None: grids.append(grids[prev]); keref.append(keref[prev]); teref.append(teref[prev]); continue
        NB = KSOL.est_N(c["nu"]); tried = []
        while True:
            keR, teR, ZR = GI.b_reference(c, NB, ph, modes[k], nr, seed + 101 * (k + 1), a, alpha); chk = KSOL.check_spectrum(ZR, NB); dsl = KSOL.rolloff(ZR, NB)
            tried.append({"N": NB, "ok": chk["ok"], "dslope": dsl})
            if (chk["ok"] and dsl <= -1.5) or NB >= 1024: break
            NB *= 2
        assert chk["ok"] and dsl <= -1.5, f"{g['name']} condition {k} of seq: not resolved up to N {NB}: {tried}"
        grids.append({"N": NB, "tries": tried}); keref.append(float(np.mean(keR))); teref.append(float(teR))
    N = max([x["N"] for x in grids] + [int(g.get("N") or 0)]); keref, teref = keref[1:], teref[1:]
    sA, wh, t_eddy = GI.spin(C0, N, n, gen, ph, modes[0], alpha, a.tspin, a.trelax); dtf = a.frame_eddy * t_eddy
    hist = [GI.frame(sA, wh)]
    for _ in range(GI.SWITCH): wh, _ = sA.advance(wh, dtf); hist.append(GI.frame(sA, wh))
    ws, wc = wh.clone(), wh.clone(); fS, fC, keS, keC, dif, spec_end = [], [], [], [], [], []
    for k, c in enumerate(segs):
        sk = GI.solver(c, N, n, ph, modes[k + 1], alpha)
        for f in range(sf):
            ws, _ = sk.advance(ws, dtf); wc, _ = sA.advance(wc, dtf); fS.append(GI.frame(sk, ws)); fC.append(GI.frame(sA, wc))
            if (len(fS)) % GI.STRIDE == 0:
                xs, xc = fS[-1].float(), fC[-1].float(); dif.append(((xs - xc).pow(2).sum((-2, -1)) / xs.pow(2).sum((-2, -1))).sqrt().numpy())
        keS.append(KSOL.ke(sk, ws)); keC.append(KSOL.ke(sA, wc))   # at the segment's end
        spec_end.append(KSOL.check_spectrum(KSOL.spectra(ws, N)[1], N))
    path = os.path.join(out_dir, f"{g['name']}.h5"); tmp = path + ".tmp"; T = GI.SWITCH + 1 + K * sf
    H = torch.stack(hist, 1).numpy(); XS = torch.stack(fS, 1).numpy(); XC = torch.stack(fC, 1).numpy()
    F = [GI.forcing_field(c, modes[i], N, ph) for i, c in enumerate(seq)]; KF = [GI.kf_of(c, modes[i]) for i, c in enumerate(seq)]
    x = np.arange(N) * 2 * np.pi / N; t = np.arange(T) * dtf
    mjs = lambda m: json.dumps({"k": np.asarray(m["k"]).tolist(), "w": np.asarray(m["w"]).tolist(), "phase": np.asarray(m["phase"]).tolist(), "kf_eff": float(m["kf_eff"])}) if m else "null"
    with h5py.File(tmp, "w") as h:
        for j in range(n):
            for key, X, sw in ((f"{j:04d}", XS, True), (f"c{j:04d}", XC, False)):
                gg = h.create_group(key); gg.create_dataset("data", data=np.concatenate([H[j], X[j]], 0)[..., None].astype(np.float16))
                gg.create_dataset("grid/t", data=t); gg.create_dataset("grid/x", data=x); gg.create_dataset("grid/y", data=x); gg.attrs["forcing_phase"] = float(ph[j])
                if sw:
                    gg.create_dataset("forcing", data=F[1][j]); gg.create_dataset("forcing_seg", data=np.stack([F[k + 1][j] for k in range(K)]).astype(np.float32))
                    gg.create_dataset("forcing_A", data=F[0][j]); gg.attrs["kf_slot"] = KF[1][j]; gg.attrs["kf_slot_seg"] = np.array([KF[k + 1][j] for k in range(K)])
                    gg.attrs["A_kf_slot"] = KF[0][j]; gg.attrs["modes_seg"] = json.dumps([json.loads(mjs(modes[k + 1][j])) for k in range(K)]); gg.attrs["modes_A"] = mjs(modes[0][j])
                    m_ = modes[1][j]; ma = modes[0][j]
                    if ma: gg.attrs["A_kf_eff"] = ma["kf_eff"]
                else:
                    gg.create_dataset("forcing", data=F[0][j]); gg.attrs["kf_slot"] = KF[0][j]; gg.attrs["modes"] = mjs(modes[0][j]); m_ = modes[0][j]
                if m_:
                    gg.attrs["modes_k"] = np.array(m_["k"]); gg.attrs["modes_w"] = np.array(m_["w"]); gg.attrs["modes_phase"] = np.array(m_["phase"]); gg.attrs["kf_eff"] = m_["kf_eff"]
        B1 = segs[0]
        for k_, v in (("nu", B1["nu"]), ("kf", float(B1.get("kf", np.mean(KF[1])))), ("amp", B1["amp"]), ("alpha", alpha), ("dt_frame", dtf), ("t_eddy", t_eddy), ("N", N),
                      ("B_dt_typical", dtf)): h.attrs[k_] = v
        for k_, v in C0.items(): h.attrs["A_" + k_] = np.array(v) if isinstance(v, list) else v
        for k_, v in B1.items(): h.attrs["B_" + k_] = np.array(v) if isinstance(v, list) else v
        h.attrs["A_kf"] = float(C0.get("kf", np.mean(KF[0]))); h.attrs["switch_frame"] = GI.SWITCH; h.attrs["seg_steps"] = seg_steps; h.attrs["seq"] = json.dumps(seq)
        h.attrs["group"] = g["name"]; h.attrs["cls"] = "in"; h.attrs["what"] = g["what"]; h.attrs["primary"] = g["primary"]; h.attrs["ms"] = 1
        h.attrs["spec_ok_seg_end"] = np.array([bool(c_["ok"]) for c_ in spec_end])
        h.attrs["group_entry"] = json.dumps(g); h.attrs["gen_code"] = CODE_MD5; h.attrs["gen_args"] = json.dumps({k_: getattr(a, k_) for k_ in ("n", "frame_eddy", "tspin", "trelax", "seed")})
        h.attrs["decay"] = 0; h.attrs["forcing"] = B1["forcing"]; h.attrs["seed"] = seed; h.attrs["kind"] = "multi-switch"
    os.replace(tmp, path)
    dif = np.array(dif); keS = np.array(keS); keC = np.array(keC)   # [K seg_steps, n], [K, n]
    return {"group": g["name"], "what": g["what"], "primary": g["primary"], "seq": seq, "N": N, "n": n, "seed": seed, "t_eddy_C0": t_eddy, "dt_frame": dtf,
            "frames": T, "file": path, "grids": grids, "spec_check_seg_end": spec_end, "spec_ok_all_seg_end": all(bool(c_["ok"]) for c_ in spec_end), "ke_ref_seg": keref, "t_eddy_ref_seg": teref, "n_ref": nr,
            "ke_switched_seg_end": keS[:, :nr].mean(1).tolist(), "ke_control_seg_end": keC[:, :nr].mean(1).tolist(),
            "ke_switched_over_ref_seg_end": [float(keS[k, :nr].mean() / keref[k]) for k in range(K)],
            "switch_effect_rel_l2": {int(m): float(dif[m - 1].mean()) for m in range(1, len(dif) + 1) if m in (1, 4, 8, 16) or m % seg_steps in (0, 1, 4)},
            "wall_s": time.time() - t0}


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--name", required=True); ap.add_argument("--groups", default=os.path.join(HERE, "groups_kolmo_ms.json"))
    ap.add_argument("--only", default="", help="comma list of group names (default all)"); ap.add_argument("--n", type=int, default=32)
    ap.add_argument("--frame_eddy", type=float, default=0.1); ap.add_argument("--tspin", type=int, default=60); ap.add_argument("--trelax", type=int, default=5)
    ap.add_argument("--seed", type=int, default=3, help="base seed (group i: seed + 1000 (i + 1)); differs from gen_switch's evaluation (0), training (1) and reference (2) seeds")
    ap.add_argument("--subdir", default="kolmo_ms", help="under $PHYSDIAL_DATA_ROOT/switch/")
    a = ap.parse_args()
    if not os.path.isabs(a.groups) and not os.path.exists(a.groups): a.groups = os.path.join(HERE, a.groups)
    out = os.path.join(HERE, "results", a.name + ".json"); os.makedirs(os.path.dirname(out), exist_ok=True)
    if os.path.exists(out): print("exists", out); return
    root = os.environ.get("PHYSDIAL_DATA_ROOT"); assert root, "PHYSDIAL_DATA_ROOT not set"
    od = os.path.join(root, "switch", a.subdir); os.makedirs(od, exist_ok=True)
    G = json.load(open(a.groups)); gs = G["groups"]; seg_steps = int(G["seg_steps"]); only = [x for x in a.only.split(",") if x]
    assert not (set(only) - {g["name"] for g in gs}), f"unknown groups {set(only) - {g['name'] for g in gs}}"
    res = {"name": a.name, "args": vars(a), "out_dir": od, "seg_steps": seg_steps, "groups": {}}
    prev = json.load(open(out + ".part"))["groups"] if os.path.exists(out + ".part") else {}
    for gi, g in enumerate(gs):
        if only and g["name"] not in only: continue
        p_, r_ = os.path.join(od, f"{g['name']}.h5"), prev.get(g["name"])
        if r_ and os.path.exists(p_):
            with h5py.File(p_, "r") as h_:
                same = int(h_.attrs["seed"]) == a.seed + 1000 * (1 + gi) and json.loads(h_.attrs["seq"]) == g["seq"] and sum(not k.startswith("c") for k in h_) == a.n \
                    and str(h_.attrs.get("gen_code", "")) == CODE_MD5 and json.loads(str(h_.attrs.get("gen_args", "{}"))) == {k_: getattr(a, k_) for k_ in ("n", "frame_eddy", "tspin", "trelax", "seed")} \
                    and json.loads(str(h_.attrs.get("group_entry", "null"))) == g
            if same: res["groups"][g["name"]] = r_; print(f"[{a.name}] {g['name']}: kept from the interrupted run", flush=True); continue
        ssg = int(g.get("seg_steps", seg_steps)); assert (len(g["seq"]) - 1) * ssg == 64, f"{g['name']}: {len(g['seq']) - 1} segments x {ssg} steps != 64"
        r = run_group(g, gi, a, od, ssg); res["groups"][g["name"]] = r
        print(f"[{a.name}] {g['name']}: N {r['N']} t_eddy {r['t_eddy_C0']:.3f} KE end/ref {np.round(r['ke_switched_over_ref_seg_end'], 2).tolist()} "
              f"effect {r['switch_effect_rel_l2']} ({r['wall_s']:.0f}s)", flush=True)
        json.dump(res, open(out + ".part.tmp", "w"), indent=1, default=float); os.replace(out + ".part.tmp", out + ".part")
    res["meta"] = runmeta.run_meta(); json.dump(res, open(out + ".tmp", "w"), indent=1, default=float); os.replace(out + ".tmp", out)
    if os.path.exists(out + ".part"): os.remove(out + ".part")
    print("saved", out, flush=True)


if __name__ == "__main__":
    main()
