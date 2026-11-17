"""Switch-trajectory data for Kolmogorov flow (single switches).

One batch of n trajectories per group (groups_kolmo*.json): spin-up under condition A as the corpus generator (data_gen/kolmo2d/kolmo_solver:
random IC, coarse spin-up on min(N, 128)^2 for --tspin time units, resize to N, --trelax units, t_eddy = 1 / mean omega_rms over 2 units,
frame interval dtf = --frame_eddy * t_eddy), then 11 stored frames (0..10) under A, then from frame 10 on the same state two continuations
of --after frames each at the same dtf:
  switched  (group '0000'..): condition B  -> frames 0..10 under A, 11.. under B
  control   (group 'c0000'..): condition A  -> frames 0..10 identical to the switched trajectory, 11.. under A
Frames are stored on the solver grid, fp16, [T, N, N, 1]; per trajectory 'forcing' = the vorticity forcing in effect after frame 10
([ix, iy] layout), 'forcing_A' on switched trajectories. File attrs: B's nu / kf / amp / alpha / dt_frame / N (what corpus.Kolmo reads) plus
A_* and the switch metadata. The grid is chosen so that B's steady state is resolved (spectrum check, doubled until it passes).
Also measured (results/<name>.json): kinetic energy of switched / control vs a reference batch run under B alone, the switched-vs-control
relative L2 per model step, the B reference's spectrum check. Writes under $PHYSDIAL_DATA_ROOT/switch/<subdir>/."""
import argparse, json, math, os, sys, time
import numpy as np, torch, h5py

HERE = os.path.dirname(os.path.abspath(__file__)); PROJ = os.path.dirname(HERE)
sys.path.insert(0, os.path.join(PROJ, "data_gen", "kolmo2d")); sys.path.insert(0, PROJ)
import kolmo_solver as KSOL   # noqa: E402  (read-only use: Solver, random_ic, resize, draw_modes, multi_field, ke, wrms, spectra, check_spectrum, est_N)
import runmeta              # noqa: E402

STRIDE = 2   # corpus.STRIDE["kolmo"]: one model step = 2 stored frames
SWITCH = 10  # last stored frame under A (= the last history frame of the evaluation window, EVAL_HIST 6 at stride 2 -> frames 0, 2, .., 10)


def forcing_field(c, modes, N, ph):
    """the vorticity forcing of condition c on the N grid, [B, N, N] float32 numpy, [ix, iy] layout (as the main stores)."""
    if c["forcing"] == "multi": return KSOL.multi_field(modes, N, c["amp"], device="cpu").numpy()
    y = np.arange(N) * 2 * np.pi / N
    return np.stack([-c["amp"] * c["kf"] * np.cos(c["kf"] * (y[None, :] + p)) * np.ones((N, 1)) for p in ph.cpu().numpy()]).astype(np.float32)


def solver(c, N, B, ph, modes, alpha):
    if c["forcing"] == "multi": return KSOL.Solver(N, c["nu"], B, alpha=alpha, amp=c["amp"], phases=ph, F2d=KSOL.multi_field(modes, N, c["amp"]))
    return KSOL.Solver(N, c["nu"], B, kf=c["kf"], alpha=alpha, amp=c["amp"], phases=ph)


def kf_of(c, modes):
    """the k_f slot value: single mode k_f; multi-mode the energy-weighted |k| per trajectory (kolmo_solver.draw_modes kf_eff)."""
    return [float(c["kf"])] * len(modes) if c["forcing"] != "multi" else [m["kf_eff"] for m in modes]


def spin(c, N, B, gen, ph, modes, alpha, tspin, trelax):
    """kolmo_solver.generate_nu's forced spin-up (stages A-C) under condition c -> (solver on N, state, t_eddy)."""
    Ns = min(N, 128); s = solver(c, Ns, B, ph, modes, alpha); wh = KSOL.random_ic(B, Ns, gen)
    for _ in range(int(tspin)): wh, _ = s.advance(wh, 1.0)
    if N != Ns:
        wh = KSOL.resize(wh, Ns, N); s = solver(c, N, B, ph, modes, alpha)
        for _ in range(int(trelax)): wh, _ = s.advance(wh, 1.0)
    wr = []
    for _ in range(4): wh, _ = s.advance(wh, 0.5); wr.append(KSOL.wrms(s, wh))
    return s, wh, 1.0 / float(np.mean(wr))


def frame(s, wh): return KSOL.to_grid(wh, s.N).half().cpu()   # [B, N, N] on the solver grid, exactly as the main stores (only the Nyquist row dropped)


def b_reference(Bc, N, ph, mB, nr, seed, a, alpha):
    """B's statistically steady state on grid N: a fresh batch spun up under B with the SAME forcing realisations / phases as the first
    nr switched trajectories, 40 frames at B's own frame interval -> (mean KE per trajectory [nr], t_eddy(B), time-mean enstrophy spectrum)."""
    gr = torch.Generator(device=KSOL.DEV); gr.manual_seed(seed + 7)
    sR, wR, teR = spin(Bc, N, nr, gr, ph[:nr], mB[:nr], alpha, a.tspin, a.trelax); keR, Zs = [], []
    for _ in range(40): wR, _ = sR.advance(wR, a.frame_eddy * teR); keR.append(KSOL.ke(sR, wR)); Zs.append(KSOL.spectra(wR, N)[1])
    return np.mean(keR, 0), teR, np.mean(Zs, 0)


def run_group(g, a, out_dir):
    A, Bc = g["A"], g["B"]; n = a.n; alpha = KSOL.ALPHA
    seed = a.seed + 1000 * (1 + [x["name"] for x in json.load(open(a.groups))["groups"]].index(g["name"]))   # unique per group
    gen = torch.Generator(device=KSOL.DEV); gen.manual_seed(seed); t0 = time.time()
    ph = torch.rand(n, device=KSOL.DEV, generator=gen) * 2 * math.pi
    mA = KSOL.draw_modes(n, gen, 4, *A["kband"]) if A["forcing"] == "multi" else [{}] * n
    mB = KSOL.draw_modes(n, gen, 4, *Bc["kband"]) if Bc["forcing"] == "multi" else [{}] * n
    nr = min(8, n); NB = KSOL.est_N(Bc["nu"]); tried = []
    while True:
        keR, teR, ZR = b_reference(Bc, NB, ph, mB, nr, seed, a, alpha); chk = KSOL.check_spectrum(ZR, NB); dsl = KSOL.rolloff(ZR, NB); tried.append({"N": NB, "ok": chk["ok"], "dslope": dsl})
        if (chk["ok"] and dsl <= -1.5) or NB >= 1024: break
        NB *= 2
    assert chk["ok"] and dsl <= -1.5, f"{g['name']}: B not resolved up to N {NB}: {tried}"
    N = max(NB, KSOL.est_N(A["nu"]), int(g.get("N") or 0))
    sA, wh, t_eddy = spin(A, N, n, gen, ph, mA, alpha, a.tspin, a.trelax); dtf = a.frame_eddy * t_eddy
    hist = [frame(sA, wh)]
    for _ in range(SWITCH): wh, _ = sA.advance(wh, dtf); hist.append(frame(sA, wh))
    sB = solver(Bc, N, n, ph, mB, alpha); wB, wA = wh.clone(), wh.clone()
    fB, fA, keB, keA, dif = [], [], [KSOL.ke(sA, wh)], [KSOL.ke(sA, wh)], []
    for f in range(a.after):
        wB, _ = sB.advance(wB, dtf); wA, _ = sA.advance(wA, dtf); fB.append(frame(sB, wB)); fA.append(frame(sA, wA))
        keB.append(KSOL.ke(sB, wB)); keA.append(KSOL.ke(sA, wA))
        if (f + 1) % STRIDE == 0:   # model step m = (f + 1) / STRIDE: switched vs control relative L2 per trajectory
            xb, xa = fB[-1].float(), fA[-1].float(); dif.append(((xb - xa).pow(2).sum((-2, -1)) / xb.pow(2).sum((-2, -1))).sqrt().numpy())
    _, ZkB = KSOL.spectra(wB, N); chk_last = KSOL.check_spectrum(ZkB, N)
    ke_ref_B = float(np.mean(keR)); ke_ref_A = float(np.mean(np.array(keA[-40:])[:, :nr]))   # both over the first nr trajectories (the reference's realisations)
    # write: switched '0000'.., control 'c0000'..
    path = os.path.join(out_dir, f"{g['name']}.h5"); tmp = path + ".tmp"; T = SWITCH + 1 + a.after
    H = torch.stack(hist, 1).numpy(); XB = torch.stack(fB, 1).numpy(); XA = torch.stack(fA, 1).numpy()
    FB, FA = forcing_field(Bc, mB, N, ph), forcing_field(A, mA, N, ph); kfB, kfA = kf_of(Bc, mB), kf_of(A, mA)
    x = np.arange(N) * 2 * np.pi / N; t = np.arange(T) * dtf
    with h5py.File(tmp, "w") as h:
        for j in range(n):
            for key, X, F, kf_, m_ in ((f"{j:04d}", XB, FB, kfB, mB), (f"c{j:04d}", XA, FA, kfA, mA)):
                gg = h.create_group(key); gg.create_dataset("data", data=np.concatenate([H[j], X[j]], 0)[..., None].astype(np.float16))
                gg.create_dataset("grid/t", data=t); gg.create_dataset("grid/x", data=x); gg.create_dataset("grid/y", data=x)
                gg.create_dataset("forcing", data=F[j]); gg.attrs["forcing_phase"] = float(ph[j]); gg.attrs["kf_slot"] = kf_[j]
                if m_[j]: gg.attrs["modes_k"] = np.array(m_[j]["k"]); gg.attrs["modes_w"] = np.array(m_[j]["w"]); gg.attrs["modes_phase"] = np.array(m_[j]["phase"]); gg.attrs["kf_eff"] = m_[j]["kf_eff"]
            gs = h[f"{j:04d}"]; gs.create_dataset("forcing_A", data=FA[j]); gs.attrs["A_kf_slot"] = kfA[j]
            if mA[j]: gs.attrs["A_kf_eff"] = mA[j]["kf_eff"]
        for k, v in (("nu", Bc["nu"]), ("kf", float(Bc.get("kf", np.mean(kfB)))), ("amp", Bc["amp"]), ("alpha", alpha), ("dt_frame", dtf), ("t_eddy", t_eddy), ("N", N),
                     ("B_dt_typical", a.frame_eddy * teR), ("B_t_eddy_ref", teR)): h.attrs[k] = v   # B_dt_typical: the frame interval B's own trajectories have in training
        for k, v in A.items(): h.attrs["A_" + k] = np.array(v) if isinstance(v, list) else v
        for k, v in Bc.items(): h.attrs["B_" + k] = np.array(v) if isinstance(v, list) else v
        h.attrs["A_kf"] = float(A.get("kf", np.mean(kfA))); h.attrs["switch_frame"] = SWITCH; h.attrs["group"] = g["name"]; h.attrs["cls"] = g["cls"]; h.attrs["what"] = g["what"]
        h.attrs["decay"] = 0; h.attrs["forcing"] = Bc["forcing"]; h.attrs["seed"] = seed; h.attrs["kind"] = "single switch"
    os.replace(tmp, path)
    keB, keA, dif = np.array(keB), np.array(keA), np.array(dif)   # [1 + after, n], [steps, n]
    steps = [m for m in (1, 4, 16, 32, 64) if m <= len(dif)]
    return {"group": g["name"], "cls": g["cls"], "what": g["what"], "A": A, "B": Bc, "N": N, "n": n, "seed": seed, "t_eddy_A": t_eddy, "dt_frame": dtf,
            "t_eddy_B_ref": teR, "B_dt_typical": a.frame_eddy * teR, "frames": T, "file": path, "grid_tries": tried, "N_B": NB, "spec_check_B_ref": chk, "rolloff_B_ref": dsl, "spec_check_B_last": chk_last,
            "ke_ref_A": ke_ref_A, "ke_ref_B": ke_ref_B, "n_ref": nr,
            "ke_ratio_switched_to_refB": {m: float(keB[STRIDE * m][:nr].mean() / ke_ref_B) for m in steps},
            "ke_ratio_control_to_refA": {m: float(keA[STRIDE * m][:nr].mean() / ke_ref_A) for m in steps},
            "switch_effect_rel_l2": {m: float(dif[m - 1].mean()) for m in steps},   # ||switched - control|| / ||switched|| at model step m
            "wall_s": time.time() - t0}


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--name", required=True); ap.add_argument("--groups", default=os.path.join(HERE, "groups_kolmo.json"))
    ap.add_argument("--only", default="", help="comma list of group names (default all)"); ap.add_argument("--n", type=int, default=32)
    ap.add_argument("--after", type=int, default=128, help="stored frames after the switch frame (64 model steps at stride 2)")
    ap.add_argument("--frame_eddy", type=float, default=0.1); ap.add_argument("--tspin", type=int, default=60); ap.add_argument("--trelax", type=int, default=5)
    ap.add_argument("--seed", type=int, default=0, help="base seed (group i: seed + 1000 (i + 1)); 0 evaluation, 1 training, 2 reference groups"); ap.add_argument("--subdir", default="kolmo", help="under $PHYSDIAL_DATA_ROOT/switch/")
    a = ap.parse_args()
    if not os.path.isabs(a.groups) and not os.path.exists(a.groups): a.groups = os.path.join(HERE, a.groups)   # a bare file name = a group file next to this script
    out = os.path.join(HERE, "results", a.name + ".json"); os.makedirs(os.path.dirname(out), exist_ok=True)
    if os.path.exists(out): print("exists", out); return
    root = os.environ.get("PHYSDIAL_DATA_ROOT"); assert root, "PHYSDIAL_DATA_ROOT not set"
    od = os.path.join(root, "switch", a.subdir); os.makedirs(od, exist_ok=True)
    gs = json.load(open(a.groups))["groups"]; only = [x for x in a.only.split(",") if x]
    assert not (set(only) - {g["name"] for g in gs}), f"unknown groups {set(only) - {g['name'] for g in gs}}"
    res = {"name": a.name, "args": vars(a), "out_dir": od, "groups": {}}
    for g in gs:
        if only and g["name"] not in only: continue
        r = run_group(g, a, od); res["groups"][g["name"]] = r
        print(f"[{a.name}] {g['name']}: N {r['N']} (tries {r['grid_tries']}) t_eddy A {r['t_eddy_A']:.3f} B {r['t_eddy_B_ref']:.3f} effect {r['switch_effect_rel_l2']} KE/refB {r['ke_ratio_switched_to_refB']} ({r['wall_s']:.0f}s)", flush=True)
        json.dump(res, open(out + ".part", "w"), indent=1, default=float)
    res["meta"] = runmeta.run_meta(); json.dump(res, open(out + ".tmp", "w"), indent=1, default=float); os.replace(out + ".tmp", out)
    if os.path.exists(out + ".part"): os.remove(out + ".part")
    print("saved", out, flush=True)


if __name__ == "__main__":
    main()
