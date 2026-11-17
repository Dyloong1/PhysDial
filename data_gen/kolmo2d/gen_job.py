#!/usr/bin/env python
"""Generate the Kolmogorov-flow and decaying-turbulence data with kolmo_solver.py (one GPU).

Sets (--set): train (training files of splits_kolmo.json), all (every (k_f, amp, nu) file of the sweep, incl. held-out columns),
dense (dense nu axis at k_f 2, amp 1), decay (freely decaying turbulence, NSU family), multif (random multi-mode forcing band),
anchor (amplitude anchors 0.7 / 1.4 at the training (k_f, nu)), drag (drag columns alpha 0.05 / 0.2),
beta (beta-plane files, amplitude 1, drag 0.1, --beta B; --cols anchor = the training (k_f, nu), training only; --cols column = the (k_f, nu)
of the amplitude column, last 4 trajectories evaluation, first 12 reference; beta < 0 is the mirror image x -> -x).
Files go to <data root>/<store>/<condition dir>/2D_kolmogorov_nu=<nu>.h5, data root = $PHYSDIAL_DATA_ROOT/2d, frames on the solver grid
every 0.1 eddy-turnover times, 300 per trajectory; seeds are fixed per set.
Usage: python data_gen/kolmo2d/gen_job.py --name <run> --set train --dst kolmo_train ..."""
import argparse, json, os, subprocess, sys, time
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, os.path.dirname(os.path.dirname(HERE)))
import runmeta
ap = argparse.ArgumentParser(); ap.add_argument("--name", required=True); ap.add_argument("--set", default="train", choices=["train", "all", "dense", "decay", "multif", "anchor", "drag", "beta"])
ap.add_argument("--ntraj", type=int, default=12); ap.add_argument("--seed_offset", type=int, default=777)
ap.add_argument("--dst", default=""); ap.add_argument("--link_from", default="kolmo_sweep"); ap.add_argument("--only", default="")
ap.add_argument("--frame_eddy", type=float, default=0.1, help="frame interval in eddy-turnover units 1/omega_rms")
ap.add_argument("--nframes", type=int, default=300, help="frames per trajectory")
ap.add_argument("--ic_wrms", type=float, default=5.0, help="--set decay: rms vorticity of the random initial condition")
ap.add_argument("--nu", type=float, default=0.0, help="--set multif: the nu value"); ap.add_argument("--kband", default="2,5", help="--set multif: kmin,kmax of the random forcing modes")
ap.add_argument("--nmodes", type=int, default=4, help="--set multif: modes per trajectory")
ap.add_argument("--N", type=int, default=0, help="force the solver grid")
ap.add_argument("--alpha", type=float, default=0.0, help="--set drag only: the linear drag of the files (0.05 or 0.2); every other set keeps the solver's 0.1")
ap.add_argument("--beta", type=float, default=0.0, help="--set beta only: the beta-plane value of the files")
ap.add_argument("--cols", default="anchor", choices=["anchor", "column"], help="--set beta: anchor = the training (k_f, nu); column = the amplitude column's (k_f, nu)")
a = ap.parse_args()
assert (a.set == "drag") == (a.alpha > 0), "--alpha goes with --set drag (and --set drag needs it); the other sets keep the solver's alpha = 0.1"
assert (a.set == "beta") == (a.beta != 0), "--beta goes with --set beta (and --set beta needs it nonzero)"
assert a.set not in ("anchor", "drag", "beta") or 0 <= a.seed_offset < 1000, "--set anchor / drag / beta: 0 <= --seed_offset < 1000 (seed = 100000 code + offset + 1000 nu index)"
os.makedirs(os.path.join(HERE, "results"), exist_ok=True); out = os.path.join(HERE, "results", a.name + ".json")
if os.path.exists(out): print("exists", out); sys.exit(0)
base = os.path.join(os.environ.get("PHYSDIAL_DATA_ROOT") or os.path.join(os.path.dirname(os.path.dirname(HERE)), "data"), "2d")   # data root of the Kolmogorov / NSU stores (corpus reads $PHYSDIAL_DATA_ROOT/2d)
dst_name = a.dst or {"all": "kolmo_sweep", "train": "kolmo_train", "dense": "kolmo_dense", "decay": "nsu", "multif": "kolmo_multif",
                     "anchor": "kolmo_anchor", "drag": "kolmo_drag", "beta": "kolmo_beta"}[a.set]; dst = os.path.join(base, dst_name)
os.makedirs(dst, exist_ok=True)
C = {(kf, amp): c for c, (kf, amp) in enumerate([(k, m) for k in (2, 3, 4, 6) for m in (0.5, 1.0)], 1)}   # seed codes of the main sweep
CA = {(2, 0.7): 61, (2, 1.4): 62, (4, 0.7): 63, (4, 1.4): 64}
CD = {(2, 0.05): 71, (2, 0.2): 72, (4, 0.05): 73, (4, 0.2): 74}


def CB(kf, beta):   # seed code of the beta-plane files: 81 + 4 round(20 |beta|) + 2 (beta < 0) + (k_f == 4), unique for beta on a 0.05 grid
    assert beta != 0 and abs(20 * beta - round(20 * beta)) < 1e-9 and kf in (2, 4), (kf, beta); return 81 + 4 * int(round(20 * abs(beta))) + 2 * (beta < 0) + (kf == 4)
if a.set == "multif":
    assert a.nu > 0, "--set multif needs --nu"; kb = a.kband.replace(",", "-"); od = f"kband{kb}"; code = 11 + sum(int(x) for x in a.kband.split(","))
    cmd = [sys.executable, os.path.join(HERE, "kolmo_solver.py"), "--nu", str(a.nu), "--only_idx", "0", "--forcing", "multi", "--kband", a.kband, "--nmodes", str(a.nmodes),
           "--kf", "2", "--amp", "1.0", "--out_dir", od, "--seed", str(100000 * code + int(round(-1000 * __import__("math").log10(a.nu))) + a.seed_offset), "--h5_root", dst,
           "--ntraj", str(a.ntraj), "--frame_eddy", str(a.frame_eddy), "--nframes", str(a.nframes)] + (["--N", str(a.N)] if a.N else [])
    print("[gen] " + " ".join(cmd[1:]), flush=True); t0 = time.time(); r = subprocess.run(cmd); assert r.returncode == 0, f"solver failed rc={r.returncode}"
    res = {"name": a.name, "set": a.set, "dst": dst, "out_dir": od, "nu": a.nu, "kband": a.kband, "nmodes": a.nmodes, "ntraj": a.ntraj, "frame_eddy": a.frame_eddy, "nframes": a.nframes,
           "N": a.N or "est_N", "wall_s": time.time() - t0, "meta": runmeta.run_meta()}
    json.dump(res, open(out, "w"), indent=1); print("saved", out, flush=True); sys.exit(0)
if a.set in ("dense", "decay"):
    n_nu, od, code = (100, "dense_kf2_amp1.0", 9) if a.set == "dense" else (101, "decay", 10)
    i0, i1 = (int(x) for x in a.only.split(",")) if a.only else (0, n_nu)
    cmd = [sys.executable, os.path.join(HERE, "kolmo_solver.py"), "--nu_list", str(n_nu), "--lo", "-4", "--hi", "-2", "--only_idx", *map(str, range(i0, i1)),
           "--kf", "2", "--amp", "1.0" if a.set == "dense" else "0.0", "--out_dir", od, "--seed", str(100000 * code + a.seed_offset), "--h5_root", dst,
           "--ntraj", str(a.ntraj), "--frame_eddy", str(a.frame_eddy), "--nframes", str(a.nframes)] + (["--decay", "--ic_wrms", str(a.ic_wrms)] if a.set == "decay" else []) + (["--N", str(a.N)] if a.N else [])
    print("[gen] " + " ".join(cmd[1:]), flush=True); t0 = time.time(); r = subprocess.run(cmd); assert r.returncode == 0, f"solver failed rc={r.returncode}"
    res = {"name": a.name, "set": a.set, "dst": dst, "out_dir": od, "nu_index_range": [i0, i1], "ntraj": a.ntraj, "frame_eddy": a.frame_eddy, "nframes": a.nframes, "N": a.N or "est_N",
           "ic_wrms": a.ic_wrms if a.set == "decay" else None, "wall_s": time.time() - t0, "meta": runmeta.run_meta()}
    json.dump(res, open(out, "w"), indent=1); print("saved", out, flush=True); sys.exit(0)
if a.set == "all": jobs = [(kf, amp, i) for (kf, amp) in C for i in range(12)]
elif a.set in ("anchor", "drag", "beta"):
    S2 = json.load(open(os.path.join(HERE, "splits_kolmo.json")))
    if a.set == "anchor": jobs = [(kf, amp, int(i)) for (kf, amp) in CA for i in S2["train"]["nu_idx"]]
    elif a.set == "beta":   # anchors at the training (k_f, nu), the column at the amplitude column's (k_f, nu); amplitude 1
        src = S2["train"] if a.cols == "anchor" else S2["extrap_amp"]; jobs = [(int(kf), 1.0, int(i)) for kf, _ in src["forcing_kf_amp"] for i in src["nu_idx"]]
    else:
        assert all((int(kf), a.alpha) in CD for kf, _ in S2["extrap_amp"]["forcing_kf_amp"]), f"--alpha {a.alpha}: no seed code in CD"
        jobs = [(int(kf), 1.0, int(i)) for kf, _ in S2["extrap_amp"]["forcing_kf_amp"] for i in S2["extrap_amp"]["nu_idx"]]
else:
    S = json.load(open(os.path.join(HERE, "splits_kolmo.json")))["train"]; jobs = [(int(kf), float(amp), int(i)) for kf, amp in S["forcing_kf_amp"] for i in S["nu_idx"]]
if a.only:
    p = a.only.split(","); jobs = [j for j in jobs if j[0] == int(p[0]) and j[1] == float(p[1]) and (len(p) < 3 or j[2] == int(p[2]))]
assert jobs, "nothing to generate"
t0 = time.time(); done = []
for kf, amp, i in jobs:
    code = CA[(kf, amp)] if a.set == "anchor" else CD[(kf, a.alpha)] if a.set == "drag" else CB(kf, a.beta) if a.set == "beta" else C[(kf, amp)]
    seed = 100000 * code + a.seed_offset   # the solver adds 1000 * nu index
    od = f"kf{kf}_amp{amp}" + (f"_alpha{a.alpha}" if a.set == "drag" else "") + (f"_beta{a.beta:g}" if a.set == "beta" else "")
    cmd = [sys.executable, os.path.join(HERE, "kolmo_solver.py"), "--nu_list", "12", "--lo", "-4", "--hi", "-2", "--only_idx", str(i), "--kf", str(kf), "--amp", str(amp),
           "--out_dir", od, "--seed", str(seed), "--h5_root", dst, "--ntraj", str(a.ntraj),
           "--frame_eddy", str(a.frame_eddy), "--nframes", str(a.nframes)] + (["--N", str(a.N)] if a.N else []) + (["--alpha", str(a.alpha)] if a.set == "drag" else []) + (["--beta", repr(a.beta)] if a.set == "beta" else [])
    print("[gen] " + " ".join(cmd[1:]), flush=True); r = subprocess.run(cmd); assert r.returncode == 0, f"solver failed rc={r.returncode}"
    done.append({"kf": kf, "amp": amp, "nu_idx": i, "seed": seed, "file_seed": seed + 1000 * i, "out_dir": od, **({"alpha": a.alpha} if a.set == "drag" else {}), **({"beta": a.beta, "cols": a.cols} if a.set == "beta" else {})})
linked = 0
if a.set == "train":   # every other sweep file of splits_kolmo.json -> the same-named file of the --link_from store (dangling until that store is generated)
    S_all = json.load(open(os.path.join(HERE, "splits_kolmo.json")))
    ref = sorted({os.path.join(*f.split("/")[1:]) for v in S_all.values() if isinstance(v, dict) and "files" in v for f in v["files"]})   # 'h5/kf2_amp1.0/<file>' -> 'kf2_amp1.0/<file>'
    for rel in ref:
        d = os.path.join(dst, rel)
        if os.path.lexists(d): continue
        os.makedirs(os.path.dirname(d), exist_ok=True)
        try: os.symlink(os.path.join(base, a.link_from, rel), d); linked += 1
        except FileExistsError: pass
        except PermissionError: print("[gen] WARNING: no symlinks on this filesystem (exFAT?); evaluation files not linked", flush=True); break
res = {"name": a.name, "set": a.set, "dst": dst, "ntraj": a.ntraj, "frame_eddy": a.frame_eddy, "nframes": a.nframes, "generated": done, "eval_files_linked": linked,
       "alpha": a.alpha if a.set == "drag" else 0.1, "seed_offset": a.seed_offset, "wall_s": time.time() - t0, "mem": runmeta.mem_peak(), "meta": runmeta.run_meta()}
print("[gen] memory " + json.dumps(res["mem"]), flush=True)
json.dump(res, open(out, "w"), indent=1); print("saved", out, flush=True)
