#!/usr/bin/env python
"""Switch trajectories for the conditional Walrus: the Kolmogorov switch training split (switch_data.KolmoSwitchTrain switch_train, groups_kolmo_train.json)
in Walrus format with the condition fields of the new condition (wcond.py), appended to the conditional anchor export, so the conditional
Walrus can be continued on them as PhysDial is in train_switch.py.

Output under $PHYSDIAL_DATA_ROOT/switch/walrus/datasets/physdial_kolmo_anchor_sw_r<grid>/ for both Kolmogorov grids (256, 512):
  - every training file of physdial_kolmo_anchor_r<grid> (walrus_cond_data.py --kolmo_extra anchor) as a symlink; valid / test -> its
    directories; stats.yaml copied;
  - sw__<group>.hdf5: the group's switched trajectories, stored frames 0, 2, .., 2 (EVAL_HIST + NW - 1) (30 frames at stride 2, NW = 24
    windows of 6 history + 1 target frame each). Windows j <= 5 contain the switch and train only their post-switch positions
    (walrus_cond_ft.py masks the loss of the earlier positions). Constant fields = the new condition's forcing and slots with the true frame interval;
  - sw__<group>__rep<r>.hdf5, r = 1 .. REP - 1: symlinks, so that every switch trajectory has the sampling weight of one 300-frame training
    trajectory (REP = 144 / NW).
Every file of a directory carries the dataset_name physdial_kolmo_anchor_r<grid> (The Well expects one name per directory). Needs the
conditional export (cond_stats.json, physdial_kolmo_anchor_r*) and the switch files."""
import argparse, glob, json, os, shutil, sys, time

HERE = os.path.dirname(os.path.abspath(__file__)); PROJ = os.path.dirname(HERE); PKG = os.path.join(PROJ, "physdial")
sys.path.insert(0, HERE); sys.path.insert(0, PROJ); sys.path.insert(0, PKG)
import numpy as np, h5py, yaml   # noqa: E402
import corpus, switch_data, wcond   # noqa: E402

EVAL_HIST, NW = 6, 24
MAIN_FRAMES = 300   # stored frames of a Kolmogorov training trajectory (time lengths of the linked files are recorded in the result)


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--name", required=True)
    a = ap.parse_args()
    out = os.path.join(HERE, "results", a.name + ".json"); os.makedirs(os.path.dirname(out), exist_ok=True)
    if os.path.exists(out): print("exists", out); return
    t0 = time.time(); switch_data.install_train(["switch_train"])
    import walrus_data as wd   # imports train_gen / corpus (main data root)
    fam = corpus.get("kolmo"); assert isinstance(fam, switch_data.KolmoSwitchTrain), type(fam)
    root = os.path.join(os.environ["PHYSDIAL_DATA_ROOT"], "switch", "walrus")
    sp = os.path.join(root, "cond_stats.json"); assert os.path.exists(sp), f"no {sp}: run walrus_cond_data.py first (the encoding must be the trained one)"
    stats = wcond.load_or_make_stats(root, fam, [])   # existing file only (asserted above); its encoding is checked against wcond's constants
    s_ = fam.stride; assert s_ == 2, s_
    w_main = (MAIN_FRAMES - 1) // s_ + 1 - EVAL_HIST; REP = w_main // NW; assert REP * NW == w_main, (w_main, NW)   # 144 / 24 = 6
    its_all = fam.items("switch_train"); assert its_all, "no switch items"
    need = (EVAL_HIST + NW) * s_ - s_ + 1   # stored frames 0 .. 2 (EVAL_HIST + NW - 1)
    assert all(it.n_frames >= need for it in its_all), min(it.n_frames for it in its_all)
    with h5py.File(its_all[0].path, "r") as h: assert int(h.attrs["switch_frame"]) == (EVAL_HIST - 1) * s_, h.attrs["switch_frame"]   # the last history frame of window 0
    keys = ("nu", "k_f", "amp"); summ = {"rep": REP, "windows_per_traj": NW, "frames": EVAL_HIST + NW, "datasets": {}, "linked_time_lengths": {}}
    byfile = {}
    for it in its_all: byfile.setdefault(it.path, []).append(it)
    for g in (256, 512):
        src = os.path.join(root, "datasets", f"physdial_kolmo_anchor_r{g}"); ds = f"physdial_kolmo_anchor_sw_r{g}"; dd = os.path.join(root, "datasets", ds)
        hn = f"physdial_kolmo_anchor_r{g}"
        assert os.path.isdir(os.path.join(src, "data", "train")), src
        os.makedirs(os.path.join(dd, "data", "train"), exist_ok=True); want = set(); n_ln = n_sw = 0
        for f in sorted(os.listdir(os.path.join(src, "data", "train"))):   # the conditional anchor export, linked
            if not f.endswith(".hdf5"): continue
            p = os.path.join(dd, "data", "train", f); t = os.path.join(src, "data", "train", f); want.add(f)
            if not (os.path.islink(p) and os.readlink(p) == t):
                if os.path.lexists(p): os.remove(p)
                os.symlink(t, p)
            n_ln += 1
            with h5py.File(t, "r") as h_: L_ = int(h_["t1_fields"]["velocity"].shape[1])
            summ["linked_time_lengths"][str(L_)] = summ["linked_time_lengths"].get(str(L_), 0) + 1
        for path, its in sorted(byfile.items()):
            if tuple(its[0].shape) != (g, g): continue
            assert len(its) <= wd.PART and len({tuple(i.shape) for i in its}) == 1, path
            grp = os.path.basename(path)[:-3]; nm = f"sw__{grp}.hdf5"; p = os.path.join(dd, "data", "train", nm); want.add(nm)
            prov = json.dumps([[it.path, str(it.traj)] for it in its] + [[0, s_, EVAL_HIST + NW], [os.path.getmtime(path)]])
            same = False
            if os.path.exists(p) and not os.path.islink(p):
                with h5py.File(p, "r") as h_: same = h_.attrs.get("physdial_source") == prov and str(h_.attrs.get("dataset_name")) == hn
            if not same:
                if os.path.lexists(p): os.remove(p)
                X = np.stack([wd.load_ws(fam, it, 0, EVAL_HIST + NW) for it in its])   # [n, EVAL_HIST + NW, 2 (u, v), H, W]
                consts = [(c, np.stack([wcond.cond_fields(fam, it, stats, fam.cond_fields)[i] for it in its])) for i, c in enumerate(wcond.COND)]
                raw = {k: its[0].raw[k] for k in keys}   # B's values (metadata; every trajectory of a group has B's nu / amp, k_f of its own slot)
                wd.write_well(p, hn, [], X, raw, wd.dt_of(fam, its[0]), fam.periodic, None, ("velocity",), None, None, consts)
                with h5py.File(p, "r+") as h_: h_.attrs["physdial_source"] = prov
            for r in range(1, REP):
                q = f"sw__{grp}__rep{r:02d}.hdf5"; pq = os.path.join(dd, "data", "train", q); want.add(q)
                if not (os.path.islink(pq) and os.readlink(pq) == nm):
                    if os.path.lexists(pq): os.remove(pq)
                    os.symlink(nm, pq)
            n_sw += len(its)
        for old in sorted(os.listdir(os.path.join(dd, "data", "train"))):
            if old not in want: os.remove(os.path.join(dd, "data", "train", old)); print(f"[{a.name}] removed stale {ds}/train/{old}", flush=True)
        for d_ in ("valid", "test"):
            l = os.path.join(dd, "data", d_); t = os.path.join(src, "data", d_)
            if not (os.path.islink(l) and os.readlink(l) == t):
                if os.path.lexists(l): os.remove(l)
                os.symlink(t, l)
        shutil.copy(os.path.join(src, "stats.yaml"), os.path.join(dd, "stats.yaml"))
        m = yaml.safe_load(open(os.path.join(src, f"physdial_kolmo_anchor_r{g}.yaml"))); m["dataset_name"] = hn; m["n_files"] = len(want)
        yaml.safe_dump(m, open(os.path.join(dd, f"{ds}.yaml"), "w"))
        wd.check_scalars(dd)
        nms = set()
        for f_ in sorted(glob.glob(os.path.join(dd, "data", "train", "*.hdf5"))):
            with h5py.File(f_, "r") as h_: nms.add(str(h_.attrs["dataset_name"]))
        assert nms == {hn}, f"{dd}: dataset_name attributes {sorted(nms)} (The Well accepts one per directory)"
        summ["datasets"][ds] = {"linked_anchor_files": n_ln, "switch_trajectories": n_sw, "switch_files": len([w for w in want if w.startswith("sw__") and "__rep" not in w])}
        print(f"[{a.name}] {ds}: {n_ln} anchor-export files linked, {n_sw} switch trajectories x {REP} ({(time.time() - t0) / 60:.1f} min)", flush=True)
    import runmeta
    res = dict(summ, name=a.name, root=root, cond=list(wcond.COND), wall_min=(time.time() - t0) / 60, meta=runmeta.run_meta())
    json.dump(res, open(out + ".tmp", "w"), indent=1); os.replace(out + ".tmp", out); print("saved", out, flush=True)


if __name__ == "__main__":
    main()
