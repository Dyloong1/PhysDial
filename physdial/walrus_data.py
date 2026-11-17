#!/usr/bin/env python
"""Data for the Walrus and DPOT-H baselines (CPU): writes the own families in The Well HDF5 format and the evaluation windows of all
families, so the baselines are fine-tuned on the same training data and scored on exactly the same frames as PhysDial
(train_gen.eval_windows: EVAL_HIST = 6 history frames, which is also Walrus's input length, then the targets). Output under
$PHYSDIAL_DATA_ROOT/walrus/:
  datasets/physdial_<family>[_anchor][_beta][_ref][_r<N>]/   own families (kolmo, nsu; _anchor / _beta: with the amplitude / beta anchors,
                           _ref: with the ref_* splits, see --kolmo_extra, --reference, --no_anchors), one dataset per grid size
                           (Kolmogorov / NSU: _r256, _r512), with
                           data/{train,valid,test}/<condition>.hdf5, <name>.yaml, stats.yaml; train/valid/test = the corpus train/val/test
                           splits (+ the training-only splits above; the evaluation trajectories of held-out conditions are never
                           written); strided or corpus-split copies of The Well families where the
                           corpus uses a stride > 1 or a subset of the published trajectories
  eval/<family>__<split>__<H>x<W>/  one dataset per evaluation split and grid: data/test/win_<i>.hdf5 (one window each; i = index in
                           train_gen.eval_windows), train/valid -> symlinks to test, stats.yaml of the family's training data (Walrus
                           normalises with them), windows.json (source trajectory, first frame); eval_f<frames>/ for --frames != 16
The other The Well families are read by Walrus from their own directories ($PHYSDIAL_EXT_DATA/the_well), held-out conditions excluded by file
filters (see baselines/walrus/cfg/data/physdial_corpus.yaml). No forcing field is given: the baselines are history-only (Walrus has no
condition input); time-constant fields of The Well (acoustic density / speed of sound) are kept as in the source."""
import argparse, glob, json, os, shutil, sys, time
HERE = os.path.dirname(os.path.abspath(__file__)); PROJ = os.path.dirname(HERE); sys.path.insert(0, PROJ); sys.path.insert(0, HERE)
import numpy as np, h5py, yaml
import runmeta, corpus
from train_gen import eval_windows, EVAL_HIST, EVAL_SEED

OWN = {"kolmo": "physdial_kolmo", "nsu": "physdial_nsu"}
PART = 16   # trajectories per written Well file (bounds the memory of one write)
CONSTS = {"acoustic": ("density", "speed_of_sound")}   # time-constant t0 fields of a Well family (its condition fields), written into our copies as in the source
VORT = {"kolmo", "nsu"}   # vorticity families: written as velocity (Walrus's own field, velocity_x / velocity_y embeddings), exact on the
                                 # periodic box; Walrus's velocity predictions are turned back into vorticity for scoring (walrus_eval.py)


def vort_to_vel(w):
    """w [..., N, M] (periodic [0, 2pi)^2, zero-mean velocity) -> u, v with w = dv/dx - du/dy (x = first grid axis, as stored)."""
    N, M = w.shape[-2:]; kx = np.fft.fftfreq(N, 1.0 / N)[:, None]; ky = np.fft.fftfreq(M, 1.0 / M)[None, :]
    wh = np.fft.fft2(w); k2 = kx ** 2 + ky ** 2; k2[0, 0] = 1.0; psi = wh / k2; psi[..., 0, 0] = 0   # -lap psi = w
    u = np.real(np.fft.ifft2(1j * ky * psi)); v = np.real(np.fft.ifft2(-1j * kx * psi))           # u = dpsi/dy, v = -dpsi/dx
    return u.astype(np.float32), v.astype(np.float32)


def vel_to_vort(u, v):
    N, M = u.shape[-2:]; kx = np.fft.fftfreq(N, 1.0 / N)[:, None]; ky = np.fft.fftfreq(M, 1.0 / M)[None, :]
    return np.real(np.fft.ifft2(1j * kx * np.fft.fft2(v) - 1j * ky * np.fft.fft2(u))).astype(np.float32)


def load_w(fam, it, t0, t1):
    """frames for Walrus on the stored grid: vorticity families -> [T, 2 (u, v), H, W]; others as stored."""
    x = fam.load(it, t0, t1)
    if fam.key in VORT: u, v = vort_to_vel(x[:, 0]); x = np.stack([u, v], 1)
    return x


def load_ws(fam, it, t0, n):
    """n frames at the family's stride (corpus.STRIDE) starting at stored frame t0."""
    return load_w(fam, it, t0, t0 + (n - 1) * fam.stride + 1)[::fam.stride]


def dt_of(fam, it):
    return 10 ** it.params["log10_dt"] * fam.stride


def write_well(path, name, fields, X, raw, dt, periodic, forcing=None, vectors=(), bcs=None, bc_src=None, consts=()):
    """X [n_traj, T, C, H, W] -> one Well file. Channels = the scalar fields `fields` (t0) then the x, y components of each vector
    field in `vectors` (t1), the order of corpus.Well.load. bcs: per-axis boundary types (default PERIODIC / WALL from `periodic`)."""
    n, T, C, H, W = X.shape; fn = fields; bcs = bcs or ["PERIODIC" if p else "WALL" for p in periodic]
    raw = {k: v for k, v in raw.items() if k not in corpus.RAW_FLAGS}
    with h5py.File(path + ".tmp", "w") as h:
        h.attrs.update({"dataset_name": name, "grid_type": "cartesian", "n_spatial_dims": 2, "n_trajectories": n,
                        "simulation_parameters": np.array(list(raw), dtype=h5py.string_dtype()), **{k: float(v) for k, v in raw.items()}})
        if bc_src:
            with h5py.File(bc_src, "r") as hs: hs.copy(hs["boundary_conditions"], h, "boundary_conditions")
        bc = h["boundary_conditions"] if bc_src else h.create_group("boundary_conditions")
        for ax, N, t in ((() if bc_src else (("x", H, bcs[0]), ("y", W, bcs[1])))):
            g = bc.create_group(f"{ax}_{t.lower()}"); m = np.zeros(N, bool); m[[0, -1]] = True; g.create_dataset("mask", data=m)
            g.attrs.update({"associated_dims": np.array([ax], dtype=h5py.string_dtype()), "associated_fields": np.array([], dtype=h5py.string_dtype()), "bc_type": t,
                            "sample_varying": False, "time_varying": False})
        d = h.create_group("dimensions"); d.attrs["spatial_dims"] = np.array(["x", "y"], dtype=h5py.string_dtype())
        for k, v, tv in (("time", np.arange(T) * dt, True), ("x", np.arange(H) / H, False), ("y", np.arange(W) / W, False)):
            ds = d.create_dataset(k, data=v.astype(np.float32)); ds.attrs.update({"sample_varying": False, "time_varying": tv})
        s = h.create_group("scalars"); s.attrs["field_names"] = np.array(list(raw), dtype=h5py.string_dtype())
        for k, v in raw.items(): ds = s.create_dataset(k, data=np.float32(v)); ds.attrs.update({"sample_varying": False, "time_varying": False})
        t0 = h.create_group("t0_fields"); names = list(fn) + [c for c, _ in consts] + (["forcing"] if forcing is not None else []); t0.attrs["field_names"] = np.array(names, dtype=h5py.string_dtype())
        for cn, cv in consts:
            ds = t0.create_dataset(cn, data=np.asarray(cv, np.float32)); ds.attrs.update({"dim_varying": np.array([True, True]), "sample_varying": True, "time_varying": False})
        for c, f in enumerate(fn):
            ds = t0.create_dataset(f, data=X[:, :, c].astype(np.float32)); ds.attrs.update({"dim_varying": np.array([True, True]), "sample_varying": True, "time_varying": True})
        if forcing is not None:
            ds = t0.create_dataset("forcing", data=forcing.astype(np.float32)); ds.attrs.update({"dim_varying": np.array([True, True]), "sample_varying": True, "time_varying": False})
        t1 = h.create_group("t1_fields"); t1.attrs["field_names"] = np.array(list(vectors), dtype=h5py.string_dtype()) if vectors else np.array([], dtype=h5py.string_dtype())
        for j, vn in enumerate(vectors):
            c0 = len(fn) + 2 * j; ds = t1.create_dataset(vn, data=np.stack([X[:, :, c0], X[:, :, c0 + 1]], -1).astype(np.float32))
            ds.attrs.update({"dim_varying": np.array([True, True]), "sample_varying": True, "time_varying": True})
        h.create_group("t2_fields").attrs["field_names"] = np.array([], dtype=h5py.string_dtype())
    os.replace(path + ".tmp", path)


def stats_of(fam, its, fields, vectors, rng, n=64):
    """The Well stats.yaml format: scalar fields -> number, vector fields -> [x, y]."""
    m1 = m2 = d1 = d2 = 0.0; k = 0                     # per-channel moments, frames of any grid size (Kolmogorov mixes 256^2 and 512^2)
    for j in rng.choice(len(its), min(n, len(its)), replace=False):
        it = its[j]; t = int(rng.integers(fam.t_start, it.n_frames - fam.stride)); x = load_ws(fam, it, t, 2).astype(np.float64); d = x[1] - x[0]
        m1 = m1 + x[0].mean((1, 2)); m2 = m2 + (x[0] ** 2).mean((1, 2)); d1 = d1 + d.mean((1, 2)); d2 = d2 + (d ** 2).mean((1, 2)); k += 1
    M = {"mean": m1 / k, "rms": np.sqrt(m2 / k), "mean_delta": d1 / k, "rms_delta": np.sqrt(d2 / k)}
    def f(A):
        o = {nm: float(A[c]) for c, nm in enumerate(fields)}
        for j, vn in enumerate(vectors): c = len(fields) + 2 * j; o[vn] = [float(A[c]), float(A[c + 1])]
        return o
    return {key: f(v) for key, v in M.items()}


def check_scalars(dd):
    """every file of a Walrus dataset directory has the same constant-scalar names (Walrus batches samples of one dataset with default_collate)."""
    sets = {}
    for f in sorted(glob.glob(os.path.join(dd, "data", "*", "*.hdf5"))):
        with h5py.File(f, "r") as h: sets.setdefault(tuple(sorted(h["scalars"].attrs["field_names"])), []).append(os.path.basename(f))
    assert len(sets) <= 1, f"{dd}: files with different scalar names: " + "; ".join(f"{k}: {len(v)} files, e.g. {v[0]}" for k, v in sets.items())


def meta_yaml(name, fields, vectors, shape, periodic, raw_keys, n_files, consts=()):
    return {"dataset_name": name, "n_spatial_dims": 2, "spatial_resolution": list(shape), "scalar_names": [], "constant_scalar_names": list(raw_keys),
            "field_names": {0: list(fields), 1: [f"{v}_{c}" for v in vectors for c in "xy"], 2: []}, "constant_field_names": {0: list(consts), 1: [], 2: []},
            "boundary_condition_types": sorted({"PERIODIC" if p else "WALL" for p in periodic}), "n_files": n_files}


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--name", required=True); ap.add_argument("--families", default="", help="default: every family")
    ap.add_argument("--eval_n", type=int, default=8); ap.add_argument("--frames", type=int, default=16); ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--skip_train", action="store_true")
    ap.add_argument("--kolmo_extra", default="", help="optional Kolmogorov groups as train_gen --kolmo_extra (anchor / beta: the amplitude / beta anchors join the training "
                    "export, written as a separate dataset <base>_anchor / <base>_anchor_beta; drag / beta: their held-out columns are exported for evaluation)")
    ap.add_argument("--reference", action="store_true", help="the training export also holds the ref_* trajectories of the held-out conditions (Walrus's in-range "
                    "reference; dataset <base>[...]_ref)")
    ap.add_argument("--no_anchors", action="store_true", help="the training export leaves out the anchor splits (anchor, beta_anchor) even where --kolmo_extra turns "
                    "their groups on: with --reference --kolmo_extra drag,beta the stage-1 reference's export (train split + every ref_* split, dataset <base>_ref)")
    a = ap.parse_args()
    if a.kolmo_extra: corpus.set_kolmo_extra(a.kolmo_extra)
    out = os.path.join(HERE, "results", a.name + ".json"); os.makedirs(os.path.dirname(out), exist_ok=True)
    if os.path.exists(out): print("exists", out); sys.exit(0)
    root = os.path.join(os.environ.get("PHYSDIAL_DATA_ROOT") or os.path.join(PROJ, "data"), "walrus"); t0 = time.time(); summary = {}
    EVD = eval_dir(a.frames)
    fams_req = [f for f in a.families.split(",") if f] or list(corpus.FAMILIES)
    for k in fams_req:
        fam = corpus.get(k); rng = np.random.default_rng(a.seed); tr = corpus.cached_items(k, "train")
        if not tr: print(f"[{a.name}] {k}: no data, skipped", flush=True); continue
        if k in VORT: fields, vectors = [], ("velocity",)
        else:   # The Well family: the scalar channels corpus.py keeps (euler drops the file's energy field) and velocity as a t1 vector
            fields = [c for c in fam.channels if not c.startswith("velocity_")]; vectors = ("velocity",) if "velocity_x" in fam.channels else ()
        st = stats_of(fam, tr, fields, vectors, rng)
        keys = [k_ for k_ in tr[0].raw if k_ not in corpus.RAW_FLAGS]   # scalar names of every training file: those of the train items (the drag / beta files'
        # extra alpha / beta are left out, since every file of a Walrus dataset needs the same scalar names; Walrus does not read them, the file name keeps them)
        sub1 = k in corpus.SUBSET_FAMILIES and fam.stride == 1
        copy = k in OWN or fam.stride > 1 or sub1   # own families, strided copies, and corpus-split subsets: Walrus must train on the corpus split, not the published files
        base = OWN.get(k, fam.DS if sub1 else f"physdial_{k}_s{fam.stride}")   # The Well families with a stride > 1 get a strided copy (Walrus has no per-dataset stride); stride-1 subsets keep The Well's name
        xtr = [sp for sp in corpus.TRAIN_ONLY_SPLITS if sp in fam.splits() and not a.no_anchors] + ([sp for sp in fam.splits() if sp.startswith("ref_")] if a.reference else [])
        base_eval = base; base = base + "".join("_" + {"anchor": "anchor", "beta_anchor": "beta"}[sp] for sp in xtr if sp in corpus.TRAIN_ONLY_SPLITS) + ("_ref" if a.reference else "")
        if copy and not a.skip_train:   # our families in Well format: one dataset per grid size (The Well: one resolution per dataset)
            shapes = sorted({tuple(it.shape) for sp in ("train", "val", "test", *xtr) for it in corpus.cached_items(k, sp)})
            for shp in shapes:
                ds = base + (f"_r{shp[0]}" if len(shapes) > 1 else ""); dd = os.path.join(root, "datasets", ds); nf = 0
                for sp, wd_ in (("train", "train"), ("val", "valid"), ("test", "test")):
                    byfile = {}
                    for it in corpus.cached_items(k, sp) + ([x for s_ in xtr for x in corpus.cached_items(k, s_)] if sp == "train" else []):
                        if tuple(it.shape) == shp: byfile.setdefault(it.path, []).append(it)
                    os.makedirs(os.path.join(dd, "data", wd_), exist_ok=True)
                    names = [os.path.basename(x) for x in byfile]; clash = len(set(names)) < len(names)
                    def file_plan(src, its_all):   # -> [(file name, items, link)]; link: a stride-1 Well file kept whole is linked, not copied
                        stem = (os.path.basename(os.path.dirname(src)) + "__" if clash else "") + os.path.basename(src).rsplit(".", 1)[0]   # source files with the same name: prefix the directory
                        if sub1 and fam.t_start == 0:
                            with h5py.File(src, "r") as h_: n_src = int(h_.attrs["n_trajectories"])
                            if [it.traj for it in its_all] == list(range(n_src)): return [(stem + ".hdf5", its_all, True)]
                        parts = [its_all[j:j + PART] for j in range(0, len(its_all), PART)]
                        return [(stem + (f"_part{pi}" if len(parts) > 1 else "") + ".hdf5", its, False) for pi, its in enumerate(parts)]
                    want = {nm for src, its_all in byfile.items() for nm, _, _ in file_plan(src, its_all)}
                    for old in sorted(os.listdir(os.path.join(dd, "data", wd_))):
                        if old.endswith((".hdf5", ".h5")) and old not in want:   # files this export does not plan (Walrus reads every file in the directory)
                            os.remove(os.path.join(dd, "data", wd_, old)); print(f"[{a.name}] removed stale {ds}/{wd_}/{old}", flush=True)
                    for src, its_all in byfile.items():
                        for nm, its, link in file_plan(src, its_all):
                            p = os.path.join(dd, "data", wd_, nm)
                            if link:
                                if os.path.islink(p) and os.readlink(p) == src: nf += 1; continue
                                if os.path.lexists(p): os.remove(p)
                                os.symlink(src, p); nf += 1; continue
                            if os.path.islink(p): os.remove(p)   # a file kept whole elsewhere is linked; a copy replaces such a link
                            prov = json.dumps([[it.path, str(it.traj)] for it in its] + [[fam.t_start, fam.stride], keys])
                            if os.path.exists(p):   # kept only if it holds exactly these trajectories (a split change that keeps a part at 16 trajectories rewrites it too)
                                with h5py.File(p, "r") as h_: same = h_.attrs.get("physdial_source") == prov
                                if same: nf += 1; continue
                                os.remove(p); print(f"[{a.name}] rewriting {ds}/{wd_}/{os.path.basename(p)} (other trajectories or no provenance)", flush=True)
                            X = np.stack([load_ws(fam, it, fam.t_start, (its[0].n_frames - fam.t_start - 1) // fam.stride + 1) for it in its])
                            write_well(p, ds, fields, X, {k_: its[0].raw[k_] for k_ in keys}, dt_of(fam, its[0]), fam.periodic, None, vectors, None, src if hasattr(fam, "DS") else None,
                                       [(n_, np.stack([c[i_] for c in map(fam.cond_fields, its)])) for i_, n_ in enumerate(CONSTS.get(k, ()))]); nf += 1; del X   # no forcing field (see below)
                            with h5py.File(p, "r+") as h_: h_.attrs["physdial_source"] = prov
                    print(f"[{a.name}] {ds}/{sp}: {len(byfile)} files ({(time.time() - t0) / 60:.1f} min)", flush=True)
                yaml.safe_dump(meta_yaml(ds, fields, vectors, shp, fam.periodic, keys, nf, CONSTS.get(k, ())), open(os.path.join(dd, f"{ds}.yaml"), "w"))
                if sub1: shutil.copy(os.path.join(os.path.dirname(fam.root), "stats.yaml"), os.path.join(dd, "stats.yaml"))   # The Well's published statistics of the dataset
                else: yaml.safe_dump(st, open(os.path.join(dd, "stats.yaml"), "w"))
                summary[ds] = {"files": nf, "grid": list(shp)}
                check_scalars(dd)
        ds_name = base_eval if copy else (fam.DS if hasattr(fam, "DS") else k)   # evaluation windows keep the base name (one grid per eval dir; shared by every export)
        written = set()
        for sp in fam.splits():
            if sp in ("train", "val") or sp.startswith("ref_") or sp in corpus.TRAIN_ONLY_SPLITS: continue
            win = eval_windows(fam, corpus.cached_items(k, sp), a.eval_n, EVAL_SEED, a.frames)
            byshape = {}
            for i, (it, t) in enumerate(win): byshape.setdefault(tuple(it.shape), []).append((i, it, t))
            for shp, ws in byshape.items():   # one evaluation dataset per (family, split, grid); window index = position in eval_windows()
                ed = os.path.join(root, EVD, f"{k}__{sp}__{shp[0]}x{shp[1]}"); os.makedirs(os.path.join(ed, "data", "test"), exist_ok=True); recs = []
                for old in os.listdir(ed):   # one dataset yaml per evaluation directory (Walrus takes the dataset name from it)
                    if old.endswith(".yaml") and old not in ("stats.yaml", f"{ds_name}.yaml"): os.remove(os.path.join(ed, old))
                keep = {f"win_{i:03d}.hdf5" for i, _, _ in ws}
                for old in os.listdir(os.path.join(ed, "data", "test")):   # only this export's windows (Walrus reads every file in the directory)
                    if old.startswith("win_") and old not in keep: os.remove(os.path.join(ed, "data", "test", old))
                for i, it, t in ws:
                    p = os.path.join(ed, "data", "test", f"win_{i:03d}.hdf5"); x = load_ws(fam, it, t, EVAL_HIST + a.frames)[None]
                    write_well(p, ds_name, fields, x, it.raw, dt_of(fam, it), fam.periodic, None, vectors, None, it.path if hasattr(fam, "DS") else None,
                               [(n_, fam.cond_fields(it)[i_][None]) for i_, n_ in enumerate(CONSTS.get(k, ()))])
                    recs.append({"window": i, "path": it.path, "traj": it.traj, "first_frame": t, "stride": fam.stride, "prediction_start": t + EVAL_HIST * fam.stride,
                                 "shape": list(it.shape)})
                for d_ in ("train", "valid"):   # Walrus's data module expects all three split directories
                    l = os.path.join(ed, "data", d_)
                    if not os.path.exists(l): os.symlink("test", l)
                json.dump(recs, open(os.path.join(ed, "windows.json"), "w"))
                check_scalars(ed)
                if k in OWN or fam.stride > 1: yaml.safe_dump(st, open(os.path.join(ed, "stats.yaml"), "w"))   # own families and the strided copies: the statistics of the dataset Walrus trains on
                else:   # The Well family: its published statistics (the format Walrus reads, vectors as [x, y] lists)
                    shutil.copy(os.path.join(os.path.dirname(fam.root), "stats.yaml"), os.path.join(ed, "stats.yaml"))
                yaml.safe_dump(meta_yaml(ds_name, fields, vectors, shp, fam.periodic, [k_ for k_ in ws[0][1].raw if k_ not in corpus.RAW_FLAGS], len(ws), CONSTS.get(k, ())), open(os.path.join(ed, f"{ds_name}.yaml"), "w"))
                written.add(os.path.basename(ed))
                summary[f"{EVD}/{k}__{sp}__{shp[0]}x{shp[1]}"] = {"windows": len(ws)}
        for old in sorted(os.listdir(os.path.join(root, EVD))) if os.path.isdir(os.path.join(root, EVD)) else []:
            if old.startswith(f"{k}__") and old not in written:   # evaluation dirs of splits the corpus does not have (walrus_eval would pool them)
                shutil.rmtree(os.path.join(root, EVD, old)); print(f"[{a.name}] removed stale eval dir {old}", flush=True)
        print(f"[{a.name}] {k}: eval windows written ({(time.time() - t0) / 60:.1f} min)", flush=True)
    res = {"name": a.name, "root": root, "summary": summary, "wall_min": (time.time() - t0) / 60, "meta": runmeta.run_meta()}
    json.dump(res, open(out, "w"), indent=1); print("saved", out, flush=True)


def eval_dir(frames):
    """evaluation-window directory under <data>/walrus/: "eval" for the 16-frame protocol, "eval_f<frames>" otherwise (walrus_eval reads the same)."""
    return "eval" if frames == 16 else f"eval_f{frames}"


if __name__ == "__main__":
    main()
