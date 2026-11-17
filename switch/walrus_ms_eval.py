#!/usr/bin/env python
"""Evaluate a Walrus run on the multi-switch windows (walrus_ms_data.py) with the Walrus evaluator (walrus_eval.py, imported, unchanged),
--families kolmo, --frames = the export's rollout length:
  --root cond  : $PHYSDIAL_DATA_ROOT/switch/ms/walrus/eval_f<frames>         views S0 (segment swaps) and U    -> conditional runs
  --root plain : $PHYSDIAL_DATA_ROOT/switch/ms_plain/walrus/eval_f<frames>   view U (no condition fields)      -> history-only runs
In this process only:
  - walrus_eval.HERE = this directory (a run's result JSON is linked into ./results), PHYSDIAL_DATA_ROOT = the export root while main() runs;
  - eval_datamodule is wrapped: for an S0 directory it also opens the same windows with the later segments' condition fields
    (.../eval_f<frames>_msseg/kolmo__switch_<group>_S<k>__HxW);
  - rollout is a copy of walrus_eval.rollout with one addition: at model step i of an S0 batch the constant fields are those of segment
    i // seg_steps;
  - to_phys is wrapped to keep ms_stats.stats of the predictions per window; accumulate_dir reseeds per group (S0 and U paired) and writes
    the statistics to results/ms/<name>/kolmo__<split>.npz.
Runs under torchrun like walrus_eval."""
import copy, glob, json, os, re, sys

HERE = os.path.dirname(os.path.abspath(__file__)); PROJ = os.path.dirname(HERE); PKG = os.path.join(PROJ, "physdial"); WDIR = os.path.join(PKG, "baselines", "walrus")
sys.path.insert(0, HERE); sys.path.insert(0, PROJ); sys.path.insert(0, PKG); sys.path.insert(0, WDIR)
import numpy as np, torch   # noqa: E402
import walrus_eval as WE   # noqa: E402
import ms_stats, switch_corpus   # noqa: E402

DIR = re.compile(r"^switch_(ms_.+)_(S\d+|U)$")
CTX = {"on": False}
NAME = [None]; SEED = [0]; RUN = [None]; ROOT = [None]


def ms_dir(ed):
    parts = os.path.basename(ed).split("__"); m = DIR.match(parts[1]) if len(parts) == 3 else None
    return (parts, m.group(1), m.group(2)) if m else None


def install(seg_steps_of):
    dm0, ro0, tp0, acc0 = WE.eval_datamodule, WE.rollout, WE.to_phys, WE.accumulate_dir

    def eval_datamodule(cfg, ed, ds, fmap, a):
        dm = dm0(cfg, ed, ds, fmap, a); info = ms_dir(ed)
        if info is None: return dm
        parts, grp, view = info; dset = dm.test_dataloaders(replicas=1, rank=0, full=True)[0].dataset; files = [os.path.basename(f) for f in dset.sub_dsets[0].files_paths]
        win = {r_["window"]: r_ for r_ in json.load(open(os.path.join(ed, "windows.json")))}; recs = [win[int(f[4:-5])] for f in files]
        K, ss = seg_steps_of(recs[0]["path"]); segs = []; src = source_of(ed); assert src["path"] == recs[0]["path"], (ed, src)
        if view == "S0":
            for k in range(1, K):
                edk = os.path.join(os.path.dirname(os.path.dirname(ed)), os.path.basename(os.path.dirname(ed)) + "_msseg", f"{parts[0]}__switch_{grp}_S{k}__{parts[2]}")
                dsk = [f[:-5] for f in os.listdir(edk) if f.endswith(".yaml") and f != "stats.yaml"][0]
                dk = dm0(cfg, edk, dsk, fmap, a).test_dataloaders(replicas=1, rank=0, full=True)[0].dataset
                assert [os.path.basename(f) for f in dk.sub_dsets[0].files_paths] == files, f"{edk}: other window files than {ed}"
                assert source_of(edk) == src, f"{edk}: exported from {source_of(edk)}, {ed} from {src}"
                segs.append(dk)
        CTX.update(on=True, ed=ed, grp=grp, view=view, recs=recs, K=K, ss=ss, segs=segs, i=0, calls=0, rec=[], src=src)
        return dm

    @torch.no_grad()
    def rollout(model, revin, formatter, batch, steps, eps, dev):
        if not CTX["on"]: return ro0(model, revin, formatter, batch, steps, eps, dev)
        B = batch["input_fields"].shape[0]; i0 = CTX["i"]; idx = list(range(i0, i0 + B)); segc = []
        for dk in CTX["segs"]:   # the same windows with segment k's constant fields
            bk = dk[idx]; assert torch.equal(bk["input_fields"], batch["input_fields"]), f"{CTX['ed']}: segment batch {idx} holds other input frames"
            segc.append(bk["constant_fields"])
        meta = batch["metadata"]
        batch = {k: (v.to(dev, non_blocking=True) if k not in {"metadata", "boundary_conditions"} else v) for k, v in batch.items()}
        segc = [c.to(dev) for c in segc]
        _, y_ref = formatter.process_input(batch, causal_in_time=model.causal_in_time, predict_delta=True, train=False)
        steps = min(steps, y_ref.shape[1]); y_ref = y_ref[:, :steps]; moving = copy.deepcopy(batch); preds = []
        assert CTX["view"] != "S0" or steps <= CTX["K"] * CTX["ss"], f"{steps} steps > {CTX['K']} segments of {CTX['ss']}"
        for i in range(steps):
            k = i // CTX["ss"]
            if CTX["view"] == "S0" and k >= 1 and i % CTX["ss"] == 0: moving["constant_fields"] = segc[k - 1]   # segment k from its first step on
            inputs, _ = formatter.process_input(moving); inputs = list(inputs)   # ---- from here as walrus_eval.rollout
            st = revin.compute_stats(inputs[0], meta, epsilon=eps)
            ni = inputs[:]; ni[0] = revin.normalize_stdmean(ni[0], st)
            y = model(ni[0], ni[1], ni[2].tolist(), metadata=meta)
            if model.causal_in_time: y = y[-1:]
            y = inputs[0][-y.shape[0]:].float() + revin.denormalize_delta(y, st)
            y = formatter.process_output(y, meta)[..., :y_ref.shape[-1]]
            pm = batch["padded_field_mask"]
            if pm.shape[0] != y.shape[-1]: pm = pm[:y.shape[-1]]
            y = y.masked_fill(~pm, 0.0)
            if i != steps - 1: moving["input_fields"] = torch.cat([moving["input_fields"][:, 1:], y[:, -1:]], dim=1)
            preds.append(y[:, -1:])
        return torch.cat(preds, 1), y_ref, batch["input_fields"][:, -2:]

    def to_phys(x, *a_, **k_):
        r = tp0(x, *a_, **k_)
        if CTX["on"]:
            c = CTX["calls"]; CTX["calls"] += 1
            if c % 3 == 0:   # P (walrus_eval.accumulate_dir: P = to_phys(yp ..), T = to_phys(yr ..), L2 = to_phys(last2 ..))
                B = r.shape[0]; recs = CTX["recs"][CTX["i"]:CTX["i"] + B]; assert len(recs) == B and r.dim() == 5 and r.shape[2] == 1, (len(recs), tuple(r.shape))
                Fr = torch.from_numpy(np.stack([ms_stats.forcings(q["path"], str(q["traj"]), CTX["K"]) for q in recs])).to(r.device)
                st = ms_stats.stats(r[:, :, 0].detach(), Fr); st["finite"] = torch.isfinite(r).flatten(1).all(1).cpu().numpy()
                for j, q in enumerate(recs): CTX["rec"].append((str(q["traj"]), {k: v[j] for k, v in st.items()}))
            if c % 3 == 2: CTX["i"] += r.shape[0]
        return r

    def accumulate_dir(ed, fam, *x, **k):
        info = ms_dir(ed)
        if info is None: return acc0(ed, fam, *x, **k)
        parts, grp, view = info
        sd = SEED[0] + int(switch_corpus.GROUP_SEEDS[grp]["walrus_seed"]); torch.manual_seed(sd); np.random.seed(sd % 2 ** 32)   # S0 and U paired
        try:
            ac = acc0(ed, fam, *x, **k)
            L = CTX["rec"]; trajs = [t for t, _ in L]; n = int(ac["n"])
            assert len(trajs) >= n and len(set(trajs[:n])) == n, f"{ed}: {len(trajs)} recorded windows for {n}"
            L = L[:n]; sp = parts[1]; od = os.path.join(HERE, "results", "ms", NAME[0]); os.makedirs(od, exist_ok=True); f = os.path.join(od, f"kolmo__{sp}.npz")
            src = CTX["src"]
            meta = {"trajs": [t for t, _ in L], "frames": int(ac["E"].shape[1]), "grid": ac["grid"], "data": src["path"], "data_mtime": src["mtime"], "eval_bs": int(x[4].eval_bs),
                    "run": RUN[0], "root": ROOT[0], "stats_md5": ms_stats.MD5}
            with open(f + ".tmp", "wb") as fh: np.savez(fh, meta=json.dumps(meta), **{kk: np.stack([d[kk] for _, d in L]) for kk in L[0][1]})
            os.replace(f + ".tmp", f); print(f"[ms] {sp}: statistics of {n} windows -> {f}", flush=True)
            return ac
        finally: CTX.clear(); CTX["on"] = False

    WE.eval_datamodule, WE.rollout, WE.to_phys, WE.accumulate_dir = eval_datamodule, rollout, to_phys, accumulate_dir


def source_of(ed):
    """walrus_ms_data's export-time provenance of a window directory; refuses windows exported from an older generation of the group file."""
    src = json.load(open(os.path.join(ed, "ms_source.json")))
    if os.path.getmtime(src["path"]) != src["mtime"]: raise SystemExit(f"{ed}: exported from {src['path']} of mtime {src['mtime']}, the file changed since; run walrus_ms_data.py again")
    return src


def seg_info(path, cache={}):
    if path not in cache:
        import h5py
        with h5py.File(path, "r") as h: cache[path] = (len(json.loads(h.attrs["seq"])) - 1, int(h.attrs["seg_steps"]))
    return cache[path]


def main():
    argv = sys.argv[1:]; root_kind = "cond"
    if "--root" in argv: i = argv.index("--root"); root_kind = argv[i + 1]; del argv[i:i + 2]
    assert root_kind in ("cond", "plain"), root_kind
    NAME[0] = argv[argv.index("--name") + 1]
    assert "--families" not in argv and "--frames" in argv, "multi-switch: --frames <the export's rollout length>; the family is kolmo"
    argv += ["--families", "kolmo"]
    SEED[0] = int(argv[argv.index("--seed") + 1]) if "--seed" in argv else 0
    run = argv[argv.index("--run") + 1] if "--run" in argv else None; ck = argv[argv.index("--ckpt") + 1] if "--ckpt" in argv else "last"
    RUN[0] = "released" if ck == "coalesced" else f"{run}@{ck}"; ROOT[0] = root_kind
    os.makedirs(os.path.join(HERE, "results"), exist_ok=True)
    if run and not os.path.exists(os.path.join(HERE, "results", run + ".json")):   # a walrus_ft.py run: its JSON is linked from the Walrus results (as walrus_cond_eval)
        src = os.path.join(WDIR, "results", run + ".json"); assert os.path.exists(src), f"no result JSON for run {run}"
        if int(os.environ.get("RANK", 0)) == 0 and not os.path.lexists(os.path.join(HERE, "results", run + ".json")): os.symlink(src, os.path.join(HERE, "results", run + ".json"))
        for _ in range(120):   # other ranks wait for rank 0's link
            if os.path.exists(os.path.join(HERE, "results", run + ".json")): break
            import time; time.sleep(1)
    WE.HERE = HERE
    if ck != "coalesced":
        cfg_txt = open(os.path.join(WE.run_dir(run), "extended_config.yaml")).read(); is_cond = "switch_cond" in cfg_txt
        assert is_cond == (root_kind == "cond"), f"{run}: {'conditional' if is_cond else 'history-only'} run evaluated on --root {root_kind}"
    else: assert root_kind == "plain", "the released Walrus reads no condition fields: --root plain"
    install(seg_info)
    real = os.environ["PHYSDIAL_DATA_ROOT"]; xr = os.path.join(real, "switch")
    os.environ["PHYSDIAL_DATA_ROOT"] = os.path.join(xr, "ms" if root_kind == "cond" else "ms_plain")
    fr = int(argv[argv.index("--frames") + 1]); ev_ = os.path.join(os.environ["PHYSDIAL_DATA_ROOT"], "walrus", "eval" if fr == 16 else f"eval_f{fr}")
    assert glob.glob(os.path.join(ev_, "kolmo__switch_ms_*")), f"no multi-switch windows under {ev_} (--frames / export missing)"
    sys.argv = [os.path.join(WDIR, "walrus_eval.py")] + argv
    try: WE.main()
    finally: os.environ["PHYSDIAL_DATA_ROOT"] = real


if __name__ == "__main__":
    main()
