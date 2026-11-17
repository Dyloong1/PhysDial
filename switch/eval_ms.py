"""Evaluate a finished PhysDial run on the multi-switch splits (ms_corpus.KolmoMS), through train_gen.main().

Installs ms_corpus.KolmoMS into this process and calls train_gen.main() (evaluation only) with --eval_splits = switch_<group>_S0 and
switch_<group>_U, --eval_chunks = frames / n_out. Inside this process only:
  - Conditioner.cond is wrapped: for a batch of S0 items (recognised through ms_corpus.REG) the chunk counter of the rollout selects the
    segment (chunk c covers model steps c n_out + 1 .. (c + 1) n_out; a chunk never straddles two segments), and the params / raw / encoded
    forcing of segment k replace the item's own from the segment's first chunk on. U batches (C0 throughout) pass unchanged.
  - nonfinite_to_inf is wrapped: the first 5-D call after a multi-switch batch's rollout carries its predictions; ms_stats.stats of them
    (kinetic energy, enstrophy, projections on every segment's forcing, energy spectrum) are kept per window;
  - evaluate_split is wrapped: a multi-switch split starts from a clean state, and its statistics are written when the split completes to
    results/ms/<name>/kolmo__<split>.npz;
  - the per-window sampling noise of S0 and U is paired.
Ensembles and --equiv_eval are not supported. Result JSON moved to ./results."""
import argparse, copy, json, os, sys

HERE = os.path.dirname(os.path.abspath(__file__)); PROJ = os.path.dirname(HERE); PKG = os.path.join(PROJ, "physdial")
sys.path.insert(0, HERE); sys.path.insert(0, PROJ); sys.path.insert(0, PKG)
import numpy as np, torch   # noqa: E402
import ms_corpus, ms_stats   # noqa: E402
import eval_switch, switch_corpus   # noqa: E402
from eval_switch import move_out   # noqa: E402  (moves <name>.json / .failed.json and pred/<name> from physdial/results to ./results)

STATE = {"key": None, "n": 0, "seen": set(), "its": None, "fz": {}}
REC = {}   # split -> list of (traj, stats dict)


def seg_fz(s, family, its, k):
    """the encoded forcing of segment k for the batch, exactly as train_gen builds fz (fam.cond_model -> encode_fixed)."""
    fam = s.fams[family]; seg = []
    for x in its:
        y = copy.copy(x); y.switch_forcing = x.switch_ms["segs"][k]["forcing"]; seg.append(y)
    cf = [fam.cond_model(y) for y in seg]; zs_ = []
    for i_ in range(len(cf[0])):
        f = torch.from_numpy(np.stack([c[i_] for c in cf])).to(s.dev).float()
        zs_.append(s.encode_fixed(f, fam.periodic))
    return torch.stack(zs_, 1)


def install_patches(train_gen, frames):
    orig_cond = train_gen.Conditioner.cond

    def cond(s, family, params, raws, fz, ref, sl, B, h, w, stride=None):
        its = [ms_corpus.REG.get(id(p)) for p in params]
        if all(x is None for x in its): return orig_cond(s, family, params, raws, fz, ref, sl, B, h, w, stride)
        assert all(x is not None for x in its), "a batch mixes multi-switch and other items"
        key = tuple(id(p) for p in params)
        if key != STATE["key"]:
            assert STATE["its"] is None, "the previous multi-switch batch was never recorded (its predictions did not reach nonfinite_to_inf)"
            assert key not in STATE["seen"], "a multi-switch batch rolled out twice (--equiv_eval / ensembles are not supported here)"
            STATE.update(key=key, n=0, its=its, fz={}); STATE["seen"].add(key)
        c = STATE["n"]; STATE["n"] += 1
        assert c < int(s.a.eval_chunks), f"conditioner call {c} of a batch with {s.a.eval_chunks} chunks (a second rollout: ensembles / --equiv_eval are not supported)"
        m0 = its[0].switch_ms; view = m0["view"]; assert all(x.switch_ms["view"] == view for x in its), "a batch mixes views"
        if view == "U": STATE["F"] = int(s.a.n_out); return orig_cond(s, family, params, raws, fz, ref, sl, B, h, w, stride)
        assert view == "S0", f"view {view}: only S0 is rolled out (with the segment swaps); S1.. exist for the Walrus export"
        F, ss, K = int(s.a.n_out), int(m0["seg_steps"]), int(m0["K"]); STATE["F"] = F
        k = (c * F) // ss; assert ((c + 1) * F - 1) // ss == k, f"chunk {c} of {F} steps straddles segments of {ss} steps"
        assert k < K, f"chunk {c}: segment {k} beyond the {K} segments"
        if k == 0: return orig_cond(s, family, params, raws, fz, ref, sl, B, h, w, stride)
        if k not in STATE["fz"]: STATE["fz"][k] = None if fz is None else seg_fz(s, family, its, k)
        pk = [x.switch_ms["segs"][k]["params"] for x in its]; rk = [x.switch_ms["segs"][k]["raw"] for x in its]
        return orig_cond(s, family, pk, rk, STATE["fz"][k], ref, sl, B, h, w, stride)

    train_gen.Conditioner.cond = cond
    orig_nf = train_gen.nonfinite_to_inf

    def nonfinite_to_inf(P, *errs):
        its = STATE["its"]
        if its is not None and torch.is_tensor(P) and P.dim() == 5 and P.shape[0] == len(its):
            assert P.shape[1] == frames and P.shape[2] == 1, f"predictions {tuple(P.shape)}: expected [B, {frames}, 1, H, W]"
            m0 = its[0].switch_ms; assert STATE["n"] * STATE["F"] == frames, f"{STATE['n']} conditioner calls of {STATE['F']} frames for {frames} frames"
            Fr = torch.from_numpy(np.stack([ms_stats.forcings(x.path, x.traj, m0["K"]) for x in its])).to(P.device)
            st = ms_stats.stats(P[:, :, 0].detach(), Fr); st["finite"] = torch.isfinite(P).flatten(1).all(1).cpu().numpy()
            sp = f"switch_{os.path.basename(its[0].path)[:-3]}_{m0['view']}"; assert sp == STATE["sp"], (sp, STATE["sp"])
            for j, it_ in enumerate(its): REC.setdefault(sp, []).append((it_.traj, {k: v[j] for k, v in st.items()}))
            STATE["its"] = None
        return orig_nf(P, *errs)

    train_gen.nonfinite_to_inf = nonfinite_to_inf
    orig_es = train_gen.evaluate_split

    def evaluate_split(model, cd, fam, its, a, gen, sp="", eval_bs=0):
        if not sp.startswith("switch_ms_"): return orig_es(model, cd, fam, its, a, gen, sp, eval_bs)
        assert int(getattr(a, "ens", 1)) <= 1 and not getattr(a, "equiv_eval", False), "multi-switch evaluation: no ensembles, no --equiv_eval"
        assert int(a.n_out) * int(a.eval_chunks) == frames, f"the run's n_out {a.n_out} x {a.eval_chunks} chunks != {frames} frames (eval_ms assumes n_out 4)"
        STATE.update(key=None, n=0, its=None, fz={}, sp=sp); STATE["seen"] = set(); REC.pop(sp, None)
        try: r = orig_es(model, cd, fam, its, a, gen, sp, eval_bs)
        except BaseException: REC.pop(sp, None); STATE.update(key=None, its=None); raise
        assert STATE["its"] is None, f"{sp}: the last batch was never recorded"
        L = REC.get(sp, []); trajs = [t for t, _ in L]
        pw = sorted(str(w["traj"]) for w in r.get("per_window", []))
        assert len(trajs) == len(set(trajs)) and sorted(trajs) == pw, f"{sp}: recorded windows {sorted(trajs)[:4]}.. ({len(trajs)}) != evaluated windows {pw[:4]}.. ({len(pw)})"
        od = os.path.join(HERE, "results", "ms", a.name); os.makedirs(od, exist_ok=True); f = os.path.join(od, f"kolmo__{sp}.npz")
        src = its[0].path; mt = {x.switch_ms["data_mtime"] for x in its}; assert len(mt) == 1 and os.path.getmtime(src) in mt, f"{src} changed during the evaluation"
        meta = {"trajs": trajs, "frames": frames, "run": a.eval_only, "eval_bs": int(eval_bs or a.eval_bs), "data_mtime": mt.pop(), "data": src,
                "ode_steps": int(a.ode_steps), "sample_shift": float(a.sample_shift or a.fm_shift), "tok": str(a.tok), "stats_md5": ms_stats.MD5}
        with open(f + ".tmp", "wb") as fh: np.savez(fh, meta=json.dumps(meta), **{k: np.stack([d[k] for _, d in L]) for k in L[0][1]})
        os.replace(f + ".tmp", f)   # (a killed write leaves '.tmp', which the read-out never globs)
        print(f"[ms] {sp}: statistics of {len(L)} windows -> {f}", flush=True)
        return r

    train_gen.evaluate_split = evaluate_split


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--name", required=True); ap.add_argument("--run", required=True, help="the finished run (its <run>.ema.pt)")
    ap.add_argument("--setup", default="physdial", choices=sorted(eval_switch.SETUP), help="as eval_switch.py: store, decoder and sampling shift")
    ap.add_argument("--groups", default="", help="comma list of groups (default: every ms_*.h5 in the data dir)"); ap.add_argument("--views", default="S0,U")
    ap.add_argument("--frames", type=int, default=64, help="predicted frames (multiple of n_out 4, <= segments x seg_steps)"); ap.add_argument("--subdir", default="kolmo_ms")
    ap.add_argument("--ode_steps", type=int, default=25); ap.add_argument("--eval_bs", type=int, default=4)
    ap.add_argument("--dump_pred", type=int, default=0, help="save fields of the first N windows of every split (train_gen --dump_pred; ms_case.py)")
    a = ap.parse_args(); S = eval_switch.SETUP[a.setup]
    out = os.path.join(HERE, "results", a.name + ".json")
    if os.path.exists(out): print("exists", out); return
    assert a.frames % 4 == 0, "--frames must be a multiple of n_out 4"
    assert set(a.views.split(",")) <= {"S0", "U"}, "the generator is evaluated on S0 (segment swaps) and U only"
    ms_dir = os.path.join(os.environ["PHYSDIAL_DATA_ROOT"], "switch", a.subdir)
    splits = ms_corpus.install(ms_dir, a.frames)
    groups = a.groups.split(",") if a.groups else sorted({switch_corpus.group_of(s)[0] for s in splits})
    want = {f"switch_{g}_{v}" for g in groups for v in a.views.split(",")}
    miss = want - set(splits); assert not miss, f"no data for {sorted(miss)} in {ms_dir}"
    import h5py
    gen_ids = {}
    for g_ in groups:
        with h5py.File(os.path.join(ms_dir, g_ + ".h5"), "r") as h_: gen_ids[g_] = (str(h_.attrs.get("gen_code", "")), str(h_.attrs.get("gen_args", "")))
    assert len(set(gen_ids.values())) == 1, f"group files of different generator runs: {gen_ids}"
    import train_gen
    assert train_gen.EVAL_HIST == ms_corpus.EVAL_HIST, (train_gen.EVAL_HIST, ms_corpus.EVAL_HIST)
    install_patches(train_gen, a.frames)
    argv = ["--store", S["store"], "--eval_n", "32", "--eval_chunks", str(a.frames // 4), "--ode_steps", str(a.ode_steps), "--seed", "0", "--tok", S["tok"],
            "--eval_only", a.run, "--kolmo_extra", "drag", "--eval_bs", str(a.eval_bs), "--name", a.name, "--eval_splits", ",".join(f"kolmo/{s}" for s in sorted(want))]
    if S["sample_shift"]: argv += ["--sample_shift", str(S["sample_shift"])]
    if a.dump_pred: argv += ["--dump_pred", str(a.dump_pred)]
    sys.argv = [os.path.join(PKG, "train_gen.py")] + argv; cwd = os.getcwd(); os.chdir(PKG)
    try: train_gen.main()
    except SystemExit as e:
        if e.code not in (0, None): raise
    finally:
        os.chdir(cwd)
        if int(os.environ.get("RANK", 0)) == 0: move_out(a.name)


if __name__ == "__main__":
    main()
