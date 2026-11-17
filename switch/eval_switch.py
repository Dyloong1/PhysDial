"""Evaluate a trained run on the Kolmogorov switch splits (KolmoSwitch views), through train_gen.main() unchanged.

Installs switch_corpus.KolmoSwitch into this process, builds an evaluation-only command line (--eval_only <run>, --eval_n 32, sampler 25
steps, --kolmo_extra drag) for one of two setups and calls train_gen.main() with --eval_splits = the requested switch_<group>_<view> splits:
  physdial  PhysDial and its switch training: the wrap-padded store wan21_wrap, its fine-tuned decoder, sampling shift 8
  study     the controlled switch study: the zero-padded store wan21, its fine-tuned decoder, the training shift (3)
--main 1 evaluates every standard column as well. The views of one group share the sampling noise of every window (switch_corpus.split_seed).
The result JSON (and --dump_pred arrays) is moved from physdial/results/ to ./results. Runs under torchrun like train_gen (rank 0 moves the files)."""
import argparse, os, shutil, sys

HERE = os.path.dirname(os.path.abspath(__file__)); PROJ = os.path.dirname(HERE); PKG = os.path.join(PROJ, "physdial")
sys.path.insert(0, HERE); sys.path.insert(0, PROJ); sys.path.insert(0, PKG)
import switch_corpus   # noqa: E402

SETUP = {"physdial": dict(store="wan21_wrap", tok="wan21ft:decoder-wrap.pt", sample_shift=8.0),
         "study": dict(store="wan21", tok="wan21ft:decoder.pt", sample_shift=0.0)}


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--name", required=True); ap.add_argument("--run", required=True, help="the trained run (its <run>.ema.pt)")
    ap.add_argument("--setup", default="physdial", choices=sorted(SETUP))
    ap.add_argument("--groups", default="", help="comma list of groups (default: every group file of the data dir)"); ap.add_argument("--views", default="B,A,N,T")
    ap.add_argument("--frames", type=int, default=16, help="predicted frames (multiple of the run's n_out 4)"); ap.add_argument("--subdir", default="kolmo")
    ap.add_argument("--ode_steps", type=int, default=25); ap.add_argument("--eval_bs", type=int, default=4)
    ap.add_argument("--dump_pred", type=int, default=0, help="save history / prediction / truth of the first N windows of every split (analyze_switch.py)")
    ap.add_argument("--main", type=int, default=0, help="1: also every standard evaluation column")
    a = ap.parse_args(); S = SETUP[a.setup]
    out = os.path.join(HERE, "results", a.name + ".json")
    if os.path.exists(out): print("exists", out); return
    assert a.frames % 4 == 0, "--frames must be a multiple of n_out 4"
    switch_dir = os.path.join(os.environ["PHYSDIAL_DATA_ROOT"], "switch", a.subdir)
    splits = switch_corpus.install(switch_dir, a.frames)
    want = {f"switch_{g}_{v}" for g in (a.groups.split(",") if a.groups else {switch_corpus.group_of(s)[0] for s in splits}) for v in a.views.split(",")}
    miss = want - set(splits); assert not miss, f"no data for {sorted(miss)} in {switch_dir}"
    import train_gen   # after install (the module reads corpus at call time; imported here so the class replacement is in place first)
    assert train_gen.EVAL_HIST == switch_corpus.EVAL_HIST, (train_gen.EVAL_HIST, switch_corpus.EVAL_HIST)
    argv = ["--store", S["store"], "--eval_chunks", str(a.frames // 4), "--ode_steps", str(a.ode_steps), "--seed", "0", "--tok", S["tok"],
            "--eval_only", a.run, "--kolmo_extra", "drag", "--eval_bs", str(a.eval_bs), "--name", a.name] + ([] if a.main else ["--eval_splits", ",".join(f"kolmo/{s}" for s in sorted(want))])
    if S["sample_shift"]: argv += ["--sample_shift", str(S["sample_shift"])]
    if a.dump_pred: argv += ["--dump_pred", str(a.dump_pred)]
    sys.argv = [os.path.join(PKG, "train_gen.py")] + argv; cwd = os.getcwd(); os.chdir(PKG)
    try: train_gen.main()
    except SystemExit as e:
        if e.code not in (0, None): raise
    finally:
        os.chdir(cwd)
        if int(os.environ.get("RANK", 0)) == 0: move_out(a.name)


def move_out(name):
    os.makedirs(os.path.join(HERE, "results", "pred"), exist_ok=True)
    for f in (name + ".json", name + ".failed.json"):
        src = os.path.join(PKG, "results", f)
        if os.path.exists(src): shutil.move(src, os.path.join(HERE, "results", f)); print("moved", src, "->", os.path.join(HERE, "results", f), flush=True)
    pd = os.path.join(PKG, "results", "pred", name)
    if os.path.isdir(pd):
        dst = os.path.join(HERE, "results", "pred", name)
        if os.path.isdir(dst): shutil.rmtree(dst)
        shutil.move(pd, dst); print("moved", pd, "->", dst, flush=True)


if __name__ == "__main__":
    main()
