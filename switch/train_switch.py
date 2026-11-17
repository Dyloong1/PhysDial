"""Switch training: continuation training of a trained run with the Kolmogorov switch-trajectory splits added (switch_data), through
train_gen.main() unchanged.
  --switch on | off     on: the switch training groups (switch_train) join training; off: the same continuation without switch data
                        (same-compute control)
  --reference_switch    also the reference groups (switch_ref: switches into the held-out values), only for the in-range reference model
                        (pass --reference to train_gen)
  --share_cap c         the switch groups take at most this share of the Kolmogorov group draws (0 = their natural share)
In this process only: switch_data.install_train, train_gen.Store = switch_data.make_store(...) (switch groups appended to the Kolmogorov
training store, frames from the switch window on) and, with --share_cap, train_gen.Batches.mix = switch_data.capped_mix(...). Everything
after '--' goes to train_gen verbatim (--init_weights <run>, ...); --name and --store are set here. Checkpoints stay in physdial/ckpt/<name>.*;
result / failed / diverged JSONs are moved to ./results. ckpt/<name>.switch.json records the switch data (a restart with other switch data
is refused)."""
import argparse, hashlib, json, os, shutil, sys

HERE = os.path.dirname(os.path.abspath(__file__)); PROJ = os.path.dirname(HERE); PKG = os.path.join(PROJ, "physdial")
sys.path.insert(0, HERE); sys.path.insert(0, PROJ); sys.path.insert(0, PKG)
import switch_data   # noqa: E402


def opt(argv, flag, default):
    return type(default)(argv[argv.index(flag) + 1]) if flag in argv else default


def main():
    if "--" not in sys.argv: sys.exit("usage: train_switch.py --name <run> --switch on|off [--reference_switch] [--share_cap c] [--H 16] [--store wan21_wrap] -- <train_gen arguments>")
    i = sys.argv.index("--"); mine, rest = sys.argv[1:i], sys.argv[i + 1:]
    ap = argparse.ArgumentParser(); ap.add_argument("--name", required=True); ap.add_argument("--switch", required=True, choices=["on", "off"])
    ap.add_argument("--reference_switch", action="store_true"); ap.add_argument("--H", type=int, default=16, help="switch windows: at most 2 H + span frames after the switch")
    ap.add_argument("--store", default="wan21_wrap"); ap.add_argument("--share_cap", type=float, default=0.0, help="cap on the switch groups' share of the Kolmogorov group draws; 0 = off")
    a = ap.parse_args(mine)
    assert "--name" not in rest and "--store" not in rest, "--name / --store are set by train_switch.py"
    assert not a.reference_switch or ("--reference" in rest and a.switch == "on"), "--reference_switch (switches into held-out values) is for the reference model only: --switch on and --reference for train_gen"
    out = os.path.join(HERE, "results", a.name + ".json"); os.makedirs(os.path.dirname(out), exist_ok=True)
    if os.path.exists(out): print("exists", out); return
    splits = (["switch_train"] if a.switch == "on" else []) + (["switch_ref"] if a.reference_switch else [])
    rank = int(os.environ.get("RANK", 0))
    if splits:
        switch_data.install_train(splits)
        import corpus, train_gen
        k_in, n_out = opt(rest, "--k_in", 2), opt(rest, "--n_out", 4); stride = corpus.STRIDE["kolmo"]; span = (k_in + n_out - 1) * stride + 1
        train_gen.Store = switch_data.make_store(train_gen.Store, a.store, splits, k_in, stride, a.H, span)
        if a.share_cap > 0: train_gen.Batches.mix = switch_data.capped_mix(train_gen.Batches.mix, a.share_cap)
        root = switch_data.latent_root(a.store)
        rec = dict(train_gen.Store.SWITCH_DATA, H=a.H, k_in=k_in, n_out=n_out, share_cap=a.share_cap,
                   index_sha1={s: hashlib.sha1(open(os.path.join(root, "kolmo", s, "index.json"), "rb").read()).hexdigest() for s in splits})
        gp = os.path.join(PKG, "ckpt", a.name + ".switch.json")
        if rank == 0:   # only rank 0 checks / writes, atomically
            if os.path.exists(gp): assert json.load(open(gp)) == json.loads(json.dumps(rec)), f"{gp}: this run was started with other switch data {json.load(open(gp))} != {rec}"
            else:
                os.makedirs(os.path.dirname(gp), exist_ok=True); tmp = gp + ".tmp"
                with open(tmp, "w") as f_: json.dump(rec, f_, indent=1)
                os.replace(tmp, gp)
        print(f"[{a.name}] switch data {rec}", flush=True)
    else:
        import train_gen
        print(f"[{a.name}] no switch data (same-compute control)", flush=True)
    sys.argv = [os.path.join(PKG, "train_gen.py")] + rest + ["--name", a.name, "--store", a.store]; cwd = os.getcwd(); os.chdir(PKG)
    try: train_gen.main()
    except SystemExit as e:
        if e.code not in (0, None): raise
    finally:
        os.chdir(cwd)
        if rank == 0:
            for f in (a.name + ".json", a.name + ".failed.json", a.name + ".diverged.json"):
                src = os.path.join(PKG, "results", f)
                if os.path.exists(src): shutil.move(src, os.path.join(HERE, "results", f)); print("moved", src, flush=True)


if __name__ == "__main__":
    main()
