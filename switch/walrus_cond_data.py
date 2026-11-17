"""Walrus-format data for the conditional Walrus (Kolmogorov with condition fields, wcond.py) and the switch evaluation windows,
written by the Walrus exporter (physdial/walrus_data.main, unchanged) into the switch data root:
  $PHYSDIAL_DATA_ROOT/switch/walrus/        (--plain 0: Kolmogorov datasets / windows with the constant fields wcond.COND)
  $PHYSDIAL_DATA_ROOT/switch/plain/walrus/  (--plain 1: switch windows without them, for the history-only Walrus runs)
In this process only: the Kolmogorov family class gets cond_fields = wcond.cond_fields (and, with --switch, switch_corpus.KolmoSwitch's switch splits,
so the views are exported as evaluation directories kolmo__switch_<group>_<view>__HxW), walrus_data.CONSTS['kolmo'] = wcond.COND, and
PHYSDIAL_DATA_ROOT points to the switch root while walrus_data.main runs. The condition-field entries of stats.yaml are set to mean 0 / rms 1
(the fields carry their own scale). --link_main links the evaluation windows of the other families into this root."""
import argparse, glob, json, os, shutil, sys

HERE = os.path.dirname(os.path.abspath(__file__)); PROJ = os.path.dirname(HERE); PKG = os.path.join(PROJ, "physdial")
sys.path.insert(0, HERE); sys.path.insert(0, PROJ); sys.path.insert(0, PKG)
import yaml   # noqa: E402
import corpus, switch_corpus, wcond   # noqa: E402


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--name", required=True); ap.add_argument("--plain", type=int, default=0)
    ap.add_argument("--switch", default="", help="switch evaluation data subdir under $PHYSDIAL_DATA_ROOT/switch (e.g. kolmo): also export its views as evaluation windows")
    ap.add_argument("--kolmo_extra", default="", help="as walrus_data (anchor: + the anchor training dataset; drag: + the drag columns)")
    ap.add_argument("--frames", type=int, default=16); ap.add_argument("--eval_n", type=int, default=32); ap.add_argument("--skip_train", action="store_true")
    ap.add_argument("--link_main", action="store_true", help="link the main evaluation directories of every non-Kolmogorov family into this root")
    a = ap.parse_args()
    assert not (a.plain and not a.skip_train), "--plain exports evaluation windows only (the history-only Walrus runs train on the main export)"
    out = os.path.join(HERE, "results", a.name + ".json"); os.makedirs(os.path.dirname(out), exist_ok=True)
    if os.path.exists(out): print("exists", out); return
    real = os.environ["PHYSDIAL_DATA_ROOT"]; xroot = os.path.join(real, "switch"); broot = os.path.join(xroot, "plain") if a.plain else xroot   # walrus_data writes <root>/walrus
    if a.switch: switch_corpus.install(os.path.join(xroot, a.switch), a.frames)
    import walrus_data   # imports train_gen / corpus with the main data root
    base = corpus.FAMILIES["kolmo"]
    if not a.plain:
        corpus.set_kolmo_extra("anchor")
        fam0 = corpus.get("kolmo"); items = corpus.cached_items("kolmo", "train") + corpus.cached_items("kolmo", "anchor"); corpus.set_kolmo_extra("")
        stats = wcond.load_or_make_stats(os.path.join(xroot, "walrus"), fam0, items)
        print(f"[{a.name}] condition encoding {json.dumps(stats)}", flush=True)

        class KolmoW(base):
            def cond_fields(self, it): return wcond.cond_fields(self, it, stats, super().cond_fields)

        corpus.FAMILIES["kolmo"] = KolmoW
        walrus_data.CONSTS["kolmo"] = wcond.COND
    argv = ["--name", a.name, "--families", "kolmo", "--eval_n", str(a.eval_n), "--frames", str(a.frames)]
    if a.kolmo_extra: argv += ["--kolmo_extra", a.kolmo_extra]
    if a.skip_train: argv += ["--skip_train"]
    sys.argv = [os.path.join(PKG, "walrus_data.py")] + argv; os.makedirs(broot, exist_ok=True); os.environ["PHYSDIAL_DATA_ROOT"] = broot
    try: walrus_data.main()
    finally:
        os.environ["PHYSDIAL_DATA_ROOT"] = real; src = os.path.join(PKG, "results", a.name + ".json")
        if os.path.exists(src): shutil.move(src, out); print("moved", src, "->", out, flush=True)
    root = os.path.join(broot, "walrus"); evd = walrus_data.eval_dir(a.frames); n_st = 0
    if not a.plain:   # condition-field entries in stats.yaml (training datasets and evaluation windows of Kolmogorov)
        for p in glob.glob(os.path.join(root, "datasets", "physdial_kolmo*", "stats.yaml")) + glob.glob(os.path.join(root, evd, "kolmo__*", "stats.yaml")):
            st = yaml.safe_load(open(p))
            for key in ("mean", "rms", "std"):
                if key in st: st[key].update({c: (0.0 if key == "mean" else 1.0) for c in wcond.COND})
            yaml.safe_dump(st, open(p + ".tmp", "w")); os.replace(p + ".tmp", p); n_st += 1
    n_ln = 0
    if a.link_main:
        md = os.path.join(real, "walrus", evd)
        for d in sorted(os.listdir(md)):
            if d.startswith("kolmo__"): continue
            q = os.path.join(root, evd, d)
            if not os.path.lexists(q): os.symlink(os.path.join(md, d), q); n_ln += 1
    r = json.load(open(out)); r.update({"switch": {"root": root, "plain": a.plain, "cond": None if a.plain else list(wcond.COND), "stats_files_updated": n_st, "linked_main_eval_dirs": n_ln}})
    json.dump(r, open(out + ".tmp", "w"), indent=1); os.replace(out + ".tmp", out); print(f"[{a.name}] stats.yaml updated {n_st}, links {n_ln}", flush=True)


if __name__ == "__main__":
    main()
