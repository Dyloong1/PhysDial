"""Walrus-format evaluation windows of the multi-switch groups (gen_ms.py), written by the Walrus exporter (walrus_data.main, unchanged)
into their own evaluation roots:
  --plain 0: $PHYSDIAL_DATA_ROOT/switch/ms/walrus/eval_f<frames>/kolmo__switch_ms_<group>_<view>__HxW   with the condition fields wcond.COND (the
             encoding statistics of the conditional runs, asserted unchanged); views S0 and U are evaluated; S1 .. S{K-1} (the condition
             fields of the later segments, same windows) are moved to .../walrus/eval_f<frames>_msseg/, where walrus_ms_eval.py reads them;
  --plain 1: $PHYSDIAL_DATA_ROOT/switch/ms_plain/walrus/eval_f<frames>/...   view U only, without condition fields (history-only runs).
Every window directory records its source group file and mtime (ms_source.json), so evaluations refuse windows of an older generation."""
import argparse, glob, json, os, shutil, sys

HERE = os.path.dirname(os.path.abspath(__file__)); PROJ = os.path.dirname(HERE); PKG = os.path.join(PROJ, "physdial")
sys.path.insert(0, HERE); sys.path.insert(0, PROJ); sys.path.insert(0, PKG)
import yaml   # noqa: E402
import corpus, ms_corpus, wcond   # noqa: E402


def roots(real, plain):
    x = os.path.join(real, "switch"); return x, os.path.join(x, "ms_plain" if plain else "ms")


def rm(p):
    """remove a previous output: a link or file is unlinked, a directory removed with its contents"""
    if os.path.islink(p) or os.path.isfile(p): os.remove(p)
    else: shutil.rmtree(p)


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--name", required=True); ap.add_argument("--plain", type=int, default=0)
    ap.add_argument("--subdir", default="kolmo_ms"); ap.add_argument("--frames", type=int, default=64); ap.add_argument("--eval_n", type=int, default=32)
    a = ap.parse_args()
    out = os.path.join(HERE, "results", a.name + ".json"); os.makedirs(os.path.dirname(out), exist_ok=True)
    if os.path.exists(out): print("exists", out); return
    real = os.environ["PHYSDIAL_DATA_ROOT"]; xroot, broot = roots(real, a.plain)
    ms_corpus.install(os.path.join(xroot, a.subdir), a.frames)
    import walrus_data   # imports train_gen / corpus with the main data root
    base = corpus.FAMILIES["kolmo"]; assert base is ms_corpus.KolmoMS
    keep_view = (lambda v: v == "U") if a.plain else (lambda v: True)
    stats = None
    if not a.plain:
        st_path = os.path.join(xroot, "walrus", "cond_stats.json"); assert os.path.exists(st_path), f"{st_path}: the conditional runs' encoding statistics are missing"
        stats = json.load(open(st_path)); assert stats["slots"] == list(wcond.SLOTS) and stats["scale"] == wcond.SCALE and stats["eps"] == wcond.EPS, stats
        tr = [f for f in glob.glob(os.path.join(xroot, "walrus", "datasets", "physdial_kolmo*", "data", "train", "*.hdf5")) if not os.path.islink(f)]
        assert tr and os.path.getmtime(st_path) <= min(os.path.getmtime(f) for f in tr), f"{st_path} is newer than the conditional training export: the runs trained on another encoding"

    class KolmoMSW(base):
        def splits(self):   # only the multi-switch views (and train / val, which walrus_data skips as evaluation columns; --skip_train)
            return [s for s in super().splits() if s in ("train", "val") or (s.startswith("switch_ms_") and keep_view(s.rsplit("_", 1)[1]))]
        if not a.plain:
            def cond_fields(self, it): return wcond.cond_fields(self, it, stats, super().cond_fields)

    corpus.FAMILIES["kolmo"] = KolmoMSW
    if not a.plain: walrus_data.CONSTS["kolmo"] = wcond.COND
    argv = ["--name", a.name, "--families", "kolmo", "--eval_n", str(a.eval_n), "--frames", str(a.frames), "--skip_train"]
    mt0 = {f: os.path.getmtime(f) for f in glob.glob(os.path.join(xroot, a.subdir, "ms_*.h5"))}
    sys.argv = [os.path.join(PKG, "walrus_data.py")] + argv; os.makedirs(broot, exist_ok=True); os.environ["PHYSDIAL_DATA_ROOT"] = broot
    try: walrus_data.main()
    finally:
        os.environ["PHYSDIAL_DATA_ROOT"] = real; src = os.path.join(PKG, "results", a.name + ".json")
        if os.path.exists(src): shutil.move(src, out); print("moved", src, "->", out, flush=True)
    root = os.path.join(broot, "walrus"); evd = walrus_data.eval_dir(a.frames); n_st = 0; moved = []
    dirs = sorted(glob.glob(os.path.join(root, evd, "kolmo__*")))
    assert dirs and all("__switch_ms_" in os.path.basename(d) for d in dirs), f"{root}/{evd}: unexpected directories {[os.path.basename(d) for d in dirs if '__switch_ms_' not in d]}"
    if not a.plain:
        for d in dirs:   # condition-field entries in stats.yaml (as walrus_cond_data)
            p = os.path.join(d, "stats.yaml"); st = yaml.safe_load(open(p))
            for key in ("mean", "rms", "std"):
                if key in st: st[key].update({c: (0.0 if key == "mean" else 1.0) for c in wcond.COND})
            yaml.safe_dump(st, open(p + ".tmp", "w")); os.replace(p + ".tmp", p); n_st += 1
        sd = os.path.join(root, evd + "_msseg"); os.makedirs(sd, exist_ok=True)
        for old in os.listdir(sd): rm(os.path.join(sd, old))
        for d in dirs:   # S1 .. S{K-1}: out of the evaluation root (walrus_eval evaluates every kolmo__* directory there)
            view = os.path.basename(d).split("__")[1].rsplit("_", 1)[1]
            if view.startswith("S") and view != "S0":
                q = os.path.join(sd, os.path.basename(d))
                if os.path.lexists(q): rm(q)
                shutil.move(d, q); moved.append(os.path.basename(d))
    for d in glob.glob(os.path.join(root, evd, "kolmo__*")) + glob.glob(os.path.join(root, evd + "_msseg", "kolmo__*")):
        src = sorted({r_["path"] for r_ in json.load(open(os.path.join(d, "windows.json")))}); assert len(src) == 1, (d, src)   # the group file, so walrus_ms_eval refuses
        assert os.path.getmtime(src[0]) == mt0[src[0]], f"{src[0]} changed during the export"                       # windows of an older generation
        json.dump({"path": src[0], "mtime": mt0[src[0]]}, open(os.path.join(d, "ms_source.json"), "w"))
    left = sorted(os.path.basename(d) for d in glob.glob(os.path.join(root, evd, "kolmo__*")))
    r = json.load(open(out)); r.update({"switch_ms": {"root": root, "plain": a.plain, "cond": None if a.plain else list(wcond.COND), "stats_files_updated": n_st, "evaluated_dirs": left, "segment_dirs": moved}})
    json.dump(r, open(out + ".tmp", "w"), indent=1); os.replace(out + ".tmp", out); print(f"[{a.name}] evaluation dirs {left}; segment dirs {len(moved)}", flush=True)


if __name__ == "__main__":
    main()
