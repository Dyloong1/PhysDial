"""KolmoMS = corpus.Kolmo plus the multi-switch splits switch_<group>_<view> (groups ms_*, files of gen_ms.py). Installed only inside a
multi-switch evaluation or export process (install()); every other split is Kolmo's own.

Views: S0 .. S{K-1} = switched truth, condition of segment k (params, raw, forcing_seg[k]); U = switched truth, condition C0 throughout (the
model is never told). PhysDial (eval_ms.py): split S0 is rolled out with the condition swapped to segment k at the start of every segment (the
S0 items carry every segment's params / raw in it.switch_ms; REG maps id(item.params) -> item, so the conditioner patch recognises a batch); U is
rolled out unchanged. The Walrus export (walrus_ms_data.py) writes every view, and walrus_ms_eval.py swaps the constant fields of S0's windows
for those of S1.. at the segment starts."""
import glob, json, math, os, sys
import numpy as np, h5py

HERE = os.path.dirname(os.path.abspath(__file__)); PROJ = os.path.dirname(HERE); PKG = os.path.join(PROJ, "physdial")
sys.path.insert(0, PROJ); sys.path.insert(0, PKG)
import corpus, switch_corpus   # noqa: E402

EVAL_HIST = 6   # train_gen.EVAL_HIST (asserted by the evaluators)
REG = {}        # id(item.params) -> item, for the S0 / U items this process created (eval_ms's conditioner patch)


def views(K): return tuple(f"S{k}" for k in range(K)) + ("U",)


def cond_of(c, kf):
    """(params slots without log10_dt, raw) of one condition dict of the group's seq, as corpus.Kolmo._item / switch_corpus write them."""
    raw = {"nu": float(c["nu"]), "k_f": float(kf), "amp": float(c["amp"])}
    if c["forcing"] == "multi": raw["multimode"] = 1.0
    return {"log10_nu": math.log10(float(c["nu"])), "kf": float(kf), "amp": float(c["amp"])}, raw


class KolmoMS(corpus.Kolmo):
    MS_DIR = ""        # set by install()
    MS_FRAMES = 64     # rollout length of the evaluation

    def _ms_files(self): return sorted(glob.glob(os.path.join(self.MS_DIR, "ms_*.h5")))

    def _K(self, path):
        with h5py.File(path, "r") as h: return len(json.loads(h.attrs["seq"])) - 1

    def splits(self): return super().splits() + [f"switch_{os.path.basename(f)[:-3]}_{v}" for f in self._ms_files() for v in views(self._K(f))]

    def items(self, split):
        if not split.startswith("switch_ms_"): return super().items(split)
        grp, view = switch_corpus.group_of(split)
        path = os.path.join(self.MS_DIR, grp + ".h5"); need = (EVAL_HIST + self.MS_FRAMES - 1) * self.stride + 1; out = []
        mtime = os.path.getmtime(path)
        with h5py.File(path, "r") as h:
            a = dict(h.attrs); seq = json.loads(a["seq"]); K = len(seq) - 1; ss = int(a["seg_steps"]); assert view in views(K), split
            assert int(a["switch_frame"]) == (EVAL_HIST - 1) * self.stride, f"{path}: switch at frame {a['switch_frame']}, the window's last history frame is {(EVAL_HIST - 1) * self.stride}"
            assert self.MS_FRAMES <= K * ss, f"{path}: {K} segments of {ss} steps < {self.MS_FRAMES} predicted frames"
            ldt = math.log10(float(a["dt_frame"]))   # the true frame interval, kept through every segment (gen_ms)
            for traj in sorted(x for x in h.keys() if not x.startswith("c")):
                g = h[traj]; T, H, W = g["data"].shape[:3]
                assert T >= need, f"{path}/{traj}: {T} frames < {need} needed for {self.MS_FRAMES} predicted frames"
                ksl = [float(x) for x in g.attrs["kf_slot_seg"]]; assert len(ksl) == K
                segs = []
                for k in range(K):
                    p, raw = cond_of(seq[k + 1], ksl[k]); p["log10_dt"] = ldt; segs.append({"params": p, "raw": raw, "forcing": ("forcing_seg", k)})
                if view == "U": p, raw = cond_of(seq[0], float(g.attrs["A_kf_slot"])); p["log10_dt"] = ldt; src = ("forcing_A", None)
                else: k = int(view[1:]); p, raw, src = dict(segs[k]["params"]), dict(segs[k]["raw"]), segs[k]["forcing"]
                it = corpus.Item(self.key, path, traj, p, raw, need, (H, W)); it.switch_forcing = src
                it.switch_ms = {"view": view, "K": K, "seg_steps": ss, "segs": segs, "data_mtime": mtime}
                REG[id(it.params)] = it; out.append(it)
        return out

    def split_seed(self, split):   # train_gen's evaluation noise seed of a split: the views of one group share the group's noise seed (groups_kolmo_ms.json)
        if split.startswith("switch_ms_"): return int(switch_corpus.GROUP_SEEDS[switch_corpus.group_of(split)[0]]["noise_seed"])
        return super().split_seed(split)

    def cond_fields(self, it):   # the forcing of this item's view (segment k: forcing_seg[k]; U: forcing_A)
        src = getattr(it, "switch_forcing", None)
        if src is None: return super().cond_fields(it)
        name, k = src
        with h5py.File(it.path, "r") as h:
            d = h[it.traj][name]; return [np.asarray(d[k] if k is not None else d, dtype=np.float32)]


def install(ms_dir, frames):
    assert os.path.isdir(ms_dir), ms_dir
    KolmoMS.MS_DIR, KolmoMS.MS_FRAMES = ms_dir, frames
    switch_corpus.install_class(KolmoMS)
    return [s for s in KolmoMS().splits() if s.startswith("switch_ms_")]
