"""KolmoSwitch = corpus.Kolmo plus the switch evaluation splits switch_<group>_<view> (group files written by gen_switch.py).

Installed only inside a switch-evaluation process (install()); every other split, item and store check is Kolmo's own (super()).
Views: B = switched truth, condition B (the new one); A = switched truth, condition A (the model is not told about the switch);
N = control truth (no switch), condition A; T = as B but the dt slot holds B's own typical frame interval instead of the true interval.
Items report n_frames = exactly the frames one window needs, so train_gen.eval_windows puts the window at the switch (history = stored
frames 0, 2, .., 10, prediction from frame 12). The views of one group share the sampling noise of every window: split_seed returns the
group's fixed noise seed (groups_kolmo.json, noise_seed)."""
import glob, json, math, os, sys
import numpy as np, h5py

HERE = os.path.dirname(os.path.abspath(__file__)); PROJ = os.path.dirname(HERE); PKG = os.path.join(PROJ, "physdial")
sys.path.insert(0, PROJ); sys.path.insert(0, PKG)
import corpus   # noqa: E402

VIEWS = ("B", "A", "N", "T")
EVAL_HIST = 6   # train_gen.EVAL_HIST (asserted in eval_switch against the imported value)
PREFIX = "switch_"
GROUP_SEEDS = {g["name"]: g for f in ("groups_kolmo.json", "groups_kolmo_ms.json") for g in json.load(open(os.path.join(HERE, f)))["groups"]}


def group_of(split):
    """'switch_<group>_<view>' -> (group, view)"""
    return tuple(split[len(PREFIX):].rsplit("_", 1))


class KolmoSwitch(corpus.Kolmo):
    SWITCH_DIR = ""        # set by install()
    SWITCH_FRAMES = 16     # rollout length of the evaluation (train_gen: n_out * eval_chunks)

    def _switch_files(self): return sorted(glob.glob(os.path.join(self.SWITCH_DIR, "*.h5")))

    def splits(self): return super().splits() + [f"{PREFIX}{os.path.basename(f)[:-3]}_{v}" for f in self._switch_files() for v in VIEWS]

    def split_seed(self, split):   # train_gen's evaluation noise seed of a split: the group's noise seed for its views, the default (by name) otherwise
        if split.startswith(PREFIX) and group_of(split)[0] in GROUP_SEEDS: return int(GROUP_SEEDS[group_of(split)[0]]["noise_seed"])
        return super().split_seed(split)

    def items(self, split):
        if not split.startswith(PREFIX): return super().items(split)
        grp, view = group_of(split); assert view in VIEWS, split
        path = os.path.join(self.SWITCH_DIR, grp + ".h5"); need = (EVAL_HIST + self.SWITCH_FRAMES - 1) * self.stride + 1; out = []
        with h5py.File(path, "r") as h:
            a = dict(h.attrs)
            assert int(a["switch_frame"]) == (EVAL_HIST - 1) * self.stride, f"{path}: switch at frame {a['switch_frame']}, the window's last history frame is {(EVAL_HIST - 1) * self.stride}"
            use_A = view in ("A", "N"); pre = "A_" if use_A else "B_"; dt = float(a["B_dt_typical"] if view == "T" else a["dt_frame"])
            nu, amp, multi = float(a[pre + "nu"]), float(a[pre + "amp"]), str(a[pre + "forcing"]) == "multi"
            for k in sorted(x for x in h.keys() if not x.startswith("c")):
                traj = "c" + k if view == "N" else k; g = h[traj]; T, H, W = g["data"].shape[:3]
                assert T >= need, f"{path}/{traj}: {T} frames < {need} needed for {self.SWITCH_FRAMES} predicted frames"
                kf = float(h[k].attrs["A_kf_slot"] if use_A else h[k].attrs["kf_slot"])
                raw = {"nu": nu, "k_f": kf, "amp": amp}
                if multi: raw["multimode"] = 1.0   # as corpus.Kolmo._item (identity text: random multi-mode forcing)
                p = {"log10_nu": math.log10(nu), "kf": kf, "amp": amp, "log10_dt": math.log10(dt)}   # the slots of Kolmo._item
                it = corpus.Item(self.key, path, traj, p, raw, need, (H, W)); it.switch_forcing = "forcing_A" if view == "A" else "forcing"
                out.append(it)
        return out

    def cond_fields(self, it):   # the forcing in effect for this view (switched trajectories carry B's as 'forcing' and A's as 'forcing_A')
        with h5py.File(it.path, "r") as h: return [np.asarray(h[it.traj][getattr(it, "switch_forcing", "forcing")], dtype=np.float32)]


def install_class(cls):
    """use cls for the Kolmogorov family in this process; items of the switch splits bypass the item cache."""
    corpus.FAMILIES["kolmo"] = cls
    orig = corpus.cached_items
    def cached_items(key, split, rebuild=False):
        return corpus.get(key).items(split) if split.startswith(PREFIX) else orig(key, split, rebuild)
    corpus.cached_items = cached_items


def install(switch_dir, frames):
    assert os.path.isdir(switch_dir), switch_dir
    KolmoSwitch.SWITCH_DIR, KolmoSwitch.SWITCH_FRAMES = switch_dir, frames
    install_class(KolmoSwitch)
    return [s for s in KolmoSwitch().splits() if s.startswith(PREFIX)]
