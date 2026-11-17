"""Training on Kolmogorov switch trajectories, installed only inside a switch-training or encoding process.

- KolmoSwitchTrain(switch_corpus.KolmoSwitch): the Kolmogorov family plus training splits of switch trajectories (gen_switch groups_kolmo_train.json ->
  switch_train; groups_kolmo_ref.json -> switch_ref, only for the in-range reference). An item is one switched trajectory ('0000'.., never the
  control 'c..'), labelled with the condition in force after the switch (B) and the true frame interval. The training splits are added to
  corpus.TRAIN_ONLY_SPLITS in this process, so they never become evaluation columns.
- make_store(...): train_gen.Store subclass that appends the latent groups of the switch splits ($PHYSDIAL_DATA_ROOT/switch/latents/<store>/kolmo/<split>/)
  and exposes only frames OFF.., OFF = SWITCH - (k_in - 1) * stride (every training window then contains or follows the switch), at most 2 H + span frames.
- capped_mix(...): wraps train_gen.Batches.mix so that the switch groups take at most a given share of the family's group draws."""
import glob, json, math, os, sys
import numpy as np, h5py

HERE = os.path.dirname(os.path.abspath(__file__)); PROJ = os.path.dirname(HERE); PKG = os.path.join(PROJ, "physdial")
sys.path.insert(0, HERE); sys.path.insert(0, PROJ); sys.path.insert(0, PKG)
import corpus, switch_corpus   # noqa: E402

SWITCH = 10   # gen_switch.SWITCH: the last stored frame under condition A (asserted per file)
TRAIN_SUBDIRS = {"switch_train": "kolmo_train", "switch_ref": "kolmo_ref"}


def data_dir(sub): return os.path.join(os.environ["PHYSDIAL_DATA_ROOT"], "switch", sub)


def latent_root(store): return os.path.join(os.environ["PHYSDIAL_DATA_ROOT"], "switch", "latents", store)


class KolmoSwitchTrain(switch_corpus.KolmoSwitch):
    TRAIN_SPLITS = ()   # set by install_train()

    def splits(self): return corpus.Kolmo.splits(self) + list(self.TRAIN_SPLITS)

    def items(self, split):
        if split not in self.TRAIN_SPLITS: return super().items(split)
        fs = sorted(glob.glob(os.path.join(data_dir(TRAIN_SUBDIRS[split]), "*.h5"))); assert fs, f"no switch files for {split} in {data_dir(TRAIN_SUBDIRS[split])}"
        out = []
        for path in fs:
            with h5py.File(path, "r") as h:
                a = dict(h.attrs); assert int(a["switch_frame"]) == SWITCH, f"{path}: switch at {a['switch_frame']}"
                nu, amp, multi, dt = float(a["B_nu"]), float(a["B_amp"]), str(a["B_forcing"]) == "multi", float(a["dt_frame"])
                for k in sorted(x for x in h.keys() if not x.startswith("c")):
                    T, H, W = h[k]["data"].shape[:3]; kf = float(h[k].attrs["kf_slot"])
                    raw = {"nu": nu, "k_f": kf, "amp": amp}
                    if multi: raw["multimode"] = 1.0   # as corpus.Kolmo._item
                    p = {"log10_nu": math.log10(nu), "kf": kf, "amp": amp, "log10_dt": math.log10(dt)}
                    it = corpus.Item(self.key, path, k, p, raw, T, (H, W)); it.switch_forcing = "forcing"   # B's forcing (switched trajectories)
                    out.append(it)
        return out


def install_train(splits):
    """use the training variant for the Kolmogorov family in this process (train_gen's end-of-run evaluation skips TRAIN_ONLY_SPLITS)."""
    assert set(splits) <= set(TRAIN_SUBDIRS), splits
    KolmoSwitchTrain.TRAIN_SPLITS = tuple(splits)
    corpus.TRAIN_ONLY_SPLITS = tuple(corpus.TRAIN_ONLY_SPLITS) + tuple(s_ for s_ in splits if s_ not in corpus.TRAIN_ONLY_SPLITS)
    switch_corpus.install_class(KolmoSwitchTrain)


def make_store(base, store, splits, k_in, stride, H, span):
    """train_gen.Store subclass (see module doc). base = train_gen.Store; span = stored frames of one training window ((L - 1) * stride + 1)."""
    off = SWITCH - (k_in - 1) * stride; assert off >= 0
    nf_max = 2 * H + span if H > 0 else None
    root = latent_root(store)

    class SwitchStore(base):
        def __init__(s, root_, family, splits_):
            super().__init__(root_, family, splits_); s.sw = set()
            if family != "kolmo" or "train" not in splits_: return
            for sp in splits:
                p = os.path.join(root, family, sp, "index.json"); assert os.path.exists(p), f"no switch store {p}"
                ix = json.load(open(p))
                assert ix.get("tok") == json.load(open(os.path.join(root_, family, "train", "index.json"))).get("tok"), f"{p}: tokenizer {ix.get('tok')} differs from the main store's"
                assert bool(ix.get("vae_wrap", False)) == bool(s.wrap), f"{p}: vae_wrap {ix.get('vae_wrap')} vs main {s.wrap}"
                for gid, meta in ix["groups"].items():
                    its = []
                    for it in ix["items"]:
                        if it["group"] != int(gid): continue
                        n = it["n_frames"] - off
                        if nf_max: n = min(n, nf_max)
                        assert n >= span, f"{it['path']} {it['traj']}: {it['n_frames']} frames, too short for one window after frame {off}"
                        its.append(dict(it, n_frames=n))
                    if its: s.groups.append((sp, int(gid), meta, its, os.path.join(root, family, sp))); s.sw.add(len(s.groups) - 1)

        def arrays(s, gi):
            Z, S, Fz = super().arrays(gi)
            return (Z[:, off:], S[:, off:], Fz) if gi in s.sw else (Z, S, Fz)

    SwitchStore.SWITCH_DATA = {"store_root": root, "splits": list(splits), "off": off, "nf_max": nf_max}
    return SwitchStore


def capped_mix(orig, share):
    """wrap train_gen.Batches.mix: within each family, scale the switch groups (split switch_*) so they take at most `share` of the family's
    group draws. Family weights (nwin, train split only) are untouched; bbar is recomputed."""
    def mix(s, stores):
        gw, nwin, bbar = orig(s, stores)
        for k, st in stores.items():
            if not st.groups: continue
            x = np.array([g[0].startswith("switch_") for g in st.groups]); w = gw[k]
            if not x.any() or x.all(): continue
            fx = w[x].sum() / w.sum()
            if fx > share: w = w.copy(); w[x] *= share * w[~x].sum() / ((1 - share) * w[x].sum()); gw[k] = w
            def B_of(meta): zc, h, ww = meta["latent"]; return int(max(1, min(s.max_bs, s.tokens // max(1, s.n_out * (h // 2) * (ww // 2)))))
            bbar[k] = float(sum(wg * B_of(g[2]) for wg, g in zip(gw[k] / gw[k].sum(), st.groups)))
            wi_ = __import__("torch").utils.data.get_worker_info()
            if s.log and (wi_ is None or wi_.id == 0): s.log(f"switch groups of {k}: natural share {fx:.3f} -> {gw[k][x].sum() / gw[k].sum():.3f} (cap {share})")
        return gw, nwin, bbar
    return mix
