#!/usr/bin/env python
"""Walrus baseline evaluation (torchrun, 1+ GPUs): score a Walrus run (walrus_ft.py) on the shared evaluation windows that walrus_data.py
wrote ($PHYSDIAL_DATA_ROOT/walrus/eval/<family>__<split>__<H>x<W>/, windows.json), with the same protocol and metric names as
train_gen.evaluate_split: 6 history frames (EVAL_HIST), autoregressive rollout of --frames steps at the family's stride, relative L2 per
frame / at 1, 4, 16, per channel at 1, persistence (repeat the last history frame), gradient-energy ratio and band retention of the last
frame. Vorticity families were given to Walrus as velocity: predictions and truth are turned back into vorticity before scoring. Walrus
has no tokenizer: tokenizer_floor@1 is null.

The rollout is Walrus's own (Trainer.rollout_model, prediction_type delta, RMS normalization built from the training corpus, delta
re-normalized every step) re-implemented here without a Trainer; the field-to-index map and model settings come from the run's
extended_config.yaml. --ckpt last|best picks the run's checkpoint; --ckpt coalesced evaluates the released Walrus zero-shot (same map
alignment as train.py). Writes results/<name>.json ({"<family>/<split>": metrics pooled over the split's grids like train_gen, plus
"<family>/<split>@<HxW>" per grid}). Seeded (Walrus's patch jitter / periodic rolls are random even in eval mode).

Environment: WALRUS_SRC (Walrus source checkout, default <repo>/ext/walrus), WALRUS_CKPT_DIR (released checkpoint folder, default
<repo>/ext_models/walrus), PHYSDIAL_DATA_ROOT, PHYSDIAL_EXT_DATA."""
import argparse, copy, glob, json, os, sys, time
HERE = os.path.dirname(os.path.abspath(__file__)); PKG = os.path.dirname(os.path.dirname(HERE)); sys.path.insert(0, os.path.dirname(PKG)); sys.path.insert(0, PKG)
import numpy as np, torch, yaml
import corpus

REPO = os.path.dirname(PKG)
VORT = {"kolmo", "nsu"}   # = walrus_data.VORT (not imported: walrus_data pulls in train_gen / diffusers)


def walrus_env():
    """Walrus source checkout; also sets the defaults of the environment variables the YAML configs in ./cfg interpolate."""
    os.environ.setdefault("WALRUS_CKPT_DIR", os.path.join(REPO, "ext_models", "walrus"))
    os.environ.setdefault("PHYSDIAL_EXT_DATA", os.path.join(REPO, "data", "ext"))
    os.environ.setdefault("PHYSDIAL_DATA_ROOT", os.path.join(REPO, "data"))
    return os.environ.get("WALRUS_SRC", os.path.join(REPO, "ext", "walrus"))


def vel_to_vort(u, v):           # = walrus_data.vel_to_vort
    N, M = u.shape[-2:]; kx = np.fft.fftfreq(N, 1.0 / N)[:, None]; ky = np.fft.fftfreq(M, 1.0 / M)[None, :]
    return np.real(np.fft.ifft2(1j * kx * np.fft.fft2(v) - 1j * ky * np.fft.fft2(u))).astype(np.float32)

EVAL_HIST = 6
from eval_metrics import BANDS, new_acc, accumulate_last_frame, physics_metrics, rel_l2, rel_l2_pooled, rel_l2_demeaned, demeaned_summary, vrmse, linear_extrap, truth_stats, apply_degenerate, nonfinite_to_inf   # the same metric code as train_gen.evaluate_split


def run_dir(name):
    c = sorted(glob.glob(os.path.join(PKG, "ckpt", "walrus_runs", name + "-[a-z][a-z][a-z][a-z][a-z]-*", "finetune", "0")))   # Walrus appends -<the first five letters of the data config name, e.g. physd>-<model>-... to the run name
    if not c: raise SystemExit(f"no run folder for {name} under {os.path.join(PKG, 'ckpt', 'walrus_runs')}")
    return c[-1]


def build(a, dev):
    """model (weights loaded), revin (training-corpus statistics), field-to-index map, run info."""
    src = walrus_env(); sys.path.insert(0, src)
    from hydra import compose, initialize_config_dir
    from hydra.utils import instantiate
    from omegaconf import OmegaConf
    info = {"run": None if a.ckpt == "coalesced" else a.run, "ckpt": a.ckpt}   # zero-shot uses no fine-tuning run (--run is ignored)
    if a.ckpt == "coalesced":   # released Walrus, zero-shot: the training config of walrus_ft.py gives the field map and the model settings
        with initialize_config_dir(config_dir=os.path.join(src, "walrus", "configs"), version_base=None):
            cfg = compose(config_name="config", overrides=[f"hydra.searchpath=[file://{os.path.join(HERE, 'cfg')}]", "data=physdial_corpus", "server=default", "logger=none",
                                                           "distribution=local", "checkpoint=physdial_finetune", "experiment=physdial_finetune", "model=isotropic_model", "trainer=globalnorm",
                                                           "model.projection_dim=48", "model.intermediate_dim=352", "model.hidden_dim=1408", "model.groups=16", "model.processor_blocks=40",
                                                           "model.drop_path=0.0", "model/processor/space_mixing=full_spatial_attention", "model.processor.space_mixing.num_heads=16",
                                                           "model.processor.time_mixing.num_heads=16", "model.causal_in_time=True", "model.jitter_patches=True", "model.override_dimensionality=0",
                                                           "++model.use_periodic_fixed_jitter=True", "++model.input_field_drop=0", "model.gradient_checkpointing_freq=2", "++trainer.epsilon=1e-8",
                                                           "finetuning_mods=all", 'trainer.prediction_type="delta"'])
        run_d = None; info["init"] = "pretrained"
    else:
        if not os.path.exists(os.path.join(HERE, "results", a.run + ".json")): raise SystemExit(f"walrus_ft {a.run} has not finished (no results/{a.run}.json)")
        run_d = run_dir(a.run); cfg = OmegaConf.load(os.path.join(run_d, "extended_config.yaml")); info["run_dir"] = run_d
        info["init"] = json.load(open(os.path.join(HERE, "results", a.run + ".json"))).get("init", "pretrained")
        assert str(cfg.get("physdial_init", "pretrained")) == info["init"], f"{a.run}: results json says init {info['init']}, run config physdial_init {cfg.get('physdial_init')}"
    # training data module: field map (fine-tuned run: the saved one) and the per-dataset normalization statistics Walrus trained with
    override = cfg.data.get("field_index_map_override", {})   # fine-tuned run: the full map train.py saved; zero-shot: the bc-only seed map
    dm_tr = instantiate(cfg.data.module_parameters, world_size=1, rank=0, data_workers=0, well_base_path=cfg.data.well_base_path,
                        field_index_map_override=override, transform=None)
    fmap = {str(k): int(v) for k, v in dm_tr.train_dataset.field_to_index_map.items()}
    if run_d is not None: assert fmap == {str(k): int(v) for k, v in override.items()}, "field map of the training corpus differs from the run's"
    if a.ckpt == "coalesced":
        rel = OmegaConf.load(os.path.join(os.path.dirname(cfg.checkpoint.coalesced_checkpoint_path), "extended_config.yaml"))   # the released config: its trainer.revin
        revin = instantiate(rel.trainer.revin)(dm_tr.train_dataset, dev); info["epsilon"] = float(rel.trainer.get("epsilon") or 1e-5)   # epsilon unset there: the Trainer default 1e-5
    else:
        revin = instantiate(cfg.trainer.revin)(dm_tr.train_dataset, dev); info["epsilon"] = float(cfg.trainer.get("epsilon") or 1e-5)
    info["revin"] = type(revin).__name__
    # acoustic density: zeroed for the released weights (pretraining's field_transforms) and for runs whose data config zeroed it
    wdi = cfg.data.module_parameters.get("well_dataset_info", {})
    run_zero = any(str((v or {}).get("field_transforms", {}).get("density", "")) == "torch.zeros_like" for k, v in wdi.items() if str(k).startswith("acoustic"))
    info["zero_density"] = a.ckpt == "coalesced" or run_zero
    model = instantiate(cfg.model, n_states=max(fmap.values()) + 1)
    if a.ckpt == "coalesced":
        sys.path.insert(0, os.path.join(src, "walrus")); from train import load_from_coalesced_checkpoint
        old = yaml.safe_load(open(cfg.config_override))["data"].get("field_index_map_override", None) if cfg.get("config_override") else None
        model = load_from_coalesced_checkpoint(model, cfg.checkpoint.coalesced_checkpoint_path, fmap, old, align_fields=True)
        if hasattr(model, "add_ft_options") and hasattr(cfg, "finetuning_mods"): model.add_ft_options(cfg.finetuning_mods)
    else:
        if hasattr(model, "add_ft_options") and hasattr(cfg, "finetuning_mods"): model.add_ft_options(cfg.finetuning_mods)   # learnable RoPE etc. before the state dict
        from walrus.trainer.checkpoints import CheckPointLoader
        cp = os.path.join(run_d, "checkpoints", a.ckpt); assert os.path.isdir(cp), f"no checkpoint {cp}"
        ld = CheckPointLoader(save_dir=os.path.join(run_d, "checkpoints"), rank=0, load_checkpoint_path=cp, prioritize_resume=False)
        epoch, vl = ld.load(model, local=True); info.update({"epoch": epoch, "val_loss": None if vl is None else float(vl), "checkpoint": cp})
    model = model.to(dev).eval(); info["n_params"] = sum(p.numel() for p in model.parameters())
    info["n_fields"] = max(fmap.values()) + 1   # field-embedding table size
    return model, revin, fmap, cfg, info


@torch.no_grad()
def rollout(model, revin, formatter, batch, steps, eps, dev):
    """Trainer.rollout_model (validation branch, prediction_type delta) -> y_pred [B, steps, H, W, C], y_ref [B, steps, H, W, C], last input [B, H, W, C]."""
    meta = batch["metadata"]
    batch = {k: (v.to(dev, non_blocking=True) if k not in {"metadata", "boundary_conditions"} else v) for k, v in batch.items()}
    _, y_ref = formatter.process_input(batch, causal_in_time=model.causal_in_time, predict_delta=True, train=False)
    steps = min(steps, y_ref.shape[1]); y_ref = y_ref[:, :steps]; moving = copy.deepcopy(batch); preds = []
    for i in range(steps):
        inputs, _ = formatter.process_input(moving); inputs = list(inputs)
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
    return torch.cat(preds, 1), y_ref, batch["input_fields"][:, -2:]   # the last two input frames (persistence and linear-extrapolation baselines)


def to_phys(x, fam, shape, pfm):
    """Walrus layout [B, T, H, W, 1, C] (2D inflated to depth 1; padded channels such as velocity_z flagged False in pfm; velocity for
    vorticity families) -> our layout [B, T, C, H, W]."""
    x = x[..., 0, :][..., pfm.to(x.device)].permute(0, 1, 4, 2, 3).float()
    if fam.key in VORT: x = torch.from_numpy(vel_to_vort(x[:, :, 0].cpu().numpy(), x[:, :, 1].cpu().numpy()))[:, :, None].to(x.device)
    assert tuple(x.shape[-2:]) == tuple(shape), (tuple(x.shape), shape)
    return x


def eval_datamodule(cfg, ed, ds, fmap, a):
    """one evaluation dataset; built directly (hydra.instantiate would *merge* well_dataset_info with the training corpus's)."""
    from omegaconf import OmegaConf
    from walrus.data import MixedWellDataModule
    mp = OmegaConf.to_container(cfg.data.module_parameters, resolve=True); mp.pop("_target_", None)
    mp.update(well_base_path=os.path.dirname(ed), well_dataset_info={ds: {"include_filters": [], "exclude_filters": [], "path": ed}}, batch_size=a.eval_bs,
              n_steps_input=EVAL_HIST, n_steps_output=a.frames, min_dt_stride=1, max_dt_stride=1, max_samples=1, world_size=1, rank=0, data_workers=a.workers,
              field_index_map_override=fmap, transform=None)
    return MixedWellDataModule(**mp)


def accumulate_dir(ed, fam, model, revin, fmap, cfg, a, dev, eps):
    """one evaluation directory -> accumulators (E, PE [n, frames, C], per-channel last-frame statistics), n windows, dataset name."""
    from walrus.data.well_to_multi_transformer import ChannelsFirstWithTimeFormatter
    ds = [f[:-5] for f in os.listdir(ed) if f.endswith(".yaml") and f != "stats.yaml"][0]
    win = json.load(open(os.path.join(ed, "windows.json"))); shape = tuple(win[0]["shape"]); n_win = len(win)
    dm = eval_datamodule(cfg, ed, ds, fmap, a)
    dset = dm.test_dataloaders(replicas=1, rank=0, full=True)[0].dataset; fmt = ChannelsFirstWithTimeFormatter(); C = len(fam.channels)
    sub = dset.sub_dsets[0]; mn = sub.metadata.dataset_name
    assert len(dset) == len(sub.files_paths) == len(win), f"{ed}: {len(dset)} samples, {len(sub.files_paths)} files, {len(win)} windows (expected one sample per window file)"
    rec = {r_["window"]: r_ for r_ in win}; recs = [rec[int(os.path.basename(f_)[4:-5])] for f_ in sub.files_paths]
    loader = torch.utils.data.DataLoader(dset, sampler=torch.utils.data.BatchSampler(torch.utils.data.SequentialSampler(dset), a.eval_bs, drop_last=False),
                                         batch_size=None, num_workers=a.workers, pin_memory=True)
    zero_c = []
    if a.zero_density and mn.startswith("acoustic_scattering"):   # convolution has no bias, so a zero channel is the same in any field slot
        cn = [c for d_ in sorted(sub.metadata.constant_field_names) for c in sub.metadata.constant_field_names[d_]]; zero_c = [cn.index("density")]
    if hasattr(revin, "name_to_normalization_stats") and mn not in revin.name_to_normalization_stats:   # evaluation datasets keep the base name; Kolmogorov trains as physdial_kolmo_r256 / _r512 (same statistics)
        alt = [k for k in revin.name_to_normalization_stats if k.startswith(mn + "_r")]
        if alt: revin.name_to_normalization_stats[mn] = revin.name_to_normalization_stats[alt[0]]
        else:   # no training statistics under this name: fall back to the evaluation dataset's own stats.yaml
            from walrus.trainer.normalization_strat import RMSGlobalRevNormalization
            print(f"[walrus_eval] WARNING: no training statistics for {mn}; using the evaluation dataset's own stats.yaml", flush=True)
            revin.name_to_normalization_stats[mn] = RMSGlobalRevNormalization(dm.train_dataset, dev).name_to_normalization_stats[mn]
    NF = [0]; E, PE, LE, V, EP, PEP, DM = [], [], [], [], [], [], []; NFW = []; GT, GR, BT, BR = new_acc(C); seen = 0; nrm = rel_l2; TS = ([], [], [])
    DUMP = []; want = getattr(a, "dump_want", None) if getattr(a, "dump_pred", 0) else None   # --dump_pred: windows to save; want = set of (path, traj, first_frame) of the matched run, None = first N in file order
    for batch in loader:
        if seen >= n_win: break
        for j in zero_c: batch["constant_fields"][..., j] = 0.0
        yp, yr, last2 = rollout(model, revin, fmt, batch, a.frames, eps, dev)
        pfm = batch["padded_field_mask"][:yr.shape[-1]].bool()
        P = to_phys(yp, fam, shape, pfm); T = to_phys(yr, fam, shape, pfm); L2 = to_phys(last2, fam, shape, pfm); L = L2[:, -1]
        assert P.shape[2] == C, f"{fam.key}: {P.shape[2]} channels from Walrus, {C} expected"
        e_, v_, ep_, dm_, nf_ = nonfinite_to_inf(P, nrm(P - T, T), vrmse(P, T), rel_l2_pooled(P - T, T), rel_l2_demeaned(P, T)); NF[0] += nf_
        NFW.append((~torch.isfinite(P)).flatten(1).any(1).cpu())
        E.append(e_.cpu()); DM.append(dm_.cpu()); PE.append(nrm(L[:, None] - T, T).cpu()); LE.append(nrm(linear_extrap(L2[:, 0], L, P.shape[1]) - T, T).cpu()); V.append(v_.cpu())
        EP.append(ep_.cpu()); PEP.append(rel_l2_pooled(L[:, None] - T, T).cpu())   # channel-pooled, as train_gen
        keep = max(0, min(P.shape[0], n_win - seen))   # last-frame statistics of the first n_win windows only, like the error arrays
        if keep: accumulate_last_frame(T[:keep], P[:keep], fam.periodic, GT, GR, BT, BR)
        for l_, v_ in zip(TS, truth_stats(T)): l_.append(v_.cpu())
        if getattr(a, "dump_pred", 0):   # history = the last two input frames (our model's K = 2), prediction, truth; physical units, our layout [frames, C, H, W], fp16
            for j in range(keep):
                r_ = recs[seen + j]; key = (r_["path"], str(r_["traj"]), int(r_["first_frame"]))
                if (key in want) if want is not None else (len(DUMP) < a.dump_pred):
                    DUMP.append(({"path": r_["path"], "traj": str(r_["traj"]), "first_frame": int(r_["first_frame"]), "window": int(r_["window"]), "grid": f"{shape[0]}x{shape[1]}"},
                                 L2[j].half().cpu().numpy(), P[j].half().cpu().numpy(), T[j].half().cpu().numpy()))
        seen += P.shape[0]
    return {"E": torch.cat(E)[:n_win], "DM": torch.cat(DM)[:n_win], "PE": torch.cat(PE)[:n_win], "LE": torch.cat(LE)[:n_win], "V": torch.cat(V)[:n_win], "EP": torch.cat(EP)[:n_win], "PEP": torch.cat(PEP)[:n_win], "GT": GT, "GR": GR, "BT": BT, "BR": BR, "n": n_win, "nonfinite": NF[0], "dataset": mn,
            "TS": [torch.cat(l_)[:n_win] for l_ in TS], "recs": recs[:n_win], "NFW": torch.cat(NFW)[:n_win],
            "stride": int(win[0].get("stride", 1)), "grid": f"{shape[0]}x{shape[1]}", "DUMP": DUMP}


def merge(accs):
    """pool the accumulators of several evaluation directories of one (family, split) (Kolmogorov: one per grid), like train_gen does."""
    C = len(accs[0]["GT"]); GT, GR, BT, BR = new_acc(C)
    for ac in accs:
        for ci in range(C): GT[ci] += ac["GT"][ci]; GR[ci] += ac["GR"][ci]; BT[ci] += ac["BT"][ci]; BR[ci] += ac["BR"][ci]
    return {"E": torch.cat([ac["E"] for ac in accs]), "DM": torch.cat([ac["DM"] for ac in accs]), "PE": torch.cat([ac["PE"] for ac in accs]), "LE": torch.cat([ac["LE"] for ac in accs]), "V": torch.cat([ac["V"] for ac in accs]), "EP": torch.cat([ac["EP"] for ac in accs]), "PEP": torch.cat([ac["PEP"] for ac in accs]), "GT": GT, "GR": GR, "BT": BT, "BR": BR,
            "TS": [torch.cat([ac["TS"][i] for ac in accs]) for i in range(3)], "recs": [r_ for ac in accs for r_ in ac["recs"]],
            "NFW": torch.cat([ac["NFW"] for ac in accs]),
            "n": sum(ac["n"] for ac in accs), "nonfinite": sum(ac.get("nonfinite", 0) for ac in accs), "dataset": accs[0]["dataset"], "stride": accs[0]["stride"], "grid": "+".join(ac["grid"] for ac in accs)}


def reduce_acc(ac, fam):
    d = apply_degenerate({k: ac[k] for k in ("E", "PE", "LE", "V", "EP", "PEP", "DM")}, *ac["TS"])   # the same undefined entries as train_gen
    from types import SimpleNamespace as NS_   # trajectory-clustered bootstrap with train_gen's clusters
    groups = [fam.ci_group(NS_(path=r_["path"], traj=r_["traj"], raw=fam._raw(r_["path"]) if hasattr(fam, "_raw") else {})) for r_ in ac["recs"]]
    r = physics_metrics(d["E"], d["PE"], None, ac["GT"], ac["GR"], ac["BT"], ac["BR"], fam.channels, 4, LE=d["LE"], V=d["V"], EP=d["EP"], PEP=d["PEP"], groups=groups, degenerate=d["degenerate"])
    E, PE = d["E"], d["PE"]; n = E.shape[1]; fv = lambda x: None if torch.isnan(x) else float(x)
    r["per_window"] = [{"window": r_["window"], "path": r_["path"], "traj": str(r_["traj"]), "first_frame": r_["first_frame"], "rel_l2@1": fv(torch.nanmean(E[i, 0])),
                        f"rel_l2@{n}": fv(torch.nanmean(E[i, -1])), "rel_l2_pooled@1": fv(d["EP"][i, 0]), f"rel_l2_pooled@{n}": fv(d["EP"][i, -1]),
                        "persistence@1": fv(torch.nanmean(PE[i, 0])), "nonfinite_pred": int(ac["NFW"][i]), "rel_l2_by_frame": [fv(torch.nanmean(E[i, f_])) for f_ in range(n)]} for i, r_ in enumerate(ac["recs"])]
    r.update(demeaned_summary(d["DM"]))
    r.update({"stride": ac["stride"], "walrus_dataset": ac["dataset"], "grids": ac["grid"], "nonfinite_pred_windows": ac.get("nonfinite", 0)}); return r


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--name", required=True); ap.add_argument("--run", default="", help="walrus_ft run name (ignored with --ckpt coalesced)")
    ap.add_argument("--ckpt", default="last", help="last | best | coalesced (released Walrus, zero-shot)")
    ap.add_argument("--families", default="", help="default: every family"); ap.add_argument("--frames", type=int, default=16)
    ap.add_argument("--eval_bs", type=int, default=2); ap.add_argument("--seed", type=int, default=0); ap.add_argument("--workers", type=int, default=2)
    ap.add_argument("--dump_pred", type=int, default=0, help="save history (last 2 input frames), prediction and truth of N windows per split to results/pred/<name>/<family>__<split>.npz in "
                    "train_gen's layout (H<i>, P<i>, T<i> [frames, C, H, W] fp16, meta json); 0 = off")
    ap.add_argument("--dump_match", default="", help="with --dump_pred: a train_gen run whose results/pred/<run>/ dumps name the windows to save (same path, traj, first_frame), so the dumps pair window by window; "
                    "a split without such a file falls back to the first N windows in file order")
    a = ap.parse_args()
    out = os.path.join(HERE, "results", a.name + ".json"); os.makedirs(os.path.dirname(out), exist_ok=True)
    if os.path.exists(out): print("exists", out); sys.exit(0)
    rank, world = int(os.environ.get("RANK", 0)), int(os.environ.get("WORLD_SIZE", 1))
    dev = torch.device(f"cuda:{int(os.environ.get('LOCAL_RANK', 0))}" if torch.cuda.is_available() else "cpu"); t0 = time.time()
    if dev.type == "cuda": torch.cuda.set_device(dev)
    model, revin, fmap, cfg, info = build(a, dev); a.zero_density = info["zero_density"]; print(f"[{a.name}] rank {rank}/{world} model ready ({info}) {time.time() - t0:.0f}s", flush=True)
    root = os.path.join(os.environ.get("PHYSDIAL_DATA_ROOT") or os.path.join(os.path.dirname(PKG), "data"), "walrus", "eval" if a.frames == 16 else f"eval_f{a.frames}")
    res = {"_info": info, "_frames": a.frames, "_eval_hist": EVAL_HIST, "_world": world}
    tasks = []
    for k in ([f for f in a.families.split(",") if f] or list(corpus.FAMILIES)):
        by_split = {}
        for ed in sorted(glob.glob(os.path.join(root, f"{k}__*"))): by_split.setdefault(os.path.basename(ed).split("__")[1], []).append(ed)
        tasks += [(k, sp, eds) for sp, eds in by_split.items()]
    size = lambda eds: sum(os.path.getsize(os.path.join(d, f)) for d in eds for f in os.listdir(d) if os.path.isfile(os.path.join(d, f)))
    load, mine = [0] * world, []
    for i, t in sorted(enumerate(tasks), key=lambda x: -size(x[1][2])):   # largest first onto the least-loaded rank (bytes ~ windows x grid x frames)
        j = load.index(min(load)); load[j] += size(t[2])
        if j == rank: mine.append((i, t))
    for i, (k, sp, eds) in sorted(mine):   # the same keys as train_gen: "<family>/<split>", every grid of the split pooled; per-grid entries as "<family>/<split>@<HxW>"
        fam = corpus.get(k)
        sd = a.seed + int.from_bytes(f"{k}/{sp}".encode(), "little") % 100003   # per-split seed (Walrus's patch jitter / periodic rolls draw random
        torch.manual_seed(sd); np.random.seed(sd % 2 ** 32)   # numbers in eval mode): a split's numbers do not depend on which splits ran before it or on the GPU count
        a.dump_want = None
        if a.dump_pred and a.dump_match:
            mf = os.path.join(PKG, "results", "pred", a.dump_match, f"{k}__{sp}.npz")
            if os.path.exists(mf):
                with np.load(mf) as z_: a.dump_want = {(w_["path"], str(w_["traj"]), int(w_["first_frame"])) for w_ in json.loads(str(z_["meta"]))["windows"]}
            else: print(f"[{a.name}] --dump_match: no {mf}; first {a.dump_pred} windows of {k}/{sp} in file order", flush=True)
        t1 = time.time(); accs = [accumulate_dir(ed, fam, model, revin, fmap, cfg, a, dev, info["epsilon"]) for ed in eds]
        dump = [d_ for ac in accs for d_ in ac.pop("DUMP")]; dump = dump if a.dump_want is not None else dump[:a.dump_pred]   # split over several grids (Kolmogorov): N in all, not per grid
        if a.dump_pred and dump:   # one file per (family, split), written by the one rank that owns the task (no concurrent writers)
            dp = os.path.join(HERE, "results", "pred", a.name); os.makedirs(dp, exist_ok=True); arr = {}
            for i_, (w_, h_, p_, t_) in enumerate(dump): arr[f"H{i_}"], arr[f"P{i_}"], arr[f"T{i_}"] = h_, p_, t_
            meta = {"windows": [w_ for w_, *_ in dump], "channels": list(fam.channels), "stride": accs[0]["stride"], "model": "walrus_eval", "matched_to": a.dump_match or None}
            fn = os.path.join(dp, f"{k}__{sp}.npz"); np.savez_compressed(fn + ".tmp.npz", meta=json.dumps(meta, default=str), **arr); os.replace(fn + ".tmp.npz", fn)
        r = reduce_acc(merge(accs) if len(accs) > 1 else accs[0], fam); r["seconds"] = time.time() - t1; res[f"{k}/{sp}"] = r
        if len(accs) > 1:
            for ac in accs: res[f"{k}/{sp}@{ac['grid']}"] = reduce_acc(ac, fam)
        rl = r["rel_l2@%d" % r["frames"]]; f4 = lambda x: "nan" if x is None else f"{x:.4f}"
        print(f"[{a.name}] {k}/{sp}: n {r['n_windows']} rel_l2@1 {f4(r['rel_l2@1'])} @{r['frames']} {f4(rl)} pers@1 {f4(r['persistence@1'])} "
              f"g {r['last_frame_grad_ratio']:.3f} top {r['last_frame_band_retention'][BANDS[-1]]:.2f} ({r['seconds']:.0f}s)", flush=True)
    res["_seconds"] = time.time() - t0
    if world > 1:
        tag = os.environ.get("TORCHELASTIC_RUN_ID") or os.environ.get("SLURM_JOB_ID", "x"); part = f"{out}.{tag}.part{rank}of{world}"
        json.dump(res, open(part + ".tmp", "w"), indent=1); os.replace(part + ".tmp", part)
        if rank != 0: print(f"[{a.name}] rank {rank}: {len(mine)} splits done", flush=True); return
        parts = [f"{out}.{tag}.part{r}of{world}" for r in range(world)]
        while not all(os.path.exists(p) for p in parts): time.sleep(10)   # torchrun stops the whole group if another rank fails
        for p in parts[1:]:
            q = json.load(open(p)); res.update({k: v for k, v in q.items() if not k.startswith("_")}); res["_seconds"] = max(res["_seconds"], q["_seconds"])
        n = sum(1 for k in res if not k.startswith("_") and "@" not in k)
        assert n == len(tasks), f"merged {n} splits, expected {len(tasks)}"
    json.dump(res, open(out + ".tmp", "w"), indent=1); os.replace(out + ".tmp", out)
    for p in (parts if world > 1 else []): os.remove(p)
    print("saved", out, flush=True)


if __name__ == "__main__":
    main()
