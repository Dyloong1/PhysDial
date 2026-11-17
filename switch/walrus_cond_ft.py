#!/usr/bin/env python
"""Conditional Walrus fine-tune. A copy of physdial/baselines/walrus/walrus_ft.py with only these differences:
  - the Hydra search path also holds ./walrus_cfg (data configs physdial_cond_corpus / physdial_cond_corpus_anchor / physdial_cond_corpus_anchor_sw: the
    Kolmogorov datasets with the constant condition fields of wcond.py; every other dataset and the dataset order as physdial_corpus / _anchor);
  - --data defaults to physdial_cond_corpus; the result JSON goes to ./results; the run config records ++switch_cond=<the wcond fields>; the
    experiment folder is physdial/ckpt/walrus_runs;
  - always from the released weights or a run's (--init_from, whose JSON is read from ./results, else from the Walrus results directory);
  - on switch data (a *_sw data config) the loss of the history positions that precede the switch is masked (Walrus's causal_in_time
    training scores every history position); switch windows are recognised by a fingerprint of their input frames.
Walrus settings (as walrus_ft.py): the released trainer settings (sample-wise RMS normalisation), acoustic density zeroed as in pretraining,
--epochs 51 --max_samples 500. Environment: WALRUS_SRC (Walrus source checkout), WALRUS_CKPT_DIR (released walrus.pt and
extended_config.yaml), PHYSDIAL_DATA_ROOT, PHYSDIAL_EXT_DATA."""
import argparse, glob, json, os, re, runpy, sys, time
HERE = os.path.dirname(os.path.abspath(__file__)); PROJ = os.path.dirname(HERE); PKG = os.path.join(PROJ, "physdial"); WDIR = os.path.join(PKG, "baselines", "walrus"); sys.path.insert(0, PROJ)
sys.path.insert(0, HERE)
import wcond   # noqa: E402

ap = argparse.ArgumentParser(); ap.add_argument("--name", required=True); ap.add_argument("--epochs", type=int, default=51)
ap.add_argument("--lr", type=float, default=1e-4); ap.add_argument("--batch", type=int, default=2); ap.add_argument("--workers", type=int, default=8)
ap.add_argument("--max_samples", type=int, default=500)
ap.add_argument("--data", default="physdial_cond_corpus", help="data config in ./walrus_cfg/data (physdial_cond_corpus, physdial_cond_corpus_anchor, physdial_cond_corpus_anchor_sw) or physdial/baselines/walrus/cfg/data")
ap.add_argument("--init_from", default="", help="walrus_ft run name: continue from its last saved epoch (model weights only)")
ap.add_argument("--warmup_epochs", type=int, default=0, help="0 = the config's (20); a continuation sets its own (the schedule spans --epochs). Walrus steps "
                "its scheduler once per epoch, after the epoch: epoch 1 runs at warmup_lr_factor 0.01 x lr whatever this value")
ap.add_argument("--cooldown_epochs", type=int, default=0, help="0 = the config's (10). NOT the number of reduced epochs: Walrus's off-ramp "
                "factor is 1 - ((e - 2 - (epochs - C)) / C)^0.5 for epoch e, so only C - 2 epochs are reduced and the last one ends at 1 - ((C-2)/C)^0.5 "
                "of the end-of-decay lr; the per-epoch lr table is printed at start")
ap.add_argument("--sw_min_batches", type=int, default=300, help="abort if no switch window was recognised in this many Kolmogorov training batches (per rank)")
a = ap.parse_args()
out = os.path.join(HERE, "results", a.name + ".json"); os.makedirs(os.path.dirname(out), exist_ok=True)
rank = int(os.environ.get("RANK", 0))
if os.path.exists(out): print("exists", out); sys.exit(0)
src = os.environ.get("WALRUS_SRC", os.path.join(PROJ, "ext", "walrus")); exp_dir = os.path.join(PKG, "ckpt", "walrus_runs")
os.environ.setdefault("WALRUS_CKPT_DIR", os.path.join(PROJ, "ext_models", "walrus")); os.environ.setdefault("PHYSDIAL_EXT_DATA", os.path.join(PROJ, "data", "ext"))
os.environ.setdefault("PHYSDIAL_DATA_ROOT", os.path.join(PROJ, "data"))   # read by the ${oc.env:...} paths of the Walrus configs
dist = "ddp" if int(os.environ.get("WORLD_SIZE", 1)) > 1 else "local"
overrides = [
    f"hydra.searchpath=[file://{os.path.join(WDIR, 'cfg')},file://{os.path.join(HERE, 'walrus_cfg')}]", f"distribution={dist}", "model=isotropic_model", f"name={a.name}", "trainer=defaults",
    "trainer.grad_acc_steps=1", "server=default", "optimizer=adam", f"optimizer.lr={a.lr}", "logger=none", "trainer.enable_amp=False",
    "model.gradient_checkpointing_freq=2", "trainer.log_interval=200", "trainer.clip_gradient=10", f"data.module_parameters.batch_size={a.batch}",
    "data.module_parameters.n_steps_input=6", "data.module_parameters.n_steps_output=1", f"data.module_parameters.max_samples={a.max_samples}",
    "model.projection_dim=48", "model.intermediate_dim=352", "model.hidden_dim=1408", "model.groups=16", "model.processor_blocks=40", "model.drop_path=0.0",
    "model/processor/space_mixing=full_spatial_attention", "model.processor.space_mixing.num_heads=16", "model.processor.time_mixing.num_heads=16",
    "model.causal_in_time=True", "model.jitter_patches=True", "trainer.short_validation_length=20", "trainer.max_rollout_steps=16",
    "lr_scheduler=inv_sqrt_w_sqrt_ramps_longer", "trainer.val_frequency=5", "trainer.rollout_val_frequency=10", "data.module_parameters.min_dt_stride=1",
    "data.module_parameters.max_dt_stride=1", 'trainer.prediction_type="delta"', f"data={a.data}", f"trainer.max_epoch={a.epochs}", f"data_workers={a.workers}",
    "model.override_dimensionality=0", "auto_resume=True", "checkpoint=physdial_finetune", "experiment=physdial_finetune", "++model.use_periodic_fixed_jitter=True",
    "++model.input_field_drop=0", "++trainer.skip_spectral_metrics=True", "finetuning_mods=all", f"++experiment_dir={exp_dir}",
    "++trainer.debug_mode=True",   # keeps every validation short (20 batches / dataset) instead of a full pass at the final epoch
    "trainer.image_validation=False", "trainer.video_validation=False"]
overrides.append("++data.module_parameters.well_dataset_info.acoustic_scattering_inclusions.field_transforms.density=torch.zeros_like")
overrides.append("++switch_cond='" + "+".join(wcond.COND) + "'")   # recorded in the run's extended_config.yaml
if a.warmup_epochs: overrides.append(f"lr_scheduler.warmup_epochs={a.warmup_epochs}")
if a.cooldown_epochs: overrides.append(f"lr_scheduler.cooldown_epochs={a.cooldown_epochs}")
def _walrus_lr(e, lr, E, W, C, wf=0.01, off=128):   # walrus/optim/schedulers.py InverseSqrtLinearWarmupSqrtCooldown closed form; epoch e (1-based) trains at last_epoch e - 1
    k = e - 1
    if k <= W: return lr * (wf + (1 - wf) * min(k, W) / W)
    if k <= E - C: return lr / (1 + (k - W + off) ** 0.5 - off ** 0.5)
    return (1 - ((k - (E - C) - 1) / C) ** 0.5) * lr / (1 + (off + E - C) ** 0.5 - off ** 0.5)
print("[walrus_ft] lr per epoch: " + " ".join(f"{e}:{_walrus_lr(e, a.lr, a.epochs, a.warmup_epochs or 20, a.cooldown_epochs or 10):.2e}" for e in range(1, a.epochs + 1)), flush=True)
SRC_CKPT = None
if a.init_from:   # the source run's last epoch: its highest checkpoints/step_<epoch>, which must be the epoch count its result json records
    src_runs = glob.glob(os.path.join(exp_dir, glob.escape(a.init_from) + "-[a-z][a-z][a-z][a-z][a-z]-*", "finetune", "*", "checkpoints"))
    assert len(src_runs) == 1, f"--init_from {a.init_from}: expected one run folder, found {src_runs}"
    steps = sorted((int(m.group(1)), d) for d in glob.glob(os.path.join(glob.escape(src_runs[0]), "step_*")) if (m := re.fullmatch(r"step_(\d+)", os.path.basename(d))))
    src_json = json.load(open(next(p_ for p_ in (os.path.join(HERE, "results", a.init_from + ".json"), os.path.join(WDIR, "results", a.init_from + ".json")) if os.path.exists(p_))))
    assert steps and steps[-1][0] == src_json["epochs"], f"--init_from {a.init_from}: last checkpoint {steps[-1:]} != its {src_json['epochs']} epochs"
    assert os.path.exists(os.path.join(steps[-1][1], "full_checkpoint.pt")), f"no full_checkpoint.pt in {steps[-1][1]}"
    SRC_CKPT = steps[-1][1]
    overrides += ["checkpoint.coalesced_checkpoint_path=null", f"checkpoint.load_checkpoint_path='{SRC_CKPT}'", f"++physdial_init_from={a.init_from}"]   # quoted: the folder name has [ ] (Hydra list syntax); saved in the run's extended_config.yaml; walrus_eval checks it against results/<name>.json
os.environ.setdefault("HYDRA_FULL_ERROR", "1"); os.environ.setdefault("HDF5_USE_FILE_LOCKING", "FALSE")
t0 = time.time(); os.chdir(os.path.join(src, "walrus")); sys.path.insert(0, src)
if dist == "ddp":
    import functools, walrus.utils.distribution_utils as _du   # every rank starts from rank 0's weights: they are broadcast at wrap time
    _du.FSDP = functools.partial(_du.FSDP, sync_module_states=True)   # (the model is already on its GPU there)
SWM = {"on": False}
if "_sw" in a.data:   # switch data: mask the loss of the pre-switch history positions
    import numpy as np, h5py, torch
    xr = os.path.join(os.environ["PHYSDIAL_DATA_ROOT"], "switch", "walrus", "datasets"); fps, npre = [], []
    for f_ in sorted(glob.glob(os.path.join(xr, "physdial_kolmo_anchor_sw_r*", "data", "train", "sw__*.hdf5"))):
        if os.path.islink(f_): continue   # the __rep<r> copies are links to the same file
        reps = [x for x in glob.glob(f_[:-5] + "__rep*.hdf5") if os.path.islink(x) and os.readlink(x) == os.path.basename(f_)]
        assert len(reps) == 5, f"{f_}: {len(reps)} rep links, walrus_sw_data writes REP - 1 = 5"
        with h5py.File(f_, "r") as h_:
            v_ = h_["t1_fields"]["velocity"]; n_, T_ = v_.shape[:2]; assert T_ == 30, (f_, v_.shape)   # walrus_sw_data NW 24: stored frames 0, 2,.., 58
            for i_ in range(n_):
                ss_ = (np.asarray(v_[i_], np.float64) ** 2).sum(axis=(1, 2, 3))
                for j_ in range(T_ - 6): fps.append(ss_[j_:j_ + 6]); npre.append(5 if j_ <= 5 else 0)   # window j: input file frames j .. j + 5
    assert fps, f"switch-window loss mask: no switch file under {xr}"
    SWM.update(on=True, fp=np.array(fps), npre=np.array(npre), kolmo=0, samples=0, matched=0, hist=[0, 0], masked_positions=0, by_dataset={}, table=len(fps), by_window=[0] * 24)   # hist: [windows with the switch, after it]
    print(f"[walrus_cond_ft] rank {rank}: switch-window loss mask on: {len(fps)} switch windows in the fingerprint table", flush=True)
    import walrus.trainer.training as _WT
    _rm0 = _WT.Trainer.rollout_model

    def _rollout_model(self, model, batch, formatter, train=True, fake_pass=False):
        out_ = _rm0(self, model, batch, formatter, train=train, fake_pass=fake_pass)
        if not train or fake_pass or not str(batch["metadata"].dataset_name).startswith("physdial_kolmo"): return out_
        y_pred, y_ref = out_; x_ = batch["input_fields"]
        fp = (x_.detach().double() ** 2).flatten(2).sum(-1).cpu().numpy()   # [B, T_in]
        rel = np.abs(SWM["fp"][None] - fp[:, None]).max(-1) / np.maximum(fp.max(-1)[:, None], 1e-30)   # [B, table]
        jj = rel.argmin(1); hit = rel[np.arange(len(jj)), jj] < 1e-6; npre_ = np.where(hit, SWM["npre"][jj], 0)
        SWM["kolmo"] += 1; SWM["samples"] += len(jj); SWM["matched"] += int(hit.sum())
        bd = SWM["by_dataset"].setdefault(str(batch["metadata"].dataset_name), [0, 0]); bd[0] += len(jj); bd[1] += int(hit.sum())
        for q_ in SWM["npre"][jj][hit]: SWM["hist"][0 if q_ > 0 else 1] += 1
        for t_ in jj[hit]: SWM["by_window"][int(t_) % 24] += 1
        if SWM["kolmo"] == a.sw_min_batches and SWM["matched"] == 0:
            print(f"[walrus_cond_ft] FATAL switch-window loss mask: no switch window recognised in {a.sw_min_batches} Kolmogorov batches (rank {rank}); the fingerprints do not match the loader", flush=True)
            os._exit(3)
        if SWM["kolmo"] % 500 == 0 and rank == 0:
            print(f"[walrus_cond_ft] switch mask: {SWM['matched']} of {SWM['samples']} Kolmogorov samples are switch windows; by window {SWM['hist']}", flush=True)
        if npre_.max() > 0:
            assert model.causal_in_time and y_pred.shape[1] == x_.shape[1] and y_ref.shape == y_pred.shape, (tuple(y_pred.shape), tuple(x_.shape), tuple(y_ref.shape))
            m_ = torch.as_tensor(np.arange(y_pred.shape[1])[None] < npre_[:, None], device=y_pred.device)
            SWM["masked_positions"] += int(m_[:, 1:].sum())
            y_pred = torch.where(m_.view(*m_.shape, *([1] * (y_pred.dim() - 2))), y_ref.detach().to(y_pred.dtype), y_pred)
        return y_pred, y_ref

    _WT.Trainer.rollout_model = _rollout_model

sys.argv = [os.path.join(src, "walrus", "train.py")] + overrides
try:
    runpy.run_path(sys.argv[0], run_name="__main__")
except SystemExit as e:
    if e.code not in (0, None): raise
if rank == 0:
    json.dump({"name": a.name, "epochs": a.epochs, "lr": a.lr, "batch_per_gpu": a.batch, "max_samples": a.max_samples, "world": int(os.environ.get("WORLD_SIZE", 1)),
               "trainer": "defaults", "data": a.data, "zero_density": True,
               "init": "pretrained", "init_from": a.init_from or None, "init_ckpt": SRC_CKPT,
               "warmup_epochs": a.warmup_epochs or None, "cooldown_epochs": a.cooldown_epochs or None,
               "experiment_dir": exp_dir, "wall_h": (time.time() - t0) / 3600, "overrides": overrides,
               "switch_loss_mask": {k_: SWM[k_] for k_ in ("on", "kolmo", "samples", "matched", "hist", "masked_positions", "by_dataset", "table", "by_window") if k_ in SWM}}, open(out, "w"), indent=1)   # rank 0's counts, last process only (auto_resume restarts them)
    print("saved", out, flush=True)
