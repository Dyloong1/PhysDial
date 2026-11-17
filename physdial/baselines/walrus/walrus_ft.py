#!/usr/bin/env python
"""Walrus baseline: fine-tune the released Walrus (PolymathicAI, 1.3B) on the corpus with Walrus's own training code
(torchrun over the GPUs). This wrapper runs walrus/train.py (source checkout in $WALRUS_SRC, default <repo>/ext/walrus) with the settings of
Walrus's fine-tuning example (walrus/run_scripts/finetuning_example_distributed_walrus.sh) and the configs in ./cfg (data = physdial_corpus: The
Well families minus the held-out conditions + the own families written by walrus_data.py; physdial_corpus_anchor: the same with the Kolmogorov
amplitude-anchor trajectories), with the released trainer settings (sample-wise RMS normalisation) and the acoustic density zeroed as in
Walrus's pretraining. Walrus gets 6 history frames (its default) and no parameter conditioning (it has no such input): a
history-conditioned foundation-model baseline.
--epochs sets the length (a Walrus "epoch" is max_samples batches per rank); the base run is --epochs 51 --max_samples 500 on 4 GPUs (lr
schedule: 20 warm-up + 10 cool-down epochs); a preemption resumes from the last epoch via auto_resume. Writes results/<name>.json (experiment
folder, epochs, wall time); evaluation: walrus_eval.py.
--init random: the from-scratch control (Walrus architecture, same data / settings / budget, random weights). train.py runs unchanged, including
load_from_coalesced_checkpoint, but the final model.load_state_dict(strict=True) only checks keys and shapes against the released weights and
copies nothing (_check_only below), so the architecture is proven to be the released one. The weights stay at Walrus's own init (rank r uses
torch.manual_seed(r); FSDP sync_module_states broadcasts rank 0's). The run config gets physdial_init: random.
Environment: WALRUS_SRC, WALRUS_CKPT_DIR (folder with the released walrus.pt and extended_config.yaml, default <repo>/ext_models/walrus),
PHYSDIAL_DATA_ROOT (default <repo>/data), PHYSDIAL_EXT_DATA (default <repo>/data/ext); the YAML configs in ./cfg interpolate the last three."""
import argparse, glob, json, os, re, runpy, sys, time
HERE = os.path.dirname(os.path.abspath(__file__)); PKG = os.path.dirname(os.path.dirname(HERE)); REPO = os.path.dirname(PKG); sys.path.insert(0, REPO)

ap = argparse.ArgumentParser(); ap.add_argument("--name", required=True); ap.add_argument("--epochs", type=int, default=51)
ap.add_argument("--lr", type=float, default=1e-4); ap.add_argument("--batch", type=int, default=2); ap.add_argument("--workers", type=int, default=8)
ap.add_argument("--max_samples", type=int, default=500)
ap.add_argument("--data", default="physdial_corpus", help="data config in ./cfg/data")
ap.add_argument("--init", default="pretrained", choices=["pretrained", "random"], help="random: released architecture, released weights not loaded (see the docstring)")
ap.add_argument("--init_from", default="", help="walrus_ft run name: continue from its last saved epoch (model weights only)")
ap.add_argument("--warmup_epochs", type=int, default=0, help="0 = the config's (20); a continuation sets its own (the schedule spans --epochs). Walrus steps "
                "its scheduler once per epoch, after the epoch: epoch 1 runs at warmup_lr_factor 0.01 x lr whatever this value")
ap.add_argument("--cooldown_epochs", type=int, default=0, help="0 = the config's (10). NOT the number of reduced epochs: Walrus's off-ramp "
                "factor is 1 - ((e - 2 - (epochs - C)) / C)^0.5 for epoch e, so only C - 2 epochs are reduced and the last one ends at 1 - ((C-2)/C)^0.5 "
                "of the end-of-decay lr; the per-epoch lr table is printed at start")
a = ap.parse_args()
assert not (a.init_from and a.init == "random"), "--init_from and --init random exclude each other"
out = os.path.join(HERE, "results", a.name + ".json"); os.makedirs(os.path.dirname(out), exist_ok=True)
rank = int(os.environ.get("RANK", 0))
if os.path.exists(out): print("exists", out); sys.exit(0)
os.environ.setdefault("WALRUS_CKPT_DIR", os.path.join(REPO, "ext_models", "walrus")); os.environ.setdefault("PHYSDIAL_EXT_DATA", os.path.join(REPO, "data", "ext"))
os.environ.setdefault("PHYSDIAL_DATA_ROOT", os.path.join(REPO, "data"))
src = os.environ.get("WALRUS_SRC", os.path.join(REPO, "ext", "walrus")); exp_dir = os.path.join(PKG, "ckpt", "walrus_runs")
dist = "ddp" if int(os.environ.get("WORLD_SIZE", 1)) > 1 else "local"
overrides = [
    f"hydra.searchpath=[file://{os.path.join(HERE, 'cfg')}]", f"distribution={dist}", "model=isotropic_model", f"name={a.name}", "trainer=defaults",
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
    "++trainer.debug_mode=True",   # keeps every validation short (20 batches / dataset) instead of a full pass over the largest dataset
    "trainer.image_validation=False", "trainer.video_validation=False"]
overrides.append("++data.module_parameters.well_dataset_info.acoustic_scattering_inclusions.field_transforms.density=torch.zeros_like")
if a.init == "random": overrides.append("++physdial_init=random")
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
    src_json = json.load(open(os.path.join(HERE, "results", a.init_from + ".json")))
    assert steps and steps[-1][0] == src_json["epochs"], f"--init_from {a.init_from}: last checkpoint {steps[-1:]} != its {src_json['epochs']} epochs"
    assert os.path.exists(os.path.join(steps[-1][1], "full_checkpoint.pt")), f"no full_checkpoint.pt in {steps[-1][1]}"
    SRC_CKPT = steps[-1][1]
    overrides += ["checkpoint.coalesced_checkpoint_path=null", f"checkpoint.load_checkpoint_path='{SRC_CKPT}'", f"++physdial_init_from={a.init_from}"]   # quoted: the folder name has [ ] (Hydra list syntax); saved in the run's extended_config.yaml; walrus_eval checks it against results/<name>.json
for p_ in glob.glob(os.path.join(exp_dir, a.name + "-[a-z][a-z][a-z][a-z][a-z]-*", "finetune", "*", "extended_config.yaml")):   # auto_resume continues any folder of this name:
    prev = "random" if re.search(r"(?m)^physdial_init: random\s*$", open(p_).read()) else "pretrained"   # never across --init
    if prev != a.init: raise SystemExit(f"{p_} was trained with --init {prev}; auto_resume would continue it as --init {a.init}. Use another --name.")
os.environ.setdefault("HYDRA_FULL_ERROR", "1"); os.environ.setdefault("HDF5_USE_FILE_LOCKING", "FALSE")
t0 = time.time(); os.chdir(os.path.join(src, "walrus")); sys.path.insert(0, src)
if dist == "ddp":
    import functools, walrus.utils.distribution_utils as _du   # new field-embedding columns are initialised randomly per process: rank 0's
    _du.FSDP = functools.partial(_du.FSDP, sync_module_states=True)   # weights are broadcast at wrap time (the model is already on its GPU there)
ARCH = {}   # --init random: set by _check_only when train.py's load of the released weights was intercepted (this process)
if a.init == "random":
    import torch
    torch.manual_seed(rank)   # Walrus seeds nothing; rank 0's init (seed 0) is what sync_module_states broadcasts, other ranks keep distinct jitter streams
    from omegaconf import OmegaConf
    released = os.path.realpath(OmegaConf.to_container(OmegaConf.load(os.path.join(HERE, "cfg", "checkpoint", "physdial_finetune.yaml")), resolve=True)["coalesced_checkpoint_path"])

    class _Released(dict): pass   # marks the released state dict; align_checkpoint_with_field_to_index_map edits it in place and returns the same object

    _torch_load, _module_load = torch.load, torch.nn.Module.load_state_dict

    def _load_marked(f, *x, **k):
        obj = _torch_load(f, *x, **k)
        if isinstance(f, (str, os.PathLike)) and os.path.realpath(f) == released: obj["app"]["model"] = _Released(obj["app"]["model"])
        return obj

    def _check_only(self, sd, *x, **k):   # train.load_from_coalesced_checkpoint: model.load_state_dict(model_checkpoint, strict=True)
        if not isinstance(sd, _Released): return _module_load(self, sd, *x, **k)   # every other load (auto_resume's own checkpoints) is untouched
        own = self.state_dict(); miss, extra = sorted(set(own) - set(sd)), sorted(set(sd) - set(own))
        bad = sorted(n for n in set(own) & set(sd) if tuple(own[n].shape) != tuple(sd[n].shape))
        if miss or extra or bad: raise RuntimeError(f"--init random: model differs from the released architecture: missing {miss[:5]} unexpected {extra[:5]} shape {bad[:5]}")
        ARCH.update(tensors=len(own), params=sum(q.numel() for q in self.parameters()))
        print(f"[walrus_ft] rank {rank}: --init random: released weights NOT loaded (random init kept); same keys and shapes as the released checkpoint "
              f"({ARCH['tensors']} tensors, {ARCH['params']:,} parameters)", flush=True)
        return torch.nn.modules.module._IncompatibleKeys([], [])   # = the strict check passed; nothing was copied

    torch.load, torch.nn.Module.load_state_dict = _load_marked, _check_only
    import walrus.utils.distribution_utils as _du2
    _distribute = _du2.distribute_model   # train.py imports distribute_model by name when it runs, so it gets the guard below

    def _distribute_checked(model, *x, **k):   # called right after the load, before any training step
        if not ARCH: raise RuntimeError("--init random: train.py did not load the released checkpoint, so the architecture check did not run")
        return _distribute(model, *x, **k)

    _du2.distribute_model = _distribute_checked
sys.argv = [os.path.join(src, "walrus", "train.py")] + overrides
try:
    runpy.run_path(sys.argv[0], run_name="__main__")
except SystemExit as e:
    if e.code not in (0, None): raise
if rank == 0:
    json.dump({"name": a.name, "epochs": a.epochs, "lr": a.lr, "batch_per_gpu": a.batch, "max_samples": a.max_samples, "world": int(os.environ.get("WORLD_SIZE", 1)),
               "trainer": "defaults", "data": a.data, "zero_density": True,
               "init": a.init, "init_seed": 0 if a.init == "random" else None, "arch_check": ARCH or None, "init_from": a.init_from or None, "init_ckpt": SRC_CKPT,
               "warmup_epochs": a.warmup_epochs or None, "cooldown_epochs": a.cooldown_epochs or None,
               "experiment_dir": exp_dir, "wall_h": (time.time() - t0) / 3600, "overrides": overrides}, open(out, "w"), indent=1)
    print("saved", out, flush=True)
