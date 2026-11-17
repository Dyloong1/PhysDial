#!/usr/bin/env python
"""DPOT-H fine-tuning (torchrun over the GPUs). Model adaptation and data layout: dpot_common.py. Data stream: the Walrus baseline's training
mixture through Walrus's own data module (--data physdial_corpus or physdial_corpus_anchor, 6 history frames, 1 target frame, batch --batch per GPU,
an epoch = --max_samples batches per rank), so DPOT-H and Walrus see the same data.

Recipe = DPOT's fine-tuning code (finetune.py / configs/dpot_finetune.yaml): one-step prediction of the next frame (T_ar = 1), relative L2
loss over the sample's present channels, Adam betas (0.9, 0.9) weight decay 1e-6, one-cycle schedule (cosine warm-up from lr / 1e4 over
--warmup_frac, cosine decay to lr / 1e8), gradient clip 1e4, no noise injection. The schedule runs on the training-time budget
(--budget_gpu_h, GPU-hours over all ranks; rank-0-agreed time), so the length does not depend on throughput. bf16 only inside the
27 blocks (AFNO in fp32); embedding and head fp32; gradient checkpointing.
Resumable (resume.Resumer). Validation: one-step loss on the first --val_batches batches of every validation dataset (Walrus valid
splits), every --val_every_min minutes and at the end. Writes ckpt/<name>.pt (weights) and results/<name>.json."""
import argparse, json, math, os, sys, time
HERE = os.path.dirname(os.path.abspath(__file__)); PKG = os.path.dirname(os.path.dirname(HERE)); sys.path.insert(0, os.path.dirname(PKG)); sys.path.insert(0, PKG); sys.path.insert(0, HERE)
import numpy as np, torch, torch.distributed as dist
import dpot_common as D
from resume import Resumer


def rel_l2(p, y, m):
    """DPOT's relative L2 per sample over the present channels; p, y [B, H, W, C], m [C] -> [B]."""
    d = ((p - y) * m).flatten(1).norm(dim=1); n = (y * m).flatten(1).norm(dim=1); return d / n.clamp_min(1e-12)


def step_inputs(batch, revin, dev):
    """Walrus batch -> DPOT input [B, H, W, T_PRE, C_IN], target [B, H, W, C_OUT] (normalized), channel mask, layout."""
    meta = batch["metadata"]; L = D.Layout(meta, batch["padded_field_mask"]); std = D.field_std(revin, meta.dataset_name, dev)
    x = L.to_dpot(batch["input_fields"].to(dev, non_blocking=True).float(), std)
    if L.Fc: x = x + L.consts(batch["constant_fields"].to(dev).float(), std, x.shape[3])
    y = L.to_dpot(batch["output_fields"].to(dev, non_blocking=True).float(), std)[:, :, :, 0, :D.C_OUT]
    return D.pad_history(x), y, L.mask.to(dev), L


def lr_at(p, a):
    """one-cycle (cos) on the training-time fraction p: lr / 1e4 -> lr over warmup_frac, then -> lr / 1e8 (OneCycleLR div 1e4, final_div 1e4)."""
    w = a.warmup_frac; lo, fin = a.lr / 1e4, a.lr / 1e8
    if p < w: return lo + (a.lr - lo) * (1 - math.cos(math.pi * p / w)) / 2
    q = min(1.0, (p - w) / max(1e-9, 1 - w)); return fin + (a.lr - fin) * (1 + math.cos(math.pi * q)) / 2


@torch.no_grad()
def validate(model, dm, revin, dev, a):
    model.eval(); out = {}
    for ld in dm.val_dataloaders(replicas=1, rank=0):
        ls, name = [], None
        for i, b in enumerate(ld):
            if i >= a.val_batches: break
            x, y, m, L = step_inputs(b, revin, dev); name = L.name
            p = model(x)   # same precision as training (bf16 blocks via model.amp)
            ls.append(rel_l2(p.float(), y, m).mean().item())
        if name: out[name] = float(np.mean(ls))
    model.train(); out["mean"] = float(np.mean([v for v in out.values()])) if out else float("nan"); return out


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--name", required=True)
    ap.add_argument("--budget_gpu_h", type=float, default=60.0)
    ap.add_argument("--lr", type=float, default=1e-4); ap.add_argument("--warmup_frac", type=float, default=0.2)
    ap.add_argument("--batch", type=int, default=2); ap.add_argument("--max_samples", type=int, default=500); ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--val_batches", type=int, default=20); ap.add_argument("--val_every_min", type=float, default=120); ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--log_every", type=int, default=100)
    ap.add_argument("--data", default="physdial_corpus", choices=list(D.DATA_CFGS), help="Walrus data config (physdial_corpus_anchor = with the Kolmogorov amplitude anchors); recorded, dpot_eval reads it")
    a = ap.parse_args()
    out = os.path.join(HERE, "results", a.name + ".json"); CK = os.path.join(PKG, "ckpt"); os.makedirs(os.path.dirname(out), exist_ok=True); os.makedirs(CK, exist_ok=True)
    if os.path.exists(out): print("exists", out); sys.exit(0)
    rank, world, lrk = int(os.environ.get("RANK", 0)), int(os.environ.get("WORLD_SIZE", 1)), int(os.environ.get("LOCAL_RANK", 0))
    dev = torch.device(f"cuda:{lrk}"); torch.cuda.set_device(dev)
    if world > 1:   # long timeout: the other ranks wait at a barrier while rank 0 validates and saves the checkpoint
        import datetime; dist.init_process_group("nccl", device_id=dev, timeout=datetime.timedelta(minutes=60))
    torch.manual_seed(a.seed * 1000 + rank); np.random.seed(a.seed * 1000 + rank)   # the Walrus sampler draws per rank from the global generator
    log = lambda s: print(f"[{a.name}] {s}", flush=True) if rank == 0 else None
    from hydra.utils import instantiate
    from walrus.trainer.normalization_strat import RMSGlobalRevNormalization
    cfg = D.walrus_cfg(a.data); t0 = time.time()
    dm = instantiate(cfg.data.module_parameters, world_size=world, rank=rank, data_workers=a.workers, well_base_path=cfg.data.well_base_path,
                     field_index_map_override=cfg.data.get("field_index_map_override", {}), transform=None, batch_size=a.batch, max_samples=a.max_samples)
    revin = RMSGlobalRevNormalization(dm.train_dataset, dev)
    model = D.DPOTAdapter(pretrained=True, grad_ckpt=True, amp=True).to(dev)
    n_params = sum(p.numel() for p in model.parameters()); log(f"DPOT-H adapter {n_params / 1e6:.1f}M params; data module {time.time() - t0:.0f}s; datasets {[s.metadata.dataset_name for s in dm.train_dataset.sub_dsets]}")
    opt = torch.optim.Adam(model.parameters(), lr=a.lr, betas=(0.9, 0.9), weight_decay=1e-6)
    rs = Resumer(CK, a.name, rank, world)
    step, ex = rs.load(model, opt); train_s, epoch, vals, seen = ex.get("train_s", 0.0), ex.get("epoch", 0), ex.get("vals", []), ex.get("samples", 0)
    if step:
        r_ = int(os.environ.get("SLURM_RESTART_COUNT", 0)) + 1; torch.manual_seed(a.seed * 1000 + rank + 7919 * r_); np.random.seed(a.seed * 1000 + rank + 7919 * r_)
    ddp = torch.nn.parallel.DistributedDataParallel(model, device_ids=[lrk]) if world > 1 else model
    budget_s = a.budget_gpu_h * 3600 / world; last_val = time.time(); done = False; hist = []
    log(f"start step {step} (restart {os.environ.get('SLURM_RESTART_COUNT', 0)}), train time {train_s / 3600:.2f} of {budget_s / 3600:.2f} h per GPU, lr {a.lr}")
    while not done:
        dl = dm.train_dataloader(); epoch += 1
        if hasattr(dl.sampler, "set_epoch"): dl.sampler.set_epoch(epoch)
        it = iter(dl)
        while True:
            tb = time.time()
            try: b = next(it)
            except StopIteration: break
            x, y, m, L = step_inputs(b, revin, dev)
            for g in opt.param_groups: g["lr"] = lr_at(min(1.0, train_s / budget_s), a)
            p = ddp(x)
            loss = rel_l2(p.float(), y, m).mean()
            fin = torch.isfinite(loss).float()
            if world > 1: dist.all_reduce(fin, op=dist.ReduceOp.MIN)
            opt.zero_grad(set_to_none=True)
            if fin.item() > 0:
                loss.backward(); gn = torch.nn.utils.clip_grad_norm_(model.parameters(), 1e4); opt.step()
            else: gn = torch.tensor(float("nan")); log(f"non-finite loss at step {step} ({L.name}): step skipped")
            step += 1; seen += x.shape[0] * world; hist.append((L.name, loss.item()))
            ctl = torch.tensor([train_s + time.time() - tb, float(time.time() - last_val > a.val_every_min * 60)], device=dev)
            if world > 1: dist.all_reduce(ctl, op=dist.ReduceOp.MAX)
            train_s, val_due = ctl[0].item(), ctl[1].item() > 0
            if step % a.log_every == 0:
                per = {}
                for k, v in hist: per.setdefault(k, []).append(v)
                log(f"it {step} ep {epoch} lr {opt.param_groups[0]['lr']:.2e} gn {gn.item():.3f} loss " + " ".join(f"{k}={np.mean(v):.4f}" for k, v in sorted(per.items()))
                    + f" | train {train_s / 3600:.2f}/{budget_s / 3600:.2f} h mem {torch.cuda.max_memory_allocated() / 2 ** 30:.0f}G"); hist = []
            stop = torch.tensor([float(train_s >= budget_s)], device=dev)
            if world > 1: dist.all_reduce(stop, op=dist.ReduceOp.MAX)   # every rank stops at the same step
            if stop.item() > 0: done = True; break
            if val_due:
                if rank == 0: v = validate(model, dm, revin, dev, a); vals.append({"step": step, "train_gpu_h": train_s * world / 3600, **v}); log(f"val {v['mean']:.4f} at step {step} | " + " ".join(f"{k} {x:.4f}" for k, x in v.items() if k != "mean"))
                if world > 1: dist.barrier()
                last_val = time.time()
            extra = {"train_s": train_s, "epoch": epoch, "vals": vals, "samples": seen}
            if rs.should_stop(step): rs.save(step, model, opt, extra=extra); rs.exit_for_requeue()
            if rs.periodic_now(step): rs.save(step, model, opt, extra=extra)
    if rank == 0:
        v = validate(model, dm, revin, dev, a); vals.append({"step": step, "train_gpu_h": train_s * world / 3600, **v}); log(f"final val {v['mean']:.4f} at step {step}")
        sd = {k: t.detach().cpu() for k, t in model.state_dict().items()}
        torch.save({"model": sd, "h_cfg": D.H_CFG, "c_in": D.C_IN, "c_out": D.C_OUT, "eps": D.EPS, "modes": D.MODES, "data": a.data, "step": step, "args": vars(a)}, os.path.join(CK, a.name + ".pt.tmp"))
        os.replace(os.path.join(CK, a.name + ".pt.tmp"), os.path.join(CK, a.name + ".pt"))
        json.dump({"name": a.name, "args": vars(a), "data": a.data, "steps": step, "epochs": epoch, "samples": seen, "world": world, "train_gpu_h": train_s * world / 3600, "wall_h": (time.time() - t0) / 3600,
                   "n_params": n_params, "final_val": v, "vals": vals, "dpot_weights": D.DPOT_WEIGHTS, "git_sha": os.environ.get("PHYSDIAL_GIT_SHA")}, open(out + ".tmp", "w"), indent=1)
        os.replace(out + ".tmp", out); rs.finish(); print("saved", out, flush=True)
    if world > 1: dist.barrier(); dist.destroy_process_group()


if __name__ == "__main__":
    main()
