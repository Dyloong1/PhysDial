#!/usr/bin/env python
"""DPOT-H evaluation (torchrun; splits sharded over ranks as in walrus_eval): scores a dpot_ft.py run on the Walrus-format evaluation
windows with walrus_eval's code unchanged (window order, vorticity from velocity, metrics, bootstrap, per-rank split sharding and
merge). Only the model and the rollout are DPOT's: walrus_eval.build / walrus_eval.rollout are replaced here, and results go
to physdial/baselines/dpot/results. Rollout: 6 history frames (EVAL_HIST; padded to DPOT's 10 by repeating the oldest), --frames
autoregressive steps at the family's stride, each step predicts the next frame (normalized with the training corpus's global RMS, as in
training). --ckpt is ignored (one checkpoint per run: ckpt/<run>.pt); --run names the dpot_ft run. Other options as walrus_eval.py."""
import json, os, sys
HERE = os.path.dirname(os.path.abspath(__file__)); PKG = os.path.dirname(os.path.dirname(HERE)); sys.path.insert(0, os.path.dirname(PKG)); sys.path.insert(0, PKG); sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(PKG, "baselines", "walrus"))
import torch
import dpot_common as D
import walrus_eval as W

def run_data(a):
    p = os.path.join(HERE, "results", a.run + ".json")
    if not os.path.exists(p): raise SystemExit(f"dpot_ft {a.run} has not finished (no results/{a.run}.json)")
    return json.load(open(p)).get("data") or (json.load(open(p)).get("args") or {}).get("data") or "physdial_corpus"


def build(a, dev):
    from hydra.utils import instantiate
    from walrus.trainer.normalization_strat import RMSGlobalRevNormalization
    data = run_data(a); cfg = D.walrus_cfg(data)   # the same global RMS statistics as in training
    dm_tr = instantiate(cfg.data.module_parameters, world_size=1, rank=0, data_workers=0, well_base_path=cfg.data.well_base_path,
                        field_index_map_override=cfg.data.get("field_index_map_override", {}), transform=None)
    fmap = {str(k): int(v) for k, v in dm_tr.train_dataset.field_to_index_map.items()}
    revin = RMSGlobalRevNormalization(dm_tr.train_dataset, dev)
    alias = D.alias_eval_stats(revin.name_to_normalization_stats)
    print(f"[dpot_eval] evaluation statistics aliased to the training datasets: {alias}", flush=True)
    model = D.DPOTAdapter(pretrained=True)
    info = {"run": a.run, "ckpt": "final", "epsilon": D.EPS, "revin": type(revin).__name__, "dpot_weights": D.DPOT_WEIGHTS, "data": data,
            "zero_density": False, "stats_alias": alias}
    ck = torch.load(os.path.join(PKG, "ckpt", a.run + ".pt"), map_location="cpu", weights_only=False)
    model.load_state_dict(ck["model"]); info.update({"step": ck["step"], "checkpoint": os.path.join(PKG, "ckpt", a.run + ".pt"), "modes": ck.get("modes", 10 ** 6)})
    for b in model.net.blocks: b.filter.modes = info["modes"]   # the run's AFNO setting (not a weight)
    model = model.to(dev).eval(); info["n_params"] = sum(p.numel() for p in model.parameters())
    return model, revin, fmap, cfg, info


@torch.no_grad()
def rollout(model, revin, formatter, batch, steps, eps, dev):
    """same contract as walrus_eval.rollout: y_pred [B, steps, H, W, 1, F], y_ref [B, steps, H, W, 1, F] (physical), last two input frames."""
    meta = batch["metadata"]; L = D.Layout(meta, batch["padded_field_mask"]); std = D.field_std(revin, meta.dataset_name, dev)
    xin = batch["input_fields"].to(dev).float(); y_ref = batch["output_fields"].to(dev).float(); steps = min(steps, y_ref.shape[1]); y_ref = y_ref[:, :steps]
    x = L.to_dpot(xin, std); cst = L.consts(batch["constant_fields"].to(dev).float(), std, D.T_PRE) if L.Fc else 0
    preds = []
    for i in range(steps):
        y = model(D.pad_history(x) + cst)   # fp32 (model.amp False), as walrus_eval
        y = y.float() * L.mask.to(dev)   # absent slots stay zero (they are never read back: from_dpot takes the present slots only)
        preds.append(L.from_dpot(y, std, y_ref))
        nx = torch.zeros_like(x[:, :, :, :1]); nx[..., :D.C_OUT] = y[:, :, :, None]
        x = torch.cat([x[:, :, :, 1:], nx], dim=3)
    return torch.cat(preds, 1), y_ref, xin[:, -2:]


W.build, W.rollout, W.HERE = build, rollout, HERE   # walrus_eval.main / accumulate_dir look these up at call time

if __name__ == "__main__":
    W.main()
