#!/usr/bin/env python
"""Decoder fine-tuning of the Wan2.1 VAE on the corpus, followed by a reconstruction gate (tok_gate).

The encoder stays frozen (so the latent space of the pretrained Wan2.1 DiT and every latent store are kept exactly); only the decoder is
fine-tuned on single physical channels, normalised per frame as the evaluation path encodes them.
Losses: L1 + gradient L1 + log radial-spectrum loss (--lam_spec) + a PatchGAN hinge term. With --vae_wrap the VAE uses wrap-around
padding on periodic axes (the decoder for the wrap-padded store). Evaluation decodes with it through train_gen --tok wan21ft:<name>.pt."""
import argparse, json, math, os, sys, time
HERE = os.path.dirname(os.path.abspath(__file__)); PROJ = os.path.dirname(HERE); sys.path.insert(0, PROJ); sys.path.insert(0, HERE)
import contextlib
import numpy as np, torch, torch.nn as nn, torch.nn.functional as F
import runmeta, corpus, tok2d, tok_gate, fieldnorm, vae_periodic
from resume import Resumer

SPEC_MIN_STD = 1e-2   # a crop can be (near) constant (floored frames are exactly 0), where the log spectrum of
                      # spec_loss is noise over a 1e-26 power floor and its gradient ~1/amplitude; crops with std < 1 % of the frame scale skip that term


class Frames(torch.utils.data.IterableDataset):
    """training crops as the evaluation path encodes frames: the whole native frame of one channel normalised by fieldnorm
    (encode_corpus / train_gen.Conditioner.encode), a channel-frame whose scale is below the scale floor (`floor` {family: [C]}, train_gen.scale_floor /
    apply_floor) -> an all-zero frame, only then the crop."""
    def __init__(s, groups, crop, pixels, max_bs, seed, floor=None, wrap=False):
        super().__init__(); s.groups, s.crop, s.pixels, s.max_bs, s.seed, s.floor = groups, crop, pixels, max_bs, seed, floor or {}
        s.wrap = wrap
    def __iter__(s):
        wi = torch.utils.data.get_worker_info(); rng = np.random.default_rng([s.seed, wi.id if wi else 0])
        fams = {k: corpus.get(k) for k in s.groups}; keys = sorted(s.groups)
        while True:
            k = keys[rng.integers(len(keys))]; gs = s.groups[k]; w = np.array([len(v) for v in gs.values()], float)
            shape = list(gs)[rng.choice(len(gs), p=w / w.sum())]; its = gs[shape]; fam = fams[k]; tg = tuple(shape)
            ch, cw = min(s.crop, tg[0]), min(s.crop, tg[1])
            if s.wrap: ch, cw = (tg[0] if fam.periodic[0] else ch), (tg[1] if fam.periodic[1] else cw)
            B = int(max(1, min(s.max_bs, s.pixels // (ch * cw)))); X = []
            for j in rng.integers(0, len(its), B):
                it = its[j]; t = int(rng.integers(fam.t_start, it.n_frames)); c = int(rng.integers(len(fam.channels)))
                x = torch.from_numpy(np.ascontiguousarray(fam.load(it, t, t + 1)[0, c:c + 1])); mu, sc = fieldnorm.scale_t(x)   # [1, H, W]: whole frame
                x = torch.zeros_like(x) if k in s.floor and float(sc[0]) < s.floor[k][c] else (x - mu[:, None, None]) / sc[:, None, None]   # below the floor: all-zero frame
                x = x[0].numpy()
                if s.wrap: x = np.roll(x, tuple(int(rng.integers(n_)) if p_ else 0 for n_, p_ in zip(tg, fam.periodic)), (0, 1))
                i0 = int(rng.integers(0, tg[0] - ch + 1)); j0 = int(rng.integers(0, tg[1] - cw + 1)); X.append(x[i0:i0 + ch, j0:j0 + cw])
            yield (torch.from_numpy(np.stack(X)[:, None].astype(np.float32)), tuple(bool(p_) for p_ in fam.periodic)) if s.wrap else torch.from_numpy(np.stack(X)[:, None].astype(np.float32))


def vae_ctx(m, per, wrap):
    return vae_periodic.periodic(m.vae, per) if wrap and any(per) else contextlib.nullcontext()


def plain_state(vae):
    """state dict of a VAE with the key names of the unpatched model: vae_periodic.install wraps every padded Conv2d / ZeroPad2d in a module
    holding it as `.orig`, which adds that segment to the parameter names (tok2d.WanTok loads the plain names)."""
    sd = vae.state_dict()
    if not getattr(vae, "_physdial_periodic", False): return sd
    return {k.replace(".orig.", "."): v for k, v in sd.items()}


class Disc(nn.Module):   # 2D PatchGAN (hinge)
    def __init__(s, ch=64):
        super().__init__()
        s.net = nn.Sequential(nn.Conv2d(1, ch, 4, 2, 1), nn.LeakyReLU(0.2), nn.Conv2d(ch, 2 * ch, 4, 2, 1), nn.GroupNorm(8, 2 * ch), nn.LeakyReLU(0.2),
                              nn.Conv2d(2 * ch, 4 * ch, 4, 1, 1), nn.GroupNorm(8, 4 * ch), nn.LeakyReLU(0.2), nn.Conv2d(4 * ch, 1, 4, 1, 1))
    def forward(s, x): return s.net(x)


def spec_loss(r, x, nb=16, keep=None):
    H, W = x.shape[-2:]; win = torch.outer(torch.hann_window(H, periodic=False, device=x.device), torch.hann_window(W, periodic=False, device=x.device))
    P = lambda y: torch.fft.rfft2(y.float() * win).abs().pow(2)
    k = torch.sqrt(torch.fft.fftfreq(H, device=x.device)[:, None] ** 2 + torch.fft.rfftfreq(W, device=x.device)[None] ** 2)
    edges = torch.logspace(math.log10(0.999 / max(H, W)), math.log10(0.5 * math.sqrt(2) * 1.0001), nb + 1, device=x.device)
    b = torch.bucketize(k, edges, right=True) - 1; kmask = (b >= 0) & (b < nb)
    Pr, Px = P(r)[..., kmask], P(x)[..., kmask]; bi = b[kmask]; cnt = torch.zeros(nb, device=x.device).index_add_(0, bi, torch.ones_like(bi, dtype=torch.float32))
    used = cnt > 0; n_ = cnt.clamp_min(1)
    Er = torch.zeros(*Pr.shape[:-1], nb, device=x.device).index_add_(-1, bi, Pr) / n_; Ex = torch.zeros(*Px.shape[:-1], nb, device=x.device).index_add_(-1, bi, Px) / n_
    eps = 1e-6 * Ex.sum(-1, keepdim=True).clamp_min(1e-20)
    d = (torch.log(Er + eps) - torch.log(Ex + eps)).abs()[..., used].mean(-1).flatten()   # per sample
    return d.mean() if keep is None else (d * keep).sum() / keep.sum().clamp_min(1.0)


def grad_l1(r, x):
    return (F.l1_loss(r[..., 1:, :] - r[..., :-1, :], x[..., 1:, :] - x[..., :-1, :]) + F.l1_loss(r[..., :, 1:] - r[..., :, :-1], x[..., :, 1:] - x[..., :, :-1])) / 2


class WanTrain(nn.Module):
    """Wan2.1 VAE with the (B, 1, H, W) interface; encoder (and quant_conv) frozen."""
    def __init__(s):
        super().__init__(); s.t = tok2d.WanTok(); s.vae = s.t.vae; s.zc = s.t.zc; s.factor = s.t.factor
        for p in s.vae.encoder.parameters(): p.requires_grad_(False)
        for p in s.vae.quant_conv.parameters(): p.requires_grad_(False)
    def moments(s, x):
        d = s.vae.encode(x.expand(-1, 3, -1, -1)[:, :, None]).latent_dist; return d.mean[:, :, 0], d.logvar[:, :, 0]
    def decode_aux(s, z): return s.vae.decode(z[:, :, None]).sample[:, :, 0].mean(1, keepdim=True), ()


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--name", required=True)
    ap.add_argument("--families", default=""); ap.add_argument("--steps", type=int, default=40000); ap.add_argument("--crop", type=int, default=256)
    ap.add_argument("--pixels", type=int, default=2 ** 21); ap.add_argument("--max_bs", type=int, default=128); ap.add_argument("--lr", type=float, default=0.0)
    ap.add_argument("--lam_grad", type=float, default=0.5); ap.add_argument("--lam_adv", type=float, default=1e-2)
    ap.add_argument("--lam_spec", type=float, default=0.1, help="weight of the log radial-spectrum loss (spec_loss)")
    ap.add_argument("--disc_start", type=int, default=5000); ap.add_argument("--warmup", type=int, default=1000)
    ap.add_argument("--workers", type=int, default=8); ap.add_argument("--seed", type=int, default=0); ap.add_argument("--n", type=int, default=24); ap.add_argument("--log_every", type=int, default=500)
    ap.add_argument("--ref_run", required=True, help="generator run whose results/<run>.json pstats give the scale floor, so the decoder "
                    "is trained on frames exactly as that run's evaluation path encodes them")
    ap.add_argument("--vae_wrap", action="store_true", help="wrap-around VAE padding on the periodic axes of each family (vae_periodic.py, the wrap-padded "
                    "store of encode_corpus --vae_wrap), frozen encoder included; periodic axes whole and randomly rolled instead of cropped")
    a = ap.parse_args()
    R = os.path.join(HERE, "results"); CK = os.path.join(HERE, "ckpt"); os.makedirs(R, exist_ok=True); os.makedirs(CK, exist_ok=True)
    out = os.path.join(R, a.name + ".json")
    if os.path.exists(out): print("exists", out); sys.exit(0)
    log = lambda *m: print(f"[{a.name}]", *m, flush=True); dev = torch.device("cuda"); torch.manual_seed(a.seed)
    torch.backends.cuda.matmul.allow_tf32 = True; torch.backends.cudnn.allow_tf32 = True
    fams = [f for f in a.families.split(",") if f] or corpus.available(); groups = {}
    for k in fams:
        g = {}
        for it in corpus.cached_items(k, "train"): g.setdefault(tuple(it.shape), []).append(it)   # train split only (SF / RB evaluation ICs, val and ref_* never enter)
        groups[k] = g
    log("training trajectories", {k: sum(len(v) for v in g.values()) for k, g in groups.items()})
    ps = json.load(open(os.path.join(R, a.ref_run + ".json")))["pstats"]; fsrc = f"pstats of {a.ref_run}"
    assert a.vae_wrap or not ps.get("vae_wrap", False), f"--ref_run {a.ref_run}: its store uses wrap-around VAE padding; this run trains the zero-padded VAE (--vae_wrap off)"   # (with --vae_wrap the run reads only the floor, which does not depend on the padding)
    floor = {k: np.asarray(v, np.float32) for k, v in ps.get("s_floor", {}).items() if k in fams}
    assert set(fams) <= set(floor), f"--ref_run {a.ref_run} has no scale floor for {sorted(set(fams) - set(floor))}"
    log(f"whole-frame normalisation; scale floor ({fsrc}): " + "; ".join(f"{k} " + ",".join(f"{v_:.3g}" for v_ in v) for k, v in floor.items()))
    data = {"families": fams, "s_floor": {k: v.tolist() for k, v in floor.items()}, "vae_wrap": bool(a.vae_wrap)}   # resume guard + results / ckpt record
    model = WanTrain().to(dev); lr = a.lr or 2e-5
    if a.vae_wrap: vae_periodic.install(model.vae)
    disc = Disc().to(dev)
    params = [p for p in model.parameters() if p.requires_grad]; npar = sum(p.numel() for p in params)
    log(f"Wan2.1 VAE decoder: {npar / 1e6:.1f}M trainable params, lr {lr}")
    opt = torch.optim.AdamW(params, lr=lr, betas=(0.5, 0.9), weight_decay=0.0)
    dopt = torch.optim.AdamW(disc.parameters(), lr=lr / 2, betas=(0.5, 0.999))
    holder = nn.ModuleDict({"model": model, "disc": disc})
    class Opts:   # one state dict for both optimisers (Resumer saves one)
        def state_dict(s): return {"g": opt.state_dict(), "d": dopt.state_dict()}
        def load_state_dict(s, d): opt.load_state_dict(d["g"]); dopt.load_state_dict(d["d"])
    rs = Resumer(CK, a.name); step, extra = rs.load(holder, Opts()); hist = extra.get("hist", []); t_train = extra.get("t_train", 0.0)
    assert extra.get("data", data) == data, f"resume of {a.name}: started on {extra.get('data')}, now {data}"
    dl = iter(torch.utils.data.DataLoader(Frames(groups, a.crop, a.pixels, a.max_bs, [a.seed, step], floor, wrap=a.vae_wrap), batch_size=None, num_workers=a.workers, persistent_workers=True, prefetch_factor=4))
    run = {}; t0 = time.time()
    while step < a.steps:
        ts = time.time()
        f = min(1.0, (step + 1) / a.warmup) * 0.5 * (1 + math.cos(math.pi * step / a.steps))
        for g in opt.param_groups: g["lr"] = lr * f
        for g in dopt.param_groups: g["lr"] = lr / 2 * f
        x, per = next(dl) if a.vae_wrap else (next(dl), (False, False)); x = x.to(dev, non_blocking=True)
        with torch.autocast("cuda", dtype=torch.bfloat16), vae_ctx(model, per, a.vae_wrap):
            z = model.moments(x)[0]   # the mean, as deployed (tok2d.encode: mode)
            r, aux = model.decode_aux(z)
        r = r.float(); l1 = F.l1_loss(r, x); lg = grad_l1(r, x); loss = l1 + a.lam_grad * lg; terms = {"l1": l1, "grad": lg}
        if a.lam_spec > 0:
            kp = (x.std((-2, -1)).flatten() >= SPEC_MIN_STD).float(); ls = spec_loss(r, x, keep=kp); loss = loss + a.lam_spec * ls; terms["spec"] = ls; terms["spec_kept"] = kp.mean()
        use_d = step >= a.disc_start and a.lam_adv > 0
        if use_d:
            for p in disc.parameters(): p.requires_grad_(False)
            with torch.autocast("cuda", dtype=torch.bfloat16): ladv = -disc(r).float().mean()
            for p in disc.parameters(): p.requires_grad_(True)
            loss = loss + a.lam_adv * ladv; terms["g_adv"] = ladv
        opt.zero_grad(set_to_none=True); loss.backward(); torch.nn.utils.clip_grad_norm_(params, 1.0); opt.step()
        if use_d:
            with torch.autocast("cuda", dtype=torch.bfloat16):
                ld = F.relu(1 - disc(x)).float().mean() + F.relu(1 + disc(r.detach())).float().mean()
            dopt.zero_grad(set_to_none=True); ld.backward(); dopt.step(); terms["d"] = ld
        step += 1; torch.cuda.synchronize(); t_train += time.time() - ts
        for k_, v_ in terms.items(): run.setdefault(k_, []).append(float(v_))
        if step % a.log_every == 0:
            row = {"step": step, **{k_: float(np.mean(v_)) for k_, v_ in run.items()}}; hist.append(row); run = {}
            log(f"it {step} " + " ".join(f"{k_}={v_:.4g}" for k_, v_ in row.items() if k_ != "step") + f" {a.log_every / (time.time() - t0):.2f} it/s mem {torch.cuda.max_memory_allocated() / 2**30:.1f}G"); t0 = time.time()
        ex = {"hist": hist, "t_train": t_train, "data": data}
        if rs.periodic_now(step): rs.save(step, holder, Opts(), None, (), ex)
        if rs.should_stop(step): rs.save(step, holder, Opts(), None, (), ex); rs.exit_for_requeue()
    model.eval(); del dl
    cfg = model.vae.config; lmean, lstd = torch.tensor(cfg.latents_mean), torch.tensor(cfg.latents_std)   # encoder unchanged: Wan's latent statistics
    ckp = os.path.join(CK, a.name + ".pt")
    torch.save({"kind": "wan21_decoder", "model": plain_state(model.vae), "lmean": lmean, "lstd": lstd, "args": vars(a), "data": data}, ckp)
    # reconstruction check (tok_gate, the deployed bf16 weights): 24 frames of every evaluation split, rel_l2 <= 0.05 and grad ratio in [0.9, 1.1]
    spec = "wan21ft:" + a.name + ".pt"; tok = tok2d.load(spec, "cuda").to(torch.bfloat16)
    res = {"name": a.name, "spec": spec, "n_trainable": npar, "steps": step, "train_h": t_train / 3600, "hist": hist, **data, "floor_source": fsrc,
           "frame_norm": "whole frame (fieldnorm), scale floor -> all-zero frame, pad, then crop", "spec_min_std": SPEC_MIN_STD, "gate": {}, "summary": {}}
    if a.vae_wrap: res["vae_wrap_note"] = "wrap-around VAE padding on periodic axes (training and gate); periodic axes whole, randomly rolled, not cropped"
    for k in fams:
        fam = corpus.get(k); rng = np.random.default_rng(a.seed); res["gate"][k] = {}; ok = True; worst = 0.0
        for sp in fam.splits():
            if sp in ("train", "val") or sp.startswith("ref_"): continue
            X = tok_gate.sample(fam, sp, a.n, rng)
            if X is None: continue
            acc = {c: tok_gate.Acc() for c in fam.channels}
            for g, Xg in X.items():
                with vae_ctx(tok, fam.periodic, a.vae_wrap): Y, _ = tok_gate.roundtrip(tok, Xg, 4)
                for i, c in enumerate(fam.channels): acc[c].add(Xg[:, i].astype(np.float64), Y[:, i].astype(np.float64), fam.periodic)
            res["gate"][k][sp] = {c: acc[c].result() for c in fam.channels}
            for m in res["gate"][k][sp].values(): ok &= tok_gate.passes(m); worst = max(worst, m["rel_l2"])
        res["summary"][k] = {"passes": bool(ok), "worst_rel_l2": worst}
        log(f"gate {k}: passes={ok} worst rel_l2={worst:.4f} | " + "; ".join(f"{sp}: " + ",".join(f"{c} l2={m['rel_l2']:.3f} top={m['band_retention']['(1/2,1]']:.2f} g={m['grad_ratio']:.3f}" for c, m in v.items()) for sp, v in res["gate"][k].items()))
    res["summary"]["_all_pass"] = all(v["passes"] for kk, v in res["summary"].items() if not kk.startswith("_"))
    res["meta"] = runmeta.run_meta(); json.dump(res, open(out, "w"), indent=1); rs.finish(); print("saved", out, flush=True)


if __name__ == "__main__":
    main()
