#!/usr/bin/env python
"""PhysDial trainer and evaluator: a pretrained text-to-video diffusion transformer (Wan2.1-T2V-1.3B) turned into an autoregressive
simulator of 2D PDE families. Flow matching in the latent space of the frozen Wan2.1 VAE: K history frames (default 2) -> F predicted frames
(default 4) per round; long rollouts chain rounds in latent space.

Data: the latent store written by encode_corpus.py (every frame and channel normalised on its own, x = mu + s * decode(z)) on each family's
native grid. Every micro-batch comes from one family and one grid size; the batch size follows a token budget (--tokens).

Frame representation (per frame, CL = zc * MAX_C + 2 * MAX_C channels): MAX_C quantity slots (corpus.QUANTITIES), each with the zc latent
channels of that physical quantity, then two constant planes per slot with its log-amplitude and mean relative to the last history frame.
A family fills only the slots of its own channels; the others are zero and masked out of the loss.

Conditions (--cond all): each physical input enters through the video generator's pathway of the same role. Scalar coefficients and the
frame spacing -> standardised numeric vector (+ presence mask and state scalars) -> MLP added to the timestep embedding (adaLN);
condition fields (Kolmogorov forcing pattern, acoustic medium) -> encoded with the same VAE and concatenated on the input channels like a
reference frame; equation text -> frozen umT5 embeddings (textemb.py) through the existing cross-attention. New input channels are zero-initialised.
History frames are concatenated to the noisy target on the channel axis (image-to-video convention) and never noised.

Main options: --kolmo_extra anchor,drag,beta (Kolmogorov amplitude anchors, drag columns, beta-plane groups), --init_weights (continue from
a trained run), --bc_pos and --rand_shift with a --vae_wrap store (periodic handling), --reference (in-range reference: held-out-condition
trajectories added to training), --tok wan21ft:<decoder> (fine-tuned decoder at evaluation), --eval_only / --dump_pred / --text_probe /
--drop_fields (evaluation and evaluation-time ablations). Ablations: --cond nonum / nofield, --init random, --objective regress, --size 14b
with --lora_r, --k_in, --pushforward, --stride_aug.
Training budget in GPU-hours (--budget_gpu_h, WSD schedule), EMA weights for evaluation, resumable, DDP via torchrun.
Outputs: results/<name>.json (evaluation of every family x test / held-out split), ckpt/<name>.ema.pt (+ .stable.pt at the start of the decay)."""
import argparse, contextlib, copy, datetime, fnmatch, gc, json, math, os, re, sys, time, traceback, uuid
HERE = os.path.dirname(os.path.abspath(__file__)); PROJ = os.path.dirname(HERE); sys.path.insert(0, PROJ); sys.path.insert(0, HERE)
import numpy as np, torch, torch.nn as nn, torch.distributed as dist
import runmeta, corpus, tok2d, gen_model, fieldnorm, vae_periodic
from resume import Resumer
from eval_metrics import new_acc, accumulate_last_frame, accumulate_pred_last_frame, physics_metrics, rel_l2, rel_l2_pooled, rel_l2_demeaned, demeaned_summary, shape_amp, shape_amp_summary, vrmse, linear_extrap, crps_rel, truth_stats, apply_degenerate, nonfinite_to_inf
from encode_corpus import store_root

MAX_C = corpus.MAX_C; A_SCALE, B_SCALE = 4.0, 2.0   # planes: A_SCALE * log(s_t / s_ref), B_SCALE * asinh((mu_t - mu_ref) / s_ref)
# (the mean plane is compressed by asinh: weak fields whose mean drifts can move far relative to their amplitude)
EVAL_SEED = 0
MAX_F = 2   # condition-field slots on the input channels (acoustic: density + sound speed; Kolmogorov: the forcing)
MB_PER_GPU_H = 4937.0   # micro-batches per GPU-hour measured on H200 GPUs: converts the GPU-hour budget into the expected number of micro-batches of a run (the pass cap)
NOISE_ROLL = [None]   # --equiv_eval: (dh, dw) latent pixels by which generate rolls its initial noise (None = off)
VAE_WRAP = [False, False]   # [the latent store was encoded with wrap-around VAE padding on periodic axes (encode_corpus --vae_wrap),
                            #  the run has a periodic family at all]
DEC_FLOOR = [None]
EVAL_HIST = 6   # every evaluation window keeps 6 frames of history before the prediction start P


SHIFT_STREAM = 0x29B2   # --rand_shift: the shift draws come from their own generator [seed, rank, restarts, worker, SHIFT_STREAM]


def shift_draws(srng, B, lat_hw, periodic, unit):
    out = np.zeros((B, 2), np.int64)
    for i in range(2):
        n = int(lat_hw[i]); assert n % unit == 0, f"--rand_shift {unit}: latent extent {n} on axis {i} is not a multiple of the unit"
        d = srng.integers(0, n // unit, size=B) * unit
        if periodic[i]: out[:, i] = d
    return out


def roll_window(x, sh, lat_hw):
    """rolls sample b of x [B, ..., h, w] by sh[b] = (dy, dx) latent cells on its last two axes (np.roll: content at y moves to y + dy).
    History, targets and condition fields of a window get the same shift, so the window is the encoding of a translated field when the store
    was encoded with wrap-around VAE padding (periodic axes; 1 cell = the VAE's stride of 8 px)."""
    assert tuple(x.shape[-2:]) == tuple(lat_hw), f"roll_window: latent {tuple(x.shape[-2:])} != {tuple(lat_hw)}"
    if not sh.any(): return x
    return np.stack([np.roll(x[b], (int(sh[b, 0]), int(sh[b, 1])), axis=(-2, -1)) if sh[b].any() else x[b] for b in range(len(x))])


# ---------------------------------------------------------------- latent store ------------------------------------------------
def reindex_num_mlp(w0, so, sn):
    """first-layer weight [dim, 2 len(so) + n_state] of the number MLP for slot list so -> [dim, 2 len(sn) + n_state] for slot list sn:
    value and mask columns move with their slot name, slots missing in so get zero columns (a run that adds the beta slot starts from
    the source run's function), the state-scalar columns are copied."""
    na, nb = len(so), len(sn); w_ = w0.new_zeros(w0.shape[0], 2 * nb + (w0.shape[1] - 2 * na))
    for j, nm in enumerate(sn):
        if nm in so: i = so.index(nm); w_[:, j] = w0[:, i]; w_[:, nb + j] = w0[:, na + i]
    w_[:, 2 * nb:] = w0[:, 2 * na:]; return w_


class Store:
    """the stored splits of one family (index + lazily opened memmaps)."""
    def __init__(s, root, family, splits):
        s.family = family; s.groups = []; s.wrap = None   # (split, gid, meta, items, dir)
        for sp in splits:
            p = os.path.join(root, family, sp, "index.json")
            if not os.path.exists(p): continue
            ix = json.load(open(p)); w_ = bool(ix.get("vae_wrap", False))
            assert s.wrap in (None, w_), f"{family}: splits of the store encoded with and without --vae_wrap"; s.wrap = w_
            for gid, meta in ix["groups"].items():
                its = [it for it in ix["items"] if it["group"] == int(gid)]
                if family in corpus.SUBSET_FAMILIES:
                    keep = {(it.path, it.traj) for it in corpus.cached_items(family, sp)}   # e.g. the initial-condition split keeps 30 of its 40 ICs for training
                    its = [it for it in its if (it["path"], it["traj"]) in keep]
                if its: s.groups.append((sp, int(gid), meta, its, os.path.join(root, family, sp)))
        s.mm = {}

    def arrays(s, gi):
        if gi not in s.mm:
            sp, gid, meta, its, d = s.groups[gi]
            s.mm[gi] = (np.load(os.path.join(d, f"z_{gid}.npy"), mmap_mode="r"), np.load(os.path.join(d, f"st_{gid}.npy"), mmap_mode="r"),
                        np.load(os.path.join(d, f"fz_{gid}.npy"), mmap_mode="r") if meta["forcing"] else None)
        return s.mm[gi]


class Batches(torch.utils.data.IterableDataset):
    """windows of L frames at the family's stride (corpus.STRIDE: one model step = stride stored frames). stride_aug (ablation): multipliers of
    the stride drawn per micro-batch. Within a family a grid group is drawn with weight ∝ trajectories x tokens per sample (training) or
    ∝ trajectories (token_weight False: the validation batches)."""
    def __init__(s, root, families, splits, t_start, strides, L, n_out, tokens, max_bs, seed, stride_aug=(), family_temp=0.0, pass_cap=0.0, n_micro=0.0, log=None, floor=None,
                 rand_shift=0, periodic=None, token_weight=True):
        super().__init__(); s.root, s.families, s.splits, s.t_start, s.strides, s.L, s.n_out, s.tokens, s.max_bs, s.seed = root, families, splits, t_start, strides, L, n_out, tokens, max_bs, seed
        s.stride_aug = list(stride_aug); s.family_temp = family_temp; s.token_weight = token_weight   # family_temp a: family weight ∝ windows^a (0 = uniform)
        s.floor = floor or {}   # scale floor: {family: (s_floor [C], {grid shape: latent of an all-zero frame})}, see apply_floor
        s.pass_cap, s.n_micro, s.log = pass_cap, n_micro, log
        s.rand_shift, s.periodic = int(rand_shift), dict(periodic or {})
        assert s.rand_shift == 0 or set(s.families) <= set(s.periodic), "--rand_shift: Batches needs every family's periodic axes"

    def mix(s, stores):
        """group weights within each family, training windows per family (train split only) and the mean micro-batch size per family."""
        def tok_of(meta): zc, h, ww = meta["latent"]; return s.n_out * (h // 2) * (ww // 2)
        gw = {k: np.array([len(g[3]) * (tok_of(g[2]) if s.token_weight else 1) for g in st.groups], float) for k, st in stores.items()}
        nwin = {k: sum(max(0, it["n_frames"] - s.t_start[k] - ((s.L - 1) * s.strides[k] + 1) + 1) for g in st.groups if g[0] == "train" or "train" not in s.splits for it in g[3])
                for k, st in stores.items()}
        def B_of(meta): zc, h, ww = meta["latent"]; return int(max(1, min(s.max_bs, s.tokens // max(1, s.n_out * (h // 2) * (ww // 2)))))
        bbar = {k: float(sum(wg * B_of(g[2]) for wg, g in zip(gw[k] / gw[k].sum(), st.groups))) if st.groups else 1.0 for k, st in stores.items()}
        return gw, nwin, bbar

    def __iter__(s):
        wi = torch.utils.data.get_worker_info(); rng = np.random.default_rng([*s.seed, wi.id if wi else 0])
        srng = np.random.default_rng([*s.seed, wi.id if wi else 0, SHIFT_STREAM]) if s.rand_shift else None   # own stream: the windows drawn (rng)
                                                                                                             # are those of the same run without --rand_shift
        stores = {k: Store(s.root, k, s.splits) for k in s.families}; keys = sorted(stores); elig = {}
        gw, nwin, bbar = s.mix(stores)
        fwd, passes = corpus.family_weights({k: nwin[k] for k in keys if stores[k].groups}, bbar, s.family_temp, s.pass_cap, s.n_micro)
        fw = np.array([fwd.get(k, 0.0) for k in keys], float); fw = fw / fw.sum() if fw.sum() > 0 else None
        if s.log and (wi is None or wi.id == 0): s.log("family weights " + ", ".join(f"{k} {fwd.get(k, 0):.3f}" + (f" ({passes[k]:.1f} passes)" if passes.get(k) is not None else "") for k in keys))
        skips = 0
        while True:
            if skips > 100000: raise RuntimeError(f"no family of {keys} has a window of {s.L} frames in splits {s.splits} (empty or too-short store)")
            k = keys[rng.integers(len(keys))] if fw is None else keys[int(rng.choice(len(keys), p=fw))]; st = stores[k]; w = gw[k]
            if not st.groups: skips += 1; continue   # e.g. a family without the requested split (validation batches)
            gi = int(rng.choice(len(st.groups), p=w / w.sum())); sp, gid, meta, its, _ = st.groups[gi]; Z, S, Fz = st.arrays(gi)
            zc, h, ww = meta["latent"]; B = int(max(1, min(s.max_bs, s.tokens // max(1, s.n_out * (h // 2) * (ww // 2)))))
            k_ = s.strides[k]
            if s.stride_aug: k_ = max(1, int(round(k_ * s.stride_aug[rng.integers(len(s.stride_aug))])))
            span = (s.L - 1) * k_ + 1   # stored frames covered by one window
            if (k, gi, k_) not in elig: elig[(k, gi, k_)] = [j for j, it in enumerate(its) if it["n_frames"] - s.t_start[k] >= span]
            ok = elig[(k, gi, k_)]
            if not ok: skips += 1; continue
            skips = 0
            zs, ss, fs, P, RW = [], [], [], [], []
            for j in rng.choice(ok, B):
                it = its[j]; t0 = int(rng.integers(s.t_start[k], it["n_frames"] - span + 1)); r = it["row"]
                zs.append(np.asarray(Z[r, t0:t0 + span:k_])); ss.append(np.asarray(S[r, t0:t0 + span:k_])); P.append(it["params"]); RW.append(it["raw"])
                if Fz is not None: fs.append(np.asarray(Fz[r]))
            zs, ss = np.stack(zs), np.stack(ss)
            if k in s.floor: zs, ss = apply_floor(zs, ss, s.floor[k][0], s.floor[k][1][tuple(meta["shape"])])
            fs = np.stack(fs) if fs else None; sh = None
            if s.rand_shift:
                sh = shift_draws(srng, len(zs), (h, ww), s.periodic[k], s.rand_shift)
                zs = roll_window(zs, sh, (h, ww)); fs = roll_window(fs, sh, (h, ww)) if fs is not None else None
            out = {"family": k, "z": torch.from_numpy(zs), "st": torch.from_numpy(ss),
                   "fz": torch.from_numpy(fs) if fs is not None else None, "params": P, "raw": RW, "stride": k_}
            if sh is not None: out["shift"] = sh
            yield out


def save_atomic(obj, path):
    """torch.save to a temporary file, then rename."""
    torch.save(obj, path + ".tmp"); os.replace(path + ".tmp", path)


FLOOR_REL = 1e-5   # scale floor relative to the median scale: numerically constant frames (e.g. Gray-Scott decayed to the uniform state, flows at rest
                   # in their initial frames) would otherwise have their rounding noise blown up to [-1, 1] by the per-frame normalisation. Such frames get
                   # s = floor and the latent of an all-zero frame; physical small-amplitude fields above the floor are kept as they are.


def scale_floor(st, rel=FLOOR_REL):
    """per-channel floor from a store's stats arrays (list of [N, T, C, 2]): rel x median of the positive scales."""
    v = np.concatenate([np.asarray(a[..., 1]).reshape(-1, a.shape[2]) for a in st]); v = v[(v > 0).all(1)]
    return (rel * np.median(v, 0)).astype(np.float32)


def apply_floor(zs, ss, fl, zzero):
    """zs [..., C, zc, h, w] latents, ss [..., C, 2] (mu, s): frames-channels with s < fl get the all-zero latent and s = fl."""
    m = ss[..., 1] < fl
    if m.any(): zs = zs.copy(); zs[m] = zzero; ss = ss.copy(); ss[..., 1] = np.maximum(ss[..., 1], fl)
    return zs, ss


# ---------------------------------------------------------------- frame representation ----------------------------------------
def to_frames(z, sl):
    """z [B, L, C, zc, h, w] (family channel order) -> latent part of the frames [B, L, MAX_C * zc, h, w]: channel i goes to quantity slot sl[i], other slots zero."""
    B, L, C, zc = z.shape[:4]; h, w = z.shape[-2:]; lat = z.new_zeros(B, L, MAX_C * zc, h, w)
    for i, s in enumerate(sl): lat[:, :, s * zc:(s + 1) * zc] = z[:, :, i]
    return lat


def planes(st, ref, sl):
    """st [B, L, C, 2] (mu, s), ref [B, C, 2] -> plane values [B, L, 2 * MAX_C] at the quantity slots: log-amplitude and mean relative to ref (A_SCALE, B_SCALE)."""
    B, L, C = st.shape[:3]
    a = torch.log(st[..., 1] / ref[:, None, :, 1]).clamp(-5, 5) * A_SCALE; b = torch.asinh((st[..., 0] - ref[:, None, :, 0]) / ref[:, None, :, 1]) * B_SCALE
    v = torch.zeros(B, L, 2 * MAX_C, device=st.device)
    for i, s in enumerate(sl): v[:, :, 2 * s] = a[..., i]; v[:, :, 2 * s + 1] = b[..., i]
    return v


def assemble(lat, pv):
    """latent part [B, L, MAX_C*zc, h, w] + plane values [B, L, 2*MAX_C] -> frames [B, L, CL, h, w]."""
    return torch.cat([lat, pv[..., None, None].expand(-1, -1, -1, *lat.shape[-2:])], 2)


def split_out(y, ref, sl, zc):
    """generated frames [B, F, CL, h, w] -> latents [B, F, C, zc, h, w] (family channel order), stats [B, F, C, 2] (mu, s)."""
    B, F = y.shape[:2]; pv = y[:, :, MAX_C * zc:].mean((-1, -2)); lat = torch.stack([y[:, :, s * zc:(s + 1) * zc] for s in sl], 2)
    a = torch.stack([pv[:, :, 2 * s] for s in sl], -1) / A_SCALE; b = torch.sinh(torch.stack([pv[:, :, 2 * s + 1] for s in sl], -1) / B_SCALE)
    sc = ref[:, None, :, 1] * torch.exp(a); mu = ref[:, None, :, 0] + b * ref[:, None, :, 1]
    return lat, torch.stack([mu, sc], -1)


def chan_mask(sl, zc, B, dev):
    """[B, CL] mask of the latent channels and planes of the present quantity slots."""
    m = torch.zeros(B, MAX_C * zc + 2 * MAX_C, device=dev)
    for s in sl: m[:, s * zc:(s + 1) * zc] = 1; m[:, MAX_C * zc + 2 * s:MAX_C * zc + 2 * s + 2] = 1
    return m


# ---------------------------------------------------------------- conditioning ------------------------------------------------
class Conditioner:
    def __init__(s, tok, a, pstats, texts, fams, dev):
        s.tok, s.a, s.dev, s.texts, s.fams = tok, a, dev, texts, fams
        s.pm = torch.tensor(pstats["mean"], device=dev); s.ps = torch.tensor(pstats["std"], device=dev)
        s.slots = list(pstats["slots"])   # the run's parameter slots
        s.max_f = MAX_F
        s.s_floor = {k: torch.tensor(v, device=dev) for k, v in pstats["s_floor"].items()}   # scale floor per family channel
        s.zc = tok.zc; s.ND = 2 * len(s.slots) + 3 * MAX_C; s.tcache = {}

    def fl(s, family, rep):   # floor per row for frames laid out as (rep, C) -> [rep * C], or None
        return s.s_floor[family].repeat(rep) if family in s.s_floor else None

    def _bs(s, H, W):   # tokenizer batch by pixel budget: --enc_bs frames of 256^2
        return max(1, min(4 * s.a.enc_bs, s.a.enc_bs * 65536 // max(1, H * W)))

    @torch.no_grad()
    def encode(s, x, periodic, fl=None):               # physical frames [N, H, W] -> latents [N, zc, h, w], stats [N, 2] (per-frame normalisation)
        mu, sc = fieldnorm.scale_t(x)                  # fl [N]: scale floor per row; rows below it are encoded as an all-zero frame with s = fl (as Batches does)
        xn = (x - mu[:, None, None]) / sc[:, None, None]
        if fl is not None: low = sc < fl; xn = torch.where(low[:, None, None], torch.zeros_like(xn), xn); sc = torch.maximum(sc, fl)
        with s._vae(periodic): z = torch.cat(s._chunked(lambda t: s.tok.encode(t[:, None]).float(), xn, s._bs(*xn.shape[-2:])))
        return z, torch.stack([mu, sc], -1)

    @torch.no_grad()
    def encode_fixed(s, x, periodic):                  # condition fields already in model scale (corpus.Family.cond_model): no fieldnorm
        with s._vae(periodic): return torch.cat(s._chunked(lambda t: s.tok.encode(t[:, None]).float(), x, s._bs(*x.shape[-2:])))

    def _vae(s, periodic):   # the store's VAE padding (VAE_WRAP); families without a periodic axis keep the unpatched layers
        return vae_periodic.periodic(s.tok.vae, periodic) if VAE_WRAP[0] and any(periodic) else contextlib.nullcontext()

    @staticmethod
    def _chunked(fn, x, bs):   # fn over chunks of bs rows; on CUDA OOM the chunk is halved
        out, i = [], 0
        while i < len(x):
            try:
                with torch.autocast("cuda", dtype=torch.bfloat16): out.append(fn(x[i:i + bs]))
                i += bs
            except torch.OutOfMemoryError:
                if bs <= 1: raise
                bs = max(1, bs // 2); torch.cuda.empty_cache()
        return out

    @torch.no_grad()
    def decode(s, z, shape, periodic):                 # [N, zc, h, w] -> normalised [N, H, W] on the family grid
        with s._vae(periodic): return torch.cat(s._chunked(lambda t: s.tok.decode(t).float()[:, 0], z, s._bs(z.shape[-2] * s.tok.factor, z.shape[-1] * s.tok.factor)))

    def cond(s, family, params, raws, fz, ref, sl, B, h, w, stride=None):
        """numeric vector [B, ND], channel mask [B, MAX_C], condition-field planes [B, max_f * (zc + 1), h, w], text [B, L, 4096]; sl = quantity
        slots of the channels; stride = the stride the window was taken at (None = the family's); fz [B, zc, h, w] (one field) or [B, nf, zc, h, w]."""
        a = s.a; dev = s.dev; nS = len(s.slots); fam = s.fams[family]
        num = torch.zeros(B, s.ND, device=dev)
        if a.cond in ("all", "nofield"):   # log10_dt includes the stride
            vm = [corpus.param_vector(corpus.effective_params(family, p, stride), s.slots) for p in params]
            v = torch.from_numpy(np.stack([x for x, _ in vm])).to(dev); m = torch.from_numpy(np.stack([y for _, y in vm])).to(dev)
            num[:, :nS] = (v - s.pm) / s.ps * m; num[:, nS:2 * nS] = m
        for i, q in enumerate(sl):   # state scalars of every present slot: log10 reference amplitude, asinh(mean / amplitude), presence
            num[:, 2 * nS + q] = torch.log10(ref[:, i, 1]); num[:, 2 * nS + MAX_C + q] = torch.asinh(ref[:, i, 0] / ref[:, i, 1]); num[:, 2 * nS + 2 * MAX_C + q] = 1
        chm = torch.zeros(B, MAX_C, device=dev); chm[:, list(sl)] = 1
        fo = torch.zeros(B, s.max_f * (s.zc + 1), h, w, device=dev)
        if a.cond in ("all", "nonum") and fz is not None and not a.drop_fields:   # --drop_fields (evaluation only): the field-condition counterfactual; --cond nofield: never
            fz = fz.to(dev).float(); fz = fz[:, None] if fz.dim() == 4 else fz   # [B, nf, zc, h, w]
            assert fz.shape[1] <= s.max_f, f"{family}: {fz.shape[1]} condition fields but {s.max_f} condition-field slots (MAX_F; fields would be dropped silently)"
            for i in range(min(fz.shape[1], s.max_f)): fo[:, i * (s.zc + 1):i * (s.zc + 1) + s.zc] = fz[:, i]; fo[:, i * (s.zc + 1) + s.zc] = 1
        txt = [fam.text_of(r) for r in raws]
        return num, chm, fo, s.text_batch(txt)

    def text_one(s, t):
        if t not in s.tcache:
            o = torch.zeros(s.a.text_len, 4096, device=s.dev)
            tp_ = getattr(s.a, "text_probe", "none")
            if tp_ == "zero": s.tcache[t] = o; return o   # 'swap': each text replaced by another text of the embedding file (sorted keys, shifted by half; mapping logged)
            if tp_ == "swap" and s.texts is not None and t:
                tkeys = sorted(k for k in s.texts if k); t2 = tkeys[(tkeys.index(t) + len(tkeys) // 2) % len(tkeys)]; assert t2 != t
                print(f"[text_probe swap] {t[:70]!r} -> {t2[:70]!r}", flush=True); n = min(s.a.text_len, s.texts[t2].shape[0]); o[:n] = s.texts[t2][:n].to(s.dev).float()
                s.tcache[t] = o; return o
            if s.texts is not None:
                e = s.texts.get(t)
                if e is None: raise KeyError(f"no text embedding for {t[:80]!r} (run textemb.py)")
                n = min(s.a.text_len, e.shape[0]); o[:n] = e[:n].to(s.dev).float()
            s.tcache[t] = o
        return s.tcache[t]

    def text_batch(s, txt):
        if len(set(txt)) == 1: return s.text_one(txt[0])[None].expand(len(txt), -1, -1)
        return torch.stack([s.text_one(t) for t in txt])


# ---------------------------------------------------------------- flow matching ------------------------------------------------
def sample_sigma(B, shift, gen, dev):
    u = torch.sigmoid(torch.randn(B, generator=gen, device=dev)); return shift * u / (1 + (shift - 1) * u)


def sigma_grid(n, shift):
    t = torch.linspace(1, 0, n + 1); return shift * t / (1 + (shift - 1) * t)


def generate(model, hist, chm, fo, num, txt, n_out, ode_steps, shift, gen, cm, objective="fm"):
    """one round: n_out frames from the history by the flow-matching ODE (ode_steps Euler steps from pure noise on the shifted sigma grid);
    objective "regress": the deterministic control, one forward pass from a zero input at sigma = 0."""
    B, K, CL, h, w = hist.shape
    if objective == "regress":
        with torch.autocast("cuda", dtype=torch.bfloat16): v = model(hist.new_zeros(B, n_out, CL, h, w), hist.new_zeros(B), hist, chm, fo, num, txt).float()
        return v * cm[:, None, :, None, None]
    x = torch.randn(B, n_out, CL, h, w, generator=gen, device=hist.device) * cm[:, None, :, None, None]
    if NOISE_ROLL[0] is not None: x = torch.roll(x, NOISE_ROLL[0], dims=(-2, -1))   # --equiv_eval: the same noise, translated with the inputs
    sg = sigma_grid(ode_steps, shift).to(hist.device)
    for i in range(ode_steps):
        with torch.autocast("cuda", dtype=torch.bfloat16): v = model(x, sg[i].expand(B), hist, chm, fo, num, txt).float() * cm[:, None, :, None, None]
        x = x + (sg[i + 1] - sg[i]) * v
    return x * cm[:, None, :, None, None]


# ---------------------------------------------------------------- evaluation ---------------------------------------------------
def eval_start(fam, it, need):
    return fam.t_start + (it.n_frames - fam.t_start - need) // 4


def eval_windows(fam, its, eval_n, seed, frames, stride=None):
    """the evaluation windows of one (family, split): [(item, first stored frame)], each window = EVAL_HIST history + `frames` targets at
    the family's stride (or `stride`), i.e. (EVAL_HIST + frames - 1) * stride + 1 stored frames; the prediction start is first + EVAL_HIST * stride."""
    stride = fam.stride if stride is None else stride
    need = (EVAL_HIST + frames - 1) * stride + 1; its = [it for it in its if it.n_frames - fam.t_start >= need]
    if not its: return []
    rng = np.random.default_rng(seed + 7)
    if len(its) >= eval_n:
        sel = [its[i] for i in rng.choice(len(its), eval_n, replace=False)]
        return [(it, eval_start(fam, it, need)) for it in sel]
    # fewer trajectories than windows: up to m non-overlapping
    # windows per trajectory, evenly spread over its usable span, at most eval_n in total
    m = -(-eval_n // len(its)); cand = []
    for it in its:
        span = it.n_frames - fam.t_start - need; mi = min(m, span // need + 1)
        starts = [fam.t_start + (span * j) // (mi - 1) for j in range(mi)] if mi > 1 else [eval_start(fam, it, need)]
        cand += [(it, t) for t in starts]
    if len(cand) > eval_n: cand = [cand[i] for i in sorted(rng.choice(len(cand), eval_n, replace=False))]
    return cand


def divergence(val_hist, ratio, n=2, span=0):
    if ratio <= 0 or len(val_hist) < n + 1: return None
    mins, flags = {}, []
    for e in val_hist:
        vf = e.get("val_fam") or {"all": e["val"]}
        for k, v in vf.items():
            if math.isfinite(v): mins[k] = min(mins.get(k, v), v)
        flags.append(float(np.median([vf[k] / max(mins.get(k, 1e-12), 1e-12) if math.isfinite(vf[k]) else math.inf for k in vf])) > ratio)
    if not all(flags[-n:]): return None
    i = len(flags) - 1
    while i > 0 and flags[i - 1]: i -= 1
    return val_hist[i]["step"] if val_hist[-1]["step"] - val_hist[i]["step"] >= span else None


def ens_match(a, fam_key, sp, pats=None):
    """does --ens_splits select (family, split): globs on the split name ('test' = every family's test) or on 'family/split' ('kolmo/test')"""
    pats = pats or (["*"] if a.ens_splits == "all" else a.ens_splits.split(","))
    return any(fnmatch.fnmatch(sp, p) or fnmatch.fnmatch(f"{fam_key}/{sp}", p) for p in pats)


def ens_window_ids(win, n, seed_off):
    rng = np.random.default_rng(EVAL_SEED + 13 + seed_off); byt = {}
    for it, t in win: byt.setdefault((it.path, it.traj), []).append((it.path, it.traj, t))
    trajs = list(byt); lists = [[byt[trajs[i]][j] for j in rng.permutation(len(byt[trajs[i]]))] for i in rng.permutation(len(trajs))]; out = []
    for r_ in range(max(map(len, lists), default=0)):
        out += [l_[r_] for l_ in lists if r_ < len(l_)]
    return set(out[:n])


def evaluate_split(model, cd, fam, its, a, gen, sp="", eval_bs=0):
    """one (family, split): --eval_n windows rolled out for --eval_chunks x F frames. Per window / frame / channel: relative L2 of the
    prediction, of persistence and of two-frame linear extrapolation, VRMSE; the tokenizer floor at frame 1; last-frame physics
    statistics vs the truth and vs the truth's encode-decode reconstruction. On the splits
    matching --ens_splits the --ens_n windows of ens_window_ids are generated --ens times: ensemble-mean error, spread, CRPS. Entries whose truth is
    numerically constant are undefined (eval_metrics.apply_degenerate)."""
    if getattr(model, "bc_pos", False): model.set_bc(fam.bc)
    K, F, Hh = a.k_in, a.n_out, EVAL_HIST; frames = F * a.eval_chunks; kst = fam.stride * a.eval_stride_mult; need = (Hh + frames - 1) * kst + 1; C = len(fam.channels); zc = cd.zc
    win = eval_windows(fam, its, a.eval_n, EVAL_SEED, frames, kst); sl = fam.slots   # eval_stride_mult > 1: the same-family-other-dt column
    if not win: return {"skipped": f"no trajectory with >= {need} frames"}
    ens_here = a.ens > 1 and ens_match(a, fam.key, sp)
    wid = lambda w: (w[0].path, w[0].traj, w[1])
    ens_ids = ens_window_ids(win, a.ens_n, fam.split_seed(sp))   # every run marks them (per_window 'ens', 'ens_subset'); only ens_here generates the ensemble
    by_shape = {}
    for it, t0 in win: by_shape.setdefault(tuple(it.shape), []).append((it, t0))
    for g_ in by_shape.values(): g_.sort(key=lambda w: wid(w) not in ens_ids)   # ensemble windows first in their grid group (stable), so few batches carry them;
    # the batch layout, and with it the task-seeded noise of every window, is the same with and without --ens
    NONFIN = [0, 0]; DUMP = []; DM = []; SH, AM, CR, EQ, EQNF = [], [], [], [], [0]; E, PE, LE, V, FL, EP, PEP, TP = [], [], [], [], [], [], [], []; NFW = []; GT, GR, BT, BR = new_acc(C); GC, BC = new_acc(C)[:2]; order = []   # order = windows in the order of E's rows
    EN = {"mean": [], "single": [], "spread": [], "crps": []}; ENR = []; TS = ([], [], [])   # ENR: rows of E that carry an ensemble; TS: truth denominators
    for shape, group in by_shape.items():
        for i0 in range(0, len(group), eval_bs or a.eval_bs):
            gw = group[i0:i0 + (eval_bs or a.eval_bs)]; g = [it for it, _ in gw]; B = len(g); n_before = len(order); order += gw
            X = torch.from_numpy(np.stack([fam.load(it, t, t + need)[::kst] for it, t in gw])).to(cd.dev)[:, Hh - K:]   # frames at the family's stride
            zh0, sth0 = cd.encode(X[:, :K].reshape(-1, *shape), fam.periodic, cd.fl(fam.key, B * K)); h_, w_ = zh0.shape[-2:]
            zh0 = zh0.view(B, K, C, zc, h_, w_); sth0 = sth0.view(B, K, C, 2)
            def enc_fields(g, px=None):   # the condition fields of the batch (translated by px pixels: --equiv_eval) -> [B, nf, zc, h, w], as encode_corpus.encode_cond
                cf = [fam.cond_model(it) for it in g]; zs_ = []
                for i_ in range(len(cf[0])):
                    f = torch.from_numpy(np.stack([c[i_] for c in cf])).to(cd.dev).float()
                    zs_.append(cd.encode_fixed(f if px is None else torch.roll(f, px, dims=(-2, -1)), fam.periodic))
                return torch.stack(zs_, 1)
            fz = enc_fields(g) if fam.cond_fields(g[0]) else None   # condition fields (Kolmogorov forcing; acoustic density, sound speed)
            def rollout(g_, zh0=zh0, sth0=sth0, fz=fz):   # one sample of the whole rollout -> physical frames [B, frames, C, H, W]
                zh, sth, Zs, Ss = zh0, sth0, [], []
                for c in range(a.eval_chunks):
                    ref = sth[:, -1]; cm = chan_mask(sl, zc, B, cd.dev)
                    hist = assemble(to_frames(zh, sl), planes(sth, ref, sl))
                    num, chm, fo, txt = cd.cond(fam.key, [it.params for it in g], [it.raw for it in g], fz, ref, sl, B, h_, w_, kst)
                    y = generate(model, hist, chm, fo, num, txt, F, a.ode_steps, a.sample_shift or a.fm_shift, g_, cm, a.objective)
                    zn, sn = split_out(y, ref, sl, zc)
                    Zs.append(zn); Ss.append(sn)
                    zh = torch.cat([zh, zn], 1)[:, -K:]; sth = torch.cat([sth, sn], 1)[:, -K:]
                Zs = torch.cat(Zs, 1); Ss = torch.cat(Ss, 1)                                                      # [B, frames, C, ...]
                xn = cd.decode(Zs.reshape(-1, zc, h_, w_), shape, fam.periodic).view(B, frames, C, *shape)
                return xn * Ss[..., 1, None, None] + Ss[..., 0, None, None]

            def recon(x):      # encode-decode of true frames [B, C, H, W] (tokenizer floor)
                z_, s_ = cd.encode(x.reshape(-1, *shape), fam.periodic, cd.fl(fam.key, B)); s_ = s_.view(B, C, 2)
                return cd.decode(z_, shape, fam.periodic).view(B, C, *shape) * s_[..., 1, None, None] + s_[..., 0, None, None]

            eq_ = a.equiv_eval and any(fam.periodic)
            if eq_: gst = gen.get_state()   # --equiv_eval: the translated rollout below replays this noise
            P = rollout(gen)
            T = X[:, K:]; xp = X[:, K - 2] if K >= 2 else X[:, K - 1]
            if eq_:
                # noise translated by one DiT token (16 pixels = 2 latent pixels x patch 2), rolled out, translated back, compared with P
                tpx = 2 * cd.tok.factor   # one DiT token (2 latent px) in physical px
                px = tuple(tpx * int(p_) for p_ in fam.periodic); lp = tuple(2 * int(p_) for p_ in fam.periodic)
                Xs = torch.roll(X[:, :K], px, dims=(-2, -1)); zs0, ss0 = cd.encode(Xs.reshape(-1, *shape), fam.periodic, cd.fl(fam.key, B * K))
                fzs = enc_fields(g, px) if fz is not None else None
                g2 = torch.Generator(device=cd.dev); g2.set_state(gst); NOISE_ROLL[0] = lp
                try: Ps = rollout(g2, zs0.view(B, K, C, zc, h_, w_), ss0.view(B, K, C, 2), fzs)
                finally: NOISE_ROLL[0] = None
                eqe_, nfq_ = nonfinite_to_inf(torch.stack([P, Ps]), rel_l2(torch.roll(Ps, tuple(-q for q in px), dims=(-2, -1)) - P, T)); EQ.append(eqe_.cpu()); EQNF[0] += nfq_   # either rollout non-finite -> +inf, counted
            if a.dump_pred and len(DUMP) < a.dump_pred:   # --dump_pred (figures only): history, predicted and true physical frames of the first windows
                rs_ = sorted({min(16, T.shape[1]), T.shape[1]}); RC_ = torch.stack([recon(T[:, s_ - 1]) for s_ in rs_], 1)   # encode-decode of the true frames at steps 16 / last (autoencoder spectra)
                for j in range(min(B, a.dump_pred - len(DUMP))): DUMP.append((gw[j], X[j, :K].half().cpu().numpy(), P[j].half().cpu().numpy(), T[j].half().cpu().numpy(), RC_[j].half().cpu().numpy(), rs_))
            e_, ep_, v_, dm_, sh_, am_, cr_, nf_ = nonfinite_to_inf(P, rel_l2(P - T, T), rel_l2_pooled(P - T, T), vrmse(P, T), rel_l2_demeaned(P, T), *shape_amp(P, T)); NONFIN[0] += nf_
            NFW.append((~torch.isfinite(P)).flatten(1).any(1).cpu())
            E.append(e_.cpu()); DM.append(dm_.cpu()); SH.append(sh_.cpu()); AM.append(am_.cpu()); CR.append(cr_.cpu()); PE.append(rel_l2(X[:, K - 1:K] - T, T).cpu())
            EP.append(ep_.cpu()); PEP.append(rel_l2_pooled(X[:, K - 1:K] - T, T).cpu())   # channels pooled: robust when one channel's truth is near zero
            LE.append(rel_l2(linear_extrap(xp, X[:, K - 1], frames) - T, T).cpu()); V.append(v_.cpu())
            rc_ = recon(T[:, 0]); FL.append(rel_l2(rc_ - T[:, 0], T[:, 0]).cpu())
            TP.append(rel_l2(recon(X[:, K - 1])[:, None] - T, T).cpu())   # persistence through the tokenizer: shares the model's floor
            accumulate_last_frame(T, P, fam.periodic, GT, GR, BT, BR); accumulate_pred_last_frame(recon(T[:, -1]), fam.periodic, GC, BC)
            for l_, v_ in zip(TS, truth_stats(T)): l_.append(v_.cpu())
            sel = [j for j, w in enumerate(gw) if wid(w) in ens_ids] if ens_here else []
            if sel:   # ensemble on the selected windows of this batch: member 0 = the sample above
                M = torch.stack([P] + [rollout(torch.Generator(device=cd.dev).manual_seed(a.seed * 1000 + 17 * m + fam.split_seed(sp) + 1009 * n_before)) for m in range(1, a.ens)])   # + window index: members differ between windows
                M = M[:, sel]; Ts = T[sel]; ENR += [n_before + j for j in sel]
                em_, esp_, ec_, nfe_ = nonfinite_to_inf(M, rel_l2(M.mean(0) - Ts, Ts), M.var(0, unbiased=True).sum((-1, -2)).sqrt() / (Ts ** 2).sum((-1, -2)).sqrt().clamp_min(1e-12), crps_rel(M, Ts))
                NONFIN[1] += nfe_
                EN["mean"].append(em_.cpu()); EN["single"].append(E[-1][sel]); EN["spread"].append(esp_.cpu()); EN["crps"].append(ec_.cpu()); del M
    d = {"E": torch.cat(E), "DM": torch.cat(DM), "SH": torch.cat(SH), "AM": torch.cat(AM), "CR": torch.cat(CR), "EQ": torch.cat(EQ) if EQ else None, "PE": torch.cat(PE), "LE": torch.cat(LE), "V": torch.cat(V), "FL": torch.cat(FL), "EP": torch.cat(EP), "PEP": torch.cat(PEP), "TP": torch.cat(TP),
         "ens": {k_: torch.cat(v_) for k_, v_ in EN.items()} if EN["mean"] else None}
    d = apply_degenerate(d, *[torch.cat(l_) for l_ in TS], ens_rows=torch.tensor(ENR, dtype=torch.long))
    E, PE, LE, V, TP = d["E"], d["PE"], d["LE"], d["V"], d["TP"]; nm = torch.nanmean; fv = lambda x: None if torch.isnan(x) else float(x)
    r = physics_metrics(E, PE, d["FL"], GT, GR, BT, BR, fam.channels, F, LE=LE, V=V, GC=GC, BC=BC, ens=d["ens"], members=a.ens, EP=d["EP"], PEP=d["PEP"], TP=TP,
                        groups=[fam.ci_group(it) for it, _ in order], degenerate=d["degenerate"])
    es = [i for i, w in enumerate(order) if wid(w) in ens_ids]   # the single-sample error on the ensemble windows (= ensemble.single_rel_l2 when there is one)
    r.update(demeaned_summary(d["DM"]))
    r.update(shape_amp_summary(d["SH"], d["AM"], d["CR"]))
    NFWc = torch.cat(NFW)
    if d["EQ"] is not None:   # --equiv_eval: ||shift^-1 f(shift x) - f(x)|| / ||truth|| per frame, mean over windows and channels
        fq = torch.nanmean(d["EQ"], (0, 2)); r.update({f"equiv_rel_l2@{j}": fv(fq[j - 1]) for j in sorted({1, min(4, F * a.eval_chunks), F * a.eval_chunks})}); r["equiv_rel_l2_by_frame"] = [fv(x) for x in fq]; r["equiv_nonfinite_windows"] = EQNF[0]
    r["nonfinite_pred_windows"] = NONFIN[0]; r["nonfinite_ens_windows"] = NONFIN[1]   # windows with any non-finite predicted value (sample / any ensemble member;
    # their errors are +inf); both must be 0 for a reportable run
    r["ens_subset"] = {"windows": len(es), **{f"rel_l2@{j}": fv(nm(E[es, j - 1])) for j in sorted({1, E.shape[1]})}} if es else None
    # per-window record: the graded extrapolation splits (nu_dense_ends, nsu extrap_ends) pool many nu values, so the error as a
    # function of the extrapolation distance is drawn from these rows; bootstrap intervals use them too
    EPd = d["EP"]
    if DUMP:   # one file per (family, split): results/pred/<name>/<family>__<split>.npz, window i = H<i> (history), P<i> (prediction), T<i> (truth), [frames, C, H, W]
        dp = os.path.join(HERE, "results", "pred", a.name); os.makedirs(dp, exist_ok=True); arr = {}
        for i, (w, h, p_, t_, rc_, _) in enumerate(DUMP): arr[f"H{i}"], arr[f"P{i}"], arr[f"T{i}"], arr[f"R{i}"] = h, p_, t_, rc_   # R<i>: [len(recon_steps), C, H, W]
        meta = {"windows": [{"path": w[0].path, "traj": str(w[0].traj), "first_frame": int(w[1]), "raw": w[0].raw} for w, *_ in DUMP], "channels": list(fam.channels), "stride": kst,
                "recon_steps": DUMP[0][5]}
        np.savez_compressed(os.path.join(dp, f"{fam.key}__{sp}.npz"), meta=json.dumps(meta, default=str), **arr)
    r["per_window"] = [{"path": it.path, "traj": str(it.traj), "first_frame": int(t0), "raw": it.raw, "ens": wid((it, t0)) in ens_ids,
                        "rel_l2@1": fv(nm(E[i, 0])), f"rel_l2@{E.shape[1]}": fv(nm(E[i, -1])),
                        "rel_l2_pooled@1": fv(EPd[i, 0]), f"rel_l2_pooled@{E.shape[1]}": fv(EPd[i, -1]),
                        "rel_l2_demeaned@1": fv(nm(d["DM"][i, 0])), f"rel_l2_demeaned@{E.shape[1]}": fv(nm(d["DM"][i, -1])),
                        "rel_l2_shape@1": fv(nm(d["SH"][i, 0])), f"rel_l2_shape@{E.shape[1]}": fv(nm(d["SH"][i, -1])), "log_amp_ratio@1": fv(nm(d["AM"][i, 0])), f"log_amp_ratio@{E.shape[1]}": fv(nm(d["AM"][i, -1])), "pattern_corr@1": fv(nm(d["CR"][i, 0])), f"pattern_corr@{E.shape[1]}": fv(nm(d["CR"][i, -1])),
                        "persistence@1": fv(nm(PE[i, 0])), "tok_persistence@1": fv(nm(TP[i, 0])), "linear@1": fv(nm(LE[i, 0])), "vrmse@1": fv(nm(V[i, 0])),
                        f"vrmse@{E.shape[1]}": fv(nm(V[i, -1])), "degenerate_channels@1": [c for ci, c in enumerate(fam.channels) if torch.isnan(E[i, 0, ci])],
                        "nonfinite_pred": int(NFWc[i]), "rel_l2_by_frame": [fv(nm(E[i, f_])) for f_ in range(E.shape[1])]} for i, (it, t0) in enumerate(order)]
    return r


# ---------------------------------------------------------------- low-rank adaptation (the 14B comparison) --------------------
IO_PREFIXES = ("num_mlp.", "bc_embed.", "net.patch_embedding.", "net.proj_out.")   # new or extended layers: input embedding, output head, number MLP, boundary embedding


def is_io(k): return k.startswith(IO_PREFIXES)


def is_lora(k): return ".lora_A." in k or ".lora_B." in k


def base_key(k):
    """parameter name in an adapted WanGen -> its plain name (peft keeps a wrapped layer X as X.base_layer and adds X.lora_A / X.lora_B.default)."""
    return re.sub(r"\.lora_[AB]\.default\.", ".", k.replace(".base_layer.", "."))


def is_body(k):
    """pretrained-body parameter = it has a W0 counterpart (blocks, condition embedders, the top-level scale-shift table); the IO layers
    and the LoRA factors are new or extended layers: they keep --wd."""
    return not is_lora(k) and not is_io(base_key(k))


def set_trainable(net, a):
    """--lora_r: trainable = LoRA factors + the IO layers + the condition embedders"""
    for k, p in net.named_parameters():
        b = base_key(k)
        p.requires_grad_(is_lora(k) or is_io(b) or b.startswith("net.condition_embedder."))


def param_table(net):
    """parameters total and trainable per group (io / body / lora) and bytes per dtype -> result json 'params'."""
    t, by = {}, {}
    for k, p in net.named_parameters():
        g = "lora" if is_lora(k) else "io" if is_io(base_key(k)) else "body"; n = p.numel(); e = t.setdefault(g, [0, 0]); e[0] += n; e[1] += n * p.requires_grad
        dt = str(p.dtype).replace("torch.", ""); by[dt] = by.get(dt, 0) + n * p.element_size()
    return {"total": sum(v[0] for v in t.values()), "trainable": sum(v[1] for v in t.values()), "groups": {g: {"total": v[0], "trainable": v[1]} for g, v in sorted(t.items())},
            "bytes_by_dtype": by}


def ptab_line(t):
    return f"trainable {t['trainable'] / 1e6:.1f}M of {t['total'] / 1e6:.1f}M (" + ", ".join(f"{g} {v['trainable'] / 1e6:.1f}M" for g, v in t["groups"].items() if v["trainable"]) + \
           "); " + ", ".join(f"{d} {b / 2**30:.1f} GB" for d, b in t["bytes_by_dtype"].items())


def load_plain(m, sd):
    """load a plain state dict (standard keys, e.g. an ema.pt) into a WanGen that may carry LoRA adapters (--init_weights with --lora_r): X.weight /
    X.bias of an adapted layer go to X.base_layer.*, the adapters keep their initialisation (B = 0). Without adapters: load_state_dict as before."""
    own = m.state_dict()
    if not any(".base_layer." in k for k in own): return m.load_state_dict(sd)
    mp = {k.replace(".base_layer.", "."): k for k in own if not is_lora(k)}; miss = [k for k in mp if k not in sd]; extra = [k for k in sd if k not in mp]
    assert not miss and not extra, f"plain weights into the adapted model: missing {miss[:5]}, unexpected {extra[:5]}"
    return m.load_state_dict({mp[k]: v for k, v in sd.items()}, strict=False)


def w0_source(a):
    """fp32 start weights of the frozen layers (standard WanGen keys) for restore_frozen: the --init_weights run's ema.pt, else the released DiT of
    --size (safetensors in the HF cache, read lazily; the 1.3B transformer is stored in fp32, so this is exactly what from_pretrained loaded)."""
    if a.init_weights:
        sd = torch.load(os.path.join(HERE, "ckpt", a.init_weights + ".ema.pt"), map_location="cpu", weights_only=False, mmap=True)["ema"]; return lambda k: sd[k]
    _, get = gen_model.released_tensors(a.size); return lambda k: get(k[len("net."):])


def export_sd(m):
    """the state dict written to ema.pt by a LoRA run: standard keys (adapters merged before), floating tensors in fp32 (a frozen
    weight whose fp32 source did not match stays bf16 in the model and is converted here), so --eval_only reads it as any other run's."""
    sd = m.state_dict(); bad = [k for k in sd if is_lora(k) or ".base_layer." in k]; assert not bad, f"ema.pt with adapter keys: {bad[:3]}"
    return {k: (v.float() if v.is_floating_point() and v.dtype != torch.float32 else v) for k, v in sd.items()}


def host_mem():
    """host memory in GB: this process's peak RSS and, from the Slurm job's unified cgroup (memory.peak includes the page cache of the memory-mapped
    stores and safetensors shards, 'anon' does not), the job's peak, current, anonymous and limit; keys missing where unreadable."""
    import resource
    out = {"rss_peak_gb": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 2**20}
    try:
        d = "/sys/fs/cgroup" + open("/proc/self/cgroup").read().strip().splitlines()[-1].split(":", 2)[2]
        while len(d) > len("/sys/fs/cgroup") and not os.path.basename(d).startswith("job_"): d = os.path.dirname(d)
        if os.path.basename(d).startswith("job_"):
            readf = lambda f: open(os.path.join(d, f)).read().strip(); mx = readf("memory.max"); st = dict(l.split() for l in readf("memory.stat").splitlines())
            out.update(job_peak_gb=int(readf("memory.peak")) / 2**30, job_current_gb=int(readf("memory.current")) / 2**30, job_anon_gb=int(st.get("anon", 0)) / 2**30,
                       job_max_gb=None if mx == "max" else int(mx) / 2**30)
    except (OSError, ValueError, IndexError): pass
    return out


# ---------------------------------------------------------------- main -------------------------------------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--name", required=True); ap.add_argument("--init", default="pretrained", choices=["pretrained", "random"], help="pretrained Wan2.1 DiT or the same architecture randomly initialised (ablation)")
    ap.add_argument("--tok", default="wan21", help="wan21 (the store's tokenizer) or, at evaluation, wan21ft:<decoder checkpoint> (train_tok2d.py)")
    ap.add_argument("--store", required=True, help="latent store (encode_corpus.py --store) made with the Wan2.1 VAE")
    ap.add_argument("--cond", default="all", choices=["all", "nonum", "nofield"], help="all = numbers, text and condition fields; nofield = numbers and text kept, condition-field channels always zero; nonum = numbers removed only, text and condition fields kept")
    ap.add_argument("--reference", action="store_true", help="in-range reference: add the ref_* trajectories of every held-out condition to training")
    ap.add_argument("--objective", default="fm", choices=["fm", "regress"], help="fm: flow matching (generative); regress: same network trained by MSE to output the frames directly (deterministic control)")
    ap.add_argument("--textemb", default="textemb"); ap.add_argument("--text_len", type=int, default=128)
    ap.add_argument("--k_in", type=int, default=2); ap.add_argument("--n_out", type=int, default=4)
    ap.add_argument("--tokens", type=int, default=16384, help="latent tokens per GPU micro-batch"); ap.add_argument("--max_bs", type=int, default=128)
    ap.add_argument("--lr", type=float, default=1e-4); ap.add_argument("--beta2", type=float, default=0.95, help="AdamW beta2"); ap.add_argument("--wd", type=float, default=0.01); ap.add_argument("--warmup", type=int, default=1000)
    ap.add_argument("--budget_steps", type=int, default=0, help="train exactly this many steps per rank, WSD schedule by steps; 0 = by --budget_gpu_h"); ap.add_argument("--budget_gpu_h", type=float, default=60.0); ap.add_argument("--max_steps", type=int, default=10 ** 9); ap.add_argument("--decay_frac", type=float, default=0.15)
    ap.add_argument("--ema", type=float, default=0.999); ap.add_argument("--fm_shift", type=float, default=3.0); ap.add_argument("--clip", type=float, default=1.0)
    ap.add_argument("--workers", type=int, default=6); ap.add_argument("--enc_bs", type=int, default=64); ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--eval_n", type=int, default=32); ap.add_argument("--eval_bs", type=int, default=4); ap.add_argument("--eval_chunks", type=int, default=4); ap.add_argument("--ode_steps", type=int, default=25)
    ap.add_argument("--div_ratio", type=float, default=3.0, help="stop (exit 3, no evaluation) when the median over families of val loss / its minimum so far exceeds this at 2 validations in a row; 0 = off")
    ap.add_argument("--grad_ckpt", action="store_true", help="gradient checkpointing of the transformer blocks (the 14B runs)"); ap.add_argument("--log_every", type=int, default=100)
    ap.add_argument("--wait_stores_h", type=float, default=8.0, help="wait up to this many hours for missing latent stores (encode jobs) before failing")
    ap.add_argument("--eval_only", default="", help="name of a trained run: load ckpt/<it>.ema.pt and only evaluate (e.g. with the fine-tuned decoder via --tok)")
    ap.add_argument("--stride_aug", default="", help="stride augmentation (ablation): comma list of multipliers of the family stride drawn per micro-batch, e.g. 1,2 (dt slot = actual stride)")
    ap.add_argument("--eval_stride_mult", type=int, default=1, help="evaluate at this multiple of the family stride (same family, other dt; the dt slot follows)")
    ap.add_argument("--val_every", type=int, default=1000, help="every N steps: loss of the EMA weights on fixed val batches (--val_batches per rank), for the divergence check")
    ap.add_argument("--val_batches", type=int, default=16)
    ap.add_argument("--ens", type=int, default=1, help="samples per window on the splits matching --ens_splits (ensemble-mean error, spread, CRPS); 1 = off")
    ap.add_argument("--ens_n", type=int, default=16, help="windows per split that get the ensemble"); ap.add_argument("--ens_splits", default="test", help="comma list of globs on the split name (test) or on family/split (kolmo/test), or all")
    ap.add_argument("--text_probe", default="none", choices=["none", "zero", "swap"], help="evaluation only (--eval_only): equation-identity text counterfactual; zero = all-zero text tokens, swap = another equation's text")
    ap.add_argument("--drop_fields", action="store_true", help="evaluation only (--eval_only): zero the condition-field channels (forcing / acoustic media) -> how much the model relies on them")
    ap.add_argument("--bc_pos", action="store_true", help="boundary-aware positions: periodic rotary harmonics on periodic axes "
                    "(exact translation equivariance on the token ring) + a zero-initialised boundary-distance embedding per non-periodic side type (corpus.Family.bc)")
    ap.add_argument("--kolmo_extra", default="", help="optional Kolmogorov groups: "
                    "anchor = the `anchor` split joins training (amplitude 0.7 / 1.4 at the training k_f and nu; the family weight still counts `train` only); "
                    "drag = the held-out drag columns kolmo/extrap_drag_lo / _hi (alpha 0.05 / 0.2, not a model input) are evaluated, their ref_* splits train only "
                    "with --reference; beta = the beta-plane groups (beta_anchor trains, extrap_beta is evaluated) and the `beta` parameter slot. "
                    "--eval_only uses the command line's")
    ap.add_argument("--equiv_eval", action="store_true", help="evaluation: also roll out every window of a periodic family with history, condition fields and noise translated by one DiT token on the periodic axes and report the translation-equivariance error")
    ap.add_argument("--rand_shift", type=int, default=0, help="training: roll every training window (history, targets, condition fields) "
                    "by a random multiple of this many latent cells (1 cell = 8 px of the tokenizer grid, 2 = one DiT token) on each periodic axis of its family; "
                    "own random stream (the windows drawn are those without the flag); needs a store encoded with --vae_wrap; validation and evaluation are not shifted; 0 = off")
    ap.add_argument("--sample_shift", type=float, default=0.0, help="evaluation only: shift of the sampling sigma grid (0 = the run's --fm_shift, which stays locked for training)")
    ap.add_argument("--family_temp", type=float, default=0.5, help="family weight ∝ (training windows)^a: 0 = uniform")
    ap.add_argument("--pass_cap", type=float, default=10.0, help="no family above this many expected passes per training window over the run (0 = off); surplus probability goes to the other families")
    ap.add_argument("--init_weights", default="", help="start from a trained run's EMA weights (ckpt/<run>.ema.pt) with a fresh optimizer, schedule, budget and parameter statistics")
    ap.add_argument("--allow_diff", default="", help="with --init_weights: k_in may differ from the source run (longer history: zero input channels are inserted for the older frames)")
    ap.add_argument("--pushforward", type=float, default=0.0, help="per micro-batch with this probability the history is replaced by the EMA model's own generated round "
                    "(--pf_ode_steps Euler steps from the true history, no gradient) and the network is trained on the NEXT true round (the model learns to continue "
                    "from its own outputs, as in the evaluation rollout); windows are k_in + 2 n_out frames; 0 = off")
    ap.add_argument("--pf_ode_steps", type=int, default=8)
    ap.add_argument("--eval_splits", default="", help="comma list of family/split to evaluate (e.g. kolmo/test,rb/test); default all; none = no evaluation")
    ap.add_argument("--dump_pred", type=int, default=0, help="save history, prediction and truth of the first N windows of every evaluated split to results/pred/<name>/ (figures); 0 = off")
    ap.add_argument("--size", default="1.3b", choices=sorted(gen_model.WAN_REPO), help="Wan backbone size: 1.3b (default) or 14b (Wan2.1-T2V-14B, same VAE latents; "
                    "architecture from its released config, weights streamed from the HF cache, fetched by fetch_hf.py); a trained run keeps its own")
    ap.add_argument("--lora_r", type=int, default=0, help="LoRA rank on every linear layer of the transformer blocks (self / cross-attention q, k, v, out; FFN in, out; peft, B = 0 so the "
                    "adapted model starts as the source model); 0 = off")
    ap.add_argument("--lora_alpha", type=float, default=32.0, help="LoRA scaling alpha / r with alpha fixed (learning rates transfer across ranks)")
    ap.add_argument("--body_wd", type=float, default=None, help="AdamW weight decay of the pretrained body (every parameter with a W0 counterpart: blocks, condition embedders); IO layers, "
                    "number MLP and LoRA factors keep --wd (a second parameter group only when it differs); default = --wd")
    ap.add_argument("--no_ema", action="store_true", help="no EMA copy: validation, evaluation, ema.pt and resume use the raw weights "
                    "(an EMA of LoRA factors is not the EMA of their product)")
    ap.add_argument("--frozen_bf16", action="store_true", help="frozen linear weights of the transformer blocks stored in bf16: "
                    "bit-identical under the bf16 autocast; restored to their fp32 source before ema.pt is written")
    a = ap.parse_args()
    assert not a.drop_fields or a.eval_only, "--drop_fields is an evaluation-time counterfactual (--eval_only)"
    assert a.text_probe == "none" or a.eval_only, "--text_probe is an evaluation-time counterfactual (--eval_only)"
    assert not a.sample_shift or a.eval_only, "--sample_shift is an evaluation-time setting (--eval_only); training keeps --fm_shift"
    assert a.rand_shift in (0, 1, 2) and not (a.rand_shift and a.eval_only), "--rand_shift: 0 (off), 1 (latent cells) or 2 (DiT tokens), training only"
    assert 0.0 <= a.pushforward <= 1.0 and a.pf_ode_steps >= 1, "--pushforward is a probability, --pf_ode_steps >= 1"
    assert a.lora_r >= 0
    src0 = os.path.join(HERE, "ckpt", a.eval_only + ".ema.pt") if a.eval_only else ""
    if src0 and os.path.exists(src0):   # the run's own input layout, conditions and objective
        ck0 = torch.load(src0, map_location="cpu", weights_only=False, mmap=True); ck_args = ck0.get("args") or {}
        for key in ("objective", "cond", "k_in", "n_out", "fm_shift", "text_len", "textemb", "bc_pos", "size", "no_ema"):
            if key in ck_args and ck_args[key] != getattr(a, key): print(f"[{a.name}] {key} {ck_args[key]} from the checkpoint", flush=True); setattr(a, key, ck_args[key])
        del ck0
    if a.eval_only and (a.lora_r or a.frozen_bf16):   # ema.pt of a LoRA run holds merged fp32 weights with the standard keys: a plain model reads them
        print(f"[{a.name}] --eval_only: --lora_r / --frozen_bf16 ignored (the checkpoint is merged)", flush=True); a.lora_r, a.frozen_bf16 = 0, False
    corpus.set_kolmo_extra(a.kolmo_extra); a.kolmo_extra = ",".join(corpus.KOLMO_EXTRA)   # (normalised: sorted, unique)
    if a.eval_only: a.val_every = 0
    ddp = "LOCAL_RANK" in os.environ
    if ddp: dist.init_process_group("nccl", timeout=datetime.timedelta(hours=6)); rank, world = dist.get_rank(), dist.get_world_size(); torch.cuda.set_device(int(os.environ["LOCAL_RANK"]))
    else: rank, world = 0, 1
    dev = torch.device("cuda"); torch.backends.cuda.matmul.allow_tf32 = True; torch.backends.cudnn.allow_tf32 = True
    R = os.path.join(HERE, "results"); CK = os.path.join(HERE, "ckpt"); os.makedirs(R, exist_ok=True); os.makedirs(CK, exist_ok=True)
    out = os.path.join(R, a.name + ".json")
    if os.path.exists(out): print("exists", out); sys.exit(0)
    log = (lambda *m: print(f"[{a.name}]", *m, flush=True)) if rank == 0 else (lambda *m: None)

    # ---- latent store and evaluation items
    root = store_root(a.store); splits = ["train"] + [sp for sp in corpus.TRAIN_ONLY_SPLITS if sp in corpus.kolmo_extra_splits()] + (sorted({sp for k in corpus.FAMILIES for sp in corpus.get(k).splits() if sp.startswith("ref_")}) if a.reference else [])
    fams_req = list(corpus.FAMILIES)   # every family; a missing store is waited for, never silently dropped
    t_w = time.time()
    need_sp = {k: [sp for sp in splits if sp == "train" or sp in corpus.get(k).splits()] + (["val"] if a.val_every and "val" in corpus.get(k).splits() else []) for k in fams_req}   # a family's own splits only; val when --val_every selects weights
    while True:
        missing = [f"{k}/{sp}" for k in fams_req for sp in need_sp[k] if not os.path.exists(os.path.join(root, k, sp, "index.json"))]
        if not missing or time.time() - t_w > a.wait_stores_h * 3600: break
        log(f"waiting for latent stores {missing} (encode jobs still running?)"); time.sleep(120)
    assert not missing, f"no latent store for {missing} under {root}"
    fams = {k: corpus.get(k) for k in fams_req}; stores = {k: Store(root, k, splits) for k in fams_req}
    for k, st in stores.items(): assert st.groups, f"no latent store for {k} under {root}"
    wr_ = {bool(st.wrap) for k, st in stores.items() if any(fams[k].periodic)}
    assert len(wr_) <= 1, f"store {a.store}: periodic families encoded with and without --vae_wrap"; VAE_WRAP[1] = bool(wr_); VAE_WRAP[0] = bool(wr_ and wr_.pop())
    if VAE_WRAP[0]: log(f"store {a.store}: wrap-around VAE padding on periodic axes; every encode / decode of this run uses it")
    assert not a.rand_shift or VAE_WRAP[0], f"--rand_shift: store {a.store} was not encoded with --vae_wrap (a rolled zero-padded latent has a seam no real field has)"
    st_tok = {json.load(open(os.path.join(g[4], "index.json"))).get("tok") for st in stores.values() for g in st.groups[:1]}
    if st_tok != {a.tok}:
        ok_ = False         # a decoder-only fine-tune (train_tok2d) leaves the latents, and so the DiT's inputs, unchanged
        if a.eval_only and st_tok == {"wan21"} and a.tok.startswith("wan21ft:"):
            ckp_ = a.tok.partition(":")[2]; ckp_ = ckp_ if os.path.isabs(ckp_) else os.path.join(HERE, "ckpt", ckp_)
            dk_ = torch.load(ckp_, map_location="cpu", weights_only=False); ok_ = dk_.get("kind") == "wan21_decoder"; dd_ = dk_["data"]
            assert bool(dd_.get("vae_wrap", False)) == VAE_WRAP[0], f"--tok {a.tok}: decoder vae_wrap {bool(dd_.get('vae_wrap', False))}, this store {VAE_WRAP[0]} (a decoder fine-tuned with wrap-around padding, train_tok2d --vae_wrap, decodes wrap-padded stores only; a zero-padded one zero-padded stores only)"
            DEC_FLOOR[0] = dd_.get("s_floor") or None
        assert ok_, f"--tok {a.tok} differs from the store's tokenizer {st_tok}: allowed only for --eval_only with a decoder-only Wan2.1 fine-tune"
        log(f"evaluation with decoder {a.tok} (encoder = the store's {st_tok.pop()}, frozen)")
    if rank == 0:
        for k in fams_req:
            for sp in fams[k].splits(): corpus.cached_items(k, sp)
        for k, st in list(stores.items()) + ([(k, Store(root, k, ["val"])) for k in fams_req] if a.val_every else []):   # the store's index must agree with the current corpus
            assert not (any(fams[k].periodic) and st.groups) or bool(st.wrap) == VAE_WRAP[0], f"store {a.store} {k}: splits encoded with and without --vae_wrap"
            for sp, gid, meta, its, _ in st.groups:
                cur = {(it.path, it.traj): it for it in corpus.cached_items(k, sp)}; js = lambda x: json.loads(json.dumps(x, default=float))   # compare as stored (JSON round trip: tuples -> lists, numpy scalars -> float)
                bad = [e for e in its if (e["path"], e["traj"]) not in cur or js(cur[(e["path"], e["traj"])].params) != e["params"] or js(cur[(e["path"], e["traj"])].raw) != e["raw"]]
                assert not bad, f"store {a.store} {k}/{sp}: {len(bad)} items disagree with the corpus (e.g. {bad[0]['path']} {bad[0]['traj']}); re-encode the store"
    if ddp: dist.barrier()
    V = [corpus.param_vector(corpus.effective_params(k, it["params"])) for k, st in stores.items() for g in st.groups for it in g[3]]
    vv = np.stack([v for v, _ in V]).astype(np.float64); mm = np.stack([m for _, m in V]).astype(np.float64); cnt = mm.sum(0).clip(1)
    pm = (vv * mm).sum(0) / cnt; ps = np.sqrt((((vv - pm) * mm) ** 2).sum(0) / cnt); ps[ps < 1e-6 * np.maximum(1.0, np.abs(pm))] = 1.0   # constant slot -> 1.0
    pstats = {"slots": list(corpus.SLOTS), "vae_wrap": VAE_WRAP[0], "mean": pm.tolist(), "std": ps.tolist(), "n_traj": {k: sum(len(g[3]) for g in st.groups) for k, st in stores.items()},
              "grids": {k: sorted({tuple(g[2]["shape"]) for g in st.groups}) for k, st in stores.items()}, "strides": {k: f.stride for k, f in fams.items()},
              "quantity_slots": {k: f.slots for k, f in fams.items()}}
    pstats["s_floor"] = {k: scale_floor([st.arrays(gi)[1] for gi, g in enumerate(st.groups) if g[0] == "train"]).tolist() for k, st in stores.items()}
    if DEC_FLOOR[0] is not None:
        bad_ = [k for k in fams if k in DEC_FLOOR[0] and not np.allclose(DEC_FLOOR[0][k], pstats.get("s_floor", {}).get(k, []), rtol=1e-4)]
        assert not bad_, f"--tok {a.tok}: scale floors differ from the decoder's for {bad_}"
    if a.kolmo_extra: pstats["kolmo_extra"] = a.kolmo_extra
    log("store", root, "train trajectories", pstats["n_traj"], "grids", pstats["grids"], "strides", pstats["strides"])
    if a.kolmo_extra and "kolmo" in stores:
        n_ = {}
        for g in stores["kolmo"].groups: n_[g[0]] = n_.get(g[0], 0) + len(g[3])
        log(f"kolmo_extra {a.kolmo_extra}: Kolmogorov training trajectories per split {n_}; evaluation adds {[sp for sp in corpus.kolmo_extra_splits() if sp not in corpus.TRAIN_ONLY_SPLITS and not sp.startswith('ref_')]}")
    if a.eval_only:   # a trained run keeps its own parameter slots and standardisation
        ck0 = torch.load(os.path.join(CK, a.eval_only + ".ema.pt"), map_location="cpu", weights_only=False, mmap=True)
        if list(ck0["pstats"]["slots"]) != list(corpus.SLOTS): log(f"parameter slots of {a.eval_only}: {ck0['pstats']['slots']} (this process {corpus.SLOTS})")
        pstats = {**pstats, "slots": list(ck0["pstats"]["slots"]), "mean": ck0["pstats"]["mean"], "std": ck0["pstats"]["std"]}
        del ck0

    # ---- tokenizer (evaluation only), texts, model
    tok = tok2d.load(a.tok, dev).to(torch.bfloat16)
    for p in tok.parameters(): p.requires_grad_(False)
    tp = os.path.join(HERE, "textemb", a.textemb + ".pt"); assert os.path.exists(tp), f"no text embeddings {tp} (run textemb.py first)"
    texts = torch.load(tp, weights_only=False)["emb"]
    cd = Conditioner(tok, a, pstats, texts, fams, dev); zc = tok.zc
    torch.manual_seed(a.seed)
    net = gen_model.WanGen(a.init, zc=zc, max_c=MAX_C, k_in=a.k_in, nd=cd.ND, grad_ckpt=a.grad_ckpt, max_f=MAX_F, bc_pos=a.bc_pos, size=a.size, device=dev).to(dev)
    bline = bool(a.lora_r or a.frozen_bf16 or a.no_ema or a.body_wd is not None or a.size != "1.3b")   # LoRA / no EMA / other DiT size
    if a.lora_r:   # before the EMA copy, the optimizer, DDP and the Resumer, so all of them see the final structure
        nl_, na_ = net.add_lora(a.lora_r, a.lora_alpha); log(f"LoRA r={a.lora_r} alpha={a.lora_alpha:g} (scaling {a.lora_alpha / a.lora_r:.4g}) on {nl_} linear layers of the blocks: {na_ / 1e6:.1f}M adapter parameters")
    if a.lora_r: set_trainable(net, a)
    if a.frozen_bf16: log(f"frozen linear weights of the blocks stored in bf16: {net.cast_frozen(torch.bfloat16) / 1e6:.1f}M parameters")
    ptab = param_table(net)
    ema = net if a.no_ema else copy.deepcopy(net).eval().requires_grad_(False)
    npar = gen_model.n_params(net); log(f"Wan {a.size} ({a.init}) {npar / 1e6:.1f}M params; tokenizer {a.tok} zc={zc} factor={tok.factor}; ND={cd.ND}")
    if bline: log(f"backbone size {a.size}, {ptab_line(ptab)}; EMA {'off (raw weights)' if a.no_ema else a.ema}; GPU {torch.cuda.memory_allocated() / 2**30:.1f} GB allocated, host {host_mem()}")
    holder = nn.ModuleDict({"model": net} if a.no_ema else {"model": net, "ema": ema})
    bwd = a.wd if a.body_wd is None else a.body_wd
    if bwd == a.wd: opt = torch.optim.AdamW([p for p in net.parameters() if p.requires_grad], lr=a.lr, betas=(0.9, a.beta2), weight_decay=a.wd)
    else:
        tr_ = [(k, p) for k, p in net.named_parameters() if p.requires_grad]; pg_ = [{"params": [p for k, p in tr_ if is_body(k)], "weight_decay": bwd}, {"params": [p for k, p in tr_ if not is_body(k)], "weight_decay": a.wd}]
        opt = torch.optim.AdamW([g_ for g_ in pg_ if g_["params"]], lr=a.lr, betas=(0.9, a.beta2))
        log(f"weight decay {bwd:g} on {sum(p.numel() for p in pg_[0]['params']) / 1e6:.1f}M pretrained-body parameters, {a.wd:g} on {sum(p.numel() for p in pg_[1]['params']) / 1e6:.1f}M new / IO parameters")
    model = nn.parallel.DistributedDataParallel(net, device_ids=[dev.index]) if ddp else net
    gen = torch.Generator(device=dev); gen.manual_seed(a.seed * 1000 + rank)
    rs = Resumer(CK, a.name, rank, world)
    step, extra = rs.load(holder, opt, None, (gen,)); fresh = step == 0 and not extra
    if fresh and not a.eval_only:
        zero = [t for t, e in texts.items() if t and float(e.float().norm()) == 0.0]
        assert not zero, f"{tp}: {len(zero)} of {len(texts) - 1} text embeddings are all zeros (umT5 embed_tokens not loaded); run textemb.py again"
    if a.init_weights and step == 0 and not extra:   # fine-tune from a finished run's EMA weights; everything else (optimizer, schedule, budget, standardisation) is this run's own
        ck = torch.load(os.path.join(CK, a.init_weights + ".ema.pt"), map_location="cpu", weights_only=False)
        sa = ck.get("args") or {}; diff = {k_: (sa[k_], getattr(a, k_)) for k_ in ("objective", "cond", "fm_shift", "text_len", "textemb", "k_in", "n_out") if k_ in sa and sa[k_] != getattr(a, k_)}
        ok_diff = {k_ for k_ in a.allow_diff.split(",") if k_}; assert ok_diff <= {"k_in"}, f"--allow_diff {sorted(ok_diff)}: only k_in"
        if ok_diff & set(diff): log(f"--init_weights {a.init_weights}: on purpose (--allow_diff) {({k_: diff[k_] for k_ in ok_diff & set(diff)})}")
        diff = {k_: v_ for k_, v_ in diff.items() if k_ not in ok_diff}
        assert not diff, f"--init_weights {a.init_weights}: settings differ from the source run (source, this): {diff}"
        k0_ = int(sa.get("k_in", a.k_in))
        if "k_in" in ok_diff and k0_ != a.k_in:
            assert a.k_in > k0_ and a.k_in <= EVAL_HIST, f"--init_weights {a.init_weights}: k_in {k0_} -> {a.k_in} (only longer, <= EVAL_HIST)"   # weights inserted
            kw_ = "net.patch_embedding.weight"; w0 = ck["ema"][kw_]; cl_ = MAX_C * zc + 2 * MAX_C; p_ = cl_; nk_ = (a.k_in - k0_) * cl_   # after x_t, before the
            w_ = w0.new_zeros(w0.shape[0], w0.shape[1] + nk_, *w0.shape[2:]); w_[:, :p_] = w0[:, :p_]; w_[:, p_ + nk_:] = w0[:, p_:]; ck["ema"][kw_] = w_   # source's history
            log(f"history {k0_} -> {a.k_in} frames: {nk_} zero input channels for the older frames inserted at {p_} in {kw_} ({tuple(w0.shape)} -> {tuple(w_.shape)})")
        sp_ = ck["pstats"]; so, sn = list(sp_["slots"]), list(pstats["slots"])
        if so and so != sn:
            kw_ = "num_mlp.0.weight"; w0 = ck["ema"][kw_]; ck["ema"][kw_] = reindex_num_mlp(w0, so, sn)   # value / mask columns by slot name; slots the source never had start at zero
            log(f"number MLP re-indexed by slot name ({tuple(w0.shape)} -> {tuple(ck['ema'][kw_].shape)}; new slots {[x for x in sn if x not in so]})")
        new_ = {k_: v_ for k_, v_ in net.state_dict().items() if k_.startswith("bc_embed.") and k_ not in ck["ema"]}
        if new_: ck["ema"].update(new_); log(f"boundary-distance embedding: {len(new_)} new zero-initialised tensors")   # --bc_pos on a source without it: keep the zero init
        load_plain(net, ck["ema"]); (load_plain(ema, ck["ema"]) if ema is not net else None); log(f"weights of {a.init_weights} (EMA at step {ck['step']}); fresh optimizer and schedule")
        if VAE_WRAP[1] and bool(sp_.get("vae_wrap", False)) != VAE_WRAP[0]: log(f"VAE padding: {a.init_weights} trained on vae_wrap={bool(sp_.get('vae_wrap', False))}, this run's store has {VAE_WRAP[0]}")
        keep = [j for j, nm in enumerate(sn) if nm in so and sp_["std"][so.index(nm)] != 1.0
                and sp_["std"][so.index(nm)] >= 1e-4 * max(1.0, abs(sp_["mean"][so.index(nm)]))]
        for j in keep: i = so.index(sn[j]); pstats["mean"][j], pstats["std"][j] = sp_["mean"][i], sp_["std"][i]
        cd = Conditioner(tok, a, pstats, texts, fams, dev); log(f"parameter standardisation of {a.init_weights} kept for slots {[sn[j] for j in keep]}")
        del ck
    if a.eval_only:
        ck = torch.load(os.path.join(CK, a.eval_only + ".ema.pt"), map_location="cpu", weights_only=False, mmap=True); step = ck["step"]
        ema.load_state_dict(ck["ema"])
        pstats = ck["pstats"]; cd = Conditioner(tok, a, pstats, texts, fams, dev)   # the run's parameter standardisation, not this subset's
        assert not VAE_WRAP[1] or bool(pstats.get("vae_wrap", False)) == VAE_WRAP[0], f"--eval_only {a.eval_only}: trained on a store with vae_wrap={not VAE_WRAP[0]}, --store {a.store} has {VAE_WRAP[0]}"
        a.max_steps = step; log(f"eval only: weights of {a.eval_only} (step {step})")
    if extra.get("pstats") and not a.eval_only:   # a restarted run keeps the statistics it started with
        pstats = extra["pstats"]; cd = Conditioner(tok, a, pstats, texts, fams, dev)
        assert not VAE_WRAP[1] or bool(pstats.get("vae_wrap", False)) == VAE_WRAP[0], f"restart: the run started on vae_wrap={pstats.get('vae_wrap', False)}, the store now has {VAE_WRAP[0]}"
        assert pstats.get("kolmo_extra", "") == a.kolmo_extra, f"restart: the run started with --kolmo_extra {pstats.get('kolmo_extra', '')!r}, this command line has {a.kolmo_extra!r} (a run keeps its own data)"

    floor = {}   # Batches' scale floor: the latent of an all-zero frame per (family, grid), through the same encode path as the store; from the final pstats
    for k, fl_ in pstats["s_floor"].items():
        if k in fams: floor[k] = (np.asarray(fl_, np.float32), {tuple(g[2]["shape"]): cd.encode(torch.zeros(1, *g[2]["shape"], device=dev), fams[k].periodic)[0][0].half().cpu().numpy()
                                                              for g in stores[k].groups + (Store(root, k, ["val"]).groups if a.val_every else [])})
    run_id = extra.get("run_id")   # identity of this run's attempt: per-rank evaluation files of another attempt under the same name are not reused
    if not run_id:
        o_ = [uuid.uuid4().hex if rank == 0 else None]
        if ddp: dist.broadcast_object_list(o_, 0)
        run_id = o_[0]
    if a.eval_only:
        st_ = os.stat(os.path.join(CK, a.eval_only + ".ema.pt")); run_id = f"eval:{a.eval_only}:{step}:{int(st_.st_mtime)}:{st_.st_size}"
    trained = extra.get("trained", False); fam_mb = extra.get("fam_mb", {})   # trained: a restart after training (during evaluation) does not train again
    train_s = extra.get("train_s", 0.0); hist_loss = extra.get("loss_hist", []); stable_saved = extra.get("stable_saved", False); restarts = extra.get("restarts", -1) + 1
    best_val, best_step, val_hist = extra.get("best_val", float("inf")), extra.get("best_step", -1), extra.get("val_hist", [])
    budget_s = a.budget_gpu_h * 3600 / world; decay_s = a.decay_frac * budget_s

    ds = Batches(root, fams_req, splits, {k: f.t_start for k, f in fams.items()}, {k: f.stride for k, f in fams.items()}, a.k_in + a.n_out * (2 if a.pushforward > 0 else 1), a.n_out, a.tokens, a.max_bs, [a.seed, rank, restarts], pass_cap=a.pass_cap, n_micro=a.budget_gpu_h * MB_PER_GPU_H, log=log,
                 stride_aug=[float(x) for x in a.stride_aug.split(",") if x], family_temp=a.family_temp, floor=floor,
                 rand_shift=a.rand_shift, periodic={k: tuple(f.periodic) for k, f in fams.items()})
    dl = iter(torch.utils.data.DataLoader(ds, batch_size=None, num_workers=a.workers, persistent_workers=a.workers > 0, prefetch_factor=4 if a.workers else None, pin_memory=True))
    if not bline: ema_p = list(ema.parameters()); net_p = list(net.parameters())
    else:   # LoRA: only trainable parameters are clipped and averaged (frozen ones have no gradient and never move)
        net_p = [p for p in net.parameters() if p.requires_grad]; ema_p = [] if a.no_ema else [q for p, q in zip(net.parameters(), ema.parameters()) if p.requires_grad]
    tm = dict.fromkeys(["data", "prep", "fb", "opt"], 0.0)
    run_loss = {}; t_log = time.time(); toks = 0; samples = 0
    mem_peak, mem_res_peak, mem_by = torch.cuda.max_memory_allocated(), torch.cuda.max_memory_reserved(), {}   # GPU peaks: whole run and per training
    # micro-batch kind (family / latent grid); the allocator's peak counters are reset at every step, the run's peak is kept here

    @torch.no_grad()
    def pushed(b, z, stt, g_):
        K, F_ = a.k_in, a.n_out; k = b["family"]; sl = fams[k].slots; B = z.shape[0]; h_, w_ = z.shape[-2:]
        ref = stt[:, K - 1]; hist = assemble(to_frames(z[:, :K], sl), planes(stt[:, :K], ref, sl))
        num, chm, fo, txt = cd.cond(k, b["params"], b["raw"], b["fz"], ref, sl, B, h_, w_, b.get("stride"))
        y = generate(ema, hist, chm, fo, num, txt, F_, a.pf_ode_steps, a.fm_shift, g_, chan_mask(sl, zc, B, dev), a.objective)
        zn, sn = split_out(y, ref, sl, zc)
        zh = torch.cat([z[:, :K], zn], 1)[:, -K:]; sh = torch.cat([stt[:, :K], sn], 1)[:, -K:]
        return torch.cat([zh, z[:, K + F_:K + 2 * F_]], 1), torch.cat([sh, stt[:, K + F_:K + 2 * F_]], 1)

    def prep(b, g_, train=False):
        """a batch -> (inputs of the network, target, channel mask); g_ = generator of sigma / noise (training: `gen`; validation: a fixed one)."""
        for m_ in (net, ema):
            if getattr(m_, "bc_pos", False): m_.set_bc(fams[b["family"]].bc)
        z = b["z"].to(dev, non_blocking=True).float(); stt = b["st"].to(dev, non_blocking=True).float()
        if train and a.pushforward > 0 and z.shape[1] >= a.k_in + 2 * a.n_out and float(torch.rand(1, generator=g_, device=dev)) < a.pushforward: z, stt = pushed(b, z, stt, g_)
        else: z, stt = z[:, :a.k_in + a.n_out], stt[:, :a.k_in + a.n_out]   # (a pushforward run's windows carry n_out more frames)
        B, L, C = z.shape[:3]; h_, w_ = z.shape[-2:]
        sl = fams[b["family"]].slots
        ref = stt[:, a.k_in - 1]; fr = assemble(to_frames(z, sl), planes(stt, ref, sl)); zh, x0 = fr[:, :a.k_in], fr[:, a.k_in:]
        num, chm, fo, txt = cd.cond(b["family"], b["params"], b["raw"], b["fz"], ref, sl, B, h_, w_, b.get("stride"))
        cm = chan_mask(sl, zc, B, dev)[:, None, :, None, None]
        if a.objective == "regress": sig = torch.zeros(B, device=dev); xt = torch.zeros_like(x0); tgt = x0
        else:
            sig = sample_sigma(B, a.fm_shift, g_, dev); eps = torch.randn(x0.shape, generator=g_, device=dev) * cm
            xt = (1 - sig[:, None, None, None, None]) * x0 + sig[:, None, None, None, None] * eps; tgt = eps - x0
        return (xt, sig, zh, chm, fo, num, txt), tgt, cm, (B, h_, w_)

    val_cache = []; val_fams = [k for k in fams_req if "val" in fams[k].splits() and Store(root, k, ["val"]).groups] if a.val_every else []
    @torch.no_grad()
    def val_loss():
        """loss of the EMA weights on fixed val batches (same windows, sigmas and noise every time): --val_batches / families (rounded up) batches per family
        with a val split on every rank, summed over ranks -> (mean over families,
        {family: loss}); a single-family run has the plain mean over its batches."""
        if not val_cache:
            nb = -(-a.val_batches // max(1, len(val_fams)))
            for k in val_fams:
                itv = iter(Batches(root, [k], ["val"], {k: fams[k].t_start}, {k: fams[k].stride}, a.k_in + a.n_out, a.n_out, a.tokens, a.max_bs, [a.seed + 7, rank, 0], floor=floor, token_weight=False))
                val_cache.extend(next(itv) for _ in range(nb))
        g_ = torch.Generator(device=dev); g_.manual_seed(4242 + rank); tot = torch.zeros(2, len(val_fams), device=dev, dtype=torch.float64)
        for b in val_cache:
            inp, tgt, cm, (B, h_, w_) = prep(b, g_)
            with torch.autocast("cuda", dtype=torch.bfloat16): v = ema(*inp).float()
            j = val_fams.index(b["family"]); tot[0, j] += float((((v - tgt) ** 2) * cm).sum() / (cm.sum() * a.n_out * h_ * w_)); tot[1, j] += 1
        if ddp: dist.all_reduce(tot, op=dist.ReduceOp.SUM)
        per = (tot[0] / tot[1].clamp_min(1)).tolist()
        return float(np.mean(per)), {k: float(v_) for k, v_ in zip(val_fams, per)}
    nonfinite = 0
    log(f"start step {step} (restart {restarts}), train time so far {train_s / 3600:.2f} h of {budget_s / 3600:.2f} h per GPU")
    dec_steps = a.decay_frac * a.budget_steps
    while step < a.max_steps and not trained:
        t_it = time.time(); prof = step % 10 == 0
        if a.budget_steps: lr = a.lr * min(1.0, (step + 1) / a.warmup) * (1.0 if step < a.budget_steps - dec_steps else max(0.0, (a.budget_steps - step) / dec_steps))
        else: lr = a.lr * min(1.0, (step + 1) / a.warmup) * (1.0 if train_s < budget_s - decay_s else max(0.0, (budget_s - train_s) / decay_s))
        for g_ in opt.param_groups: g_["lr"] = lr
        mem_peak = max(mem_peak, torch.cuda.max_memory_allocated()); mem_res_peak = max(mem_res_peak, torch.cuda.max_memory_reserved()); torch.cuda.reset_peak_memory_stats()   # (validation since the last step included)
        b = next(dl)
        if prof: torch.cuda.synchronize()
        t1 = time.time(); tm["data"] += (t1 - t_it) * prof
        inp, tgt, cm, (B, h_, w_) = prep(b, gen, train=True)
        if prof: torch.cuda.synchronize()
        t2 = time.time(); tm["prep"] += (t2 - t1) * prof
        with torch.autocast("cuda", dtype=torch.bfloat16): v = model(*inp).float()
        loss = (((v - tgt) ** 2) * cm).sum() / (cm.sum() * a.n_out * h_ * w_)
        opt.zero_grad(set_to_none=True); loss.backward()
        if prof: torch.cuda.synchronize()
        t3 = time.time(); tm["fb"] += (t3 - t2) * prof
        gn = torch.nn.utils.clip_grad_norm_(net_p, a.clip)
        if torch.isfinite(gn): opt.step()
        else:
            opt.zero_grad(set_to_none=True); nonfinite += 1; log(f"WARNING: non-finite gradient norm at step {step + 1} (skipped; {nonfinite} so far)")
            if nonfinite > 20 and a.div_ratio > 0:
                log("DIVERGED: more than 20 non-finite gradient steps; stopping without evaluation")
                if rank == 0: json.dump({"name": a.name, "diverged_at_step": step + 1, "stopped_at_step": step + 1, "reason": "non-finite gradients", "val_hist": val_hist, "loss_hist": hist_loss, "args": vars(a)}, open(os.path.join(R, a.name + ".diverged.json"), "w"), indent=1)
                if ddp: dist.barrier()
                raise SystemExit(3)
        step += 1
        if not a.no_ema:
            with torch.no_grad(): torch._foreach_lerp_(ema_p, net_p, 1 - min(a.ema, (1 + step) / (10 + step)))
        torch.cuda.synchronize(); tm["opt"] += (time.time() - t3) * prof; train_s += time.time() - t_it
        pk_ = torch.cuda.max_memory_allocated(); kb_ = f"{b['family']}/{h_}x{w_}"; mem_by[kb_] = max(mem_by.get(kb_, 0), pk_); mem_peak = max(mem_peak, pk_)
        if step == 1 and bline: log(f"first step ({kb_}, B={B}): GPU peak {pk_ / 2**30:.1f} GB allocated, {torch.cuda.max_memory_reserved() / 2**30:.1f} GB reserved; host {host_mem()}")
        toks += B * a.n_out * (h_ // 2) * (w_ // 2); samples += B; run_loss.setdefault(b["family"], []).append(float(loss)); fam_mb[b["family"]] = fam_mb.get(b["family"], 0) + 1
        if step % a.log_every == 0:
            el = time.time() - t_log; msg = " ".join(f"{k}={np.mean(v):.4f}" for k, v in sorted(run_loss.items())); tt = sum(tm.values()) or 1.0
            log(f"it {step} lr {lr:.2e} gn {float(gn):.2f} loss[{msg}] {toks / el:.0f} tok/s {samples / el:.1f} samp/s mem {mem_peak / 2**30:.1f}G "
                f"train {train_s / 3600:.2f}/{budget_s / 3600:.2f} h | time% data {100 * tm['data'] / tt:.0f} prep {100 * tm['prep'] / tt:.0f} fwdbwd {100 * tm['fb'] / tt:.0f} opt {100 * tm['opt'] / tt:.0f}")
            hist_loss.append({"step": step, "train_h": train_s / 3600, "lr": lr, "tok_s": toks / el, **{k: float(np.mean(v)) for k, v in run_loss.items()}})
            run_loss = {}; t_log = time.time(); toks = 0; samples = 0; tm = dict.fromkeys(tm, 0.0)
        if ddp: t = torch.tensor([train_s], device=dev, dtype=torch.float64); dist.all_reduce(t, op=dist.ReduceOp.MAX); train_s = float(t.item())   # one clock for all ranks
        done_local = step >= a.budget_steps if a.budget_steps else train_s >= budget_s
        if not stable_saved and (step >= a.budget_steps - dec_steps if a.budget_steps else train_s >= budget_s - decay_s):
            if rank == 0: save_atomic({"step": step, "model": net.state_dict(), **({} if a.no_ema else {"ema": ema.state_dict()}), "opt": opt.state_dict(), "train_s": train_s, "args": vars(a), "pstats": pstats}, os.path.join(CK, a.name + ".stable.pt"))
            stable_saved = True; log(f"decay starts at step {step}: stable checkpoint saved")
        if a.val_every and (step % a.val_every == 0 or done_local):   # validation of the EMA weights (collective: every rank computes its share)
            vl, vfam = val_loss(); val_hist.append({"step": step, "train_h": train_s / 3600, "val": vl, "val_fam": vfam}); better = vl < best_val
            if better: best_val, best_step = vl, step
            log(f"val {vl:.5f} at step {step}" + (" (best)" if better else f" (best {best_val:.5f} at {best_step})") + (" | " + " ".join(f"{k} {v_:.5f}" for k, v_ in vfam.items()) if len(vfam) > 1 else ""))
            dv = divergence(val_hist, a.div_ratio, span=int(round(3 / max(1 - a.ema, 1e-6))))
            if not math.isfinite(vl) and a.div_ratio > 0: dv = step
            if dv is not None:
                log(f"DIVERGED: the validation loss has stayed above {a.div_ratio}x its minimum (median over families) since step {dv} (longer than 3 EMA memories); stopping without evaluation")
                if rank == 0: json.dump({"name": a.name, "diverged_at_step": dv, "stopped_at_step": step, "val_hist": val_hist, "loss_hist": hist_loss, "args": vars(a)}, open(os.path.join(R, a.name + ".diverged.json"), "w"), indent=1)
                if ddp: dist.barrier()
                raise SystemExit(3)
        ex = {"run_id": run_id, "pstats": pstats, "fam_mb": fam_mb, "train_s": train_s, "loss_hist": hist_loss, "stable_saved": stable_saved, "restarts": restarts, "best_val": best_val, "best_step": best_step, "val_hist": val_hist}
        if done_local:
            if not a.eval_only: rs.save(step, holder, opt, None, (gen,), {**ex, "trained": True})   # a requeue during evaluation resumes evaluation, not training
            break
        if rs.periodic_now(step): rs.save(step, holder, opt, None, (gen,), ex)
        if rs.should_stop(step): rs.save(step, holder, opt, None, (gen,), ex); rs.exit_for_requeue()
    log(f"training done: step {step}, {train_s / 3600:.2f} h per GPU")
    if not a.eval_only and a.div_ratio > 0 and divergence(val_hist, a.div_ratio, n=1, span=0) is not None:
        log(f"DIVERGED: the last validation (step {val_hist[-1]['step']}) is above {a.div_ratio}x its minimum at the end of the budget; the last weights are not scored")   # span rule
        if rank == 0: json.dump({"name": a.name, "diverged_at_step": val_hist[-1]["step"], "stopped_at_step": step, "reason": "flagged at the end of the budget", "val_hist": val_hist, "loss_hist": hist_loss, "args": vars(a)}, open(os.path.join(R, a.name + ".diverged.json"), "w"), indent=1)
        if ddp: dist.barrier()   # cannot wait 3 EMA memories past the end of training, and the last weights would be the collapsed ones
        raise SystemExit(3)
    if a.no_ema: net.eval().requires_grad_(False)   # --no_ema: the raw weights are scored; evaluate_split relies on frozen weights, as the EMA copy is
    lora_merge = None
    if (a.lora_r or a.frozen_bf16) and not a.eval_only:   # every rank: the weights that are saved and scored = fp32, standard keys (as --eval_only reads ema.pt)
        opt.state.clear(); net.zero_grad(set_to_none=True); torch.cuda.empty_cache()   # Adam moments and gradients are done with
        if a.frozen_bf16:
            ok_, bad_ = ema.restore_frozen(w0_source(a)); log(f"frozen weights restored to their fp32 source ({'--init_weights ' + a.init_weights if a.init_weights else 'released ' + a.size}): {ok_} layers"
                                                               + (f"; WARNING: {bad_} layers are not the bf16 rounding of the source and stay bf16 (their export carries the rounding)" if bad_ else ""))
        if a.lora_r:
            nm_ = ema.merge_lora(); v_ad = val_hist[-1] if val_hist and val_hist[-1]["step"] == step else None; lora_merge = {"layers": nm_, "val_step": step, "val_adapter": v_ad and v_ad["val"]}
            log(f"LoRA merged in fp32 into {nm_} layers (standard keys, no adapters)")
            if a.val_every:   # identity check: the merged model on the same validation batches, sigmas and noise as the adapter model's last validation
                vm_, vfm_ = val_loss(); lora_merge.update(val_merged=vm_, val_fam_adapter=v_ad and v_ad["val_fam"], val_fam_merged=vfm_, rel_diff=(vm_ - v_ad["val"]) / v_ad["val"] if v_ad else None)
                log(f"LoRA merge check: validation loss {vm_:.6f} merged vs {v_ad['val']:.6f} adapter (relative difference {lora_merge['rel_diff']:+.2e}; bf16 rounding of W0 + dW vs W0 and dW apart)" if v_ad else f"LoRA merge check: validation loss {vm_:.6f} merged")
    if rank == 0 and not a.eval_only: save_atomic({"ema": export_sd(ema) if bline else ema.state_dict(), "args": vars(a), "pstats": pstats, "step": step}, os.path.join(CK, a.name + ".ema.pt"))
    selected_step = step
    del dl, opt, model; net.zero_grad(set_to_none=True); b = inp = tgt = v = loss = None; gc.collect(); torch.cuda.empty_cache()   # optimizer states, DDP buckets, gradients and the last batch are not needed for evaluation
    mem_peak = max(mem_peak, torch.cuda.max_memory_allocated()); mem_res_peak = max(mem_res_peak, torch.cuda.max_memory_reserved())
    log(f"evaluation: {torch.cuda.memory_allocated() / 2**30:.1f} GB resident, {torch.cuda.memory_reserved() / 2**30:.1f} GB reserved (peak in training {mem_peak / 2**30:.1f} GB)")
    mem = {"gpu_peak_alloc_gb": mem_peak / 2**30, "gpu_peak_reserved_gb": mem_res_peak / 2**30, "gpu_peak_alloc_gb_by_batch": {k: v / 2**30 for k, v in sorted(mem_by.items())}, **host_mem()}   # training phase
    if ddp: mall_ = [None] * world; dist.all_gather_object(mall_, mem); mem = {"per_rank": mall_, **{f"max_{k}": max(m_[k] for m_ in mall_) for k in ("gpu_peak_alloc_gb", "gpu_peak_reserved_gb", "rss_peak_gb")}}
    if bline: log(f"memory (training): {json.dumps(mem if not ddp else {k: v for k, v in mem.items() if k != 'per_rank'})}")
    # ---- evaluation: (family, split) tasks round-robin over ranks, resumable per task
    tasks = [(k, sp) for k, f in fams.items() for sp in f.splits() if sp not in ("train", "val") + corpus.TRAIN_ONLY_SPLITS and not sp.startswith("ref_")]
    if a.eval_splits: tasks = [t_ for t_ in tasks if f"{t_[0]}/{t_[1]}" in a.eval_splits.split(",")]; log(f"evaluating only {len(tasks)} splits: {a.eval_splits}")   # --eval_splits (figures)
    if a.ens > 1 and a.ens_splits != "all":
        dead = [p for p in a.ens_splits.split(",") if not any(ens_match(a, k, sp, [p]) for k, sp in tasks)]
        if dead: log(f"WARNING --ens_splits patterns matching no evaluated split: {dead}")
        assert any(ens_match(a, k, sp) for k, sp in tasks), f"--ens {a.ens} but --ens_splits {a.ens_splits} matches no evaluated split"
    evp = os.path.join(CK, f"{a.name}.eval.rank{rank}.json"); done = json.load(open(evp)) if os.path.exists(evp) else {}
    sha_ = runmeta.run_meta().get("git_sha")
    if done.get("__step__") != selected_step or done.get("__run__") != run_id or done.get("__sha__") != sha_: done = {"__step__": selected_step, "__run__": run_id, "__sha__": sha_}   # results of other weights or of another
    done = {kk: vv for kk, vv in done.items() if not (isinstance(vv, dict) and "error" in vv)}   # attempt under the same name are not reused; failed splits are retried
    egen = torch.Generator(device=dev)
    for i, (k, sp) in enumerate(tasks):
        if i % world != rank or f"{k}/{sp}" in done: continue
        t0 = time.time(); bs = a.eval_bs; base = torch.cuda.memory_allocated() / 2**30; torch.cuda.reset_peak_memory_stats()
        while True:   # CUDA OOM: halve the window batch and redo the split (same seed; the noise-to-window assignment changes with the batch, so eval_bs_used is recorded)
            egen.manual_seed(a.seed + 99 + fams[k].split_seed(sp))   # by split, not list position: a family subset / other order draws the same noise
            try: r = evaluate_split(ema, cd, fams[k], corpus.cached_items(k, sp), a, egen, sp, bs); break
            except torch.OutOfMemoryError as e:
                torch.cuda.empty_cache()
                if bs <= 1: r = {"error": f"{type(e).__name__}: {str(e)[:300]}"}; break
                bs = max(1, bs // 2); log(f"eval {k}/{sp}: CUDA out of memory -> retry with eval_bs {bs}")
            except Exception as e:
                tb = traceback.extract_tb(e.__traceback__)[-1]; r = {"error": f"{type(e).__name__}: {str(e)[:300]} (at {os.path.basename(tb.filename)}:{tb.lineno})"}; break
        r["seconds"] = time.time() - t0; r["eval_bs_used"] = bs; r["mem_base_gb"] = base; r["mem_peak_gb"] = torch.cuda.max_memory_allocated() / 2**30
        done[f"{k}/{sp}"] = r; json.dump(done, open(evp + ".tmp", "w")); os.replace(evp + ".tmp", evp)
        print(f"[{a.name}] eval {k}/{sp}: " + ", ".join(f"{kk}={vv:.4f}" for kk, vv in r.items() if isinstance(vv, float)) + (f" {r['error']}" if "error" in r else ""), flush=True)
    _, nwin_, bbar_ = ds.mix({k: Store(root, k, splits) for k in fams_req})
    passes_done = {k: fam_mb.get(k, 0) * world * bbar_[k] / max(1, nwin_[k]) for k in fams_req}   # micro-batch counts (the start-up log prints the planned ones)
    log("realised passes per training window: " + ", ".join(f"{k} {v_:.1f}" for k, v_ in passes_done.items()))
    if ddp: allr = [None] * world; dist.all_gather_object(allr, done); merged = {kk: vv for d_ in allr for kk, vv in d_.items()}
    else: merged = done
    nerr = sorted(f"{k}/{sp}" for k, sp in tasks if not isinstance(merged.get(f"{k}/{sp}"), dict) or "error" in merged[f"{k}/{sp}"])
    if rank == 0:
        res = {"name": a.name, "args": vars(a), "n_params": npar, "steps": step, "train_gpu_h": train_s * world / 3600, "world": world, "restarts": restarts,
               "pstats": pstats, "family_microbatches_rank0": fam_mb if rank == 0 else None, "realised_passes": passes_done, "loss_hist": hist_loss, "val_hist": val_hist, "selected_step": selected_step, "eval_stride_mult": a.eval_stride_mult, "vae_wrap": VAE_WRAP[0], "rand_shift": a.rand_shift, "bc_pos": a.bc_pos, "drop_fields": a.drop_fields, "text_probe": a.text_probe, "ens": {"members": a.ens, "windows": a.ens_n, "splits": a.ens_splits},
               "eval_seconds": sum(float(v.get("seconds", 0)) for v in merged.values() if isinstance(v, dict)),
               "eval": {f"{k}/{sp}": merged.get(f"{k}/{sp}") for k, sp in tasks}, "meta": runmeta.run_meta(),
               "params": ptab, "mem": {**mem, "gpu_peak_alloc_gb_eval_rank0": max([float(v.get("mem_peak_gb", 0)) for v in done.values() if isinstance(v, dict)] or [0.0]), "end_rank0": host_mem()},
               "tok_s_median": float(np.median([h["tok_s"] for h in hist_loss[1:]])) if len(hist_loss) > 1 else None, "lora_merge": lora_merge}   # tok/s per GPU, first log interval (start-up) left out
        nf = sum(int(v.get(f"nonfinite_{w}_windows", 0)) for v in merged.values() if isinstance(v, dict) for w in ("pred", "ens"))
        if nf: log(f"WARNING: {nf} evaluation windows with non-finite predictions (scored +inf; see nonfinite_pred_windows / nonfinite_ens_windows per split)")
        res["nonfinite_windows_total"] = nf; res["failed_splits"] = nerr
        if nerr:   # results kept for inspection under .failed.json; checkpoints and the rank files stay, so a requeue redoes only the failed splits
            fo = out[:-5] + ".failed.json"; json.dump(res, open(fo, "w"), indent=1); log(f"evaluation failed on {nerr}; saved {fo}, not {out}")
        else:
            json.dump(res, open(out, "w"), indent=1); print("saved", out, flush=True)
            for r_ in range(world):
                try: os.remove(os.path.join(CK, f"{a.name}.eval.rank{r_}.json"))
                except FileNotFoundError: pass
            rs.finish()
    if ddp: dist.barrier(); dist.destroy_process_group()
    if nerr: sys.exit(2)


if __name__ == "__main__":
    main()
