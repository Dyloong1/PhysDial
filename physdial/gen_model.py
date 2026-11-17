"""Generative backbone of PhysDial: the pretrained Wan2.1-T2V diffusion transformer (WanGen, flow matching; 1.3B, or 14B for the
scale comparison).

Each physical input enters through the pathway of the video generator that has the same role:
  - scalar coefficients (+ the frame spacing) -> numeric vector -> small MLP added to the timestep embedding -> adaLN modulation
  - condition fields (forcing pattern, acoustic medium) -> latents of the same VAE, concatenated on the input channels like a reference frame
  - equation text -> frozen umT5 embedding through the existing cross-attention
New input channels and the output of the numeric MLP are zero-initialised, so the pretrained model is unchanged at step 0.
Periodic domains: BCRope replaces the rotary frequencies on periodic axes by integer harmonics of the token ring; BCEmbed adds a learned
boundary-distance embedding on non-periodic sides.

  inputs   x_t    [B, F, CL, h, w]  noisy latents of the F future frames (CL = zc * MAX_C + 2 * MAX_C: MAX_C quantity slots shared by all
                                     families (corpus.QUANTITIES: vorticity, velocity x / y, pressure, density, buoyancy, tracer, species
                                     1 / 2), each with zc latent channels, then two constant planes per slot carry its per-frame
                                     log-amplitude and mean relative to the last history frame; slots a family does not have are zero
                                     and masked)
           sigma  [B]               flow-matching noise level in [0, 1] (timestep = 1000 sigma, the Wan convention; 0 for --objective regress)
           hist   [B, K, CL, h, w]  clean latents of the K history frames (concatenated on the channel axis at every output position)
           chmask [B, MAX_C]        which quantity slots are present
           force  [B, max_f * (zc + 1), h, w] latents of the condition fields (Kolmogorov forcing; acoustic density, sound speed), each with a presence plane (zeros when absent)
           num    [B, ND]           numeric vector: standardised parameter slots + their masks + state scalars (amplitude per channel)
           text   [B, L, 4096]      umT5 embedding of the equation-identity text (--cond none: the embedding of the empty string, a constant)
  output   v      [B, F, CL, h, w]  flow prediction

Entrances:
  - numbers  -> a small MLP added to the output of Wan's time embedder -> adaLN modulation of every block
  - text     -> cross-attention (Wan's own text projection)"""
import contextlib, json, math, os
import torch, torch.nn as nn

WAN21 = "Wan-AI/Wan2.1-T2V-1.3B-Diffusers"
WAN_REPO = {"1.3b": WAN21, "14b": "Wan-AI/Wan2.1-T2V-14B-Diffusers"}   # --size -> released text-to-video DiT (both use the Wan2.1 VAE latent space)
WAN_ARCH = dict(patch_size=(1, 2, 2), num_attention_heads=12, attention_head_dim=128, text_dim=4096, freq_dim=256, ffn_dim=8960,
                num_layers=30, cross_attn_norm=True, qk_norm="rms_norm_across_heads", eps=1e-6)
LORA_TARGETS = r"net\.blocks\.\d+\.(attn1|attn2)\.(to_q|to_k|to_v|to_out\.0)|net\.blocks\.\d+\.ffn\.net\.(0\.proj|2)"
# layer of the transformer blocks (self- and cross-attention q / k / v / out, FFN in / out; re.fullmatch on WanGen module names). IO layers
# (patch_embedding, proj_out), the number MLP and the condition embedders are not adapted: train_gen keeps them fully trainable.


def released_dir(size):
    """transformer/ of the released DiT of this size in the HF cache. Never downloads: a training job must not fetch a large checkpoint on its own; the
    CPU script fetch_hf.py puts the files there (the 1.3B transformer is downloaded by from_pretrained)."""
    from huggingface_hub import snapshot_download
    try: return os.path.join(snapshot_download(WAN_REPO[size], allow_patterns=["transformer/*"], local_files_only=True), "transformer")
    except Exception as e: raise FileNotFoundError(f"{WAN_REPO[size]}: transformer not in the HF cache ({type(e).__name__}: {e}); run fetch_hf.py first") from e


def released_tensors(size):
    """(sorted keys, get(key) -> CPU tensor in the stored dtype) of the released transformer (its own key names, no 'net.' prefix), read lazily
    from the safetensors shards (memory-mapped: the ranks of a job share one page-cache copy instead of one host copy each)."""
    from safetensors import safe_open
    d = released_dir(size); ix = os.path.join(d, "diffusion_pytorch_model.safetensors.index.json")
    if os.path.exists(ix): wm = json.load(open(ix))["weight_map"]
    else: f0 = "diffusion_pytorch_model.safetensors"; wm = {k: f0 for k in safe_open(os.path.join(d, f0), "pt").keys()}
    fh = {f: safe_open(os.path.join(d, f), "pt") for f in sorted(set(wm.values()))}
    return sorted(wm), lambda k: fh[wm[k]].get_tensor(k)


def wan_arch(size):
    """architecture of the DiT of this size: 1.3b = WAN_ARCH; others from the released config.json (every
    constructor argument except the channel counts, which the conditioning layout sets). Text-to-video only (no image cross-attention)."""
    if size == "1.3b": return dict(WAN_ARCH)
    cfg = json.load(open(os.path.join(released_dir(size), "config.json")))
    assert cfg.get("image_dim") is None and cfg.get("added_kv_proj_dim") is None, f"{WAN_REPO[size]}: image cross-attention (not a text-to-video DiT)"
    arch = {k: v for k, v in cfg.items() if not k.startswith("_") and k not in ("in_channels", "out_channels")}
    arch["patch_size"] = tuple(arch["patch_size"]); return arch


def io_init(own, src, zc, max_c, cl, dim, p):
    """the pretrained IO initialisation (shared by every size): own = the new DiT's state dict (tensors written in place), src(key) -> released
    tensor. Output head zero except W0's head copied into every quantity slot; input embedding zero except W0's patch embedding copied into the
    x_t channels of every slot (the same Wan2.1 VAE latents, zc = 16)."""
    W, B = own["proj_out.weight"].view(p, cl, dim), own["proj_out.bias"].view(p, cl); W.zero_(); B.zero_()
    pe, pw, pb = src("patch_embedding.weight"), src("proj_out.weight"), src("proj_out.bias")
    assert pe.shape[1] == zc, f"latent width {zc} != the pretrained DiT's {pe.shape[1]}"
    own["patch_embedding.weight"].zero_()
    for q in range(max_c):   # every quantity slot of x_t: pretrained input embedding and output head
        own["patch_embedding.weight"][:, q * zc:(q + 1) * zc] = pe
        W[:, q * zc:(q + 1) * zc] = pw.view(p, zc, dim); B[:, q * zc:(q + 1) * zc] = pb.view(p, zc)


def mlp(nd, dim, zero_last=True):
    m = nn.Sequential(nn.Linear(nd, dim), nn.SiLU(), nn.Linear(dim, dim))
    if zero_last: nn.init.zeros_(m[2].weight); nn.init.zeros_(m[2].bias)
    return m


class _AddToTimeEmb(nn.Module):
    """wraps Wan's time_embedder: output + the numeric-condition embedding set just before the forward."""
    def __init__(s, te): super().__init__(); s.te = te; s.extra = None
    def forward(s, t):
        out = s.te(t)
        return out if s.extra is None else out + s.extra.to(out.dtype)


BC_TYPES = ("wall", "open")   # non-periodic side types (corpus.Family.bc)


class BCRope(nn.Module):
    """replaces Wan's rotary position embedding. bc = ((low, high) of axis 0 = height, (low, high) of axis 1 = width), set per micro-batch
    (one family). On a periodic axis of N tokens every rotary frequency w_j becomes 2 pi m_j / N with m_j = clip(round(w_j N / 2 pi), 0, N // 2),
    an integer harmonic of the token ring: attention scores then depend on (n - n') mod N only (translation equivariance on the torus).
    m = 0 for the slow pairs, so a fine-tune from the non-periodic model starts close to it. Bounded axes and time keep the pretrained frequencies."""
    def __init__(s, orig, theta=10000.0):
        super().__init__(); s.orig, s.bc = orig, None
        s.base = [1.0 / theta ** (torch.arange(0, d, 2, dtype=torch.float64)[: d // 2] / d) for d in (orig.h_dim, orig.w_dim)]   # = get_1d_rotary_pos_embed

    @staticmethod
    def harmonics(w, N): return 2 * math.pi * torch.clamp(torch.round(w * N / (2 * math.pi)), 0, N // 2) / N   # m = 0: the slow pairs stay (almost) unrotated

    def forward(s, hidden_states):
        cos, sin = s.orig(hidden_states)
        if s.bc is None or all(b[0] != "periodic" for b in s.bc): return cos, sin
        o = s.orig; _, _, F_, H_, W_ = hidden_states.shape; pt, ph, pw = o.patch_size; n = (F_ // pt, H_ // ph, W_ // pw); D = cos.shape[-1]
        cos = cos.view(*n, D).clone(); sin = sin.view(*n, D).clone()
        for ax, (N, dim, off) in enumerate(((n[1], o.h_dim, o.t_dim), (n[2], o.w_dim, o.t_dim + o.h_dim))):
            if s.bc[ax][0] != "periodic": continue
            ang = torch.arange(N, dtype=torch.float64, device=cos.device)[:, None] * s.harmonics(s.base[ax].to(cos.device), N)[None]   # [N, dim / 2]
            c, si = ang.cos().repeat_interleave(2, 1).to(cos.dtype), ang.sin().repeat_interleave(2, 1).to(sin.dtype)
            shp = (1, N, 1, dim) if ax == 0 else (1, 1, N, dim)
            cos[..., off:off + dim] = c.view(shp); sin[..., off:off + dim] = si.view(shp)
        return cos.reshape(1, -1, 1, D), sin.reshape(1, -1, 1, D)


class BCEmbed(nn.Module):
    """learned embedding of the distance to each non-periodic side (wall / open), added to the patch embedding; zero-initialised."""
    L = (0.5, 1.0, 2.0, 4.0)

    def __init__(s, dim):
        super().__init__(); s.lin = nn.ModuleDict({t: nn.Linear(len(s.L), dim) for t in BC_TYPES})
        for m in s.lin.values(): nn.init.zeros_(m.weight); nn.init.zeros_(m.bias)

    def forward(s, bc, h, w, device):   # -> [dim, h, w], or a zero scalar for a fully periodic grid
        out = sum(p.sum() for p in s.parameters()) * 0.0   # every parameter is in the graph on every rank (DDP: micro-batches of other families)
        dt = s.lin["wall"].weight.dtype; lam = torch.tensor(s.L, device=device, dtype=dt)
        for ax, N in ((0, h), (1, w)):
            for side, t in enumerate(bc[ax]):
                if t == "periodic": continue
                pos = torch.arange(N, device=device, dtype=dt) + 0.5; d = pos if side == 0 else N - pos
                e = s.lin[t](torch.exp(-d[:, None] / lam[None])).t()                   # [dim, N]
                e = e[:, :, None] if ax == 0 else e[:, None, :]
                out = out + e
        return out


class WanGen(nn.Module):
    def __init__(s, init, zc, max_c, k_in, nd, grad_ckpt=False, max_f=2, bc_pos=False, size="1.3b", device=None):
        """init: pretrained (the released DiT body) or random (the same architecture, Wan's initialisation; ablation). zc: latent channels of the
        VAE; max_c: quantity slots (corpus.MAX_C); k_in: history frames; nd: size of the numeric vector (train_gen.Conditioner.ND: 2 x parameter
        slots + 3 x quantity slots). max_f: condition-field slots on the input channels (2: acoustic density + sound speed; the Kolmogorov forcing
        uses the first), each zc latent channels + a presence plane."""
        super().__init__()
        from diffusers import WanTransformer3DModel
        s.zc, s.max_c, s.k_in, s.max_f, s.size = zc, max_c, k_in, max_f, size; cl = zc * max_c + 2 * max_c; cin = cl + k_in * cl + max_c + max_f * (zc + 1)
        arch = wan_arch(size); dim = arch["num_attention_heads"] * arch["attention_head_dim"]
        on_dev = size != "1.3b" and device is not None
        with (torch.device(device) if on_dev else contextlib.nullcontext()): s.net = WanTransformer3DModel(in_channels=cin, out_channels=cl, **arch)
        skip = {"patch_embedding.weight", "proj_out.weight", "proj_out.bias"}; p = math.prod(arch["patch_size"])
        if init == "pretrained" and size == "1.3b":
            src = WanTransformer3DModel.from_pretrained(WAN21, subfolder="transformer", torch_dtype=torch.float32).state_dict()
            own = s.net.state_dict()
            missing = [k for k in own if k not in src and k not in skip]; assert not missing, missing[:5]
            with torch.no_grad():
                for k, v in src.items():
                    if k not in skip: own[k].copy_(v)
                io_init(own, src.__getitem__, zc, max_c, cl, dim, p)
        elif init == "pretrained":
            keys, get = released_tensors(size); own = s.net.state_dict(); key_set = set(keys)
            missing = [k for k in own if k not in key_set and k not in skip]; extra = [k for k in keys if k not in own and "norm_added_q" not in k]   # (diffusers ignores
            assert not missing and not extra, f"{WAN_REPO[size]}: missing {missing[:5]}, unexpected {extra[:5]}"                         # norm_added_q on load too)
            with torch.no_grad():
                for k in keys:
                    if k in skip or k not in own: continue
                    t = get(k); assert tuple(t.shape) == tuple(own[k].shape), f"{k}: released {tuple(t.shape)}, model {tuple(own[k].shape)}"
                    own[k].copy_(t)   # stored dtype (fp32 shards) -> fp32, onto the device
                io_init(own, lambda k: get(k).to(own["proj_out.weight"].device), zc, max_c, cl, dim, p)
        else:
            s._wan_init()
        s.addt = _AddToTimeEmb(s.net.condition_embedder.time_embedder); s.net.condition_embedder.time_embedder = s.addt
        s.num_mlp = mlp(nd, dim)
        s.bc_pos, s.bc = bc_pos, None
        if bc_pos:
            s.net.rope = BCRope(s.net.rope); s.bc_embed = BCEmbed(dim)
            s.net.patch_embedding.register_forward_hook(s._add_bc)
        if grad_ckpt: s.net.enable_gradient_checkpointing()

    def set_bc(s, bc):
        """boundary types of the next forward passes (corpus.Family.bc); a no-op without bc_pos."""
        if s.bc_pos: s.bc = tuple(tuple(x) for x in bc); s.net.rope.bc = s.bc

    # ---- low-rank adaptation and frozen-weight storage (the 14B comparison); train_gen calls these before the EMA copy, DDP and the Resumer
    def add_lora(s, r, alpha):
        """peft LoRA (rank r, scaling alpha / r, no dropout, no bias) on every linear layer of the transformer blocks (LORA_TARGETS), injected in
        place into this module. A is kaiming-uniform, B = 0, so the function is unchanged at step 0. Call while the base weights are fp32.
        -> (layers adapted, adapter parameters)"""
        from peft import LoraConfig, inject_adapter_in_model
        from peft.tuners.lora import LoraLayer
        inject_adapter_in_model(LoraConfig(r=r, lora_alpha=alpha, lora_dropout=0.0, bias="none", init_lora_weights=True, target_modules=LORA_TARGETS), s)
        lay = [m for m in s.modules() if isinstance(m, LoraLayer)]
        for m in lay: m.cast_input_dtype_enabled = False
        ad = [p for m in lay for d_ in (m.lora_A, m.lora_B) for p in d_.parameters()]
        assert all(p.dtype == torch.float32 for p in ad), "LoRA factors must be fp32 (inject before casting the frozen weights)"
        return len(lay), sum(p.numel() for p in ad)

    def cast_frozen(s, dtype=torch.bfloat16):
        """frozen nn.Linear weights of the transformer blocks (LoRA bases) stored in `dtype`. Every use of
        them is a matmul under the bf16 autocast, which casts fp32 weights to bf16 with the same rounding at each call, so the forward and backward
        are bit-identical; biases, norms and scale-shift tables stay fp32. Saves 2 bytes per frozen weight. -> parameters cast"""
        n = 0
        for m in s.net.blocks.modules():
            if isinstance(m, nn.Linear) and not m.weight.requires_grad and m.weight.dtype != dtype: m.weight.data = m.weight.data.to(dtype); n += m.weight.numel()
        return n

    @torch.no_grad()
    def restore_frozen(s, src):
        """undo cast_frozen with the fp32 source of those weights (src(standard WanGen key) -> tensor: the released W0 or the --init_weights checkpoint),
        so the export holds W0 exactly (otherwise a weight analysis would read the bf16 rounding as a weight change). Each
        weight is replaced only if it is exactly the bf16 rounding of its source value. -> (restored, mismatched: left in bf16)"""
        ok = bad = 0
        for name, m in s.named_modules():
            if not isinstance(m, nn.Linear) or m.weight.dtype == torch.float32: continue
            w = src(name.replace(".base_layer", "") + ".weight").to(m.weight.device, torch.float32)
            if tuple(w.shape) == tuple(m.weight.shape) and torch.equal(w.to(m.weight.dtype), m.weight): m.weight.data = w; ok += 1
            else: bad += 1
        return ok, bad

    @torch.no_grad()
    def merge_lora(s):
        """fold every LoRA adapter into its base layer in place: W = W_base + (alpha / r) B A with the product and the sum in float64, stored fp32,
        and replace the peft layer by that plain nn.Linear. -> layers merged"""
        from peft.tuners.lora import LoraLayer
        n = 0
        for name, m in list(s.named_modules()):
            if not isinstance(m, LoraLayer): continue
            (ad,) = list(m.lora_A.keys()); base = m.get_base_layer()
            dev_ = base.weight.device   # fp64 sum and product on the CPU, one layer at a time
            w = base.weight.detach().to("cpu", torch.float64) + m.scaling[ad] * (m.lora_B[ad].weight.detach().to("cpu", torch.float64) @ m.lora_A[ad].weight.detach().to("cpu", torch.float64))
            base.weight.data = w.float().to(dev_); base.weight.requires_grad_(False); del w   # in place: the DDP wrapper keeps every Parameter object
            if base.bias is not None: base.bias.data = base.bias.detach().float(); base.bias.requires_grad_(False)   # alive, so a new Parameter left the old fp32 weight on the GPU
            par, _, ch = name.rpartition("."); setattr(s.get_submodule(par), ch, base); n += 1
        if "peft_config" in s.__dict__: del s.peft_config   # set by peft's injection
        return n

    def _add_bc(s, mod, inp, out):   # forward hook of the patch embedding: [B, dim, F, h, w] + boundary-distance embedding
        assert s.bc is not None, "bc_pos model: call set_bc(family.bc) before the forward pass"
        e = s.bc_embed(s.bc, out.shape[-2], out.shape[-1], out.device)
        return out + (e[None, :, None] if e.dim() == 3 else e).to(out.dtype)

    def _wan_init(s):   # the Wan reference initialisation (wan/modules/model.py init_weights)
        for m in s.net.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None: nn.init.zeros_(m.bias)
        nn.init.xavier_uniform_(s.net.patch_embedding.weight.flatten(1)); nn.init.zeros_(s.net.patch_embedding.bias)
        ce = s.net.condition_embedder
        for m in list(ce.text_embedder.modules()) + list(ce.time_embedder.modules()):
            if isinstance(m, nn.Linear): nn.init.normal_(m.weight, std=0.02)
        nn.init.zeros_(s.net.proj_out.weight); nn.init.zeros_(s.net.proj_out.bias)

    def forward(s, x_t, sigma, hist, chmask, force, num, text):
        B, F, CL, h, w = x_t.shape
        cond = torch.cat([hist.flatten(1, 2), chmask[:, :, None, None].expand(-1, -1, h, w), force], 1)        # [B, Cc, h, w]
        inp = torch.cat([x_t.transpose(1, 2), cond[:, :, None].expand(-1, -1, F, -1, -1)], 1)                  # [B, Cin, F, h, w]
        s.addt.extra = s.num_mlp(num)
        out = s.net(inp, timestep=sigma * 1000, encoder_hidden_states=text, return_dict=False)[0]               # [B, CL, F, h, w]
        s.addt.extra = None
        return out.transpose(1, 2)


def n_params(m): return sum(p.numel() for p in m.parameters())
