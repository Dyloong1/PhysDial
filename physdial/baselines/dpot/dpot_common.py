"""DPOT-H baseline: the released DPOT-H (Hao et al., ICML 2024; 1.03B; weights from the HF repository hzk17/DPOT, code from
github.com/HaoZhongkai/DPOT, imported from a checkout in $DPOT_SRC and not copied here) adapted to the corpus. Shared by
dpot_ft.py (fine-tuning) and dpot_eval.py (scoring on the Walrus-format evaluation windows).

Data: the Walrus-format corpus written by walrus_data.py (the Walrus baseline's training mixture, The Well sampler, held-out conditions
excluded) and the same evaluation windows, so DPOT-H, Walrus and PhysDial are scored on identical windows. Normalization: Walrus's global
per-field RMS of the training corpus (RMSGlobalRevNormalization), scale only; DPOT-H itself was pretrained on raw fields (normalize=0).

Adaptation (the model is otherwise the pretrained network, every weight loaded):
- grid: native grids (128^2 ... 512^2, non-square; no resizing). The 16 x 16 learned position embedding is interpolated (bicubic) to
  the patch grid of each input; the 3 coordinate channels DPOT appends are linspace(0, 1) per axis.
- AFNO modes: pretrained at a 16 x 16 patch grid with modes=32, i.e. every Fourier mode went through the block MLP (the weights do not
  depend on the mode). At 64 x 64 patches modes=32 would keep only the non-negative x-frequencies, so every mode is kept (modes = 10^6).
- history: DPOT-H takes 10 frames, the protocol gives every model 6 (EVAL_HIST); the oldest frame is repeated 4 times in front, so the
  pretrained time aggregator is used unchanged.
- channels: DPOT-H has 4 positional channels; its pretraining data put [vx, vy, density, pressure] (PDEBench CNS) and the species of
  two-species systems in channels 0-1. Here: velocity_x -> 0, velocity_y -> 1, pressure -> 3, the other scalar (density / buoyancy /
  tracer) -> 2; the family without velocity (gs A, B) -> 0, 1. Absent slots are zero and masked out of the loss.
  Time-constant fields (acoustic density, speed of sound) enter as 2 extra input channels whose first-layer weights start at zero.
  Vorticity families are given as velocity (as for Walrus; scored as vorticity by walrus_eval.to_phys).
- the dataset-classification head (cls_head) is dropped: DPOT's fine-tuning loss does not use it.
Environment: DPOT_SRC (default <repo>/ext/DPOT), DPOT_WEIGHTS (default <repo>/ext/dpot_weights/model_H.pth), WALRUS_SRC, WALRUS_CKPT_DIR."""
import os, re, sys
import torch, torch.nn as nn, torch.nn.functional as F

REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
DPOT_SRC = os.environ.get("DPOT_SRC", os.path.join(REPO, "ext", "DPOT"))
DPOT_WEIGHTS = os.environ.get("DPOT_WEIGHTS", os.path.join(REPO, "ext", "dpot_weights", "model_H.pth"))
# = the `args` stored in model_H.pth (width 2048, 27 layers, 8 blocks, mlp_ratio 4, out_layer_dim 128, patch 8, modes 32, T_in 10, res 128)
H_CFG = dict(img_size=128, patch_size=8, mixing_type="afno", in_channels=4, out_channels=4, in_timesteps=10, out_timesteps=1, n_blocks=8,
             embed_dim=2048, out_layer_dim=128, depth=27, modes=32, mlp_ratio=4, n_cls=12, normalize=False, act="gelu", time_agg="exp_mlp")
C_OUT, N_CONST = 4, 2; C_IN = C_OUT + N_CONST; T_PRE = 10; EPS = 1e-8
MODES = 10 ** 6   # AFNO modes kept: every mode (see the module docstring)


def slot_map(names):
    """dynamic field names of a dataset (Walrus order, padded velocity_z excluded) -> DPOT channel per field."""
    vel = [n for n in names if n.startswith("velocity_")]
    if not vel:
        assert len(names) <= 2, f"{names}: more than 2 scalars without velocity"; return list(range(len(names)))
    other = [n for n in names if n not in ("velocity_x", "velocity_y", "pressure")]
    assert len(other) <= 1, f"{names}: more than one scalar besides pressure"
    return [{"velocity_x": 0, "velocity_y": 1, "pressure": 3}.get(n, 2) for n in names]


def _afno_fp32():
    """AFNO's FFTs and block MLPs run in fp32 under bf16 autocast (cuFFT has no bf16 for non-power-of-2 sizes; spectral weights stay exact)."""
    from models.dpot import AFNO2D
    if getattr(AFNO2D, "_physdial_fp32", False): return
    orig = AFNO2D.forward
    def fwd(self, x, spatial_size=None):
        with torch.autocast("cuda", enabled=False): return orig(self, x.float())
    AFNO2D.forward = fwd; AFNO2D._physdial_fp32 = True


class DPOTAdapter(nn.Module):
    """x [B, X, Y, T_PRE, C_IN] (normalized) -> next frame [B, X, Y, C_OUT] (normalized); DPOTNet.forward with normalize=False, the
    position embedding interpolated to the patch grid and the patch-size assert skipped."""
    def __init__(s, pretrained=True, grad_ckpt=False, amp=False, modes=MODES):
        super().__init__()
        sys.path.insert(0, DPOT_SRC); from models.dpot import DPOTNet; _afno_fp32()
        net = DPOTNet(**H_CFG)
        if pretrained:
            sd = torch.load(DPOT_WEIGHTS, map_location="cpu", weights_only=False)["model"]; net.load_state_dict(sd)   # strict: every weight
        old = net.patch_embed.proj[0]; new = nn.Conv2d(C_IN + 3, old.out_channels, old.kernel_size, old.stride)
        with torch.no_grad():   # data channels 0-3 and the 3 coordinate channels keep their weights; the constant-field channels start at zero
            new.weight.zero_(); new.weight[:, :C_OUT] = old.weight[:, :C_OUT]; new.weight[:, C_IN:] = old.weight[:, C_OUT:]; new.bias.copy_(old.bias)
        net.patch_embed.proj[0] = new
        for b in net.blocks: b.filter.modes = modes   # every mode (see the module docstring)
        del net.cls_head
        s.net, s.grad_ckpt, s.amp = net, grad_ckpt, amp

    def forward(s, x):
        n = s.net; B, X, Y, T, _ = x.shape
        x = torch.cat((x, n.get_grid_3d(x)), dim=-1)
        x = x.permute(0, 3, 4, 1, 2).reshape(B * T, -1, X, Y)                       # b x y t c -> (b t) c x y
        x = n.patch_embed.proj(x)
        pe = n.pos_embed if tuple(n.pos_embed.shape[-2:]) == tuple(x.shape[-2:]) else F.interpolate(n.pos_embed, size=x.shape[-2:], mode="bicubic", align_corners=False)
        x = x + pe
        h, w = x.shape[-2:]
        x = x.reshape(B, T, -1, h, w).permute(0, 3, 4, 1, 2)                         # (b t) c x y -> b x y t c
        x = n.time_agg_layer(x).permute(0, 3, 1, 2)                                   # b x y c -> b c x y
        with torch.autocast("cuda", dtype=torch.bfloat16, enabled=s.amp):
            for blk in n.blocks:   # only the blocks run in bf16; embedding, time aggregation and output head stay fp32 (DPOT predicts the full state)
                x = torch.utils.checkpoint.checkpoint(blk, x, use_reentrant=False) if (s.grad_ckpt and s.training) else blk(x)
        return n.out_layer(x.float()).permute(0, 2, 3, 1)


class Layout:
    """Walrus batch <-> DPOT tensors for one dataset (fields from the batch metadata, Walrus's padded-field mask)."""
    def __init__(s, meta, pfm):
        from the_well.data.utils import flatten_field_names
        dyn = flatten_field_names(meta, include_constants=False); allc = flatten_field_names(meta, include_constants=True)
        pfm = pfm.bool().cpu()[:len(dyn)]
        s.F, s.Fc = len(dyn), len(allc) - len(dyn); assert s.Fc <= N_CONST, f"{meta.dataset_name}: {s.Fc} constant fields"
        s.idx = [i for i in range(s.F) if pfm[i]]                                     # present dynamic fields (velocity_z dropped)
        s.slot = slot_map([dyn[i] for i in s.idx]); s.name = meta.dataset_name
        s.mask = torch.zeros(C_OUT); s.mask[s.slot] = 1

    def to_dpot(s, x, std):
        """x [B, T, H, W, 1, F] physical (+ constants [B, H, W, 1, Fc] via cat), std [F + Fc] -> [B, H, W, T, C_IN] normalized."""
        B, T, H, W = x.shape[:4]; out = x.new_zeros(B, H, W, T, C_IN)
        for i, c in zip(s.idx, s.slot): out[..., c] = x[:, :, :, :, 0, i].permute(0, 2, 3, 1) / std[i]
        return out

    def consts(s, cf, std, T):
        """constant fields [B, H, W, 1, Fc] -> the C_OUT.. channels of [B, H, W, T, C_IN] (zeros elsewhere)."""
        B, H, W = cf.shape[:3]; out = cf.new_zeros(B, H, W, T, C_IN)
        for j in range(s.Fc): out[..., C_OUT + j] = (cf[:, :, :, 0, j] / std[s.F + j])[..., None]
        return out

    def from_dpot(s, y, std, like):
        """y [B, H, W, C_OUT] normalized -> physical [B, 1, H, W, 1, F] in Walrus layout (padded fields zero)."""
        out = like.new_zeros(like.shape[0], 1, *like.shape[2:4], 1, s.F)
        for i, c in zip(s.idx, s.slot): out[:, 0, :, :, 0, i] = y[..., c].to(out.dtype) * std[i]
        return out


def pad_history(x):
    """[B, H, W, T, C] with T <= T_PRE -> T_PRE frames (the oldest frame repeated in front)."""
    k = T_PRE - x.shape[3]; return torch.cat([x[:, :, :, :1].expand(-1, -1, -1, k, -1), x], dim=3) if k > 0 else x


def field_std(revin, name, dev):
    return revin.name_to_normalization_stats[name].sample_std.reshape(-1).to(dev).float() + EPS


GRID_NAME = re.compile(r"(.+?)(?:_anchor)?_r\d+")   # physdial_kolmo_r256, physdial_kolmo_anchor_r512, physdial_nsu_r256 -> physdial_kolmo, physdial_kolmo, physdial_nsu
STAT_FIELDS = ("sample_mean", "sample_std", "delta_mean", "delta_std")


def alias_eval_stats(st):
    """The evaluation datasets keep the family's base name (physdial_kolmo, physdial_nsu) while the training datasets carry the grid and, for the
    anchor export, the export suffix (physdial_kolmo_anchor_r256 / _r512). Every `<base>[_anchor]_r<N>` group of the training statistics `st`
    (name -> NormalizationStats) is aliased to `<base>` after asserting that the group's statistics are identical. Returns {base: group}."""
    groups = {}
    for k in list(st):
        m = GRID_NAME.fullmatch(k)
        if m: groups.setdefault(m.group(1), []).append(k)
    out = {}
    for base, names in sorted(groups.items()):
        if base in st: continue
        names = sorted(names); ref = st[names[0]]
        for k in names[1:]:
            for f in STAT_FIELDS:
                a, b = getattr(st[k], f, None), getattr(ref, f, None)
                assert (a is None and b is None) or (a is not None and b is not None and a.shape == b.shape and torch.equal(a.cpu(), b.cpu())), \
                    f"{k} vs {names[0]}: {f} differ; aliasing the evaluation dataset {base} would depend on the grid"
        st[base] = ref; out[base] = names
    return out


DATA_CFGS = ("physdial_corpus", "physdial_corpus_anchor")   # physdial_corpus_anchor = the Walrus mixture with the Kolmogorov anchors; the run records its data config


def walrus_cfg(data="physdial_corpus"):
    """the composed Walrus config of the baseline (data = physdial_corpus or physdial_corpus_anchor; only the data part is used here)."""
    assert data in DATA_CFGS, f"data config {data!r} not in {DATA_CFGS}"
    os.environ.setdefault("WALRUS_CKPT_DIR", os.path.join(REPO, "ext_models", "walrus")); os.environ.setdefault("PHYSDIAL_EXT_DATA", os.path.join(REPO, "data", "ext"))
    os.environ.setdefault("PHYSDIAL_DATA_ROOT", os.path.join(REPO, "data"))
    src = os.environ.get("WALRUS_SRC", os.path.join(REPO, "ext", "walrus")); sys.path.insert(0, src)
    from hydra import compose, initialize_config_dir
    here = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "walrus")
    with initialize_config_dir(config_dir=os.path.join(src, "walrus", "configs"), version_base=None):
        return compose(config_name="config", overrides=[f"hydra.searchpath=[file://{os.path.join(here, 'cfg')}]", f"data={data}", "server=default", "logger=none",
                                                         "distribution=local", "checkpoint=physdial_finetune", "experiment=physdial_finetune", "model=isotropic_model", "trainer=globalnorm"])

