"""Wrap-around (circular) padding for the Wan VAE on periodic axes: patches the padded convolutions of a diffusers AutoencoderKLWan
so that, inside the `periodic(vae, per)` context, periodic axes are padded by wrapping and the other axes by zeros."""
import contextlib
import torch, torch.nn as nn, torch.nn.functional as F

_PER = [None]   # (periodic along H, periodic along W) of the current context, or None


def wrap2d(x, top, bottom, left, right, per):
    """pad the last two dims of x: wrap-around on periodic axes, zeros otherwise."""
    H, W = x.shape[-2:]
    if per[0]: x = torch.cat([x[..., H - top:, :], x, x[..., :bottom, :]], -2) if top or bottom else x
    else: x = F.pad(x, (0, 0, top, bottom))
    if per[1]: x = torch.cat([x[..., W - left:], x, x[..., :right]], -1) if left or right else x
    else: x = F.pad(x, (left, right, 0, 0))
    return x


class _WrapPad2d(nn.Module):
    def __init__(s, orig): super().__init__(); s.orig = orig; s.p = orig.padding   # ZeroPad2d: (left, right, top, bottom)
    def forward(s, x):
        if _PER[0] is None: return s.orig(x)
        l, r, t, b = s.p; return wrap2d(x, t, b, l, r, _PER[0])


class _WrapConv2d(nn.Module):
    def __init__(s, orig): super().__init__(); s.orig = orig
    def forward(s, x):
        c = s.orig
        if _PER[0] is None: return c(x)
        ph, pw = c.padding; return F.conv2d(wrap2d(x, ph, ph, pw, pw, _PER[0]), c.weight, c.bias, c.stride, 0, c.dilation, c.groups)


def install(vae):
    """patch a diffusers AutoencoderKLWan in place (idempotent); returns it."""
    if getattr(vae, "_physdial_periodic", False): return vae
    from diffusers.models.autoencoders.autoencoder_kl_wan import WanCausalConv3d
    if not getattr(WanCausalConv3d, "_physdial_periodic", False):
        orig = WanCausalConv3d.forward
        def fwd(self, x, cache_x=None):
            if _PER[0] is None or not any(_PER[0]): return orig(self, x, cache_x)
            p = list(self._padding)   # (w_left, w_right, h_top, h_bottom, t_front, t_back)
            if cache_x is not None and p[4] > 0:
                x = torch.cat([cache_x.to(x.device), x], dim=2); p[4] -= cache_x.shape[2]
            x = wrap2d(x, p[2], p[3], p[0], p[1], _PER[0])
            return nn.Conv3d.forward(self, F.pad(x, (0, 0, 0, 0, p[4], p[5])))
        WanCausalConv3d.forward = fwd; WanCausalConv3d._physdial_periodic = True
    for name, m in list(vae.named_modules()):
        for cn, ch in list(m.named_children()):
            if isinstance(ch, nn.ZeroPad2d): setattr(m, cn, _WrapPad2d(ch))
            elif isinstance(ch, nn.Conv2d) and any(ch.padding): setattr(m, cn, _WrapConv2d(ch))
    vae._physdial_periodic = True; return vae


@contextlib.contextmanager
def periodic(vae, per):
    """wrap-around padding on the axes flagged in per = (H, W) for every encode / decode inside the block."""
    install(vae); old = _PER[0]; _PER[0] = tuple(bool(p) for p in per)
    try: yield vae
    finally: _PER[0] = old
