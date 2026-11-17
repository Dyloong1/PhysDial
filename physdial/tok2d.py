"""2D tokenizer of the corpus: the Wan2.1 video VAE applied to single frames. Every physical channel is compressed on its own, after
per-frame normalisation, so one tokenizer serves every family whatever its channels are (a scalar field is replicated to RGB; the decoded
RGB is averaged).

Interface: tok = load(spec); z = tok.encode(x) with x [B, 1, H, W] normalised to about [-1, 1]; x' = tok.decode(z);
tok.factor (spatial downsampling, 8), tok.zc (latent channels per physical channel, 16).

Specs:
  wan21            Wan2.1 VAE, frozen: the latent space of the pretrained Wan2.1 DiT
  wan21ft:<ckpt>   the same VAE with the decoder fine-tuned on the corpus (train_tok2d.py; relative paths are under physdial/ckpt/)"""
import os, torch

WAN21 = "Wan-AI/Wan2.1-T2V-1.3B-Diffusers"


class WanTok(torch.nn.Module):
    def __init__(self, ckpt=None, dtype=torch.float32):
        super().__init__()
        from diffusers import AutoencoderKLWan
        self.vae = AutoencoderKLWan.from_pretrained(WAN21, subfolder="vae", torch_dtype=dtype)
        cfg = self.vae.config; self.zc = cfg.z_dim; self.factor = 8; lm, ls = cfg.latents_mean, cfg.latents_std
        if ckpt:   # fine-tuned decoder (train_tok2d.py)
            ck = torch.load(ckpt, map_location="cpu", weights_only=False); self.vae.load_state_dict(ck["model"]); lm, ls = ck["lmean"], ck["lstd"]
        self.register_buffer("lmean", torch.as_tensor(lm, dtype=torch.float32).view(1, -1, 1, 1), persistent=False)
        self.register_buffer("lstd", torch.as_tensor(ls, dtype=torch.float32).view(1, -1, 1, 1), persistent=False)

    def encode(self, x, sample=False):            # x [B, 1, H, W] -> z [B, zc, H/f, W/f] (Wan latent normalisation applied)
        d = self.vae.encode(x.expand(-1, 3, -1, -1)[:, :, None]).latent_dist
        z = (d.sample() if sample else d.mode())[:, :, 0]
        return (z - self.lmean) / self.lstd

    def decode(self, z):                           # -> [B, 1, H, W] (mean of the three RGB outputs)
        y = self.vae.decode((z * self.lstd + self.lmean)[:, :, None]).sample[:, :, 0]
        return y.mean(1, keepdim=True)


def load(spec, device="cuda"):
    kind, _, ckpt = spec.partition(":")
    if ckpt and not os.path.isabs(ckpt): ckpt = os.path.join(os.path.dirname(os.path.abspath(__file__)), "ckpt", ckpt)
    if kind == "wan21": tok = WanTok()
    elif kind == "wan21ft": tok = WanTok(ckpt)
    else: raise ValueError(spec)
    return tok.to(device).eval()
