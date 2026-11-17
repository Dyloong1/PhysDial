"""Per-frame, per-channel normalisation of a physical field before the tokenizer (encode_corpus, train_gen.Conditioner, tok_gate share it):
x_n = (x - mu) / s with mu the spatial mean and s = max |x - mu|, so every frame reaches the tokenizer in [-1, 1].
The stored amplitude / mean planes carry (mu, s), so the model sees the physical level of every frame."""
import numpy as np


def scale_t(x):
    """x [N, H, W] torch -> mu [N], s [N] (s >= 1e-12)."""
    mu = x.mean((-1, -2)); s = (x - mu[:, None, None]).abs().amax((-1, -2))
    return mu, s.clamp_min(1e-12)


def scale_np(x):
    """x [n, C, H, W] numpy -> mean, sc [n, C, 1, 1] (tok_gate's layout)."""
    mu = x.mean((2, 3), keepdims=True); s = np.abs(x - mu).max((2, 3), keepdims=True)
    return mu, np.maximum(s, 1e-12)
