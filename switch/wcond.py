"""Condition inputs of the conditional Walrus baseline (Kolmogorov only; every other family as in the plain Walrus fine-tune).

Walrus has one native input for conditions: constant fields (time-invariant fields, as the acoustic media), concatenated to the input
channels. Every input channel is divided by max(rms over time and space of this sample and channel, epsilon) (sample-wise RMS
normalisation; epsilon = the trainer default 1e-5). So:
  cond_forcing   = the trajectory's vorticity forcing / 16 (corpus.Kolmo.FORCE_SCALE): its shape survives the normalisation, its amplitude not;
  cond_<slot>    = a constant field of value SCALE * z, z = (slot value - mean) / std over the Kolmogorov training items (+ the amplitude
                   anchors), for slot in log10_nu, kf, amp, log10_dt (log10 of the model step = stored dt x stride):
                   |SCALE * z| < epsilon, so the normalisation divides by epsilon and the model sees 0.15 z (linear, sign kept, invariant
                   to Walrus's periodic patch jitter). A plain constant would become sign(z).
The statistics are computed once (from the corpus) and stored in <root>/cond_stats.json; every export asserts the same file, so training
and evaluation see the same encoding."""
import json, math, os
import numpy as np

COND = ("cond_forcing", "cond_log10_nu", "cond_kf", "cond_amp", "cond_log10_dt")
SLOTS = ("log10_nu", "kf", "amp", "log10_dt")
SCALE = 1.5e-6; EPS = 1e-5   # normalised value = SCALE * z / EPS = 0.15 z, valid while |z| < EPS / SCALE = 6.7 (asserted with margin)
ZMAX = 0.9 * EPS / SCALE


def slot_values(fam, it):
    p = it.params
    return {"log10_nu": float(p["log10_nu"]), "kf": float(p["kf"]), "amp": float(p["amp"]), "log10_dt": math.log10(10 ** float(p["log10_dt"]) * fam.stride)}


def compute_stats(fam, items):
    V = np.array([[slot_values(fam, it)[s] for s in SLOTS] for it in items], np.float64)
    mu, sd = V.mean(0), V.std(0)
    return {"slots": list(SLOTS), "mean": mu.tolist(), "std": [float(x) if x > 1e-9 else 1.0 for x in sd], "n_items": len(items), "scale": SCALE, "eps": EPS}


def load_or_make_stats(root, fam, items):
    p = os.path.join(root, "cond_stats.json")
    if os.path.exists(p):
        st = json.load(open(p)); assert st["slots"] == list(SLOTS) and st["scale"] == SCALE and st["eps"] == EPS, f"{p}: other encoding {st}"
        return st
    st = compute_stats(fam, items); os.makedirs(root, exist_ok=True); json.dump(st, open(p + ".tmp", "w"), indent=1); os.replace(p + ".tmp", p)
    return st


def cond_fields(fam, it, stats, base_cond_fields):
    """[cond_forcing, cond_log10_nu, cond_kf, cond_amp, cond_log10_dt] as float32 [H, W]; base_cond_fields = the family's own cond_fields (the forcing)."""
    f = np.asarray(base_cond_fields(it)[0], np.float32); assert f.ndim == 2, f.shape
    out = [f / 16.0]; sv = slot_values(fam, it)
    for j, s in enumerate(SLOTS):
        z = (sv[s] - stats["mean"][j]) / stats["std"][j]
        assert abs(z) < ZMAX, f"{it.path} {it.traj}: {s} = {sv[s]} is {z:.2f} std from the training mean, beyond the linear range {ZMAX:.1f} of the encoding"
        out.append(np.full(f.shape, SCALE * z, np.float32))
    return out
