"""Per-frame statistics of Kolmogorov vorticity fields for the multi-switch read-out.

stats(w, F) with w [n, T, H, W] vorticity (physical units, [ix, iy] layout, periodic box 2 pi) and F [n, J, H, W] reference forcing fields
-> dict of float32 numpy arrays:
  ke    [n, T]      kinetic energy per unit area, 0.5 sum_k |w_k|^2 / |k|^2 (w_k = FFT / (H W))
  ens   [n, T]      enstrophy 0.5 mean (w - mean w)^2
  proj  [n, T, J]   mean(w F_j) / rms(F_j): how much of the flow follows forcing pattern j
  spec  [n, T, NK]  energy spectrum in integer shells, NK = min(H, W) // 2
Computed in float64 in chunks of frames. Used for the predictions (eval_ms.py, walrus_ms_eval.py) and the truth (analyze_ms.py)."""
import hashlib, os
import numpy as np, torch

MD5 = hashlib.md5(open(os.path.abspath(__file__), "rb").read()).hexdigest()


def stats(w, F, chunk=8):
    w = torch.as_tensor(w); F = torch.as_tensor(F).to(w.device, torch.float64)
    n, T, H, W = w.shape; assert F.shape[0] == n and F.shape[-2:] == (H, W), (w.shape, F.shape)
    kx = torch.fft.fftfreq(H, 1.0 / H, device=w.device, dtype=torch.float64); ky = torch.fft.fftfreq(W, 1.0 / W, device=w.device, dtype=torch.float64)
    k2 = kx[:, None] ** 2 + ky[None, :] ** 2; inv = torch.where(k2 > 0, 1.0 / k2.clamp(min=1e-30), torch.zeros_like(k2))
    NK = min(H, W) // 2; sh = torch.round(k2.sqrt()).long(); m = sh < NK
    fr = F.pow(2).mean((-2, -1)).sqrt().clamp(min=1e-30)   # [n, J]
    ke, ens, proj, spec = [], [], [], []
    for t0 in range(0, T, chunk):
        x = w[:, t0:t0 + chunk].to(torch.float64); e2 = torch.fft.fft2(x).abs().pow(2) / float(H * W) ** 2   # [n, c, H, W]
        ke.append(0.5 * (e2 * inv).sum((-2, -1))); ens.append(0.5 * (x - x.mean((-2, -1), keepdim=True)).pow(2).mean((-2, -1)))
        proj.append(torch.einsum("nchw,njhw->ncj", x, F) / float(H * W) / fr[:, None, :])
        s = torch.zeros(n, x.shape[1], NK, device=w.device, dtype=torch.float64)
        s.index_add_(2, sh[m], (0.5 * e2 * inv)[..., m]); spec.append(s)
    cat = lambda L: torch.cat(L, 1).float().cpu().numpy()
    return {"ke": cat(ke), "ens": cat(ens), "proj": cat(proj), "spec": cat(spec)}


def forcings(path, traj, K):
    import h5py
    with h5py.File(path, "r") as h:
        g = h[traj]; assert g["forcing_seg"].shape[0] == K, (path, traj, g["forcing_seg"].shape, K)
        return np.concatenate([np.asarray(g["forcing_A"], np.float32)[None], np.asarray(g["forcing_seg"], np.float32)], 0)
