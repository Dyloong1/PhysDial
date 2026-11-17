#!/usr/bin/env python
"""2D Kolmogorov-flow generator: pseudo-spectral vorticity solver in PyTorch (GPU), restartable per nu.

Equation on [0,2pi)^2, periodic:  dw/dt + u.grad w + beta v = nu lap w - alpha w + F(y),
  velocity forcing f = (sin(n (y+phi)), 0)  =>  vorticity forcing F = curl f = -n cos(n (y+phi)),  alpha = 0.1, beta = 0 by default.
k_f, amplitude, alpha and beta are options; --forcing multi draws a random multi-mode forcing per trajectory; --decay gives freely
decaying turbulence (no forcing, no drag). The resolution is estimated per nu and doubled until the enstrophy spectrum passes
the dissipation check; frames are stored on the solver grid. Normally driven by gen_job.py."""
import argparse, json, math, os, time
import numpy as np, torch, h5py

HERE = os.path.dirname(os.path.abspath(__file__))
DEV = "cuda"
KF, ALPHA, AMP = 4, 0.1, 1.0


class Solver:
    def __init__(self, N, nu, B, kf=KF, alpha=ALPHA, phases=None, cfl=0.5, amp=AMP, F2d=None, beta=0.0):
        """F2d [B, N, N] (torch, DEV): an explicit vorticity-forcing field per trajectory instead of the single sinusoidal mode.
        beta: beta-plane term beta v in the vorticity equation (v = velocity along y, the axis the forcing varies along; v_hat = -i kx psi_hat,
        psi_hat = w_hat / k^2), i.e. +i beta kx / k^2 in the linear operator (Rossby waves), integrated exactly by the integrating factor."""
        self.N, self.nu, self.B, self.cfl = N, nu, B, cfl
        k = torch.fft.fftfreq(N, d=1.0 / N, device=DEV)
        kr = torch.fft.rfftfreq(N, d=1.0 / N, device=DEV)
        self.kx, self.ky = k[:, None], kr[None, :]
        self.k2 = self.kx ** 2 + self.ky ** 2
        self.k2i = torch.where(self.k2 > 0, 1.0 / self.k2, torch.zeros_like(self.k2))
        kc = N // 3
        self.mask = ((self.kx.abs() <= kc) & (self.ky <= kc)).float()  # 2/3 rule (square)
        self.L = -(nu * self.k2 + alpha)
        self.beta = float(beta)
        if self.beta: self.L = self.L + 1j * self.beta * self.kx * self.k2i
        y = torch.arange(N, device=DEV) * (2 * math.pi / N)
        ph = phases if phases is not None else torch.zeros(B, device=DEV)
        if F2d is not None: self.Fh = torch.fft.rfft2(F2d.to(DEV).float().contiguous()) * self.mask
        else:
            F = -kf * amp * torch.cos(kf * (y[None, :] + ph[:, None]))  # [B,N] depends on y
            self.Fh = torch.fft.rfft2(F[:, None, :].expand(B, N, N).contiguous()) * self.mask
        self.dx = 2 * math.pi / N

    def vel(self, wh):
        ph = wh * self.k2i
        return torch.fft.irfft2(1j * self.ky * ph, s=(self.N,) * 2), torch.fft.irfft2(-1j * self.kx * ph, s=(self.N,) * 2)

    def N_(self, wh):
        ph = wh * self.k2i
        u = torch.fft.irfft2(1j * self.ky * ph, s=(self.N,) * 2); v = torch.fft.irfft2(-1j * self.kx * ph, s=(self.N,) * 2)
        wx = torch.fft.irfft2(1j * self.kx * wh, s=(self.N,) * 2); wy = torch.fft.irfft2(1j * self.ky * wh, s=(self.N,) * 2)
        return -torch.fft.rfft2(u * wx + v * wy) * self.mask + self.Fh

    def umax(self, wh):
        u, v = self.vel(wh)
        return torch.maximum(u.abs().amax(), v.abs().amax()).item()

    def step(self, wh, dt):
        E = torch.exp(self.L * dt); E2 = torch.exp(self.L * dt / 2)
        k1 = self.N_(wh)
        k2 = self.N_(E2 * (wh + dt / 2 * k1))
        k3 = self.N_(E2 * wh + dt / 2 * k2)
        k4 = self.N_(E * wh + dt * E2 * k3)
        return E * wh + dt / 6 * (E * k1 + 2 * E2 * (k2 + k3) + k4)

    def advance(self, wh, T):
        """advance by exactly T time units; returns wh, nsteps"""
        t, n = 0.0, 0
        while t < T - 1e-12:
            um = max(self.umax(wh), 1e-3)
            dt = min(self.cfl * self.dx / um, T - t)
            wh = self.step(wh, dt); t += dt; n += 1
        return wh, n


def draw_modes(B, gen, nmodes=4, kmin=2, kmax=5):
    """random multi-mode vorticity forcing: per trajectory `nmodes` integer wavevectors k with kmin <= |k| <= kmax (ky >= 0,
    no duplicates), weights w with sum w^2 = 1 and phases; returns a list of dicts (grid-independent)."""
    cand = [(kx, ky) for kx in range(-kmax, kmax + 1) for ky in range(0, kmax + 1) if kmin ** 2 <= kx * kx + ky * ky <= kmax ** 2 and not (ky == 0 and kx < 0)]
    out = []
    for _ in range(B):
        idx = torch.randperm(len(cand), generator=gen, device=DEV)[:nmodes].cpu().tolist(); kvecs = [cand[i] for i in idx]
        w = torch.rand(nmodes, generator=gen, device=DEV) * 0.5 + 0.5; w = (w / w.pow(2).sum().sqrt()).cpu().tolist()
        ph = (torch.rand(nmodes, generator=gen, device=DEV) * 2 * math.pi).cpu().tolist()
        kf_eff = sum(wi * wi * math.sqrt(kx * kx + ky * ky) for wi, (kx, ky) in zip(w, kvecs))   # energy-weighted mean |k| -> the kf slot
        out.append({"k": kvecs, "w": w, "phase": ph, "kf_eff": kf_eff})
    return out


def multi_field(modes, N, amp, device=DEV):
    """vorticity forcing F(x, y) = amp * sum_i w_i |k_i| cos(k_i . x + phi_i) on the N^2 grid ([ix, iy] layout) -> [B, N, N] (rms ~ the single mode's
    amp * kf / sqrt(2))."""
    x = torch.arange(N, device=device, dtype=torch.float32) * (2 * math.pi / N); X, Y = x[:, None], x[None, :]; out = []
    for m in modes:
        F = torch.zeros(N, N, device=device)
        for (kx, ky), w, ph in zip(m["k"], m["w"], m["phase"]): F += w * math.sqrt(kx * kx + ky * ky) * torch.cos(kx * X + ky * Y + ph)
        out.append(amp * F)
    return torch.stack(out)


def random_ic(B, N, gen, amp=1.0, k0=6.0):
    w = torch.randn(B, N, N, device=DEV, generator=gen)
    wh = torch.fft.rfft2(w)
    k = torch.fft.fftfreq(N, d=1.0 / N, device=DEV)[:, None]; kr = torch.fft.rfftfreq(N, d=1.0 / N, device=DEV)[None, :]
    kk = torch.sqrt(k ** 2 + kr ** 2)
    wh = wh * torch.exp(-(kk / k0) ** 2) * (kk > 0)
    w = torch.fft.irfft2(wh, s=(N, N))
    w = w / w.pow(2).mean((-2, -1), keepdim=True).sqrt() * amp
    return torch.fft.rfft2(w)


def resize(wh, N_old, N_new):
    """spectral zero-pad / truncate rfft coefficients [B,N_old,N_old//2+1] -> N_new (vorticity amplitude kept)."""
    B = wh.shape[0]; out = torch.zeros(B, N_new, N_new // 2 + 1, dtype=wh.dtype, device=wh.device)
    kmax = min(N_old, N_new) // 2 - 1
    kyn = min(kmax + 1, wh.shape[2], out.shape[2])
    out[:, :kmax + 1, :kyn] = wh[:, :kmax + 1, :kyn]
    out[:, -kmax:, :kyn] = wh[:, -kmax:, :kyn]
    return out * (N_new / N_old) ** 2


def to_grid(wh, N):
    """the stored frame on the solver grid: modes |kx|, |ky| <= N/2 - 1 (the Nyquist mode dropped)."""
    out = resize(wh, N, N)
    return torch.fft.irfft2(out, s=(N, N))  # [B,N,N] [ix,iy]


def spectra(wh, N):
    """shell-summed energy spectrum E(k), k=1..N//2 (averaged over batch), returns (E,Z) with Z=k^2 E."""
    full = torch.fft.fft2(torch.fft.irfft2(wh, s=(N, N))) / N ** 2
    k = torch.fft.fftfreq(N, d=1.0 / N, device=wh.device)
    kk = torch.sqrt(k[:, None] ** 2 + k[None, :] ** 2)
    k2 = kk ** 2; k2[0, 0] = 1
    e = 0.5 * (full.abs() ** 2 / k2).mean(0); e[0, 0] = 0   # |u_hat|^2 = |w_hat|^2/k^2
    idx = torch.round(kk).long().flatten().clamp(max=N // 2)
    Ek = torch.zeros(N // 2 + 1, device=wh.device, dtype=torch.float64).index_add_(0, idx, e.flatten().double())
    kv = torch.arange(N // 2 + 1, device=wh.device, dtype=torch.float64)
    return Ek.cpu().numpy(), (kv ** 2 * Ek).cpu().numpy()


def check_spectrum(Z, N):
    """Z: enstrophy spectrum k=0..N//2. kc = N//3. Pass if tail (0.8kc..kc) mean >= 3 decades below peak and no pile-up."""
    kc = N // 3; Zp = Z[1:kc + 1].max()
    tail = Z[int(0.8 * kc):kc + 1].mean(); prev = Z[int(0.6 * kc):int(0.8 * kc)].mean()
    drop = np.log10(Zp / max(tail, 1e-300))
    return dict(drop_decades=float(drop), pileup_ratio=float(tail / prev), ok=bool(drop >= 3.0 and tail < prev), kc=kc)


def rolloff(Z, N):
    """dslope = (log-slope of Z over 0.7-0.95 k_c) - (log-slope over k=12..32); <= -1.5 means a visible dissipation roll-off."""
    kc = N // 3; k = np.arange(len(Z), dtype=np.float64)
    def sl(a, b):
        m = (k >= a) & (k <= b) & (Z > 0); return np.polyfit(np.log(k[m]), np.log(Z[m]), 1)[0]
    return float(sl(0.7 * kc, 0.95 * kc) - sl(12, 32))


def est_N(nu):
    """first-guess resolution per nu: N ~ nu^-1/2 (dissipation wavenumber; thresholds calibrated with the roll-off criterion), verified + escalated post hoc by the spectrum check."""
    for N, nu_min in ((256, 4.4e-4), (512, 1.0e-4)):
        if nu >= nu_min: return N
    return 1024


def generate_nu(nu, idx, ntraj, nframes, frame_eddy, seed, N, tspin, trelax, verbose=True, kf=KF, amp=AMP, decay=False, ic_wrms=5.0, forcing="single", nmodes=4, kband=(2, 5), alpha=ALPHA, beta=0.0):
    """decay=True: no forcing, no drag, no spin-up; random IC (k0 = 6) with rms vorticity ic_wrms
    directly on the N grid, half an initial eddy time of development, then t_eddy = 1 / omega_rms at that moment and frames every
    frame_eddy * t_eddy from there (the flow decays, so later frames change less per step).
    alpha: the linear drag of the forced runs; recorded in meta and as the file attribute alpha. beta: the beta-plane term (Solver);
    recorded as the file attribute beta when nonzero."""
    gen = torch.Generator(device=DEV); gen.manual_seed(seed)
    ph = (torch.rand(ntraj, device=DEV, generator=gen) * 2 * math.pi)
    modes = draw_modes(ntraj, gen, nmodes, *kband) if forcing == "multi" else None
    F_at = (lambda n: multi_field(modes, n, amp)) if modes else (lambda n: None)
    t0 = time.time(); kt = []  # (time, KE mean over traj)
    if decay:
        amp = 0.0; Ns = N; s = Solver(N, nu, ntraj, kf=kf, alpha=0.0, amp=0.0, phases=ph); wh = random_ic(ntraj, N, gen, amp=ic_wrms)
        tt, nstep = 0.0, 0; wh, n = s.advance(wh, 0.5 / ic_wrms); tt += 0.5 / ic_wrms; nstep += n; kt.append((tt, ke(s, wh)))
        wr = [wrms(s, wh)]; tspin = trelax = 0
    else:
        Ns = min(N, 128)
        s = Solver(Ns, nu, ntraj, kf=kf, alpha=alpha, amp=amp, phases=ph, F2d=F_at(Ns), beta=beta)
        wh = random_ic(ntraj, Ns, gen)
        tt, nstep = 0.0, 0
    if not decay:
        # 1. spin-up on a coarse grid, KE recorded every time unit
        for _ in range(int(tspin)):
            wh, n = s.advance(wh, 1.0); tt += 1.0; nstep += n
            kt.append((tt, ke(s, wh)))
        if N != Ns:   # 2. relaxation on the solver grid
            wh = resize(wh, Ns, N); s = Solver(N, nu, ntraj, kf=kf, alpha=alpha, amp=amp, phases=ph, F2d=F_at(N), beta=beta)
            for _ in range(int(trelax)):
                wh, n = s.advance(wh, 1.0); tt += 1.0; nstep += n
                kt.append((tt, ke(s, wh)))
        # 3. t_eddy measured over 2 time units
        wr = []
        for _ in range(4):
            wh, n = s.advance(wh, 0.5); tt += 0.5; nstep += n
            wr.append(wrms(s, wh)); kt.append((tt, ke(s, wh)))
    t_eddy = 1.0 / float(np.mean(wr)); dtf = frame_eddy * t_eddy
    # 4. the stored frames, every frame_eddy * t_eddy
    frames, K = [], []; Zs = []
    for f in range(nframes):
        if f > 0:
            wh, n = s.advance(wh, dtf); nstep += n; tt += dtf
        frames.append(to_grid(wh, N).half().cpu())
        k_ = ke(s, wh); K.append(k_); kt.append((tt, k_))
        Ek, Zk = spectra(wh, N); Zs.append(Zk);
    frames = torch.stack(frames, 1)  # [B,T,N,N]
    Zm = np.mean(Zs, 0); chk = check_spectrum(Zm, N)
    # stats from full-res final-window statistics (computed on last frame batch + spectra average)
    Ek_m = np.mean([spectra_from(Z) for Z in Zs], 0)
    kv = np.arange(len(Zm), dtype=np.float64)
    E_tot = Ek_m.sum(); Z_tot = (kv ** 2 * Ek_m).sum()          # E=<|u|^2>/2, Z=<w^2>/2 (mean over frames, trajs)
    urms = math.sqrt(E_tot)                                         # per-component rms = sqrt(<|u|^2>/2) = sqrt(E)
    wrms_ = math.sqrt(2 * Z_tot); lam = math.sqrt(E_tot / Z_tot)    # Taylor microscale sqrt(<|u|^2>/<w^2>)
    Karr = np.array(K); Kn = Karr.mean(1)
    slope = np.polyfit(np.arange(nframes) * dtf, Kn, 1)[0] * nframes * dtf / Kn.mean()   # relative drift over window
    meta = dict(amp=amp, nu=nu, idx=idx, N=N, N_spin=Ns, T_spin=tspin, T_relax=trelax, t_eddy=t_eddy, dt_frame=dtf, frame_eddy=frame_eddy,
                rolloff_dslope=rolloff(Zm, N), n_traj=ntraj, n_frames=nframes, energy=float(E_tot), u_rms=urms, w_rms=wrms_, enstrophy=float(Z_tot), taylor=lam,
                Re_lambda=urms * lam / nu, Re_nu=1.0 / nu, K_rel_drift_window=float(slope), K_rel_std_window=float(Kn.std() / Kn.mean()),
                spec_check=chk, nsteps=nstep, wall_s=time.time() - t0, seed=seed, phases=ph.cpu().tolist(),
                alpha=0.0 if decay else float(alpha), decay=bool(decay), ic_wrms=float(ic_wrms) if decay else None,
                forcing=forcing, modes=modes, nmodes=nmodes if modes else None, kband=list(kband) if modes else None,
                kf=(float(np.mean([m["kf_eff"] for m in modes])) if modes else kf))   # multi: the file-level kf attr = mean energy-weighted |k|
    if beta: meta["beta"] = float(beta)
    kt = np.array([(t_, float(np.mean(k_))) for t_, k_ in kt]); return frames.numpy(), meta, kt, (Ek_m, Zm), s


def spectra_from(Z):
    kv = np.arange(len(Z), dtype=np.float64); out = np.zeros_like(Z); out[1:] = Z[1:] / kv[1:] ** 2; return out


def ke(s, wh):
    u, v = s.vel(wh); return (0.5 * (u ** 2 + v ** 2).mean((-2, -1))).cpu().numpy()


def wrms(s, wh):
    return torch.fft.irfft2(wh, s=(s.N,) * 2).pow(2).mean().sqrt().item()


def fname(nu, out): return os.path.join(out, f"2D_kolmogorov_nu={nu!r}.h5")


def write_h5(path, frames, meta):
    T, M = frames.shape[1], frames.shape[2]
    x = np.arange(M) * 2 * np.pi / M; t = np.arange(T) * meta["dt_frame"]
    tmp = path + ".tmp"
    with h5py.File(tmp, "w") as h:
        for j in range(frames.shape[0]):
            g = h.create_group(f"{j:04d}")
            g.create_dataset("data", data=frames[j][..., None].astype(np.float16))   # [T,N,N,1]
            g.create_dataset("grid/t", data=t); g.create_dataset("grid/x", data=x); g.create_dataset("grid/y", data=x)
            g.attrs["forcing_phase"] = meta["phases"][j]
            if meta.get("modes"):
                F = multi_field([meta["modes"][j]], M, meta["amp"], device="cpu")[0].numpy(); g.create_dataset("forcing", data=F.astype(np.float32))
                g.attrs["modes_k"] = np.array(meta["modes"][j]["k"]); g.attrs["modes_w"] = np.array(meta["modes"][j]["w"]); g.attrs["modes_phase"] = np.array(meta["modes"][j]["phase"]); g.attrs["kf_eff"] = meta["modes"][j]["kf_eff"]
            else:   # vorticity forcing F = -amp*kf*cos(kf(y+phi)) on the grid, [ix,iy] like `data`
                F = -meta["amp"] * meta["kf"] * np.cos(meta["kf"] * (x[None, :] + meta["phases"][j])) * np.ones((M, 1))
                g.create_dataset("forcing", data=F.astype(np.float32))
        for k in ("nu", "t_eddy", "dt_frame", "N", "alpha", "kf", "amp"):
            h.attrs[k] = meta.get(k, {"alpha": ALPHA, "kf": KF}.get(k))
        h.attrs["decay"] = int(bool(meta.get("decay", False)))
        h.attrs["forcing"] = meta.get("forcing", "single")
        if meta.get("modes"): h.attrs["nmodes"] = meta["nmodes"]; h.attrs["kband"] = np.array(meta["kband"])
        if meta.get("beta"): h.attrs["beta"] = float(meta["beta"])
    os.replace(tmp, path)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--nu_list", type=int, default=60); ap.add_argument("--lo", type=float, default=-5.0); ap.add_argument("--hi", type=float, default=-2.0)
    ap.add_argument("--ntraj", type=int, default=12); ap.add_argument("--nframes", type=int, default=300)
    ap.add_argument("--frame_eddy", type=float, default=0.1); ap.add_argument("--tspin", type=int, default=60); ap.add_argument("--trelax", type=int, default=5)
    ap.add_argument("--only_idx", type=int, nargs="*", default=None); ap.add_argument("--nu", type=float, nargs="*", default=None)
    ap.add_argument("--N", type=int, default=None, help="force resolution (else estimated + escalated)")
    ap.add_argument("--max_N", type=int, default=1024); ap.add_argument("--force", action="store_true"); ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--kf", type=int, default=KF); ap.add_argument("--amp", type=float, default=AMP)
    ap.add_argument("--alpha", type=float, default=ALPHA, help="linear drag; ignored with --decay (no drag)")
    ap.add_argument("--beta", type=float, default=0.0, help="beta-plane term (written as the file attribute beta when nonzero)")
    ap.add_argument("--decay", action="store_true", help="decaying turbulence: no forcing, no drag, no spin-up; frames from a random IC")
    ap.add_argument("--ic_wrms", type=float, default=5.0, help="--decay: rms vorticity of the random initial condition (sets the initial Re)")
    ap.add_argument("--forcing", default="single", choices=["single", "multi"], help="multi: random multi-mode vorticity forcing per trajectory, --nmodes integer wavevectors with |k| in --kband")
    ap.add_argument("--nmodes", type=int, default=4); ap.add_argument("--kband", default="2,5", help="kmin,kmax of the multi-mode forcing")
    ap.add_argument("--h5_root", default=None, help="root of the files (default ./h5)")
    ap.add_argument("--out_dir", required=True, help="<h5_root>/<out_dir>/ (files) and <h5_root>/meta/<out_dir>/ (per-nu meta and series)")
    a = ap.parse_args()
    root = a.h5_root or os.path.join(HERE, "h5"); out = os.path.join(root, a.out_dir); meta_dir = os.path.join(root, "meta", a.out_dir)
    os.makedirs(out, exist_ok=True); os.makedirs(meta_dir, exist_ok=True)
    lg = np.linspace(a.hi, a.lo, a.nu_list)[::-1]   # ascending nu order idx 0 = smallest nu
    nus = [float(f"{10 ** x:.6e}") for x in lg]
    todo = list(range(len(nus)))[::-1] if a.only_idx is None else a.only_idx
    for i in todo:
        nu = nus[i] if a.nu is None else a.nu[0]
        path = fname(nu, out); mp = os.path.join(meta_dir, f"nu={nu!r}.json")
        if os.path.exists(path) and os.path.exists(mp) and not a.force:
            print(f"[{i}] nu={nu:.3e} exists, skip", flush=True); continue
        N = a.N or est_N(nu)
        while True:
            fr, meta, kt, (Ek, Zk), s = generate_nu(nu, i, a.ntraj, a.nframes, a.frame_eddy, a.seed + 1000 * i, N, a.tspin, a.trelax, kf=a.kf, amp=a.amp, decay=a.decay, ic_wrms=a.ic_wrms,
                                                    forcing=a.forcing, nmodes=a.nmodes, kband=tuple(int(x) for x in a.kband.split(",")), alpha=a.alpha, beta=a.beta)
            c = meta["spec_check"]
            if c["ok"] and meta["rolloff_dslope"] > -1.5:   # the roll-off criterion (dslope <= -1.5) as a second resolution check
                c["ok"] = False; c["rolloff_fail"] = True
            print(f"[{i}] kf={a.kf} amp={a.amp} nu={nu:.3e} N={N} dslope={meta['rolloff_dslope']:.2f} t_eddy={meta['t_eddy']:.3f} drop={c['drop_decades']:.2f} pile={c['pileup_ratio']:.2f} "
                  f"ok={c['ok']} wall={meta['wall_s']:.0f}s" + (f" alpha={a.alpha:g}" if a.alpha != ALPHA else "") + (f" beta={a.beta:g}" if a.beta else "") + f" gpu_peak={torch.cuda.max_memory_allocated() / 2**30:.1f}GB", flush=True)
            if c["ok"] or N >= a.max_N or a.N: break
            N *= 2
        write_h5(path, fr, meta)
        np.savez(os.path.join(meta_dir, f"nu={nu!r}_series.npz"), kt=kt, Ek=Ek, Zk=Zk, snap=fr[0, -1].astype(np.float32))
        json.dump(meta, open(mp, "w"), indent=1)
        torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
