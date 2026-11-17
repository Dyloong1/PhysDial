#!/usr/bin/env python
"""Physical-constraint projection of the predicted frames (output post-processing) and extra metrics, on the `train_gen --dump_pred`
dumps (CPU, numpy).
  python3 projection.py [--run NAME] [--ref NAME] [--out FILE]

Inputs (in physdial/results): the model evaluated with the standard protocol (fine-tuned decoder, 25 steps, the standard
windows and noise) with --dump_pred 32 on every column (stage2-noperiodic-s0-eval-dump: result json + pred/<name>/<family>__
<split>.npz, float16 history / prediction / truth in physical units, [frames, C, H, W]); the same for its in-range reference
(stage2-noperiodic-ref-s0-eval-dump; optional: absent -> the extrapolation-cost lines read 'missing'); the training result jsons of both
models (stage2-noperiodic-s0, stage2-noperiodic-ref-s0) for the checkpoint check.
Checks (a failing column is 'missing(...)' and left out of every family value; a missing required file aborts):
  - the dump run is the expected evaluation (decoder TOK, store STORE, --dump_pred 32, the listed --eval_splits, 25 steps, the trained
    checkpoint and its selected_step, no failed split);
  - every window of a column is dumped, and the pooled relative L2 recomputed from the dump at frames 1 and 16 equals the json's
    per-window rel_l2_pooled within |x - y| <= 1e-2 y + 2e-3 (train_gen computes the json value from the float32 fields BEFORE the float16
    cast of the dump; the float16 rounding bound and why the absolute floor is needed: `consistent`). A larger difference = the dump is
    not that evaluation.
Constraints (a constraint applies to a family only if the TRUTH of the dumped windows satisfies it):
  zero_mean  kolmo, nsu vorticity: doubly periodic vorticity has zero spatial mean. Applies if the 99th percentile over truth frames of
             |mean| / rms <= 1e-3. Projection: subtract the predicted frame's mean (orthogonal projection).
  leray      sf velocity_x / velocity_y (incompressible, doubly periodic): the divergence-free part (Fourier, index wavenumbers, square
             cells). The axis of velocity_x (H or W) is the one under which the truth's divergence is smaller; applies if the 99th percentile
             over truth frames of ||k.u|| / ||k|| ||u|| (spectral) <= 1e-2. Projection: u <- u - k (k.u) / |k|^2 (orthogonal projection).
             First derivatives use k = 0 on the Nyquist row / column of an even grid (with k = -pi there the projected spectrum is not
             Hermitian, so the real part would be neither exactly divergence-free nor idempotent); the projection is then exact.
  nonneg     gs A, B; euler density, pressure; trl density, pressure (concentrations, densities, pressures of compressible gas). Applies if
             the truth minimum >= -1e-6 x max|truth| over every dumped frame. Projection: max(x, 0) (projection onto a convex set).
  Each projection P is non-expansive, so ||P(x) - t|| <= ||P(x) - P(t)|| + ||P(t) - t|| <= ||x - t|| + dist(t, set): a frame whose
  projected error exceeds raw x (1 + 1e-3) + dist(truth, set) (+ 1e-12; all relative to ||truth||) is counted and the window FLAGGED (a
  bug). dist(truth, set) is the truth's own distance to the set (within the applicability tolerance; without it a window predicted to
  float16 precision, e.g. a GS steady state, whose truth has a -1e-7 undershoot would flag).
  dist(truth, set) is computed by closed forms that never call project() (`truth_dist`): zero_mean |mean| sqrt(H W) per constrained
  channel; nonneg ||min(t, 0)||; leray the norm of the gradient part of the analytic Helmholtz split, sqrt(sum_k |k.U|^2 / |k|^2 / (H W))
  (Parseval, the index wavenumbers of `kvec_odd`); per frame the constrained channels add in quadrature (they are disjoint, so the set is
  a product). An allowance ||project(t) - t|| would let a buggy projection that also moves the truth absorb its own bug. Second check, also
  per frame: an orthogonal projection onto the set moves the truth by exactly dist(truth, set), so | ||project(t) - t|| - dist | >
  1e-9 + 1e-6 dist (relative to ||truth||) is counted ('truth_move_bad'; it also catches a projection that does nothing) and FLAGs.
Quantity: per column, half (tables/devsplit) and frame f in (1, 16): mean over the half's windows of the pooled relative L2 after
  projection / the same before projection (both recomputed from the dump). The projection is applied to every predicted frame on its own;
  the history fed to the model is unchanged (no feedback: frame 1 is exact for an in-loop projection, frame 16 is not). A half whose raw
  mean is exactly 0 (every window predicted to float16 precision): ratio 1 if the projection moved it by no more than the mean truth
  distance (+ 1e-12), else +inf.
Verdict per projected family (family value = geomean over its in-range columns, per half): 'worse' iff any half at frame 1 or 16 is
  > 1.05 (a non-expansive projection cannot raise a family value above 1 by more than the truth's distance to the set, which the
  applicability tolerances keep far below 5 %; so it marks a bug and flags the overall verdict); else 'useful' iff both halves <= 0.95 at
  frame 1 OR at frame 16; otherwise 'small (<= 5 %)'; 'not applicable(<truth statistic>)' if the truth check fails; 'missing' if no
  usable in-range column. MAIN: 'projection useful: <families>' if any family is useful, else 'projection changes the error by <= 5 %';
  FLAG (a non-expansiveness or truth-move window, or a family 'worse') -> 'flagged, no verdict' (unflagged kept).
Reference (report only): the reference dumps are projected with the constraints (applicability and velocity axis) decided on the model's
  dumps, so both sides of every extrapolation cost get the same projection; the reference's own truth statistics are reported and a
  disagreement is a warning; reference windows failing either guard are counted and listed (the extrapolation-cost lines are then
  marked), and a different evaluation code of the two dump runs (EVAL_FILES) is noted.
Report only: extrapolation cost with and without projection (model / in-range reference per held-out column, frames 1 / 16); per channel
  ratios; extra metrics per column at frames 1 / 16 (window means, model and reference): VRMSE, radial-spectrum log distance (integer
  wavenumber shells in units of the longer axis's fundamental, square cells, shells 1 ... Nyquist), Wasserstein-1 of the value
  distribution / truth std, and derived quantities: vorticity families - enstrophy and energy relative errors; velocity families -
  kinetic-energy relative error, energy-weighted direction error 1 - cos, vorticity-from-velocity relative L2, divergence ratio."""
import argparse, json, math, os, subprocess, sys
from statistics import fmean
import numpy as np
HERE = os.path.dirname(os.path.abspath(__file__)); REPO = os.path.dirname(HERE); PKG = os.path.join(REPO, "physdial")
sys.path.insert(0, os.path.join(REPO, "tables"))
import devsplit as DS   # noqa: E402

RESULTS = os.path.join(PKG, "results")
RUN = "stage2-noperiodic-s0-eval-dump"; REF = "stage2-noperiodic-ref-s0-eval-dump"
RUN_TRAIN = "stage2-noperiodic-s0"; REF_TRAIN = "stage2-noperiodic-ref-s0"
TOK = "wan21ft:decoder.pt"; STORE = "wan21"   # fine-tuned decoder; the zero-padded latent store
# every evaluated column
COLS_RUN = ["acoustic/test", "euler/extrap_gamma_high", "euler/extrap_gamma_low", "euler/test", "gs/extrap_bubbles", "gs/interp_maze", "gs/test",
            "kolmo/extrap_amp", "kolmo/extrap_drag_hi", "kolmo/extrap_drag_lo", "kolmo/extrap_joint", "kolmo/extrap_kband", "kolmo/extrap_kf",
            "kolmo/extrap_nu_high", "kolmo/extrap_nu_low", "kolmo/interp_kf", "kolmo/interp_nu", "kolmo/multif_test", "kolmo/nu_dense_ends",
            "kolmo/nu_dense_interp", "kolmo/test", "nsu/extrap_ends", "nsu/interp_test", "nsu/test", "rb/extrap_Ra", "rb/interp_Pr", "rb/test",
            "sf/extrap_Re", "sf/interp_Sc", "sf/test", "trl/extrap_tcool", "trl/interp_tcool", "trl/test"]
COLS_REF = [c for c in COLS_RUN if "drag" not in c]   # the reference has no drag columns
SPLITS_RUN, SPLITS_REF = ",".join(COLS_RUN), ",".join(COLS_REF)
IN_RANGE = {"kolmo": {"test", "interp_nu", "interp_kf", "nu_dense_interp", "multif_test"}, "nsu": {"test", "interp_test"}, "gs": {"test", "interp_maze"},
            "sf": {"test", "interp_Sc"}, "rb": {"test", "interp_Pr"}, "trl": {"test", "interp_tcool"},
            "euler": {"test"}, "acoustic": {"test"}}
PERIODIC = {"kolmo": (True, True), "nsu": (True, True), "gs": (True, True), "sf": (True, True), "rb": (True, False), "trl": (True, False),
            "euler": (False, False), "acoustic": (False, False)}   # = corpus Family.periodic
ZERO_MEAN = {"kolmo": ["vorticity"], "nsu": ["vorticity"]}
LERAY = {"sf": ("velocity_x", "velocity_y")}
NONNEG = {"gs": ["A", "B"], "euler": ["density", "pressure"], "trl": ["density", "pressure"]}
PROJ_FAMS = sorted(set(ZERO_MEAN) | set(LERAY) | set(NONNEG))
TOL_MEAN, TOL_DIV, TOL_NEG, TOL_NONEXP = 1e-3, 1e-2, 1e-6, 1e-3
TOL_MOVE_ABS, TOL_MOVE_REL = 1e-9, 1e-6   # | ||project(t) - t|| - dist(t, set) | <= TOL_MOVE_ABS + TOL_MOVE_REL dist (relative to ||t||)
FR = (1, 16); FAM_LO, FAM_HI = 0.95, 1.05; TOL = 1e-12
USEFUL, SMALL = "useful", "small (<= 5 %)"
WORSE = "worse (> 1.05: impossible for a non-expansive projection, code error -> flagged)"
FLAGGED = "flagged, no verdict"
TOL_CONS_REL, TOL_CONS_ABS = 1e-2, 2e-3   # dump vs json: |x - y| <= TOL_CONS_REL y + TOL_CONS_ABS (see consistent)
VEL_FAMS = ("sf", "rb", "trl", "euler", "acoustic"); VORT_FAMS = ("kolmo", "nsu")
# files of the evaluation path (relative to the repository root): two dump runs on different versions of them are noted
EVAL_FILES = ["physdial/" + f for f in ("train_gen.py", "eval_metrics.py", "gen_model.py", "corpus.py", "fieldnorm.py", "tok2d.py",
                                        "vae_periodic.py", "encode_corpus.py")] + ["resume.py", "data_gen/kolmo2d/splits_kolmo.json"]
GIT = ["git"]


# ---------------------------------------------------------------- result files -------------------------------------------------------
class Abort(Exception): pass


def le(x, t): return x <= t * (1 + TOL)
def fam(c): return c.split("/", 1)[0]
def in_range(c): return c.split("/", 1)[1] in IN_RANGE.get(fam(c), ())
def isnum(x): return isinstance(x, (int, float)) and not isinstance(x, bool)
def ev_of(top): return (top or {}).get("eval") or {}
def args_of(top): return (top or {}).get("args") or {}
def sha_of(top): return ((top or {}).get("meta") or {}).get("git_sha")
def skipped(v): return isinstance(v, dict) and "skipped" in v and not v.get("per_window")


def load(path):
    try: return json.load(open(path))
    except FileNotFoundError: return None


def gm(xs, what="values"):
    """geometric mean; a ratio 0 or NaN makes it undefined -> Abort."""
    xs = list(xs)
    bad = [x for x in xs if not (x > 0)]
    if bad: raise Abort(f"geometric mean of {what} undefined: {bad}")
    if not xs: raise Abort(f"geometric mean of {what}: no values")
    return DS.geomean(xs)


def sha_check(sa, sb):
    """(flag reason or None, differing EVAL_FILES) for two evaluation commits."""
    if not sa or not sb: return f"git sha unknown ({sa!r} / {sb!r})", []
    if sa == sb: return None, []
    def git(*a): return subprocess.run(GIT + ["-C", REPO] + list(a), capture_output=True, text=True, timeout=60)
    try: r = git("rev-parse", "--git-dir")
    except (OSError, subprocess.SubprocessError) as e: return f"git unavailable ({type(e).__name__}); evaluation files cannot be compared", []
    if r.returncode: return f"no git repository at {REPO}", []
    for x in (sa, sb):
        if git("cat-file", "-e", f"{x}^{{commit}}").returncode: return f"commit {x} unknown to the local repository", []
    d = git("diff", "--name-only", sa, sb, "--", *EVAL_FILES)
    if d.returncode: return f"git diff failed ({d.stderr.strip()[:120]})", []
    ch = d.stdout.split()
    return ("evaluation files differ: " + ", ".join(ch), ch) if ch else (None, [])


def col_problem(k, c, v):
    """reason a (non-skipped) column is unusable, or None: per-window rel_l2_pooled@1 / @16 present and not NaN (+inf kept: a diverged
    window), no duplicate windows, stored aggregate = window mean."""
    if not (isinstance(v, dict) and isinstance(v.get("per_window"), list) and v["per_window"]): return f"{k} {c}: no per-window records"
    ks = [DS.window_key(w) for w in v["per_window"]]
    if len(set(ks)) != len(ks): return f"{k} {c}: duplicate windows"
    for f in FR:
        key = f"rel_l2_pooled@{f}"; xs = [w.get(key) for w in v["per_window"]]
        if any(x is None or not isnum(x) or math.isnan(x) for x in xs): return f"{k} {c}: per-window {key} missing or undefined"
        m, s = fmean(float(x) for x in xs), v.get(key)
        if not isnum(s): return f"{k} {c}: stored {key} {s!r}"
        if math.isinf(m) != math.isinf(s) or (math.isfinite(m) and abs(m - s) > 1e-3 * abs(s)): return f"{k} {c}: stored {key} {s} != window mean {m}"
    return None


def eval_problem(k, files, expect, ckpt=None):
    """reason an evaluation result is not the expected one, or None. expect: arg -> required value (None = must be falsy / absent);
    ckpt: key of the training result the evaluation must load (eval_only = its name, selected_step = its, max_steps = selected_step)."""
    top = files.get(k)
    if top is None: return "file not found"
    a = top.get("args")
    if not isinstance(a, dict): return "no args (not a train_gen result json)"
    if "failed_splits" not in top: return "no 'failed_splits' (not a finished evaluation json)"
    if top["failed_splits"] != []: return f"failed_splits {top['failed_splits']}"
    if a.get("ode_steps") != 25: return f"ode_steps {a.get('ode_steps')} (25 expected)"
    for x, want in expect.items():
        got = a.get(x)
        if want is None:
            if got: return f"{x} {got!r} set (expected off)"
        elif got != want: return f"{x} {got!r}, expected {want!r}"
    if ckpt is not None:
        t = files.get(ckpt)
        if t is None: return f"training result {ckpt} absent: the checkpoint cannot be checked"
        if a.get("eval_only") != args_of(t).get("name"): return f"eval_only {a.get('eval_only')!r}, expected {args_of(t).get('name')!r}"
        if top.get("selected_step") is None or top.get("selected_step") != t.get("selected_step"): return f"selected_step {top.get('selected_step')} != {ckpt}'s {t.get('selected_step')}"
        if "max_steps" in a and a["max_steps"] != top["selected_step"]: return f"max_steps {a['max_steps']} != selected_step {top['selected_step']}"
    ev = ev_of(top)
    if not ev: return "no eval"
    for c, v in ev.items():
        if skipped(v): continue
        p = col_problem(k, c, v)
        if p: return p
    return None


# ---------------------------------------------------------------- field operations ---------------------------------------------------
def pooled(d, t):
    """relative L2 with the channel axis pooled ([..., C, H, W] -> [...]), as eval_metrics.rel_l2_pooled."""
    return np.sqrt((d ** 2).sum((-1, -2, -3))) / np.maximum(np.sqrt((t ** 2).sum((-1, -2, -3))), 1e-12)


def kgrid(H, W):
    return np.fft.fftfreq(H)[:, None] * 2 * np.pi, np.fft.fftfreq(W)[None, :] * 2 * np.pi


def kvec_odd(n):
    """index wavenumbers of a FIRST derivative: the Nyquist mode of an even grid gets k = 0. Its mode is real (self-conjugate) and i k of it
    has no real-valued counterpart; with k = -pi the Leray projection of the Nyquist row / column would not be Hermitian, so its real part
    would be neither exactly divergence-free nor idempotent. With k = 0 there every spectral operator below is Hermitian and exact."""
    k = np.fft.fftfreq(n) * 2 * np.pi
    if n % 2 == 0: k[n // 2] = 0.0
    return k


def kgrid_odd(H, W):
    return kvec_odd(H)[:, None], kvec_odd(W)[None, :]


def div_ratio(u, v, axis_x):
    """spectral ||k.u|| / (||k|| ||u||) of a velocity field [..., H, W]; axis_x: the array axis velocity_x points along (0 = H, 1 = W)."""
    k0, k1 = kgrid_odd(*u.shape[-2:]); U, V = np.fft.fft2(u), np.fft.fft2(v)
    A, B = (U, V) if axis_x == 0 else (V, U)   # the component along H, the component along W
    d = np.abs(k0 * A + k1 * B) ** 2; g = (k0 ** 2 + k1 ** 2) * (np.abs(U) ** 2 + np.abs(V) ** 2)
    return np.sqrt(d.sum((-1, -2)) / np.maximum(g.sum((-1, -2)), 1e-300))


def leray(u, v, axis_x):
    """orthogonal projection onto the divergence-free fields (spectral, Nyquist derivative 0: exact and idempotent on any grid; the modes
    with k = 0, i.e. the mean and the Nyquist-only modes, are left unchanged)."""
    k0, k1 = kgrid_odd(*u.shape[-2:]); U, V = np.fft.fft2(u), np.fft.fft2(v)
    A, B = (U, V) if axis_x == 0 else (V, U)
    k2 = k0 ** 2 + k1 ** 2; s = (k0 * A + k1 * B) / np.where(k2 > 0, k2, 1.0)
    A, B = A - k0 * s, B - k1 * s
    U2, V2 = (A, B) if axis_x == 0 else (B, A)
    return np.fft.ifft2(U2).real, np.fft.ifft2(V2).real


def deriv(x, axis, periodic):
    """d/d(index) along a spatial axis (-2 or -1): spectral on a periodic axis (Nyquist derivative 0), second-order differences otherwise."""
    if periodic:
        n = x.shape[axis]; k = kvec_odd(n); sh = [1] * x.ndim; sh[axis] = n
        return np.fft.ifft(1j * k.reshape(sh) * np.fft.fft(x, axis=axis), axis=axis).real
    return np.gradient(x, axis=axis)


def vort_of(u, v, axis_x, per):
    """vorticity d(v_y)/dx - d(v_x)/dy with x the axis velocity_x points along."""
    ax, ay = (-2, -1) if axis_x == 0 else (-1, -2); px, py = (per[0], per[1]) if axis_x == 0 else (per[1], per[0])
    return deriv(v, ax, px) - deriv(u, ay, py)


def shells(H, W):
    """integer wavenumber shell of every FFT mode of an [H, W] grid with square cells: |k| in units of the fundamental of the longer axis
    (M = max(H, W)), rounded; the Nyquist wavenumber is M / 2 on both axes. Integer shells keep the low modes (|n| < 8, the Kolmogorov
    forcing scale) that bins of a fixed fraction of the Nyquist wavenumber would merge with the mean."""
    M = max(H, W)
    kap = np.sqrt((np.fft.fftfreq(H) * M)[:, None] ** 2 + (np.fft.fftfreq(W) * M)[None, :] ** 2)
    return np.rint(kap).astype(int), M // 2


def radial_logdist(p, t, per):
    """mean over integer wavenumber shells 1 ... Nyquist (`shells`; full shells only, the corners beyond the Nyquist circle are left out)
    of |log10 E_p - log10 E_t| (radial power spectrum, Hann window on non-periodic axes, shells where the truth holds > 1e-10 of its
    energy); p, t [H, W]. Report only."""
    H, W = t.shape; w = np.ones((H, W))
    if not per[0]: w *= np.hanning(H)[:, None]
    if not per[1]: w *= np.hanning(W)[None, :]
    b, nyq = shells(H, W); n = int(b.max()) + 1
    Ep = np.bincount(b.ravel(), (np.abs(np.fft.fft2(p * w)) ** 2).ravel(), n)[1:nyq + 1]
    Et = np.bincount(b.ravel(), (np.abs(np.fft.fft2(t * w)) ** 2).ravel(), n)[1:nyq + 1]
    m = Et > 1e-10 * max(Et.sum(), 1e-300)
    if not m.any(): return float("nan")
    return float(np.mean(np.abs(np.log10(np.maximum(Ep[m], 1e-300)) - np.log10(Et[m]))))


def w1_norm(p, t, n=65536):
    """Wasserstein-1 between the value distributions of p and t (quantiles), / std(t)."""
    a, b = p.ravel(), t.ravel(); st = max(len(a) // n, 1); a, b = np.sort(a[::st]), np.sort(b[::st])
    return float(np.mean(np.abs(a - b)) / max(float(t.std()), 1e-12))


def direction_err(up, vp, ut, vt):
    """energy-weighted 1 - cos(angle) between predicted and true velocity."""
    e = ut ** 2 + vt ** 2; dot = up * ut + vp * vt; nrm = np.sqrt((up ** 2 + vp ** 2) * e)
    cos = np.where(nrm > 1e-30, dot / np.maximum(nrm, 1e-30), 1.0)
    return float((e * (1 - cos)).sum() / max(e.sum(), 1e-300))


def vort_energy(w):
    """energy of a doubly periodic vorticity field (stream function from the inverse Laplacian), index units."""
    k0, k1 = kgrid(*w.shape); k2 = k0 ** 2 + k1 ** 2; k2[0, 0] = np.inf
    return float((np.abs(np.fft.fft2(w)) ** 2 / k2).sum())


# ---------------------------------------------------------------- constraints ----------------------------------------------------------
class TruthAcc:
    """streaming truth statistics of one family (window by window; the whole family never sits in memory)."""
    def __init__(s, f, chans): s.f, s.chans = f, chans; s.zm, s.dv = [], {0: [], 1: []}; s.mn = {c: math.inf for c in NONNEG.get(f, [])}; s.mx = {c: 0.0 for c in NONNEG.get(f, [])}
    def add(s, T):   # T [frames, C, H, W] float64 (history and truth frames of one window)
        f, ch = s.f, s.chans
        for c in ZERO_MEAN.get(f, []):
            x = T[:, ch.index(c)]; s.zm.append(np.abs(x.mean((-1, -2))) / np.maximum(np.sqrt((x ** 2).mean((-1, -2))), 1e-300))
        if f in LERAY:
            iu, iv = (ch.index(c) for c in LERAY[f])
            for ax in (0, 1): s.dv[ax].append(div_ratio(T[:, iu], T[:, iv], ax))
        for c in NONNEG.get(f, []):
            x = T[:, ch.index(c)]; s.mn[c] = min(s.mn[c], float(x.min())); s.mx[c] = max(s.mx[c], float(np.abs(x).max()))
    def merge(s, o):
        s.zm += o.zm; s.dv[0] += o.dv[0]; s.dv[1] += o.dv[1]
        for c in s.mn: s.mn[c] = min(s.mn[c], o.mn[c]); s.mx[c] = max(s.mx[c], o.mx[c])
    def result(s):
        """{constraint: (applies, statistic, detail[, axis of velocity_x])}."""
        out, f = {}, s.f
        if f in ZERO_MEAN and s.zm:
            v = float(np.percentile(np.concatenate(s.zm), 99)); out["zero_mean"] = (v <= TOL_MEAN, v, f"p99 |mean|/rms {v:.2e} (tolerance {TOL_MEAN})")
        if f in LERAY and s.dv[0]:
            d = {ax: float(np.percentile(np.concatenate(s.dv[ax]), 99)) for ax in (0, 1)}; ax = min(d, key=d.get)
            out["leray"] = (d[ax] <= TOL_DIV, d[ax], f"p99 divergence ratio {d[ax]:.2e} with velocity_x along axis {'HW'[ax]} (other axis {d[1 - ax]:.2e}; tolerance {TOL_DIV})", ax)
        if f in NONNEG:
            bad = [f"{c} min {s.mn[c]:.3g} (max|x| {s.mx[c]:.3g})" for c in NONNEG[f] if s.mn[c] < -TOL_NEG * s.mx[c]]
            out["nonneg"] = (not bad, len(bad), "; ".join(bad) or f"truth >= -{TOL_NEG} max|truth| on {', '.join(NONNEG[f])}")
        return out


def project(f, chans, P, cons):
    """P [frames, C, H, W] float64 -> projected copy, applying the constraints of `cons` that apply."""
    Q = P.copy()
    if cons.get("zero_mean", (False,))[0]:
        for ch in ZERO_MEAN[f]: i = chans.index(ch); Q[:, i] -= Q[:, i].mean((-1, -2), keepdims=True)
    if cons.get("leray", (False,))[0]:
        iu, iv = (chans.index(c) for c in LERAY[f]); Q[:, iu], Q[:, iv] = leray(Q[:, iu], Q[:, iv], cons["leray"][3])
    if cons.get("nonneg", (False,))[0]:
        for ch in NONNEG[f]: i = chans.index(ch); Q[:, i] = np.maximum(Q[:, i], 0.0)
    return Q


def grad_part_sq(u, v, axis_x):
    """squared norm of the gradient (curl-free) part of a velocity field [..., H, W] in the analytic Helmholtz split, per leading index:
    sum over modes with |k| > 0 of |k.U|^2 / |k|^2, / (H W) (Parseval for numpy's unnormalised fft2). Index wavenumbers of `kvec_odd` (the
    set the Leray projection maps onto); it does not call leray()."""
    k0, k1 = kgrid_odd(*u.shape[-2:]); U, V = np.fft.fft2(u), np.fft.fft2(v)
    A, B = (U, V) if axis_x == 0 else (V, U)
    k2 = k0 ** 2 + k1 ** 2; num = np.abs(k0 * A + k1 * B) ** 2
    return np.where(k2 > 0, num / np.where(k2 > 0, k2, 1.0), 0.0).sum((-1, -2)) / (u.shape[-1] * u.shape[-2])


def truth_dist(f, chans, T, cons):
    """per frame of T [frames, C, H, W]: distance of T[j] to the set of the constraints of `cons` that apply, / ||T[j]|| (all channels),
    from closed forms independent of project() (the allowance of the non-expansiveness check must not be computed by the code it checks).
    zero_mean: |mean| sqrt(H W) per channel; leray: grad_part_sq; nonneg: ||min(x, 0)||. The constrained channels of a family are disjoint
    (asserted), so the squared distances add."""
    d2 = np.zeros(T.shape[0]); used = []
    if cons.get("zero_mean", (False,))[0]:
        for ch in ZERO_MEAN[f]: x = T[:, chans.index(ch)]; d2 += x.shape[-1] * x.shape[-2] * x.mean((-1, -2)) ** 2; used.append(ch)
    if cons.get("leray", (False,))[0]:
        iu, iv = (chans.index(c) for c in LERAY[f]); d2 += grad_part_sq(T[:, iu], T[:, iv], cons["leray"][3]); used += list(LERAY[f])
    if cons.get("nonneg", (False,))[0]:
        for ch in NONNEG[f]: d2 += (np.minimum(T[:, chans.index(ch)], 0.0) ** 2).sum((-1, -2)); used.append(ch)
    assert len(used) == len(set(used)), f"{f}: constraints share a channel {used} (the distance would not be a sum)"
    return np.sqrt(d2) / np.maximum(np.sqrt((T ** 2).sum((-1, -2, -3))), 1e-12)


# ---------------------------------------------------------------- dumps ----------------------------------------------------------------
def open_column(top, root, name, col):
    """(npz, meta, {key: per-window record}, [keys in dump order]) of a column, or a reason string: every window of the json dumped, no
    window that the json lacks."""
    p = os.path.join(root, "pred", name, col.replace("/", "__") + ".npz")
    if not os.path.exists(p): return f"dump {p} not found"
    z = np.load(p, allow_pickle=False); meta = json.loads(str(z["meta"])); pw = {DS.window_key(w): w for w in ev_of(top)[col]["per_window"]}
    keys = [(w["path"], str(w["traj"]), int(w["first_frame"])) for w in meta["windows"]]
    if any(k not in pw for k in keys): return f"dumped window {next(k for k in keys if k not in pw)} not in the json's per-window records"
    if set(keys) != set(pw): return f"{len(set(pw) - set(keys))} windows of the column not dumped (--dump_pred too small?)"
    return z, meta, pw, keys


def windows(z, keys):
    """yield (key, H, P, T) float64 one window at a time."""
    for i, k in enumerate(keys): yield k, z[f"H{i}"].astype(np.float64), z[f"P{i}"].astype(np.float64), z[f"T{i}"].astype(np.float64)


def cons_tol(y): return TOL_CONS_REL * abs(y) + TOL_CONS_ABS


def consistent(P, T, rec):
    """reason the dumped window does not reproduce the json's pooled errors at frames 1 / 16, or None. train_gen computes the json value
    from the float32 prediction and truth and casts the dump to float16 afterwards (train_gen.py, --dump_pred), so the recomputed value x
    differs from the json value y by the float16 rounding of both fields: with |dP| <= u |P|, |dT| <= u |T| per element (u = 2^-11, normal
    range), |x - y| <= 2 u (1 + y) / (1 - u) ~ 9.8e-4 (1 + y); subnormal values add at most 2^-25 per element. Tolerance: |x - y| <= 1e-2 y +
    2e-3, at least twice the bound for every y >= 0 (2e-3 vs 9.8e-4 at y = 0, slope 1e-2 vs 9.8e-4). A relative-only tolerance would reject
    every window with a tiny error (e.g. GS steady windows: json 3e-9 ... 4e-8, recomputed 0 from the float16 dump). A dump of another
    evaluation (other noise or checkpoint) differs by far more on the windows with errors ~ 0.01 - 1."""
    for f in FR:
        x, y = float(pooled(P[f - 1] - T[f - 1], T[f - 1])), float(rec[f"rel_l2_pooled@{f}"])
        if not (math.isfinite(x) and math.isfinite(y)): return f"non-finite error at frame {f} (dump {x}, json {y})"
        if abs(x - y) > cons_tol(y): return f"frame {f}: pooled error from the dump {x:.6g} != json {y:.6g} (|diff| {abs(x - y):.3g} > {TOL_CONS_REL} y + {TOL_CONS_ABS}): the dump is not this evaluation"
    return None


def metrics(f, chans, P, T, axis_x=0):
    """report-only metrics of one window at frames 1 / 16: {name@f: value}."""
    per = PERIODIC[f]; out = {}
    for fr in FR:
        p, t = P[fr - 1], T[fr - 1]
        out[f"vrmse@{fr}"] = float(np.mean([np.sqrt(((p[i] - t[i]) ** 2).mean()) / max(float(t[i].std()), 1e-12) for i in range(len(chans))]))
        out[f"spec_logdist@{fr}"] = float(np.nanmean([radial_logdist(p[i], t[i], per) for i in range(len(chans))]))
        out[f"w1@{fr}"] = float(np.mean([w1_norm(p[i], t[i]) for i in range(len(chans))]))
        if f in VORT_FAMS:
            w_p, w_t = p[chans.index("vorticity")], t[chans.index("vorticity")]
            out[f"enstrophy_err@{fr}"] = float(abs((w_p ** 2).mean() - (w_t ** 2).mean()) / max((w_t ** 2).mean(), 1e-300))
            out[f"energy_err@{fr}"] = abs(vort_energy(w_p) - vort_energy(w_t)) / max(vort_energy(w_t), 1e-300)
        if f in VEL_FAMS:
            iu, iv = chans.index("velocity_x"), chans.index("velocity_y"); up, vp, ut, vt = p[iu], p[iv], t[iu], t[iv]
            ke_p, ke_t = float((up ** 2 + vp ** 2).sum()), float((ut ** 2 + vt ** 2).sum())
            out[f"ke_err@{fr}"] = abs(ke_p - ke_t) / max(ke_t, 1e-300)
            out[f"dir_err@{fr}"] = direction_err(up, vp, ut, vt)
            wp, wt = vort_of(up, vp, axis_x, per), vort_of(ut, vt, axis_x, per)
            out[f"vort_rel_l2@{fr}"] = float(np.sqrt(((wp - wt) ** 2).sum() / max((wt ** 2).sum(), 1e-300)))
            if all(per): out[f"div_ratio_pred@{fr}"] = float(div_ratio(up, vp, axis_x)); out[f"div_ratio_true@{fr}"] = float(div_ratio(ut, vt, axis_x))
    return out


def read_model(top, root, name, cols, warn, cons_fixed=None):
    """two passes over the dumps, one window in memory at a time. Pass 1: per column, the dump's identity with the json (a failing column
    is 'missing' and contributes nothing) and the truth statistics; merged per family -> the constraints that apply (cons_fixed given, e.g.
    the model's for the reference dumps: those are used instead; a family absent from it is not projected). Pass 2: raw and projected
    pooled errors at FR, per-channel errors, the truth's distance to the set (independent closed forms), the non-expansiveness and
    truth-move counts and the report-only metrics per window. Returns (columns, constraints used, the dumps' own truth statistics)."""
    colres, opened, acc = {}, {}, {}
    for c in cols:
        if c not in ev_of(top): colres[c] = "missing(column not in the json)"; continue
        o = open_column(top, root, name, c)
        if isinstance(o, str): colres[c] = f"missing({o})"; warn.append(f"{name} {c}: {o}"); continue
        z, meta, pw, keys = o; chans = list(meta["channels"]); a = TruthAcc(fam(c), chans); bad = None
        for k, Hh, P, T in windows(z, keys):
            bad = consistent(P, T, pw[k])
            if bad: bad = f"window {k}: {bad}"; break
            a.add(np.concatenate([Hh, T], 0))
        if bad: colres[c] = f"missing({bad})"; warn.append(f"{name} {c}: {bad}"); continue
        opened[c] = (z, keys, chans)
        if fam(c) in acc: acc[fam(c)].merge(a)
        else: acc[fam(c)] = a
    own = {f: a.result() for f, a in acc.items()}
    cons = own if cons_fixed is None else {f: cons_fixed.get(f, {}) for f in acc}
    for c, (z, keys, chans) in opened.items():
        f = fam(c); ax = cons[f]["leray"][3] if "leray" in cons[f] else 0; rows = []
        for k, Hh, P, T in windows(z, keys):
            Q = project(f, chans, P, cons[f]) if f in PROJ_FAMS else P
            row = {"key": k}
            dist = truth_dist(f, chans, T, cons[f]) if f in PROJ_FAMS else np.zeros(T.shape[0])   # independent of project()
            for fr in FR:
                row[f"raw@{fr}"] = float(pooled(P[fr - 1] - T[fr - 1], T[fr - 1])); row[f"proj@{fr}"] = float(pooled(Q[fr - 1] - T[fr - 1], T[fr - 1])); row[f"dist@{fr}"] = float(dist[fr - 1])
                row[f"chan_raw@{fr}"] = [float(np.linalg.norm(P[fr - 1, i] - T[fr - 1, i]) / max(np.linalg.norm(T[fr - 1, i]), 1e-12)) for i in range(len(chans))]
                row[f"chan_proj@{fr}"] = [float(np.linalg.norm(Q[fr - 1, i] - T[fr - 1, i]) / max(np.linalg.norm(T[fr - 1, i]), 1e-12)) for i in range(len(chans))]
            row["nonexp_bad"] = sum(1 for j in range(P.shape[0]) if pooled(Q[j] - T[j], T[j]) > pooled(P[j] - T[j], T[j]) * (1 + TOL_NONEXP) + dist[j] + 1e-12)
            QT = project(f, chans, T, cons[f]) if f in PROJ_FAMS else T   # an orthogonal projection moves the truth by exactly its distance to the set
            row["truth_move_bad"] = sum(1 for j in range(T.shape[0]) if abs(float(pooled(QT[j] - T[j], T[j])) - dist[j]) > TOL_MOVE_ABS + TOL_MOVE_REL * dist[j])
            row["metrics"] = metrics(f, chans, P, T, ax); rows.append(row)
        colres[c] = {"windows": rows, "channels": chans}
    return colres, cons, own


def half_mean(rows, maps, col, key):
    out = {"dev": [], "test": []}
    for r in rows: out[maps[col][DS.unit_of(r["key"][0], r["key"][1])]].append(r[key])
    return {h: (float(np.mean(v)) if v else float("nan")) for h, v in out.items()}


def half_ratio(rows, maps, col, fr):
    """{half: mean projected / mean raw pooled error at frame fr}. A half whose raw mean is exactly 0: 1.0 if the projection moved it by no
    more than the mean truth distance (+ 1e-12), else +inf; no window in a half: nan."""
    p, r, d = (half_mean(rows, maps, col, f"{k}@{fr}") for k in ("proj", "raw", "dist"))
    out = {}
    for h in ("dev", "test"):
        if math.isnan(r[h]): out[h] = float("nan")
        elif r[h] > 0: out[h] = p[h] / r[h]
        else: out[h] = 1.0 if p[h] <= d[h] * (1 + 1e-9) + 1e-12 else math.inf
    return out


def fam_verdict(at1, at16):
    """any half > 1.05 at frame 1 or 16 -> WORSE (a bug; flags the overall verdict), checked first."""
    if any(x > FAM_HI * (1 + TOL) for x in tuple(at1) + tuple(at16)): return WORSE
    return USEFUL if (all(le(x, FAM_LO) for x in at1) or all(le(x, FAM_LO) for x in at16)) else SMALL


def run_train(root): return load(os.path.join(root, RUN_TRAIN + ".json"))


def readout(run, ref, root):
    warn, L = [], []
    p = eval_problem("run", {"run": run, "t": run_train(root)}, {"tok": TOK, "store": STORE, "dump_pred": 32, "eval_splits": SPLITS_RUN, "kolmo_extra": "drag",
                                                                 "sample_shift": 0.0, "eval_stride_mult": 1}, "t")
    if p: raise Abort(f"{args_of(run).get('name', RUN)}: {p}")
    colres, cons, _ = read_model(run, root, args_of(run)["name"], COLS_RUN, warn)
    maps = DS.split_maps({c: v for c, v in ev_of(run).items() if c in COLS_RUN})
    res = {"warnings": warn, "constraints": {f: {k: list(v[:3]) for k, v in d.items()} for f, d in cons.items()}, "columns": {}, "family": {}}
    nonexp = tmove = 0
    for c, v in colres.items():
        if isinstance(v, str): res["columns"][c] = v; continue
        row = {fr: half_ratio(v["windows"], maps, c, fr) for fr in FR}
        row["channels"] = {ch: {fr: float(np.mean([w[f"chan_proj@{fr}"][i] for w in v["windows"]]) / max(float(np.mean([w[f"chan_raw@{fr}"][i] for w in v["windows"]])), 1e-12)) for fr in FR} for i, ch in enumerate(v["channels"])}
        row["metrics"] = {m: float(np.mean([w["metrics"][m] for w in v["windows"]])) for m in v["windows"][0]["metrics"]}
        row["nonexp_bad_windows"] = sum(1 for w in v["windows"] if w["nonexp_bad"]); nonexp += row["nonexp_bad_windows"]
        row["truth_move_bad_windows"] = sum(1 for w in v["windows"] if w["truth_move_bad"]); tmove += row["truth_move_bad_windows"]
        res["columns"][c] = row
    for f in PROJ_FAMS:
        if f not in cons: res["family"][f] = "missing(no usable column)"; continue
        na = [f"{k}: {v[2]}" for k, v in cons[f].items() if not v[0]]
        if na and not any(v[0] for v in cons[f].values()): res["family"][f] = f"not applicable({'; '.join(na)})"; continue
        cs = [c for c in COLS_RUN if fam(c) == f and in_range(c) and isinstance(res["columns"].get(c), dict)]
        if not cs: res["family"][f] = "missing(no usable in-range column)"; continue
        v = {fr: tuple(gm((res["columns"][c][fr][h] for c in cs), f"{f} f{fr}") for h in ("dev", "test")) for fr in FR}
        res["family"][f] = {"values": v, "verdict": fam_verdict(v[1], v[16]), "columns": cs, "not_applied": na}
    useful = [f for f, x in res["family"].items() if isinstance(x, dict) and x["verdict"] == USEFUL]
    worse = [f for f, x in res["family"].items() if isinstance(x, dict) and x["verdict"] == WORSE]
    main = f"projection useful: {', '.join(useful)}" if useful else "projection changes the error by <= 5 %"
    res["main_unflagged"] = main; res["nonexp_bad_windows"] = nonexp; res["truth_move_bad_windows"] = tmove; res["worse_families"] = worse
    res["main"] = FLAGGED if (nonexp or tmove or worse) else main
    # extrapolation cost against the in-range reference (report only)
    if ref is None: res["extrap_cost"] = "missing: reference dump absent"
    else:
        p = eval_problem("ref", {"ref": ref, "t": load(os.path.join(root, REF_TRAIN + ".json"))}, {"tok": TOK, "store": STORE, "dump_pred": 32, "eval_splits": SPLITS_REF, "kolmo_extra": None}, "t")
        if p: res["extrap_cost"] = f"missing: {args_of(ref).get('name', REF)}: {p}"
        else:
            cr, _, own_ref = read_model(ref, root, args_of(ref)["name"], COLS_REF, warn, cons_fixed=cons); ec = {}   # the model's constraints on both sides
            res["constraints_ref_own"] = {f: {k: list(v[:3]) for k, v in d.items()} for f, d in own_ref.items()}
            for f, d in own_ref.items():
                for k, v in d.items():
                    m = cons.get(f, {}).get(k)
                    if m is not None and (bool(v[0]) != bool(m[0]) or (k == "leray" and v[0] and m[0] and v[3] != m[3])):
                        warn.append(f"reference dumps, {f} {k}: their own truth check gives {'applies' if v[0] else 'NOT applicable'} ({v[2]}); projected with the constraints chosen on the model's dumps")
            bad_ref = {c: (sum(1 for w in b["windows"] if w["nonexp_bad"]), sum(1 for w in b["windows"] if w["truth_move_bad"])) for c, b in cr.items() if isinstance(b, dict)}
            res["ref_guard_bad"] = {c: v for c, v in bad_ref.items() if any(v)}
            res["extrap_cost_code_flag"] = sha_check(sha_of(run), sha_of(ref))[0]   # report only: the two dump runs on different EVAL_FILES
            for c in COLS_REF:
                if in_range(c): continue
                a, b = res["columns"].get(c), cr.get(c)
                if not isinstance(a, dict) or not isinstance(b, dict): ec[c] = "missing"; continue
                wa, wb = colres[c]["windows"], b["windows"]
                if {w["key"] for w in wa} != {w["key"] for w in wb}: ec[c] = "missing(reference windows differ)"; continue
                ec[c] = {fr: {"raw": np.mean([w[f"raw@{fr}"] for w in wa]) / np.mean([w[f"raw@{fr}"] for w in wb]),
                              "proj": np.mean([w[f"proj@{fr}"] for w in wa]) / np.mean([w[f"proj@{fr}"] for w in wb])} for fr in FR}
                ec[c]["metrics_ref"] = {m: float(np.mean([w["metrics"][m] for w in wb])) for m in wb[0]["metrics"]}
                g = (a["nonexp_bad_windows"] + a["truth_move_bad_windows"], sum(bad_ref.get(c, (0, 0))))
                if any(g): ec[c]["flag"] = f"guard windows: model {g[0]}, reference {g[1]}"
            res["extrap_cost"] = ec
    L.append("## constraints (truth check; applied only where the truth satisfies them)")
    for f, d in res["constraints"].items():
        for k, v in d.items(): L.append(f"{f:<9} {k:<10} {'applies' if v[0] else 'NOT applicable'}: {v[2]}")
    L.append("## projected / raw pooled error (dev / test), frames 1 and 16")
    for c, row in res["columns"].items():
        if isinstance(row, str): L.append(f"{c:<26} {row}"); continue
        if fam(c) not in PROJ_FAMS: continue
        L.append(f"{c:<26}{'in' if in_range(c) else 'extrap':>7}" + "".join(f"  f{fr} {row[fr]['dev']:.4f} / {row[fr]['test']:.4f}" for fr in FR)
                 + "  channels " + ", ".join(f"{ch} {v[1]:.3f}/{v[16]:.3f}" for ch, v in row["channels"].items()) + (f"  NONEXP {row['nonexp_bad_windows']}" if row["nonexp_bad_windows"] else "")
                 + (f"  TRUTH-MOVE {row['truth_move_bad_windows']}" if row["truth_move_bad_windows"] else ""))
    for f, x in res["family"].items():
        L.append(f"family {f:<9} " + (x if isinstance(x, str) else f"f1 {x['values'][1][0]:.4f}/{x['values'][1][1]:.4f} f16 {x['values'][16][0]:.4f}/{x['values'][16][1]:.4f} -> {x['verdict']}" + (f" (not applied: {x['not_applied']})" if x["not_applied"] else "")))
    L.append("## extra metrics (report only; window means, model): " + "; ".join(f"{c}: " + ", ".join(f"{m} {v:.3g}" for m, v in row["metrics"].items()) for c, row in res["columns"].items() if isinstance(row, dict)))
    if isinstance(res["extrap_cost"], str): L.append(f"extrapolation cost: {res['extrap_cost']}")
    else:
        if res.get("extrap_cost_code_flag"): L.append(f"extrapolation cost: the two dump runs differ in evaluation code ({res['extrap_cost_code_flag']}); report only, read with care")
        L += [f"extrap cost {c:<24} " + (v if isinstance(v, str) else "  ".join(f"f{fr} raw {v[fr]['raw']:.3f} proj {v[fr]['proj']:.3f}" for fr in FR) + (f"  FLAG {v['flag']}" if v.get("flag") else "")) for c, v in res["extrap_cost"].items()]
    L += [f"WARNING {x}" for x in warn]
    why = [s for s, n in ((f"{nonexp} windows where the projection increased the error beyond the truth's distance to the set", nonexp),
                          (f"{tmove} windows where the projection did not move the truth by exactly its distance to the set", tmove),
                          (f"family value > 1.05: {', '.join(worse)}", worse)) if n]
    L.append(f"## MAIN physical-constraint projection (output post-processing, no feedback): {res['main']}" + (f" (FLAG: {'; '.join(why)}; unflagged: {main})" if why else ""))
    return "\n".join(L), res


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--run", default=RUN, help="the model's evaluation with dumps"); ap.add_argument("--ref", default=REF, help="the in-range reference's evaluation with dumps")
    ap.add_argument("--out", default=os.path.join(HERE, "results", "projection.json"))
    a = ap.parse_args(argv)
    run = load(os.path.join(RESULTS, a.run + ".json"))
    if run is None: print(f"ABORT: {a.run}.json not found"); return 1
    try: text, res = readout(run, load(os.path.join(RESULTS, a.ref + ".json")), RESULTS)
    except Abort as e: print(f"ABORT: {e}"); return 1
    os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
    print(text); json.dump(res, open(a.out, "w"), indent=1, ensure_ascii=False, default=str); print(f"saved {a.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
