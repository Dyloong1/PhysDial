"""Unified 2D corpus of the eight PDE families. Every family is read on the grid its solver or publisher used; grids differ
between families and are never resampled here.

One item = one trajectory: frames [T, C, H, W] float32 (C physical channels, family-specific), optional condition fields
[H, W] (Kolmogorov forcing pattern, acoustic medium), the physical parameters, and the frame spacing. A sample for the
generative model is a window of consecutive frames.

Families (key: source, grid, channels, conditions):
  kolmo    : generated Kolmogorov flow (data_gen/kolmo2d), 256^2-512^2 by nu, vorticity; nu, k_f, amplitude + forcing field;
             the main sweep (splits_kolmo.json) plus a dense nu axis (KOLMO_DENSE: k_f 2, amp 1, 100 nu in [1e-4, 1e-2])
             and random multi-mode forcing trajectories (KOLMO_MULTIF)
  nsu      : generated freely decaying turbulence (same solver, no forcing / drag; 101 nu in [1e-4, 1e-2]), vorticity; nu
  trl      : The Well turbulent_radiative_layer_2D, 128x384, density, pressure, vx, vy; t_cool
  gs       : The Well gray_scott_reaction_diffusion, 128^2, (A, B); F, k
  rb       : The Well rayleigh_benard, 512x128, buoyancy, pressure, vx, vy; Rayleigh, Prandtl
  sf       : The Well shear_flow, 256x512, tracer, pressure, vx, vy; Reynolds, Schmidt
  euler    : The Well euler_multi_quadrants_openBC, 512^2, density, pressure, vx, vy; gamma
  acoustic : The Well acoustic_scattering_inclusions, 256^2, pressure, vx, vy; density and sound-speed fields as conditions

Splits (same names for every family):
  train  : training trajectories (training conditions)
  val    : a few held-out trajectories of training conditions (validation loss)
  test   : held-out trajectories of training conditions (in-range result)
  interp_* / extrap_* : conditions never seen in training, evaluation trajectories only
  ref_*  : the OTHER trajectories of those held-out conditions; added to training only for the in-range reference run,
           so the reference is evaluated on exactly the same trajectories as the main model.
  Optional Kolmogorov groups (set_kolmo_extra; train_gen / encode_corpus --kolmo_extra):
    anchor                         amplitude anchors (forcing amplitude 0.7 / 1.4 at the training k_f and nu), training only
    extrap_drag_lo / _hi (+ ref_)  linear drag 0.05 / 0.2 instead of 0.1 (evaluation columns)
    beta_anchor, extrap_beta (+ ref_)  beta-plane term: anchors at beta = +-2.4 (training only), held-out column beta = 4

Paths: $PHYSDIAL_DATA_ROOT (generated data and latent stores) and $PHYSDIAL_EXT_DATA (The Well; default ./data/ext)."""
import glob, json, math, os, re
from dataclasses import dataclass, field
import numpy as np, h5py

HERE = os.path.dirname(os.path.abspath(__file__)); PROJ = os.path.dirname(HERE)
DATA_ROOT = os.environ.get("PHYSDIAL_DATA_ROOT", ""); EXT = os.environ.get("PHYSDIAL_EXT_DATA", os.path.join(PROJ, "data", "ext"))

# parameter slots of the numeric condition vector (value + presence mask per slot); `beta` is appended while the beta groups are on
SLOTS = ["log10_nu", "kf", "amp", "log10_Re", "log10_tcool", "gs_F", "gs_k", "log10_Ra", "log10_Pr", "log10_Sc", "log10_dt", "gamma"]
# Every physical channel of every family is assigned to a fixed QUANTITY SLOT shared across families (the field index map of MPP /
# Walrus): vorticity always enters and leaves the model through the same weights whichever family it comes from, velocity and pressure
# share their slots across the flow families, etc. The equation-identity text names the fields as well.
QUANTITIES = ["vorticity", "velocity_x", "velocity_y", "pressure", "density", "buoyancy", "tracer", "species_1", "species_2"]
QSLOT = {"vorticity": 0, "velocity_x": 1, "velocity_y": 2, "pressure": 3, "density": 4, "buoyancy": 5, "tracer": 6,
         "A": 7, "B": 8}   # A / B: Gray-Scott species
MAX_C = len(QUANTITIES)   # channel slots of a frame in the generative model

# Model step per family: one model step = STRIDE[family] stored frames;
# history, outputs and rollout steps are all STRIDE frames apart and the frame-spacing slot log10_dt is shifted by log10(STRIDE)
# (effective_params). The stride of each family is chosen so that one model step changes the field by a comparable amount
# across families.
STRIDE = {"kolmo": 2, "nsu": 1, "trl": 1, "gs": 16, "rb": 1, "sf": 8, "euler": 2, "acoustic": 1}
KOLMO_ALPHA = 0.1   # linear drag of every Kolmogorov file except the drag columns


def effective_params(key, params, stride=None):
    """parameter slots as the model sees them: log10_dt of the stored frames + log10 of the stride (the family's, or `stride` when a
    window was taken at another stride: stride augmentation). Kolmogorov items without a beta-plane term carry beta = 0 (read only by
    runs with the beta slot)."""
    if key == "kolmo" and "beta" not in params: params = dict(params, beta=0.0)
    k = STRIDE.get(key, 1) if stride is None else int(stride)
    if k == 1 or "log10_dt" not in params: return params
    p = dict(params); p["log10_dt"] = p["log10_dt"] + math.log10(k); return p


@dataclass
class Item:
    family: str
    path: str
    traj: object                       # group key (str) or trajectory index (int)
    params: dict                       # slot name -> value (already transformed)
    raw: dict = field(default_factory=dict)   # parameter values as stored in the data files (for reports)
    n_frames: int = 0
    shape: tuple = ()                  # (H, W)


class Family:
    key = ""; channels = []; periodic = (True, True); text = ""
    BC_SIDES = {}   # axis -> (low side, high side) for non-periodic axes
    @property
    def bc(self): return tuple(("periodic", "periodic") if self.periodic[i] else self.BC_SIDES[i] for i in (0, 1))
    t_start = 0                        # first usable frame
    @property
    def slots(self):   # quantity slot of every channel (QUANTITIES)
        return [QSLOT[c] for c in self.channels]
    @property
    def stride(self): return STRIDE.get(self.key, 1)            # stored frames per model step
    def items(self, split): raise NotImplementedError
    def load(self, it, t0=0, t1=None): raise NotImplementedError          # -> frames [t1-t0, C, H, W] float32
    def cond_fields(self, it): return []                                  # -> list of [H, W] float32 condition fields
    def splits(self): raise NotImplementedError
    def ci_group(self, it): return f"{it.path}|{it.traj}"
    def split_seed(self, split):   # offset of the evaluation noise seed of a split (train_gen): fixed by the split name, so a split draws
        import zlib                 # the same noise whatever else is evaluated
        return zlib.crc32(f"{self.key}/{split}".encode()) % 100003
    def text_of(self, raw=None): return self.text                         # equation-identity text of one trajectory
    def texts(self): return [self.text]                                   # every identity text the family can produce (textemb)
    def cond_fixed(self, fields): return fields
    def cond_model(self, it):
        """the condition fields as the model encodes them: a fixed per-family transform into about [-1, 1] (cond_fixed), encoded without
        per-frame normalisation, so absolute levels and contrasts survive (Kolmogorov forcing amp x k_f; acoustic media)."""
        fs = self.cond_fields(it)
        return self.cond_fixed(fs) if fs else []


SUBSET_FAMILIES = ("sf", "rb", "acoustic")   # families whose latent store holds more trajectories than the corpus split keeps (encoded from The Well's files;
                            # SF and RB split by initial condition, acoustic drops a diverged trajectory); train_gen.Store
                            # and walrus_data use the corpus items only (Walrus: copies of these families instead of the published directories)
RAW_FLAGS = ("multimode",)   # raw keys that mark a kind of trajectory, not a parameter value (never written as a parameter: Walrus scalars)


def _traj_slice(n, part, n_eval):
    """trajectory indices of a file: for held-out conditions the last n_eval are evaluation ('eval'), the rest go to the reference run ('ref')."""
    n_eval = min(n_eval, n)
    return list(range(n - n_eval, n)) if part == "eval" else list(range(0, n - n_eval))


def _held(split):   # 'ref_<column>' -> (column, 'ref'); '<column>' -> (column, 'eval')
    return (split[4:], "ref") if split.startswith("ref_") else (split, "eval")


# ---------------------------------------------------------------- Kolmogorov (own) -------------------------------------------
KOLMO_BIG, KOLMO_NATIVE = "kolmo_train", "kolmo_sweep"
KOLMO_DENSE = os.path.join("kolmo_dense", "dense_kf2_amp1.0")
NSU_ROOT = os.path.join("nsu", "decay")
KOLMO_MULTIF = "kolmo_multif"; MULTIF_TRAIN, MULTIF_EXTRAP = "kband2-5", "kband6-7"
NU_CLASS = {"hull": (-3.4545454, -2.7272728), "interp_every": 5, "tol": 1e-6}


def nu_class(i, lg):
    """split class of the i-th nu value (log10 value lg) of a dense axis: 'ends' (outside the hull) / 'interp' / 'train'."""
    lo, hi = NU_CLASS["hull"]; tol = NU_CLASS["tol"]
    if lg < lo - tol or lg > hi + tol: return "ends"
    return "interp" if i % NU_CLASS["interp_every"] == 0 else "train"


def nu_files(d):
    """[(index, log10 nu, path)] of a dense-axis directory, nu ascending (index = position on the axis)."""
    fs = sorted(glob.glob(os.path.join(d, "2D_kolmogorov_nu=*.h5")), key=lambda f: float(re.search(r"nu=([0-9.eE+-]+)\.h5$", f)[1])) if os.path.isdir(d) else []
    lg = [math.log10(float(re.search(r"nu=([0-9.eE+-]+)\.h5$", f)[1])) for f in fs]
    if len(lg) > 2:   # a missing file would shift every later class silently: the axis must be log-uniform on [1e-4, 1e-2]
        st_ = (lg[-1] - lg[0]) / (len(lg) - 1)
        assert abs(lg[0] + 4) < 1e-3 and abs(lg[-1] + 2) < 1e-3 and all(abs(x - (lg[0] + i * st_)) < 0.2 * st_ for i, x in enumerate(lg)), f"dense nu axis {d}: {len(lg)} files, not log-uniform on [1e-4, 1e-2] (missing file?)"
    return [(i, x, f) for i, (x, f) in enumerate(zip(lg, fs))]


DENSE_TRAIN, DENSE_VAL, DENSE_EVAL = 12, 2, 4   # of the 16 trajectories per nu: training-nu files 12 train / 2 val / 2 test; held-out-nu files last 4 eval, first 12 reference


def dense_split(n, split):
    """trajectory indices of a training-nu file with n (16) trajectories: train / val / test."""
    a_, b_ = DENSE_TRAIN, DENSE_TRAIN + DENSE_VAL
    return {"train": range(0, a_), "val": range(a_, b_), "test": range(b_, n)}[split]


KOLMO_ANCHOR, KOLMO_DRAG, KOLMO_BETA = "kolmo_anchor", "kolmo_drag", "kolmo_beta"
ANCHOR_AMPS = (0.7, 1.4)
DRAG_SPLITS = {"extrap_drag_lo": 0.05, "extrap_drag_hi": 0.2}
BETA_ANCHORS = (-2.4, 2.4); BETA_SPLITS = {"extrap_beta": 4.0}   # beta = 4: the largest beta that stays turbulent at k_f 2 and 4; anchors at +-0.6 of it
TRAIN_ONLY_SPLITS = ("anchor", "beta_anchor")   # splits that only ever join training (no evaluation trajectories; train_gen never evaluates them)
KOLMO_EXTRA = []                  # the optional groups turned on (set_kolmo_extra); empty = off


def set_kolmo_extra(names):
    """turn the optional Kolmogorov groups on for this process (train_gen / encode_corpus --kolmo_extra): names = comma string or iterable, a subset
    of {anchor, drag, beta}; empty = off (default). beta also adds the parameter slot `beta` (the beta-plane coefficient; 0 in every other file)."""
    ns = sorted({n.strip() for n in (names.split(",") if isinstance(names, str) else (names or ())) if n and n.strip()})
    bad = [n for n in ns if n not in ("anchor", "drag", "beta")]
    assert not bad, f"kolmo_extra: unknown {bad} (allowed: anchor, drag, beta)"
    KOLMO_EXTRA[:] = ns
    if "beta" in ns and "beta" not in SLOTS: SLOTS.append("beta")
    if "beta" not in ns and "beta" in SLOTS: SLOTS.remove("beta")


def kolmo_extra_splits():
    """the Kolmogorov splits the optional groups add, in the order Kolmo.splits() appends them (empty when off)."""
    return ((["anchor"] if "anchor" in KOLMO_EXTRA else []) + ([s for k in DRAG_SPLITS for s in (k, "ref_" + k)] if "drag" in KOLMO_EXTRA else [])
            + (["beta_anchor"] + [s for k in BETA_SPLITS for s in (k, "ref_" + k)] if "beta" in KOLMO_EXTRA else []))


def _kf_of(rel):   # 'h5/kf2_amp1.0/2D_kolmogorov_nu=....h5' -> 2
    return int(re.match(r"kf(\d+)_amp", rel.split("/")[1])[1])


class Kolmo(Family):
    key = "kolmo"; channels = ["vorticity"]; periodic = (True, True)
    text = ("Two-dimensional incompressible Navier-Stokes (Kolmogorov flow) with linear drag and a steady sinusoidal body force, "
            "periodic square domain. Field: vorticity.")
    TEXT_MULTIF = ("Two-dimensional incompressible Navier-Stokes (Kolmogorov-type flow) with linear drag and a steady random multi-mode body force "
                   "given as a field, periodic square domain. Field: vorticity.")   # random multi-mode forcing trajectories
    FORCE_SCALE = 16.0   # forcing field (the vorticity forcing, max = amp x k_f: single mode 1-6, random multi-mode 5-13) / 16, no
                         # mean removal, so amplitude and strength stay visible
    def cond_fixed(self, fields): return [f / self.FORCE_SCALE for f in fields]
    def text_of(self, raw=None): return self.TEXT_MULTIF if raw and raw.get("multimode") else self.text
    def texts(self): return [self.text, self.TEXT_MULTIF]
    TEST_BIG = 12; VAL_BIG = 6; N_EVAL = 4   # training files (162 trajectories): last 12 test, 6 before them val; held-out files: last 4 eval

    def __init__(self):
        root = os.path.join(DATA_ROOT, "2d") if DATA_ROOT else os.path.join(PROJ, "data", "2d")
        # every Kolmogorov store: frames every 0.1 eddy-turnover times (1/omega_rms), 300 per trajectory.
        self.big = os.path.join(root, KOLMO_BIG); self.native = os.path.join(root, KOLMO_NATIVE); self.dense = os.path.join(root, KOLMO_DENSE)
        self.multif = os.path.join(root, KOLMO_MULTIF)
        self.anchor = os.path.join(root, KOLMO_ANCHOR); self.drag = os.path.join(root, KOLMO_DRAG); self.beta_dir = os.path.join(root, KOLMO_BETA)
        self.data_dirs = [self.multif]   # (cache signature)
        self.extra_dirs = [d for n, d in (("anchor", self.anchor), ("drag", self.drag), ("beta", self.beta_dir)) if n in KOLMO_EXTRA]   # listed only when turned on
        self.S = json.load(open(os.path.join(PROJ, "data_gen", "kolmo2d", "splits_kolmo.json")))
        hull = self.S.get("hull_log10_nu")   # the dense-axis classes are tied to this hull
        assert hull is None or all(abs(a - b) < 1e-4 for a, b in zip(hull, NU_CLASS["hull"])), (hull, NU_CLASS["hull"])

    def _p(self, rel, big):   # 'h5/kf2_amp1.0/2D_kolmogorov_nu=...h5' -> store path
        return os.path.join(self.big if big else self.native, *rel.split("/")[1:])

    def _item(self, path, traj):
        with h5py.File(path, "r") as h:
            a = dict(h.attrs); g = h[traj]; T, H, W = g["data"].shape[:3]; kf_traj = g.attrs.get("kf_eff", None)   # random multi-mode forcing: the trajectory's own energy-weighted |k|
        raw = {"nu": float(a["nu"]), "k_f": float(kf_traj if kf_traj is not None else a["kf"]), "amp": float(a["amp"])}
        if kf_traj is not None: raw["multimode"] = 1.0
        al = float(a.get("alpha", KOLMO_ALPHA))
        if abs(al - KOLMO_ALPHA) > 1e-9: raw["alpha"] = al   # drag columns: recorded, not a model input
        p = {"log10_nu": math.log10(raw["nu"]), "kf": raw["k_f"], "amp": raw["amp"], "log10_dt": math.log10(float(a["dt_frame"]))}
        if float(a.get("beta", 0.0)): raw["beta"] = p["beta"] = float(a["beta"])   # beta-plane files
        return Item(self.key, path, traj, p, raw, T, (H, W))

    def _keys(self, path):
        with h5py.File(path, "r") as h: return sorted(h.keys())

    @staticmethod
    def _modeset(path, traj):   # the random forcing's wavevectors of one trajectory, each up to sign (k and -k are the same mode)
        with h5py.File(path, "r") as h: m = np.asarray(h[traj].attrs["modes_k"]).reshape(-1, 2).astype(int).tolist()
        return tuple(sorted(max(tuple(v), tuple(-x for x in v)) for v in m))

    DENSE_SPLITS = {"nu_dense_interp": "interp", "nu_dense_ends": "ends"}

    def _multif(self, band): return sorted(glob.glob(os.path.join(self.multif, band, "2D_kolmogorov_nu=*.h5"))) if os.path.isdir(self.multif) else []

    def splits(self):
        names = ["train", "val", "test"]
        for k in self.S:
            if isinstance(self.S[k], dict) and "files" in self.S[k] and k != "train": names += [k, "ref_" + k]
        for k in self.DENSE_SPLITS: names += [k, "ref_" + k]
        names += ["multif_test", "extrap_kband", "ref_extrap_kband"]   # unseen realisations of the random forcing; the held-out wavenumber band
        return names + kolmo_extra_splits()

    def _anchor_files(self):
        return [os.path.join(self.anchor, f"kf{_kf_of(rel)}_amp{amp}", rel.split("/")[-1]) for amp in ANCHOR_AMPS for rel in self.S["train"]["files"]]

    def _drag_files(self, alpha):
        return [os.path.join(self.drag, f"kf{_kf_of(rel)}_amp1.0_alpha{alpha}", rel.split("/")[-1]) for rel in self.S["extrap_amp"]["files"]]

    def _beta_files(self, beta, which):   # anchors at the training files' (k_f, nu), the column at extrap_amp's
        return [os.path.join(self.beta_dir, f"kf{_kf_of(rel)}_amp1.0_beta{beta:g}", rel.split("/")[-1]) for rel in self.S["train" if which == "anchor" else "extrap_amp"]["files"]]

    def _extra_files(self, name, files):   # the option must be on and the generated store complete: a missing file would silently shrink a training split
        assert name in KOLMO_EXTRA, f"kolmo: the {name} splits need corpus.set_kolmo_extra(['{name}']) (--kolmo_extra {name})"
        miss = [f for f in files if not os.path.exists(f)]
        assert not miss, f"kolmo --kolmo_extra {name}: {len(miss)} of {len(files)} files missing (e.g. {miss[0]}); generate them first (gen_job --set {name})"
        return files

    def _dense(self, cls):   # dense-axis files of one split class
        return [f for i, lg, f in nu_files(self.dense) if nu_class(i, lg) == cls]

    def items(self, split):
        out = []
        if split in ("train", "val", "test"):
            for rel in self.S["train"]["files"]:
                path = self._p(rel, big=True); tkeys = self._keys(path); n = len(tkeys)
                lo_test = n - self.TEST_BIG; lo_val = lo_test - self.VAL_BIG
                sel = {"train": range(0, lo_val), "val": range(lo_val, lo_test), "test": range(lo_test, n)}[split]
                out += [self._item(path, tkeys[i]) for i in sel]
            for path in self._dense("train"):   # dense nu axis, training values: 12 / 2 / 2 of 16 trajectories
                tkeys = self._keys(path); out += [self._item(path, tkeys[i]) for i in dense_split(len(tkeys), split)]
            if split != "test":   # random-forcing training band, 162 train / 6 val per file; its last 12 are the multif_test column, not the plain test
                for path in self._multif(MULTIF_TRAIN):
                    tkeys = self._keys(path); n = len(tkeys); lo_test = n - self.TEST_BIG; lo_val = lo_test - self.VAL_BIG
                    out += [self._item(path, tkeys[i]) for i in {"train": range(0, lo_val), "val": range(lo_val, lo_test)}[split]]
            return out
        if split == "multif_test":
            for path in self._multif(MULTIF_TRAIN): tkeys = self._keys(path); out += [self._item(path, tkeys[i]) for i in range(max(0, len(tkeys) - self.TEST_BIG), len(tkeys))]
            return out
        if split == "anchor":   # amplitude anchors: every trajectory of every anchor file trains
            for path in self._extra_files("anchor", self._anchor_files()): out += [self._item(path, k) for k in self._keys(path)]
            return out
        if split == "beta_anchor":   # beta anchors: every trajectory of every anchor file trains
            for path in self._extra_files("beta", [f for b in BETA_ANCHORS for f in self._beta_files(b, "anchor")]): out += [self._item(path, k) for k in self._keys(path)]
            return out
        base, part = _held(split)
        if base in DRAG_SPLITS:
            for path in self._extra_files("drag", self._drag_files(DRAG_SPLITS[base])):
                tkeys = self._keys(path); out += [self._item(path, tkeys[i]) for i in _traj_slice(len(tkeys), part, self.N_EVAL)]
            return out
        if base in BETA_SPLITS:
            for path in self._extra_files("beta", self._beta_files(BETA_SPLITS[base], "column")):
                tkeys = self._keys(path); out += [self._item(path, tkeys[i]) for i in _traj_slice(len(tkeys), part, self.N_EVAL)]
            return out
        if base == "extrap_kband":   # the held-out wavenumber band, 24 trajectories per file: last 12 evaluation, first 12 reference
            for path in self._multif(MULTIF_EXTRAP):
                tkeys = self._keys(path); sel = list(_traj_slice(len(tkeys), part, self.TEST_BIG))
                if part == "eval":
                    seen = {self._modeset(path, tkeys[i]) for i in _traj_slice(len(tkeys), "ref", self.TEST_BIG)}
                    sel = [i for i in sel if self._modeset(path, tkeys[i]) not in seen]
                out += [self._item(path, tkeys[i]) for i in sel]
            return out
        if base in self.DENSE_SPLITS:
            for path in self._dense(self.DENSE_SPLITS[base]):
                tkeys = self._keys(path); out += [self._item(path, tkeys[i]) for i in _traj_slice(len(tkeys), part, DENSE_EVAL)]
            return out
        for rel in self.S[base]["files"]:
            path = self._p(rel, big=False)
            if not os.path.exists(path): print(f"[corpus] WARNING: kolmo/{split}: missing file {path}", flush=True); continue
            tkeys = self._keys(path); out += [self._item(path, tkeys[i]) for i in _traj_slice(len(tkeys), part, self.N_EVAL)]
        return out

    def load(self, it, t0=0, t1=None):
        with h5py.File(it.path, "r") as h: x = h[it.traj]["data"][t0:t1]          # [T, x, y, 1] float16
        return np.ascontiguousarray(np.moveaxis(np.asarray(x, dtype=np.float32), -1, 1))

    def cond_fields(self, it):
        with h5py.File(it.path, "r") as h: return [np.asarray(h[it.traj]["forcing"], dtype=np.float32)]


class NSU(Family):
    key = "nsu"; channels = ["vorticity"]; periodic = (True, True)
    text = ("Two-dimensional incompressible Navier-Stokes, freely decaying turbulence (no body force, no drag) from a random initial "
            "condition, periodic square domain. Field: vorticity.")
    SPLITS = {"interp_test": "interp", "extrap_ends": "ends"}
    t_start = 20   # skip the start-up transient from the smooth random initial condition

    def __init__(self):
        root = os.path.join(DATA_ROOT, "2d") if DATA_ROOT else os.path.join(PROJ, "data", "2d")
        self.root = os.path.join(root, NSU_ROOT)

    def splits(self):
        s = ["train", "val", "test"]
        for k in self.SPLITS: s += [k, "ref_" + k]
        return s

    def _files(self, cls): return [f for i, lg, f in nu_files(self.root) if nu_class(i, lg) == cls]

    def _keys(self, path):
        with h5py.File(path, "r") as h: return sorted(k for k in h.keys() if k != "meta")

    def _item(self, path, traj):
        with h5py.File(path, "r") as h:
            a = dict(h.attrs); g = h[traj]; T, H, W = g["data"].shape[:3]
        nu = float(a["nu"])
        return Item(self.key, path, traj, {"log10_nu": math.log10(nu), "log10_dt": math.log10(float(a["dt_frame"]))}, {"nu": nu}, T, (H, W))

    def items(self, split):
        out = []
        if split in ("train", "val", "test"):
            for path in self._files("train"):
                tkeys = self._keys(path); out += [self._item(path, tkeys[i]) for i in dense_split(len(tkeys), split)]
            return out
        base, part = _held(split)
        for path in self._files(self.SPLITS[base]):
            tkeys = self._keys(path); out += [self._item(path, tkeys[i]) for i in _traj_slice(len(tkeys), part, DENSE_EVAL)]
        return out

    def load(self, it, t0=0, t1=None):
        with h5py.File(it.path, "r") as h: x = h[it.traj]["data"][t0:t1]          # [T, x, y, 1] float16
        return np.ascontiguousarray(np.moveaxis(np.asarray(x, dtype=np.float32), -1, 1))


# ---------------------------------------------------------------- The Well ----------------------------------------------------
class Well(Family):
    """Generic reader of a The Well 2D dataset: t0_fields (scalars) then t1_fields (vectors, components x, y); tensor fields (t2)
    are not used. Held-out conditions: HOLDOUT maps a split name to a predicate on the file's parameters; files of held-out
    conditions are removed from training; their data/test (+ valid) trajectories are the evaluation set, their data/train
    trajectories the reference set."""
    DS = ""; PARAMS = {}; HOLDOUT = {}; N_VAL = 2
    N_EVAL_TRAIN = 0   # held-out conditions: this many of the file's data/train trajectories join the evaluation set (the rest are the reference set);
                       # The Well gives only 1 valid + 1 test trajectory per condition, so TRL uses 4

    def __init__(self): self.root = os.path.join(EXT, "the_well", self.DS, "data")

    def _files(self, d): return sorted(glob.glob(os.path.join(self.root, d, "*.hdf5")))

    def _raw(self, f):
        with h5py.File(f, "r") as h: return {n: float(h.attrs[n]) for n in h.attrs["simulation_parameters"]}

    def _held(self, raw): return [k for k, pred in self.HOLDOUT.items() if pred(raw)]

    def splits(self):
        s = ["train", "val", "test"]
        for k in self.HOLDOUT: s += [k, "ref_" + k]
        return s

    def _items_of(self, f, trajs=None):
        with h5py.File(f, "r") as h:
            t = h["dimensions/time"][:2]
            name0 = [n for n in h["t0_fields"].attrs["field_names"] if h[f"t0_fields/{n}"].attrs.get("time_varying", True)]   # constant fields (acoustic: density, sound speed) are conditions, not channels
            first = h[f"t0_fields/{name0[0]}"] if len(name0) else h["t1_fields/velocity"]
            n = int(h.attrs["n_trajectories"]); T, H, W = first.shape[1:4]
            if n <= 0 or n > first.shape[0]: n = int(first.shape[0])   # some files carry an invalid n_trajectories attribute; the data shape is the truth
        raw = self._raw(f); p = {slot: fn(raw) for slot, fn in self.PARAMS.items()}; p["log10_dt"] = math.log10(float(t[1] - t[0]))
        return [Item(self.key, f, i, p, raw, T, (H, W)) for i in (range(n) if trajs is None else trajs)]

    def items(self, split):
        out = []
        if split in ("train", "val", "test"):
            d = {"train": "train", "val": "valid", "test": "test"}[split]
            for f in self._files(d):
                if self._held(self._raw(f)): continue
                its = self._items_of(f); out += its[: self.N_VAL] if split == "val" else its
            return out
        base, part = _held(split)
        for d in ("valid", "test", "train"):
            for f in self._files(d):
                if base not in self._held(self._raw(f)): continue
                if d != "train":
                    if part == "eval": out += self._items_of(f)
                else:
                    its = self._items_of(f); out += its[:self.N_EVAL_TRAIN] if part == "eval" else its[self.N_EVAL_TRAIN:]
        return out

    def load(self, it, t0=0, t1=None):
        with h5py.File(it.path, "r") as h:
            ch = [np.asarray(h[f"t0_fields/{n}"][it.traj, t0:t1], dtype=np.float32) for n in h["t0_fields"].attrs["field_names"] if h[f"t0_fields/{n}"].attrs.get("time_varying", True)]
            for n in h["t1_fields"].attrs["field_names"]:
                v = np.asarray(h[f"t1_fields/{n}"][it.traj, t0:t1], dtype=np.float32); ch += [v[..., 0], v[..., 1]]
        return np.ascontiguousarray(np.stack(ch, 1))


class TRL(Well):
    key = "trl"; DS = "turbulent_radiative_layer_2D"; channels = ["density", "pressure", "velocity_x", "velocity_y"]; periodic = (True, False)
    BC_SIDES = {1: ("open", "open")}
    N_EVAL_TRAIN = 4   # 10 trajectories per cooling time (8 / 1 / 1): evaluation = 1 valid + 1 test + 4 train, reference = the other 4
    t_start = 24   # trajectories at other t_cool share initial conditions and stay close early on, so windows start at frame 24
    text = ("Two-dimensional compressible turbulent radiative mixing layer between hot and cold gas with radiative cooling, periodic in x "
            "and open in y. Fields: density, pressure, velocity x, velocity y.")
    PARAMS = {"log10_tcool": lambda r: math.log10(r["tcool"])}
    HOLDOUT = {"extrap_tcool": lambda r: r["tcool"] > 3.0, "interp_tcool": lambda r: abs(r["tcool"] - 0.32) < 1e-3}


class GS(Well):
    key = "gs"; DS = "gray_scott_reaction_diffusion"; channels = ["A", "B"]; periodic = (True, True)
    text = "Two-dimensional Gray-Scott reaction-diffusion system, periodic square domain. Fields: concentrations of species A and B."
    PARAMS = {"gs_F": lambda r: r["F"], "gs_k": lambda r: r["k"]}
    # six (F, k) pattern regimes; 'maze' (F 0.029, k 0.057) lies inside the training hull, 'bubbles' (F 0.098, the largest F) outside
    HOLDOUT = {"interp_maze": lambda r: abs(r["F"] - 0.029) < 1e-4, "extrap_bubbles": lambda r: abs(r["F"] - 0.098) < 1e-4}


class _ICSplit(Well):
    """The Well ran the same initial conditions (ICs) at every condition and split them into its train / valid / test files differently
    per condition, so an evaluation trajectory would have training trajectories with its IC at other conditions (SF: even bit-identical
    copies). These families split by IC instead: ICs with id % EVAL_IC_EVERY == EVAL_IC_EVERY - 1 are evaluation ICs.
    Training keeps the training ICs;
    every evaluation split takes the evaluation ICs from any file at its conditions, one trajectory per (IC, condition); val = the stored val
    trajectories with an evaluation IC. No evaluation IC is ever trained on, at any condition. N_IC: the number of distinct ICs (asserted)."""
    EVAL_IC_EVERY = 4; N_IC = 0; _GROUPS = {}

    def _ic_cluster(self, keys):   # -> list of clusters (lists of indices into keys)
        raise NotImplementedError

    def _ic_groups(self):
        """{(path, traj): IC id 0..N_IC-1} over every trajectory of The Well's train / valid / test files (all conditions), numbered by first member."""
        if self.key not in _ICSplit._GROUPS:
            keys = [(f, i) for d in ("train", "valid", "test") for f in self._files(d) for i in range(len(self._items_of(f)))]
            order = sorted(self._ic_cluster(keys), key=lambda m: min(keys[a] for a in m))
            assert len(order) == self.N_IC, f"{self.key}: {len(order)} initial-condition clusters, expected {self.N_IC} (data changed?)"
            _ICSplit._GROUPS[self.key] = {keys[a]: g for g, m in enumerate(order) for a in m}
        return _ICSplit._GROUPS[self.key]

    def ic_id(self, it): return self._ic_groups()[(it.path, it.traj)]

    def _cond_key(self, raw): return tuple(sorted((k, v) for k, v in raw.items() if k not in RAW_FLAGS))

    def items(self, split):
        g = self._ic_groups(); ev = lambda it: g[(it.path, it.traj)] % self.EVAL_IC_EVERY == self.EVAL_IC_EVERY - 1
        if split == "train" or split.startswith("ref_"): return [it for it in super().items(split) if not ev(it)]
        if split == "val": return [it for it in super().items(split) if ev(it)]
        out, seen = [], set()
        for d in ("test", "valid", "train"):
            for f in self._files(d):
                raw = self._raw(f); held = self._held(raw)
                if (held if split == "test" else split not in held): continue
                for it in self._items_of(f):
                    key = (g[(f, it.traj)], self._cond_key(raw))
                    if split == "test" and d == "valid" and it.traj < self.N_VAL: continue   # the val trajectories stay out of test (by trajectory)
                    if ev(it) and key not in seen: seen.add(key); out.append(it)
        return out


class RB(_ICSplit):
    key = "rb"; DS = "rayleigh_benard"; channels = ["buoyancy", "pressure", "velocity_x", "velocity_y"]; periodic = (True, False)
    BC_SIDES = {1: ("wall", "wall")}
    text = ("Two-dimensional Rayleigh-Benard convection between a hot bottom and a cold top wall, periodic in x, walls in y. "
            "Fields: buoyancy, pressure, velocity x, velocity y.")
    PARAMS = {"log10_Ra": lambda r: math.log10(r["Rayleigh"]), "log10_Pr": lambda r: math.log10(r["Prandtl"])}
    # Ra in {1e6 .. 1e10} x Pr in {0.1, 0.2, 0.5, 1, 2, 5, 10}: Ra = 1e10 held out (extrapolation up in Ra), Pr = 2 held out (interpolation)
    HOLDOUT = {"extrap_Ra": lambda r: r["Rayleigh"] > 5e9, "interp_Pr": lambda r: abs(r["Prandtl"] - 2) < 1e-6 and r["Rayleigh"] < 5e9}
    N_IC = 50

    def _ic_cluster(self, keys):
        import hashlib
        cl, rep = {}, {}
        for a, (f, i) in enumerate(keys):
            with h5py.File(f, "r") as h:
                x = [np.asarray(h[f"t0_fields/{n}"][i, 0], np.float32) for n in h["t0_fields"].attrs["field_names"]]
                x += [np.asarray(h[f"t1_fields/{n}"][i, 0], np.float32) for n in h["t1_fields"].attrs["field_names"]]
            m = hashlib.sha1()
            for v in x: m.update(np.ascontiguousarray(v).tobytes())
            k = m.hexdigest(); cl.setdefault(k, []).append(a)
            if k not in rep: rep[k] = np.concatenate([v.astype(np.float64).ravel() for v in x])
        tkeys = list(cl); par = list(range(len(tkeys)))
        def find(a):
            while par[a] != a: par[a] = par[par[a]]; a = par[a]
            return a
        for a in range(len(tkeys)):
            for b in range(a + 1, len(tkeys)):
                if rep[tkeys[a]].shape == rep[tkeys[b]].shape and np.abs(rep[tkeys[a]] - rep[tkeys[b]]).max() < 1e-6: par[find(a)] = find(b)
        out = {}
        for a, k in enumerate(tkeys): out.setdefault(find(a), []).extend(cl[k])
        return list(out.values())

    def ci_group(self, it): return f"rb-ic{self.ic_id(it)}|Ra{it.raw['Rayleigh']:g}"


class SF(_ICSplit):
    key = "sf"; DS = "shear_flow"; channels = ["tracer", "pressure", "velocity_x", "velocity_y"]; periodic = (True, True)
    text = ("Two-dimensional incompressible shear flow (two shear layers) with a passive tracer, periodic domain. "
            "Fields: tracer, pressure, velocity x, velocity y.")
    PARAMS = {"log10_Re": lambda r: math.log10(r["Reynolds"]), "log10_Sc": lambda r: math.log10(r["Schmidt"])}
    # Re in {1e4, 5e4, 1e5, 5e5} x Sc in {0.1 .. 10}: Re = 5e5 held out (extrapolation), Sc = 2 held out (interpolation)
    HOLDOUT = {"extrap_Re": lambda r: r["Reynolds"] > 2e5, "interp_Sc": lambda r: abs(r["Schmidt"] - 2) < 1e-6 and r["Reynolds"] < 2e5}
    N_IC = 40
    # The tracer is passive, so velocity and pressure at one Re are the same (to 1e-6) at every Sc: ICs = clusters of the frame-0 velocity

    def _ic_cluster(self, keys):
        X = []
        for f, i in keys:
            with h5py.File(f, "r") as h: X.append(np.asarray(h["t1_fields/velocity"][i, 0, ::8, ::8], np.float64).ravel())
        X = np.stack(X); nr = np.linalg.norm(X, axis=1); sq = nr ** 2; par = list(range(len(keys)))
        D = np.sqrt(np.maximum(sq[:, None] + sq[None, :] - 2 * X @ X.T, 0)) / np.maximum(nr[:, None], 1e-12)
        def find(a):
            while par[a] != a: par[a] = par[par[a]]; a = par[a]
            return a
        for a, b in zip(*np.nonzero(np.triu(D < 1e-3, 1))): par[find(int(a))] = find(int(b))
        cl = {}
        for a in range(len(keys)): cl.setdefault(find(a), []).append(a)
        return list(cl.values())

    def ci_group(self, it): return f"sf-ic{self.ic_id(it)}"


class Euler(Well):
    """The Well euler_multi_quadrants_openBC, 2D compressible Euler from random piecewise-constant quadrants (shocks, rarefactions,
    contact discontinuities), 512^2, 101 frames, open boundaries; gamma in {1.13 .. 1.76} (10 values). Only the first 40-trajectory train
    chunk (chunk_0) and the 10-trajectory test tail (chunk_40) of every gamma are downloaded: training 8 gammas x 38,
    val = the last 2 of each chunk, test = the test tails; gamma 1.13 / 1.76 held out (two-sided extrapolation; their chunk_0 is the reference set,
    their test tail the evaluation set). Energy is dropped (ideal gas: it follows from density, pressure and velocity); velocity = momentum / density."""
    key = "euler"; DS = "euler_multi_quadrants_openBC"; channels = ["density", "pressure", "velocity_x", "velocity_y"]; periodic = (False, False)
    BC_SIDES = {0: ("open", "open"), 1: ("open", "open")}
    text = ("Two-dimensional compressible Euler equations of an ideal gas from piecewise-constant initial states (Riemann quadrants) with shocks, "
            "open boundaries. Fields: density, pressure, velocity x, velocity y.")
    PARAMS = {"gamma": lambda r: r["gamma"]}
    HOLDOUT = {"extrap_gamma_low": lambda r: r["gamma"] < 1.2, "extrap_gamma_high": lambda r: r["gamma"] > 1.7}

    def items(self, split):
        if split in ("train", "val"):   # no valid files downloaded: the last N_VAL trajectories of each 40-trajectory train chunk are the val split
            out = []
            for f in self._files("train"):
                if self._held(self._raw(f)): continue
                its = self._items_of(f); out += its[:-self.N_VAL] if split == "train" else its[-self.N_VAL:]
            return out
        return super().items(split)

    def load(self, it, t0=0, t1=None):
        with h5py.File(it.path, "r") as h:
            rho = np.asarray(h["t0_fields/density"][it.traj, t0:t1], dtype=np.float32); p = np.asarray(h["t0_fields/pressure"][it.traj, t0:t1], dtype=np.float32)
            m = np.asarray(h["t1_fields/momentum"][it.traj, t0:t1], dtype=np.float32)
        u = m / np.maximum(rho, 1e-12)[..., None]
        return np.ascontiguousarray(np.stack([rho, p, u[..., 0], u[..., 1]], 1))


class Acoustic(Well):
    """The Well acoustic_scattering_inclusions, 2D acoustic wave equation (pressure, velocity) in a medium whose density and sound
    speed are constant fields with 1-15 elliptic inclusions; 256^2, 102 frames. Files: train chunks 0-27 (2,798 trajectories; chunks 8 and 27
    hold 99 each), valid chunks 32-35 and test chunks 36-39 (400 trajectories each); the train split reads every file of data/train, i.e.
    chunks 0-27. One training trajectory whose solution diverges is left out (DIVERGED): 2,797 training trajectories. No scalar parameters:
    the medium enters as condition fields."""
    key = "acoustic"; DS = "acoustic_scattering_inclusions"; channels = ["pressure", "velocity_x", "velocity_y"]; periodic = (False, False)
    BC_SIDES = {0: ("wall", "open"), 1: ("wall", "open")}
    text = ("Two-dimensional acoustic wave propagation in a heterogeneous medium with elliptic inclusions of different density, wall and open "
            "boundaries; the material density and sound speed are given as fields. Fields: pressure, velocity x, velocity y.")
    PARAMS = {}; HOLDOUT = {}
    # log10 density / 5 and log10 sound speed / 2.5, no mean removal, so the background level and the inclusion contrasts are kept
    def cond_fixed(self, fields): return [np.log10(np.maximum(fields[0], 1e-3)) / 5.0, np.log10(np.maximum(fields[1], 1e-3)) / 2.5]

    DIVERGED = {("acoustic_scattering_inclusions_chunk_11.hdf5", 47)}   # one published trajectory diverges; it is left out

    def items(self, split):
        return [it for it in super().items(split) if (os.path.basename(it.path), it.traj) not in self.DIVERGED]

    def cond_fields(self, it):   # constant per trajectory: material density and speed of sound [H, W]
        with h5py.File(it.path, "r") as h: return [np.asarray(h[f"t0_fields/{n}"][it.traj], dtype=np.float32) for n in ("density", "speed_of_sound")]


# ---------------------------------------------------------------- the corpus -------------------------------------------------
FAMILIES = {c.key: c for c in (Kolmo, NSU, TRL, GS, RB, SF, Euler, Acoustic)}


def family_weights(nwin, bbar, temp=0.0, cap=0.0, n_micro=0.0):
    """sampling weights of the families.
    nwin: {family: training windows}; bbar: {family: expected micro-batch size (windows drawn per micro-batch)}; temp: weight ∝ windows^temp
    (0 = uniform; training uses 0.5); cap > 0 with n_micro (expected micro-batches of the run): no family's expected passes per window
    n_micro * w_k * bbar_k / nwin_k may exceed cap — such a family is held at the cap and its surplus probability goes to the others in
    proportion to their uncapped weights.
    Returns ({family: weight}, {family: expected passes})."""
    keys = sorted(nwin); base = {k: max(nwin[k], 1) ** temp for k in keys}; fixed = {}
    for _ in range(len(keys) + 1):
        free = [k for k in keys if k not in fixed]; rem = max(0.0, 1.0 - sum(fixed.values())); tot = sum(base[k] for k in free)
        w = {**fixed, **{k: rem * base[k] / tot for k in free}} if free and tot > 0 else dict(fixed)
        if not (cap > 0 and n_micro > 0): break
        over = [k for k in free if n_micro * w[k] * bbar[k] / max(nwin[k], 1) > cap * (1 + 1e-9)]
        if not over: break
        for k in over: fixed[k] = cap * max(nwin[k], 1) / (n_micro * bbar[k])
    tot = sum(w.values()); w = {k: v / tot for k, v in w.items()} if tot > 0 else w   # all families capped: the corpus is too small for the run; keep the ratios
    passes = {k: (n_micro * w[k] * bbar[k] / max(nwin[k], 1)) if n_micro > 0 else None for k in keys}
    return w, passes


def get(key): return FAMILIES[key]()


CACHE_VERSION = 1


def _src_sig():   # split logic and item parameters live in this file and the Kolmogorov split file: any edit invalidates the item cache
    import hashlib; h = hashlib.md5(open(__file__, "rb").read())
    sp = os.path.join(PROJ, "data_gen", "kolmo2d", "splits_kolmo.json")
    if os.path.exists(sp): h.update(open(sp, "rb").read())
    return h.hexdigest()[:8]
SRC_SIG = _src_sig()


def cached_items(key, split, rebuild=False):
    """items of a family split, cached as a pickle under cache/; the cache key covers this file, the Kolmogorov split file,
    the data root and the number / total size of the data files, so any change rebuilds it."""
    import pickle, hashlib
    f = get(key); root = getattr(f, "root", None) or getattr(f, "big", "")
    fs = glob.glob(os.path.join(root, "*", "*.h*5")) + glob.glob(os.path.join(root, "*.h*5"))
    for extra in (getattr(f, "native", ""), getattr(f, "dense", ""), *getattr(f, "data_dirs", []), *getattr(f, "extra_dirs", [])):   # Kolmogorov: the held-out store, the dense axis, the random forcing, optional groups
        if extra and extra != root and os.path.isdir(extra): fs += glob.glob(os.path.join(extra, "*", "*.h*5")) + glob.glob(os.path.join(extra, "*.h*5"))
    sig = f"{CACHE_VERSION}|{SRC_SIG}|" + root + "|" + str(len(fs)) + "|" + str(sum(os.path.getsize(x) for x in fs))
    if key == "kolmo" and KOLMO_EXTRA: sig += "|kolmo_extra=" + ",".join(KOLMO_EXTRA)
    p = os.path.join(HERE, "cache", f"items_{key}_{split}_{hashlib.md5(sig.encode()).hexdigest()[:8]}.pkl")
    if not rebuild and os.path.exists(p): return pickle.load(open(p, "rb"))
    its = f.items(split); os.makedirs(os.path.dirname(p), exist_ok=True)
    tmp = f"{p}.{os.getpid()}.tmp"; pickle.dump(its, open(tmp, "wb")); os.replace(tmp, p); return its   # per-process tmp: parallel jobs build the same cache


def available(keys=None):
    """families whose data are present here (The Well ones appear once downloaded)."""
    out = []
    for k in (keys or FAMILIES):
        try:
            f = get(k)
            if f.items("test"): out.append(k)
        except Exception:
            pass
    return out


def param_vector(params, slots=None):
    """slot values + presence mask (float32 [len(slots)] each); missing slots are 0 with mask 0. `slots` defaults to SLOTS; a trained run is
    evaluated with the slot list stored in its checkpoint (pstats["slots"])."""
    slots = SLOTS if slots is None else slots
    v = np.zeros(len(slots), np.float32); m = np.zeros(len(slots), np.float32)
    for k, x in params.items():
        if k in slots: i = slots.index(k); v[i] = x; m[i] = 1
    return v, m


if __name__ == "__main__":
    import sys
    for k in (sys.argv[1:] or FAMILIES):
        try:
            f = get(k); print(f"== {k}: root ok, splits {f.splits()}")
            for s in f.splits():
                its = f.items(s)
                if its: print(f"   {s:24s} {len(its):6d} traj  T={its[0].n_frames} grid={its[0].shape} params={its[0].raw}")
                else: print(f"   {s:24s} (none)")
        except Exception as e:
            print(f"== {k}: unavailable ({type(e).__name__}: {str(e)[:120]})")
