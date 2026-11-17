#!/usr/bin/env python3
"""Dev / test halves of the evaluation windows (standard library only), used by the table scripts.

Rule: every evaluation column has 32 windows (fewer for a few columns).
They are split BY TRAJECTORY into two fixed halves:
  dev  - used only to choose settings / samplers;
  test - used only for reported numbers.

Split rule
  * trajectory = (basename of the file path, traj id); all windows of one trajectory share a half.
  * hash(trajectory) = first 8 bytes of sha256("<basename>\\0<traj>") as an integer (not Python's
    salted hash()).
  * per column: sort the column's distinct trajectories by (hash, key) and alternate
    dev, test, dev, test, ... (rank 0 -> dev, so an odd count gives dev the extra one).
  * SF and RB: these families run the same initial conditions (ICs) at every
    condition; windows of one IC are identical (SF velocity / pressure at every Sc) or near-duplicates
    (SF across Re, RB across Pr at one Ra).  Their unit is the IC, and the halves are assigned once per
    family over its evaluation ICs (sorted by hash of "<fam>-ic\\0<ic>", alternating), so an IC is in the
    same half in every column of the family.  These columns are therefore not balanced one by one.
    The IC of a window comes from devsplit_ic.json next to this file (map of every SF / RB evaluation
    trajectory to its initial condition).
Functions: window_key (the identity of a window), unit_of (the unit a window's half is drawn for), split_maps (column -> {unit: half})."""
import hashlib
import json
import math
import os
from statistics import fmean

HALVES = ("dev", "test")


def traj_key(path, traj):
    """(basename, traj): distinct trajectories that share a basename (e.g. the same nu under two k_f / amp directories) form one unit."""
    return (str(path).replace("\\", "/").split("/")[-1], str(traj))


def _hash(key):
    return int.from_bytes(hashlib.sha256(f"{key[0]}\0{key[1]}".encode()).digest()[:8], "big")


IC_FILE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "devsplit_ic.json")
_IC = None


def _ic_tables():
    """(IC of '<dir>/<basename>' + traj, family-wide half of every evaluation IC), loaded once from IC_FILE."""
    global _IC
    if _IC is None:
        d = json.load(open(IC_FILE))
        look, halves = {}, {}
        for fam, v in d.items():
            for f, ics in v["files"].items():
                for t, ic in enumerate(ics):
                    if ic >= 0:
                        look[(f, str(t))] = (f"{fam}-ic", str(ic))
            order = sorted(((f"{fam}-ic", str(ic)) for ic in v["eval_ics"]), key=lambda k: (_hash(k), k))
            halves.update({k: HALVES[i % 2] for i, k in enumerate(order)})
        _IC = (look, halves, set(d))
    return _IC


def unit_of(path, traj):
    """The unit a window's half is drawn for: its IC for SF / RB, otherwise its trajectory key."""
    parts = str(path).replace("\\", "/").split("/")
    u = _ic_tables()[0].get(("/".join(parts[-2:]), str(traj)))
    return u if u is not None else traj_key(path, traj)


def _pairs(windows):
    for w in windows:
        if isinstance(w, dict):
            yield w["path"], w["traj"]
        else:
            yield w[0], w[1]


def assign_column(windows, family=None):
    """Split of one column: {unit: 'dev'|'test'}; trajectory units balanced within the column, IC units (SF / RB) by the family-wide
    table.  `family` (the column's family): for SF / RB every window must have an IC (else KeyError)."""
    look, ic_half, ic_fams = _ic_tables()
    units = {unit_of(p, t) for p, t in _pairs(windows)}
    if family in ic_fams:
        bad = [u for u in units if u not in ic_half]
        if bad:
            raise KeyError(f"{family}: {len(bad)} window unit(s) without an evaluation IC in {IC_FILE}, e.g. {bad[0]}")
    out = {u: ic_half[u] for u in units if u in ic_half}
    order = sorted((u for u in units if u not in ic_half), key=lambda k: (_hash(k), k))
    out.update({k: HALVES[i % 2] for i, k in enumerate(order)})
    return out


def base_of(col):
    return col.split("@", 1)[0]


def window_key(w):
    return (w["path"], str(w["traj"]), w["first_frame"])


def split_maps(*column_sets):
    """column name -> {unit: half}, from the union of windows over the given files;
    a slice column reuses its base column's map when the base is present."""
    wins = {}
    for cols in column_sets:
        for c, v in cols.items():
            wins.setdefault(c, []).extend(v["per_window"])
    maps = {}
    for c in wins:
        b = base_of(c)
        maps[c] = assign_column(wins[b] if b in wins else wins[c], family=b.split("/", 1)[0])
    return maps


def geomean(xs):
    xs = list(xs)
    return math.inf if any(math.isinf(x) for x in xs) else math.exp(fmean(math.log(x) for x in xs))
