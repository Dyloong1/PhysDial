"""Case-study figures of multiple condition switches in one rollout, CPU.

Reads the field dumps of an eval_ms.py run with --dump_pred N (results/pred/<eval>/kolmo__switch_<group>_S0.npz = told every segment's
condition, ..._U.npz = not told: the first condition throughout) and the group files.
Segment k covers model steps k*seg_steps + 1 .. (k+1)*seg_steps and has the condition seq[k + 1]. Per window: err(P) = ||P - S|| / ||S||
at every step; the windows with the largest mean gap err(not told) - err(told) are drawn (top --top per group):
  - <group>_wNN.png/.pdf: columns = steps after each switch (--after) and segment ends; rows = forcing in effect, switched truth, told,
    not told, control truth;
  - <group>_wNN_curve.png/.pdf: projection of the field on the current forcing (or its mean square when the forcing pattern never changes);
  - <group>_wNN_anim.gif (--gif).
Writes results/<name>.json and results/case/<name>/."""
import argparse, json, os
import numpy as np, h5py
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))
EVAL_HIST = 6


def nrm(x): return float(np.sqrt((x.astype(np.float64) ** 2).sum()))


def load(pdir, g, v):
    f = os.path.join(pdir, f"kolmo__switch_{g}_{v}.npz")
    if not os.path.exists(f): return None
    z = np.load(f, allow_pickle=False); meta = json.loads(str(z["meta"])); n = len(meta["windows"])
    return meta, [z[f"P{i}"].astype(np.float32) for i in range(n)], [z[f"T{i}"].astype(np.float32) for i in range(n)]


def truth(w, st, n):
    t0 = w["first_frame"] + EVAL_HIST * st
    with h5py.File(w["path"], "r") as h:
        g, c = h[w["traj"]], h["c" + w["traj"]]
        S = np.moveaxis(np.asarray(g["data"][t0:t0 + n * st:st], dtype=np.float32), -1, 1)
        C = np.moveaxis(np.asarray(c["data"][t0:t0 + n * st:st], dtype=np.float32), -1, 1)
        fA = np.asarray(g["forcing_A"], dtype=np.float32); fs = np.asarray(g["forcing_seg"], dtype=np.float32)
        seq = json.loads(h.attrs["seq"]); ss = int(h.attrs["seg_steps"])
    return S, C, fA, fs, seq, ss


def seg_of(m, ss): return (m - 1) // ss   # model step m (1-based) -> segment index


def short(c):
    if c.get("forcing") == "multi": s = f"multi-mode |k| {c['kband'][0]}–{c['kband'][1]}"
    else: s = f"k_f {c['kf']:g}"
    return f"{s}, A {c['amp']:g}, ν {c['nu']:.1e}"


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--name", required=True); ap.add_argument("--eval", required=True)
    ap.add_argument("--groups", default="ms_kf,ms_action,ms_mixed,ms_amp,ms_nu"); ap.add_argument("--top", type=int, default=2)
    ap.add_argument("--after", default="4"); ap.add_argument("--gif", action="store_true")
    a = ap.parse_args()
    pdir = os.path.join(HERE, "results", "pred", a.eval); od = os.path.join(HERE, "results", "case", a.name); os.makedirs(od, exist_ok=True)
    after = [int(x) for x in a.after.split(",") if x]; out = {"name": a.name, "eval": a.eval, "groups": {}}
    for g in [x for x in a.groups.split(",") if x]:
        vS, vU = load(pdir, g, "S0"), load(pdir, g, "U")
        if vS is None or vU is None: print(f"[ms_case] {g}: no dumps, skipped", flush=True); continue
        meta, PS, TS = vS; mU, PU, _ = vU; st = meta["stride"]
        keyU = {(w["path"], w["traj"], int(w["first_frame"])): i for i, w in enumerate(mU["windows"])}
        rec = []
        for i, w in enumerate(meta["windows"]):
            j = keyU.get((w["path"], w["traj"], int(w["first_frame"])))
            if j is None: continue
            n = PS[i].shape[0]; S, C, fA, fs, seq, ss = truth(w, st, n)
            assert np.abs(S - TS[i]).max() <= 1e-2 * max(np.abs(S).max(), 1e-6), f"{g} window {i}: dumped truth != file"
            eT = [nrm(PS[i][m] - S[m]) / nrm(S[m]) for m in range(n)]; eN = [nrm(PU[j][m] - S[m]) / nrm(S[m]) for m in range(n)]
            rec.append({"window": i, "u_index": j, "traj": w["traj"], "err_told_mean": float(np.mean(eT)), "err_not_told_mean": float(np.mean(eN)),
                        "err_told_last": eT[-1], "err_not_told_last": eN[-1], "gap": float(np.mean(eN) - np.mean(eT))})
        picks = sorted(rec, key=lambda x: -x["gap"])[:a.top]
        for x in picks:
            i, j = x["window"], x["u_index"]; w = meta["windows"][i]; n = PS[i].shape[0]; S, C, fA, fs, seq, ss = truth(w, st, n)
            K = min(len(seq) - 1, (n + ss - 1) // ss); T_, N_ = PS[i], PU[j]
            cols = []
            for k in range(K):
                for d in after:
                    if k * ss + d <= n: cols.append(k * ss + d)
                if (k + 1) * ss <= n: cols.append((k + 1) * ss)
            cols = sorted(set(cols)); lim = max(np.percentile(np.abs(S[:, 0]), 99.5), 1e-6)
            frc = lambda m: fs[min(seg_of(m, ss), fs.shape[0] - 1)]
            vf = float(max(np.abs(fs).max(), np.abs(fA).max(), 1e-12))
            rows = [("forcing in effect", None), ("truth, conditions change", S), ("ours, told each change", T_), ("ours, not told", N_), ("truth, first condition kept", C)]
            fig, ax = plt.subplots(len(rows), len(cols), figsize=(1.7 * len(cols), 1.75 * len(rows)), squeeze=False)
            for r_, (nm, X) in enumerate(rows):
                for c_, m in enumerate(cols):
                    A = ax[r_, c_]; A.set_xticks([]); A.set_yticks([])
                    if X is None: A.imshow(frc(m).T, cmap="PuOr", vmin=-vf, vmax=vf, origin="lower")
                    else: A.imshow(X[m - 1, 0].T, cmap="RdBu_r", vmin=-lim, vmax=lim, origin="lower", interpolation="nearest")
                    if r_ == 0: A.set_title(f"step {m}\n(segment {seg_of(m, ss) + 1})", fontsize=7)
                ax[r_, 0].set_ylabel(nm, fontsize=7)
            fig.suptitle(f"Kolmogorov / {g}, window {i}: " + " → ".join(short(c) for c in seq[:K + 1]) +
                         f"; mean error to the truth: told {x['err_told_mean']:.2f}, not told {x['err_not_told_mean']:.2f}", fontsize=7)
            fig.tight_layout(rect=(0, 0, 1, 0.95)); base = os.path.join(od, f"{g}_w{i:02d}")
            for e in ("png", "pdf"): fig.savefig(f"{base}.{e}", dpi=150)
            plt.close(fig)
            # curve: projection on the current forcing (or mean square when the forcing pattern never changes)
            pat = len({json.dumps({k: v for k, v in c.items() if k in ("kf", "forcing", "kband")}, sort_keys=True) for c in seq}) > 1
            def q(X, m):
                if pat: f = frc(m); return float((X[m - 1, 0].astype(np.float64) * f).sum()) / max(nrm(f) * nrm(S[m - 1, 0]), 1e-12)
                return float((X[m - 1, 0].astype(np.float64) ** 2).mean())
            ms_ = np.arange(1, n + 1); fig, axc = plt.subplots(figsize=(6.5, 2.6))
            for X, lab, kw in ((S, "truth, conditions change", dict(color="k", lw=1.4)), (T_, "ours, told", dict(color="C1", lw=1.8)),
                               (N_, "ours, not told", dict(color="C0", ls="--", lw=1.2)), (C, "truth, first condition kept", dict(color="0.6", lw=1.2))):
                axc.plot(ms_, [q(X, m) for m in ms_], label=lab, **kw)
            for k in range(1, K): axc.axvline(k * ss + 0.5, color="0.8", lw=0.8)
            axc.set_xlabel("model step"); axc.set_ylabel("projection on the current forcing" if pat else "mean square of the field", fontsize=8)
            axc.legend(fontsize=7, ncol=2); axc.set_title(f"{g}, window {i}", fontsize=8); fig.tight_layout()
            for e in ("png", "pdf"): fig.savefig(f"{base}_curve.{e}", dpi=150)
            plt.close(fig)
            if a.gif:
                from matplotlib.animation import PillowWriter
                fig, axg = plt.subplots(1, 4, figsize=(10.4, 3.0)); ims = []
                for k_, (nm, X) in enumerate((("truth, conditions change", S), ("ours, told each change", T_), ("ours, not told", N_))):
                    axg[k_].set_xticks([]); axg[k_].set_yticks([]); axg[k_].set_title(nm, fontsize=9); ims.append(axg[k_].imshow(X[0, 0].T, cmap="RdBu_r", vmin=-lim, vmax=lim, origin="lower"))
                axg[3].set_xticks([]); axg[3].set_yticks([]); axg[3].set_title("forcing in effect", fontsize=9); imf = axg[3].imshow(frc(1).T, cmap="PuOr", vmin=-vf, vmax=vf, origin="lower")
                txt = fig.suptitle("", fontsize=10); fig.tight_layout(rect=(0, 0, 1, 0.9)); wr = PillowWriter(fps=4)
                with wr.saving(fig, f"{base}_anim.gif", dpi=80):
                    for m in range(1, n + 1):
                        for im, X in zip(ims, (S, T_, N_)): im.set_data(X[m - 1, 0].T)
                        imf.set_data(frc(m).T); txt.set_text(f"step {m}, segment {seg_of(m, ss) + 1}: {short(seq[seg_of(m, ss) + 1])}"); wr.grab_frame()
                plt.close(fig)
        out["groups"][g] = {"windows": rec, "picked": [x["window"] for x in picks]}
        print(f"[ms_case] {g}: {len(rec)} windows, drawn {[x['window'] for x in picks]}", flush=True)
    json.dump(out, open(os.path.join(HERE, "results", a.name + ".json"), "w"), indent=1); print("saved", flush=True)


if __name__ == "__main__":
    main()
