"""Case-study figures of single condition switches, CPU.

Reads the prediction dumps of a switch evaluation (eval_switch.py / eval_switch.py --dump_pred: results/pred/<eval>/<family>__switch_<group>_<view>.npz,
views B = told the new condition, A = not told) and the group files (switched trajectory '<traj>', control 'c<traj>' with the old condition
kept, forcing fields, switch_frame).
Per window and model step m: effect e = ||S - C|| / ||S||, err_S(P) = ||P - S|| / ||S||, err_C(P) = ||P - C|| / ||S||, response
r = <P - C, S - C> / ||S - C||^2 (0 = stays on the old condition's future, 1 = follows the switch).
A window is a case when, at the last step: e >= --min_effect, B is closer to the switched truth than to the control
(err_S(B) <= --margin * err_C(B)), and A stays closer to the control than to the switched truth (err_S(A) > err_C(A), which is the
same condition as r(A) < 0.5). Cases are ranked by e * (err_C(B) - err_S(B)); the
top --top per group are drawn, and every window's numbers go to the JSON.
Figures: per case, rows = switched truth, control truth, told, not told, true change S - C, predicted change B - C; optional comparison
with other models (--compare), mosaic of all windows (--mosaic), animation (--gif), single-frame export (--export); per group a curve of
median errors vs step. Writes results/<name>.json and results/case/<name>/."""
import argparse, glob, json, os
import numpy as np, h5py
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))
EVAL_HIST = 6   # history frames before the prediction start, as in analyze_switch.py


def nrm(x): return float(np.sqrt((x.astype(np.float64) ** 2).sum()))


def load_view(pdir, fam, g, v):
    f = os.path.join(pdir, f"{fam}__switch_{g}_{v}.npz")
    if not os.path.exists(f): return None
    z = np.load(f, allow_pickle=False); meta = json.loads(str(z["meta"]))
    return meta, [z[f"P{i}"].astype(np.float32) for i in range(len(meta["windows"]))], [z[f"T{i}"].astype(np.float32) for i in range(len(meta["windows"]))]


def frames(w, st, n):
    """history frame at the switch-side end of the history, control and switched truth at the predicted frames, forcing before / after"""
    t0 = w["first_frame"] + EVAL_HIST * st; idx = [t0 + m * st for m in range(n)]
    with h5py.File(w["path"], "r") as h:
        g, c = h[w["traj"]], h["c" + w["traj"]]
        S = np.moveaxis(np.asarray(g["data"][idx[0]:idx[-1] + 1:st], dtype=np.float32), -1, 1)
        C = np.moveaxis(np.asarray(c["data"][idx[0]:idx[-1] + 1:st], dtype=np.float32), -1, 1)
        H = np.moveaxis(np.asarray(g["data"][t0 - st], dtype=np.float32), -1, 0)   # last history frame
        H1 = np.moveaxis(np.asarray(g["data"][t0 - 2 * st], dtype=np.float32), -1, 0)   # the one before (the model reads two)
        fb = np.asarray(g["forcing"], dtype=np.float32) if "forcing" in g else None
        fa = np.asarray(g["forcing_A"], dtype=np.float32) if "forcing_A" in g else None
        attrs = {k: ((v.item() if np.size(v) == 1 else np.asarray(v).tolist()) if hasattr(v, "item") else (v.decode() if isinstance(v, bytes) else v)) for k, v in h.attrs.items()}
        sw = int(h.attrs.get("switch_frame", -1))
    return H, S, C, fb, fa, attrs, sw, t0, H1


def label(attrs):
    keys = sorted({k[2:] for k in attrs if k.startswith(("A_", "B_"))} - {"dt_typical", "t_eddy_ref"})
    attrs = {**{f"A_{k}": "-" for k in keys}, **{f"B_{k}": "-" for k in keys}, **attrs}   # a quantity set on one side only (e.g. B_kband)
    ch = [f"{k}: {attrs['A_' + k]:g} → {attrs['B_' + k]:g}" if all(isinstance(attrs[x + k], (int, float)) for x in ("A_", "B_")) else f"{k}: {attrs['A_' + k]} → {attrs['B_' + k]}"
          for k in keys if str(attrs["A_" + k]) != str(attrs["B_" + k])]
    return ", ".join(ch) or attrs.get("what", "?")


def draw(path, fam, g, wi, ch, H, S, C, PB, PA, fb, fa, steps, st, attrs, sw, t0, mets):
    cols = [None] + [m for m in steps if m <= S.shape[0]]
    rows = [("switched truth", S), ("control truth (old condition)", C), ("ours, told the new condition", PB), ("ours, not told", PA),
            ("true change  S − C", S - C), ("predicted change  B − C", PB - C)]
    lim = max(np.percentile(np.abs(np.concatenate([S[:, ch].ravel(), C[:, ch].ravel()])), 99.5), 1e-6)
    dlim = max(np.percentile(np.abs((S - C)[:, ch].ravel()), 99.5), 1e-6)
    nf = 1 if fb is None else 2
    fig, ax = plt.subplots(len(rows), len(cols) + nf, figsize=(1.9 * (len(cols) + nf), 1.9 * len(rows)), squeeze=False)
    for i, (nm, X) in enumerate(rows):
        L = dlim if i >= 4 else lim
        for j, m in enumerate(cols):
            a = ax[i, j]; a.set_xticks([]); a.set_yticks([])
            img = (H[ch] if i in (0, 1) else None) if m is None else X[m - 1, ch]
            if img is not None: a.imshow(img, cmap="RdBu_r", vmin=-L, vmax=L, origin="lower", interpolation="nearest")
            else: a.axis("off")
            if i == 0: a.set_title("last history (switch)" if m is None else f"step {m}  (frame +{(m) * st})", fontsize=8)
        for k in range(nf):
            a = ax[i, len(cols) + k]; a.axis("off")
        ax[i, 1].set_ylabel(nm, fontsize=7)
    if fb is not None:
        for k, (F, t) in enumerate(((fa, "forcing before"), (fb, "forcing after"))):
            if F is None: continue
            a = ax[0, len(cols) + k]; a.axis("on"); a.set_xticks([]); a.set_yticks([])
            F2 = F[..., 0] if F.ndim == 3 else F; a.imshow(F2, cmap="PuOr", origin="lower"); a.set_title(t, fontsize=8)
    m_ = mets
    fig.suptitle(f"{fam} / {g}  window {wi} ({label(attrs)}); step {cols[-1]}: effect {m_['e']:.2f}, "
                 f"err to switched truth B {m_['errS_B']:.2f} / A {m_['errS_A']:.2f}, response r B {m_['r_B']:.2f} / A {m_['r_A']:.2f}", fontsize=8)
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    for ext in ("png", "pdf"): fig.savefig(f"{path}.{ext}", dpi=150)
    plt.close(fig)


def export(od, fam, g, wi, ch, H1, H, S, C, PB, PA, fb, fa, mets):
    """single-frame PNGs: hist1, hist2 (the two history frames, shared),
    A_stepNN = control truth (old condition), B_stepNN = switched truth, P_stepNN = ours told the new condition, N_stepNN = ours not told,
    forcing_A / forcing_B. Kolmogorov arrays are [x, y] and are transposed (y vertical, increasing upwards);
    one colour scale per sequence (99.5th percentile of |control|), forcing on its own scale."""
    tr = (lambda z: z.T) if fam == "kolmo" else (lambda z: z)
    v = max(float(np.percentile(np.abs(C[:, ch]), 99.5)), 1e-6)
    for cm in ("RdBu_r", "coolwarm", "PuOr_r", "twilight_shifted"):
        d = os.path.join(od, f"{fam}_{g}_w{wi:02d}_{cm}"); os.makedirs(d, exist_ok=True)
        sv = lambda n, z, L=v, c=cm: plt.imsave(os.path.join(d, n + ".png"), tr(z), vmin=-L, vmax=L, cmap=c, origin="lower")
        sv("hist1", H1[ch]); sv("hist2", H[ch])
        for m in range(1, S.shape[0] + 1):
            sv(f"A_step{m:02d}", C[m - 1, ch]); sv(f"B_step{m:02d}", S[m - 1, ch]); sv(f"P_step{m:02d}", PB[m - 1, ch]); sv(f"N_step{m:02d}", PA[m - 1, ch])
        dl = max(float(np.percentile(np.abs((S - C)[:, ch]), 99.5)), 1e-6)
        for m in range(1, S.shape[0] + 1):
            sv(f"dTrue_step{m:02d}", (S - C)[m - 1, ch], dl); sv(f"dPred_step{m:02d}", (PB - C)[m - 1, ch], dl)
        if fb is not None:
            vf = float(max(np.abs(fb).max(), np.abs(fa).max() if fa is not None else 0, 1e-12))
            for n, F in (("forcing_A", fa), ("forcing_B", fb)):
                if F is not None: sv(n, F[..., 0] if F.ndim == 3 else F, vf, "BrBG")
        json.dump(mets, open(os.path.join(d, "metrics.json"), "w"), indent=1)


def load_compare(spec, fam, g):
    """--compare 'eval:label,eval:label': the told (B) predictions of other models on the same group, keyed by (path, traj, first_frame)."""
    out = []
    for item in [x for x in spec.split(",") if x]:
        ev, lab = item.split(":", 1); v = load_view(os.path.join(HERE, "results", "pred", ev), fam, g, "B")
        if v is None: print(f"[compare] {ev}: no {g} dump, skipped", flush=True); continue
        meta, Ps, _ = v; out.append((lab.replace("_", " "), {(w["path"], w["traj"], int(w["first_frame"])): Ps[i] for i, w in enumerate(meta["windows"])}))
    return out


def resp(P, S, C):
    d = S[-1] - C[-1]; dd = float((d.astype(np.float64) ** 2).sum())
    return (float(((P[-1] - C[-1]).astype(np.float64) * d).sum()) / dd if dd > 0 else float("nan")), nrm(P[-1] - S[-1]) / nrm(S[-1])


def draw_compare(path, fam, g, wi, ch, H, S, C, PB, others, fb, fa, steps, st, attrs, ours_label):
    cols = [m for m in steps if m <= S.shape[0]]
    rows = [("switched truth (new condition)", S, None), ("control truth (old condition)", C, None), (ours_label, PB, resp(PB, S, C))]
    rows += [(lab, P, resp(P, S, C)) for lab, P in others]
    lim = max(np.percentile(np.abs(np.concatenate([S[:, ch].ravel(), C[:, ch].ravel()])), 99.5), 1e-6)
    nf = 0 if fb is None else 2
    fig, ax = plt.subplots(len(rows), len(cols) + nf, figsize=(1.9 * (len(cols) + nf), 1.9 * len(rows)), squeeze=False)
    for i, (nm, X, rm) in enumerate(rows):
        for j, m in enumerate(cols):
            a = ax[i, j]; a.set_xticks([]); a.set_yticks([]); a.imshow(X[m - 1, ch], cmap="RdBu_r", vmin=-lim, vmax=lim, origin="lower", interpolation="nearest")
            if i == 0: a.set_title(f"step {m}", fontsize=8)
        import textwrap; ax[i, 0].set_ylabel(textwrap.fill(nm, 24) + ("" if rm is None else f"\nr = {rm[0]:.2f}, err = {rm[1]:.2f}"), fontsize=7)
        for k in range(nf): ax[i, len(cols) + k].axis("off")
    if fb is not None:
        for k, (F, tt) in enumerate(((fa, "forcing before"), (fb, "forcing after"))):
            if F is None: continue
            a = ax[0, len(cols) + k]; a.axis("on"); a.set_xticks([]); a.set_yticks([]); a.imshow(F[..., 0] if F.ndim == 3 else F, cmap="PuOr", origin="lower"); a.set_title(tt, fontsize=8)
    fig.suptitle(f"{fam} / {g}  window {wi} ({label(attrs)}): every model is told the new condition; r = response at step {cols[-1]} "
                 f"(0 = old future, 1 = new future), err = relative L2 to the switched truth", fontsize=8)
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    for ext in ("png", "pdf"): fig.savefig(f"{path}.{ext}", dpi=150)
    plt.close(fig)


def draw_mosaic(path, fam, g, items, ch, last):
    """all windows of a group at the last step: switched truth / ours told / ours not told / control truth, one column per window."""
    for part in range(0, len(items), 16):
        it = items[part:part + 16]
        fig, ax = plt.subplots(4, len(it), figsize=(1.25 * len(it), 5.2), squeeze=False)
        for j, (wi, ok, S, C, PB, PA) in enumerate(it):
            lim = max(np.percentile(np.abs(np.concatenate([S[last, ch].ravel(), C[last, ch].ravel()])), 99.5), 1e-6)
            for i, X in enumerate((S, PB, PA, C)):
                a = ax[i, j]; a.set_xticks([]); a.set_yticks([]); a.imshow(X[last, ch], cmap="RdBu_r", vmin=-lim, vmax=lim, origin="lower", interpolation="nearest")
                if not ok:
                    for sp in a.spines.values(): sp.set_color("0.6"); sp.set_linestyle(":")
            ax[0, j].set_title(f"w{wi}" + ("" if ok else " (–)"), fontsize=7)
        for i, nm in enumerate(("switched truth", "ours, told", "ours, not told", "control truth")): ax[i, 0].set_ylabel(nm, fontsize=7)
        fig.suptitle(f"{fam} / {g}: step {last + 1}, windows {it[0][0]}–{it[-1][0]} ((–) = does not qualify under the rule)", fontsize=8)
        fig.tight_layout(rect=(0, 0, 1, 0.95))
        for ext in ("png", "pdf"): fig.savefig(f"{path}_part{part // 16 + 1}.{ext}", dpi=150)
        plt.close(fig)


def draw_gif(path, fam, g, wi, ch, S, C, PB, PA, fb, fa):
    """animation over the model steps: control truth | switched truth | ours told | ours not told (+ forcing before / after)."""
    from matplotlib.animation import PillowWriter
    lim = max(np.percentile(np.abs(np.concatenate([S[:, ch].ravel(), C[:, ch].ravel()])), 99.5), 1e-6)
    panels = [("old condition (solver)", C), ("new condition (solver)", S), ("ours, told the new condition", PB), ("ours, not told", PA)]
    nf = 0 if fb is None else 2
    fig, ax = plt.subplots(1, 4 + nf, figsize=(2.6 * (4 + nf), 3.0))
    ims = []
    for k, (nm, X) in enumerate(panels):
        ax[k].set_xticks([]); ax[k].set_yticks([]); ax[k].set_title(nm, fontsize=9); ims.append(ax[k].imshow(X[0, ch], cmap="RdBu_r", vmin=-lim, vmax=lim, origin="lower", interpolation="nearest"))
    if fb is not None:
        for k, (F, tt) in enumerate(((fa, "forcing before"), (fb, "forcing after"))):
            a = ax[4 + k]; a.set_xticks([]); a.set_yticks([]); a.set_title(tt, fontsize=9)
            if F is not None: a.imshow(F[..., 0] if F.ndim == 3 else F, cmap="PuOr", origin="lower")
    txt = fig.suptitle("", fontsize=10); fig.tight_layout(rect=(0, 0, 1, 0.92))
    w = PillowWriter(fps=3)
    with w.saving(fig, f"{path}.gif", dpi=90):
        for m in range(S.shape[0]):
            for im, (_, X) in zip(ims, panels): im.set_data(X[m, ch])
            txt.set_text(f"{fam} / {g}, window {wi}: step {m + 1} after the change"); w.grab_frame()
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--name", required=True); ap.add_argument("--eval", required=True); ap.add_argument("--family", default="kolmo", choices=["kolmo"])
    ap.add_argument("--groups", default=""); ap.add_argument("--top", type=int, default=3); ap.add_argument("--min_effect", type=float, default=0.25)
    ap.add_argument("--margin", type=float, default=0.6); ap.add_argument("--steps", default="1,4,8,12,16"); ap.add_argument("--export", action="store_true")
    ap.add_argument("--compare", default="", help="eval:label,... other models (told) on the same windows, one extra figure per picked window")
    ap.add_argument("--ours_label", default="ours, switch-trained"); ap.add_argument("--mosaic", action="store_true", help="all windows of a group at the last step")
    ap.add_argument("--gif", action="store_true", help="an animation per picked window")
    a = ap.parse_args()
    pdir = os.path.join(HERE, "results", "pred", a.eval); od = os.path.join(HERE, "results", "case", a.name); os.makedirs(od, exist_ok=True)
    steps = [int(x) for x in a.steps.split(",")]
    groups = [x for x in a.groups.split(",") if x] or sorted({os.path.basename(f)[len(a.family) + 6:].rsplit("_", 1)[0]
                                                             for f in glob.glob(os.path.join(pdir, f"{a.family}__switch_*_B.npz"))})
    out = {"name": a.name, "eval": a.eval, "family": a.family, "rule": {"min_effect": a.min_effect, "margin": a.margin, "top": a.top}, "groups": {}}
    for g in groups:
        vb, va = load_view(pdir, a.family, g, "B"), load_view(pdir, a.family, g, "A")
        if vb is None or va is None: continue
        meta, PBs, TBs = vb; _, PAs, _ = va; st = meta["stride"]; chs = meta["channels"]; ch = 0
        rec, curves = [], {"errS_B": [], "errC_B": [], "errS_A": [], "errC_A": []}
        for i, w in enumerate(meta["windows"]):
            PB, PA, T = PBs[i], PAs[i], TBs[i]; n = PB.shape[0]
            H, S, C, fb, fa, attrs, sw, t0, _ = frames(w, st, n)
            assert np.abs(S - T).max() <= 1e-2 * max(np.abs(S).max(), 1e-6), f"{g} window {i}: dumped truth != file frames"
            cur = {k: [] for k in curves}
            for m in range(n):
                s, c, ns = S[m], C[m], nrm(S[m])
                for v, P in (("B", PB), ("A", PA)): cur[f"errS_{v}"].append(nrm(P[m] - s) / ns); cur[f"errC_{v}"].append(nrm(P[m] - c) / ns)
            for k in curves: curves[k].append(cur[k])
            s, c = S[-1], C[-1]; d = s - c; dd = float((d.astype(np.float64) ** 2).sum()); ns = nrm(s)
            r = lambda P: float(((P[-1] - c).astype(np.float64) * d).sum()) / dd if dd > 0 else float("nan")
            m_ = {"window": i, "traj": w["traj"], "e": nrm(d) / ns, "errS_B": cur["errS_B"][-1], "errC_B": cur["errC_B"][-1], "errS_A": cur["errS_A"][-1],
                  "errC_A": cur["errC_A"][-1], "r_B": r(PB), "r_A": r(PA), "switch": label(attrs), "switch_frame": sw, "pred_start_frame": t0}
            m_["case"] = bool(m_["e"] >= a.min_effect and m_["errS_B"] <= a.margin * m_["errC_B"] and m_["errS_A"] > m_["errC_A"])
            m_["score"] = m_["e"] * (m_["errC_B"] - m_["errS_B"]); rec.append(m_)
        picks = sorted([x for x in rec if x["case"]], key=lambda x: -x["score"])[:a.top]
        cmp_ = load_compare(a.compare, a.family, g) if a.compare else []
        if a.mosaic:
            items = []
            for x in rec:
                i = x["window"]; H, S, C, *_ = frames(meta["windows"][i], st, PBs[i].shape[0]); items.append((i, x["case"], S, C, PBs[i], PAs[i]))
            draw_mosaic(os.path.join(od, f"{a.family}_{g}_mosaic"), a.family, g, items, ch, PBs[0].shape[0] - 1)
        for x in picks:
            i = x["window"]; w = meta["windows"][i]; H, S, C, fb, fa, attrs, sw, t0, H1 = frames(w, st, PBs[i].shape[0])
            key = (w["path"], w["traj"], int(w["first_frame"]))
            if cmp_:
                oth = [(lab, d[key]) for lab, d in cmp_ if key in d]
                x["compare"] = {lab: dict(zip(("r", "err"), resp(P, S, C))) for lab, P in oth}
                draw_compare(os.path.join(od, f"{a.family}_{g}_w{i:02d}_compare"), a.family, g, i, ch, H, S, C, PBs[i], oth, fb, fa, steps, st, attrs, a.ours_label)
            if a.gif: draw_gif(os.path.join(od, f"{a.family}_{g}_w{i:02d}_anim"), a.family, g, i, ch, S, C, PBs[i], PAs[i], fb, fa)
            if a.export: export(os.path.join(od, "frames"), a.family, g, i, ch, H1, H, S, C, PBs[i], PAs[i], fb, fa, x)
            draw(os.path.join(od, f"{a.family}_{g}_w{i:02d}"), a.family, g, i, ch, H, S, C, PBs[i], PAs[i], fb, fa, steps, st, attrs, sw, t0, x)
        if rec:
            fig, axc = plt.subplots(figsize=(4, 3)); xs = np.arange(1, len(curves["errS_B"][0]) + 1)
            for k, sty in (("errS_B", "C0-"), ("errC_B", "C0--"), ("errS_A", "C3-"), ("errC_A", "C3--")):
                axc.plot(xs, np.median(np.array(curves[k]), 0), sty, label={"errS_B": "told: to switched truth", "errC_B": "told: to control",
                                                                           "errS_A": "not told: to switched truth", "errC_A": "not told: to control"}[k])
            axc.set_xlabel("model step after the switch"); axc.set_ylabel("relative L2 (median over windows)"); axc.set_title(f"{a.family} / {g}", fontsize=9); axc.legend(fontsize=6)
            fig.tight_layout(); [fig.savefig(os.path.join(od, f"{a.family}_{g}_curves.{e}"), dpi=150) for e in ("png", "pdf")]; plt.close(fig)
        out["groups"][g] = {"channel": chs[ch], "n_windows": len(rec), "n_cases": sum(x["case"] for x in rec), "picked": [x["window"] for x in picks], "windows": rec}
        print(f"[case] {g}: {sum(x['case'] for x in rec)} / {len(rec)} windows qualify; drawn {[x['window'] for x in picks]}", flush=True)
    json.dump(out, open(os.path.join(HERE, "results", a.name + ".json"), "w"), indent=1); print("saved", os.path.join(HERE, "results", a.name + ".json"), flush=True)


if __name__ == "__main__":
    main()
