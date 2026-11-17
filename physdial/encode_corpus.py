#!/usr/bin/env python
"""Encode the training part of the corpus into the latent space of the Wan2.1 VAE once, so training reads latents instead of running
the tokenizer on every step (standard for video diffusion models).

Every frame and physical channel is normalised on its own, x_n = (x - mu) / s with s = max |x - mu| (fieldnorm, as in tok_gate), and
encoded with the tokenizer on its own grid. Stored per family / split / grid-size group:
  z_<g>.npy   float16 memmap [N_traj, T, C, zc, h, w]   latents (frames t >= family.t_start; earlier frames are zeros)
  st_<g>.npy  float32 [N_traj, T, C, 2]                  (mu, s) per frame and channel, so x = mu + s * decode(z)
  fz_<g>.npy  float16 [N_traj, zc, h, w] (one field) or [N_traj, nf, zc, h, w]   latents of the condition fields (corpus.Family.cond_model:
              a fixed per-family transform, no per-frame normalisation)
  index.json  items (path, traj, params, raw, n_frames, shape, group, row), tokenizer spec, latent shapes
Splits: train, val and every ref_* split. Evaluation splits are not stored: evaluation encodes its few history frames itself.
Resumable per group (progress file); output under $PHYSDIAL_DATA_ROOT/latents/<--store>/<family>/<split>/. train_gen refuses a store whose
index disagrees with the corpus.
--vae_wrap: wrap-around padding of the VAE on periodic axes (the store of the periodic handling).
--kolmo_extra anchor,drag,beta: switches the optional Kolmogorov groups on (corpus.set_kolmo_extra) and restricts this call to the splits the
switch adds (anchor, extrap_drag_*, beta_anchor, ...), e.g. --splits anchor into an existing store: every other split directory of the store
is left untouched."""
import argparse, contextlib, json, os, sys, time
HERE = os.path.dirname(os.path.abspath(__file__)); PROJ = os.path.dirname(HERE); sys.path.insert(0, PROJ); sys.path.insert(0, HERE)
import numpy as np, torch
import runmeta, corpus, tok2d, fieldnorm, vae_periodic


def store_root(store): return os.path.join(os.environ.get("PHYSDIAL_DATA_ROOT") or os.path.join(PROJ, "data"), "latents", store)


def own_splits(a, k, fam):
    """the splits of family k this call may write: all of them, or with --kolmo_extra only the Kolmogorov splits the switch adds."""
    return [s for s in fam.splits() if k == "kolmo" and s in corpus.kolmo_extra_splits()] if a.kolmo_extra else fam.splits()


def wrap_ctx(tok, periodic, wrap):
    return vae_periodic.periodic(tok.vae, periodic) if wrap and any(periodic) else contextlib.nullcontext()


@torch.no_grad()
def encode_frames(tok, X, periodic, bs, wrap=False):
    """X [N, H, W] float32 numpy -> latents [N, zc, h, w] float16, stats [N, 2] (mu, s)."""
    x = torch.from_numpy(X).cuda().float(); mu, s = fieldnorm.scale_t(x)
    xn = (x - mu[:, None, None]) / s[:, None, None]; out = []
    for i in range(0, len(xn), bs):
        with torch.autocast("cuda", dtype=torch.bfloat16), wrap_ctx(tok, periodic, wrap): out.append(tok.encode(xn[i:i + bs, None]).float())
    return torch.cat(out).half().cpu().numpy(), torch.stack([mu, s], -1).cpu().numpy()


@torch.no_grad()
def encode_cond(tok, fam, it, wrap=False):
    """condition fields of one trajectory -> latents [nf, zc, h, w] float16 (corpus.Family.cond_model)."""
    out = []
    for f in fam.cond_model(it):
        x = torch.from_numpy(np.asarray(f, np.float32)).cuda()[None]
        with torch.autocast("cuda", dtype=torch.bfloat16), wrap_ctx(tok, fam.periodic, wrap): out.append(tok.encode(x[:, None]).float().half().cpu().numpy()[0])
    return out


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--name", required=True); ap.add_argument("--tok", required=True); ap.add_argument("--store", required=True)
    ap.add_argument("--families", required=True); ap.add_argument("--bs", type=int, default=64); ap.add_argument("--splits", default="train,val,ref", help="comma list; ref = every ref_* split")
    ap.add_argument("--vae_wrap", action="store_true", help="wrap-around padding of the Wan VAE on the periodic axes; "
                    "recorded in every index.json, train_gen encodes / decodes with the store's setting")
    ap.add_argument("--kolmo_extra", default="", help="comma list of anchor, drag, beta (corpus.set_kolmo_extra); this call then writes only the Kolmogorov splits the switch adds")
    a = ap.parse_args(); corpus.set_kolmo_extra(a.kolmo_extra)
    out = os.path.join(HERE, "results", a.name + ".json"); os.makedirs(os.path.dirname(out), exist_ok=True)
    if os.path.exists(out): print("exists", out); sys.exit(0)
    tok = tok2d.load(a.tok).to(torch.bfloat16); t0 = time.time(); summary = {}
    for k in a.families.split(","):
        fam = corpus.get(k); want = [s for s in own_splits(a, k, fam) if s in a.splits.split(",") or ("ref" in a.splits.split(",") and s.startswith("ref_"))]
        if a.kolmo_extra: print(f"[{a.name}] --kolmo_extra {a.kolmo_extra}: {k} splits written: {want or 'none'} (existing groups untouched)", flush=True)
        for sp in want:
            its = corpus.cached_items(k, sp)
            if not its: continue
            d = os.path.join(store_root(a.store), k, sp); os.makedirs(d, exist_ok=True)
            if os.path.islink(os.path.join(store_root(a.store), k)) or os.path.realpath(d) != os.path.abspath(d): print(f"[{a.name}] {k}/{sp}: {d} is written through a link, real path {os.path.realpath(d)}", flush=True)
            groups = {}
            for it in its: groups.setdefault(tuple(it.shape), []).append(it)
            index = {"tok": a.tok, "family": k, "split": sp, "channels": fam.channels, "groups": {}, "items": [], "vae_wrap": a.vae_wrap}
            old_ix = json.load(open(os.path.join(d, "index.json"))) if os.path.exists(os.path.join(d, "index.json")) else None
            assert old_ix is None or old_ix.get("vae_wrap", False) == a.vae_wrap, f"{k}/{sp}: the existing store was encoded with vae_wrap={old_ix.get('vae_wrap', False)}"
            for gi, (shape, git) in enumerate(sorted(groups.items())):
                T = max(it.n_frames for it in git); C = len(fam.channels)
                h, w = shape[0] // tok.factor, shape[1] // tok.factor; N = len(git)
                zf, sf, ff, pf = (os.path.join(d, f"{p}_{gi}.npy") for p in ("z", "st", "fz", "progress"))
                mode = "r+" if os.path.exists(zf) else "w+"
                nf = len(fam.cond_fields(git[0])); has_f = nf > 0   # condition fields (Kolmogorov: the forcing; acoustic: density + sound speed)
                rows = [[it.path, it.traj] for it in git]; rf = os.path.join(d, f"rows_{gi}.json")
                if mode == "r+":   # an existing store is only resumed if it is the same group (shape, trajectories in the same rows); open_memmap
                                   # r+ ignores shape, and rows below the progress count were skipped by position whatever they held
                    prev = json.load(open(rf)) if os.path.exists(rf) else ([[e["path"], e["traj"]] for e in old_ix["items"] if e["group"] == gi] if old_ix else None)
                    assert prev == rows, f"{k}/{sp} group {gi}: the existing store holds other trajectories / rows; use a new --store"
                else: json.dump(rows, open(rf, "w"))
                mf = os.path.join(d, f"meta_{gi}.json"); m_ = {"vae_wrap": a.vae_wrap}
                if mode == "r+" and os.path.exists(mf): assert json.load(open(mf)) == m_, f"{k}/{sp} group {gi}: resumed with {m_}, encoded with {json.load(open(mf))}"
                elif mode == "w+": json.dump(m_, open(mf, "w"))
                Z = np.lib.format.open_memmap(zf, mode=mode, dtype=np.float16, shape=(N, T, C, tok.zc, h, w))
                S = np.lib.format.open_memmap(sf, mode=mode, dtype=np.float32, shape=(N, T, C, 2))
                Fz = np.lib.format.open_memmap(ff, mode=mode, dtype=np.float16, shape=(N, tok.zc, h, w) if nf == 1 else (N, nf, tok.zc, h, w)) if has_f else None
                assert Z.shape == (N, T, C, tok.zc, h, w) and S.shape == (N, T, C, 2) and (Fz is None or Fz.shape[0] == N), f"{k}/{sp} group {gi}: existing arrays have another shape"
                done = int(open(pf).read()) if os.path.exists(pf) else 0
                index["groups"][str(gi)] = {"shape": list(shape), "latent": [tok.zc, h, w], "n": N, "T": T, "forcing": nf}
                for r, it in enumerate(git):
                    index["items"].append({"path": it.path, "traj": it.traj, "params": it.params, "raw": it.raw, "n_frames": it.n_frames, "shape": list(shape), "group": gi, "row": r})
                    if r < done: continue
                    x = fam.load(it, fam.t_start, it.n_frames)                      # [T', C, H, W]
                    z, st = encode_frames(tok, x.reshape(-1, *shape), fam.periodic, a.bs, a.vae_wrap)
                    Z[r, fam.t_start:it.n_frames] = z.reshape(x.shape[0], C, tok.zc, h, w); S[r, fam.t_start:it.n_frames] = st.reshape(x.shape[0], C, 2)
                    if has_f:
                        lat = encode_cond(tok, fam, it, a.vae_wrap)
                        if nf == 1: Fz[r] = lat[0]
                        else: Fz[r] = np.stack(lat)
                    if (r + 1) % 50 == 0 or r + 1 == N:
                        Z.flush(); S.flush(); Fz is not None and Fz.flush(); open(pf, "w").write(str(r + 1))
                        print(f"[{a.name}] {k}/{sp} group {gi} {shape}: {r + 1}/{N} traj ({(time.time() - t0) / 60:.1f} min)", flush=True)
                del Z, S, Fz
            ip = os.path.join(d, "index.json"); json.dump(index, open(ip + ".tmp", "w")); os.replace(ip + ".tmp", ip)   # atomic: train_gen polls for index.json
            summary[f"{k}/{sp}"] = {"n_traj": len(its), "groups": index["groups"]}
    res = {"name": a.name, "tok": a.tok, "store": a.store, "root": store_root(a.store), "vae_wrap": a.vae_wrap, "kolmo_extra": a.kolmo_extra, "summary": summary,
           "wall_min": (time.time() - t0) / 60, "mem": runmeta.mem_peak(), "meta": runmeta.run_meta()}
    print(f"[{a.name}] memory " + json.dumps(res["mem"]), flush=True)
    json.dump(res, open(out, "w"), indent=1); print("saved", out, flush=True)


if __name__ == "__main__":
    main()
