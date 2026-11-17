#!/usr/bin/env python
"""Download The Well 2D datasets (Hugging Face polymathic-ai/<dataset>, CC-BY-4.0) on their published grids into $PHYSDIAL_EXT_DATA/the_well.

Usage: python data_gen/the_well/fetch_well.py --name fetch-well --datasets gray_scott_reaction_diffusion,shear_flow,...
       [--include 'train:_chunk_0.hdf5,test:_chunk_40.hdf5']"""
import argparse, json, os, sys, time
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, os.path.dirname(os.path.dirname(HERE)))
import runmeta
from huggingface_hub import snapshot_download

ap = argparse.ArgumentParser(); ap.add_argument("--name", required=True); ap.add_argument("--datasets", required=True)
ap.add_argument("--dst", default=os.path.join(os.environ.get("PHYSDIAL_EXT_DATA", os.path.join(os.path.dirname(os.path.dirname(HERE)), "data", "ext")), "the_well")); ap.add_argument("--workers", type=int, default=8)
ap.add_argument("--include", default="", help="fetch only part of a dataset: comma list of specs; each spec becomes the allow_pattern '*<spec>*' with ':' "
                "standing for '*' (no shell glob needed), e.g. 'train:_chunk_0.hdf5,test:_chunk_40.hdf5' -> '*train*_chunk_0.hdf5*', "
                "'*test*_chunk_40.hdf5*' for euler_multi_quadrants_openBC")
a = ap.parse_args()
patterns = ["*" + s.replace(":", "*") + "*" for s in a.include.split(",") if s] or None
if patterns: print("[fetch] allow_patterns", patterns, flush=True)
a.dst = os.path.abspath(a.dst)
os.makedirs(os.path.join(HERE, "results"), exist_ok=True); out = os.path.join(HERE, "results", a.name + ".json")
if os.path.exists(out): print("exists", out); sys.exit(0)
t0 = time.time(); per = {}
for ds in [x for x in a.datasets.split(",") if x]:
    for attempt in range(30):   # rate limits / network hiccups: the download resumes where it stopped
        try:
            snapshot_download(f"polymathic-ai/{ds}", repo_type="dataset", local_dir=os.path.join(a.dst, ds), max_workers=a.workers,
                              allow_patterns=patterns); break
        except Exception as e:
            print(f"[fetch] {ds}: attempt {attempt + 1} failed: {type(e).__name__}: {str(e)[:200]} -> retry in 60 s", flush=True); time.sleep(60)
    else: sys.exit(f"[fetch] {ds}: giving up after 30 attempts")
    n = nb = 0
    for root, _, fs in os.walk(os.path.join(a.dst, ds)):
        if "/.cache" in root: continue
        for f in fs: n += 1; nb += os.path.getsize(os.path.join(root, f))
    per[ds] = {"n_files": n, "bytes": nb}; print(f"[fetch] {ds}: {n} files, {nb / 1e9:.1f} GB ({(time.time() - t0) / 60:.1f} min)", flush=True)
res = {"name": a.name, "datasets": per, "dst": a.dst, "n_files": sum(v["n_files"] for v in per.values()), "bytes": sum(v["bytes"] for v in per.values()),
       "wall_s": time.time() - t0, "meta": runmeta.run_meta()}
json.dump(res, open(out, "w"), indent=1); print("saved", out, flush=True)
