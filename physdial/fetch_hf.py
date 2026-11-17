#!/usr/bin/env python
"""Download model weights from the Hugging Face Hub into the HF cache ($HF_HOME, or the library default) and check the safetensors shards
against their index.

Usage: python physdial/fetch_hf.py --name fetch-wan21 --repo Wan-AI/Wan2.1-T2V-1.3B-Diffusers --subfolders transformer,vae,text_encoder,tokenizer"""
import argparse, json, math, os, struct, sys, time
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, os.path.dirname(HERE))
import runmeta
from huggingface_hub import snapshot_download


def header(path):
    """safetensors header: 8-byte little-endian length, then JSON {tensor: {dtype, shape, data_offsets}, '__metadata__': ...}."""
    with open(path, "rb") as f: n = struct.unpack("<Q", f.read(8))[0]; return {k: v for k, v in json.loads(f.read(n)).items() if k != "__metadata__"}


def check(sub):
    """-> (summary dict, list of problems) of one fetched subfolder."""
    fs = sorted(os.listdir(sub)); bad = []; per = {"n_files": len(fs), "bytes": sum(os.path.getsize(os.path.join(sub, f)) for f in fs)}
    st = [f for f in fs if f.endswith(".safetensors")]; ix = [f for f in fs if f.endswith(".safetensors.index.json")]
    for i in ix:
        wm = json.load(open(os.path.join(sub, i)))["weight_map"]
        for sh in sorted(set(wm.values())):
            if sh not in st: bad.append(f"{i}: shard {sh} missing"); continue
            want = {k for k, v in wm.items() if v == sh}; have = set(header(os.path.join(sub, sh)))
            if want != have: bad.append(f"{sh}: {len(want - have)} indexed tensors absent, {len(have - want)} unindexed")
    n_t = n_p = 0; dt = {}
    for sh in st:
        for k, v in header(os.path.join(sub, sh)).items(): n_t += 1; n = math.prod(v["shape"]); n_p += n; dt[v["dtype"]] = dt.get(v["dtype"], 0) + n
    if not st: bad.append("no safetensors file")
    per.update(n_shards=len(st), n_tensors=n_t, n_params=n_p, params_by_dtype=dt, indexes=ix)
    return per, bad


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--name", required=True); ap.add_argument("--repo", required=True, help="model repo id, e.g. Wan-AI/Wan2.1-T2V-1.3B-Diffusers")
    ap.add_argument("--subfolders", default="transformer", help="comma list of repo subfolders to fetch (each becomes the pattern '<sub>/*')")
    ap.add_argument("--workers", type=int, default=8)
    a = ap.parse_args()
    os.makedirs(os.path.join(HERE, "results"), exist_ok=True); out = os.path.join(HERE, "results", a.name + ".json")
    if os.path.exists(out): print("exists", out); sys.exit(0)
    hf = os.environ.get("HF_HOME", "(default Hugging Face cache)")
    subs = [s for s in a.subfolders.split(",") if s]; pats = [f"{s}/*" for s in subs] + ["model_index.json"]
    t0 = time.time(); print(f"[fetch_hf] {a.repo} {pats} -> {hf}", flush=True)
    for attempt in range(30):   # rate limits / network hiccups: the download resumes where it stopped
        try: d = snapshot_download(a.repo, allow_patterns=pats, max_workers=a.workers); break
        except Exception as e: print(f"[fetch_hf] attempt {attempt + 1} failed: {type(e).__name__}: {str(e)[:200]} -> retry in 60 s", flush=True); time.sleep(60)
    else: sys.exit(f"[fetch_hf] {a.repo}: giving up after 30 attempts")
    per, bad = {}, []
    for s in subs:
        p_, b_ = check(os.path.join(d, s)); per[s] = p_; bad += [f"{s}/{x}" for x in b_]
        print(f"[fetch_hf] {s}: {p_['n_shards']} shards, {p_['n_tensors']} tensors, {p_['n_params'] / 1e9:.3f}B parameters {p_['params_by_dtype']}, {p_['bytes'] / 1e9:.1f} GB "
              f"({(time.time() - t0) / 60:.1f} min)", flush=True)
    if bad: print("[fetch_hf] CHECK FAILED:", *bad, sep="\n  ", flush=True); sys.exit(1)
    res = {"name": a.name, "repo": a.repo, "snapshot": d, "subfolders": per, "wall_s": time.time() - t0, "meta": runmeta.run_meta()}
    json.dump(res, open(out, "w"), indent=1); print("saved", out, flush=True)


if __name__ == "__main__":
    main()
