#!/usr/bin/env python
"""Frozen umT5 text embeddings of every equation-identity text of the corpus (the cross-attention input of the generator).

Usage: python physdial/textemb.py --name textemb
Writes physdial/textemb/<name>.pt ({text: [n_tokens, 4096] bf16}) and physdial/results/<name>.json."""
import argparse, json, os, sys, time
HERE = os.path.dirname(os.path.abspath(__file__)); PROJ = os.path.dirname(HERE); sys.path.insert(0, PROJ); sys.path.insert(0, HERE)
import torch
import runmeta, corpus
from gen_model import WAN21

ap = argparse.ArgumentParser(); ap.add_argument("--name", required=True); ap.add_argument("--max_len", type=int, default=128)
a = ap.parse_args()
os.makedirs(os.path.join(HERE, "results"), exist_ok=True); os.makedirs(os.path.join(HERE, "textemb"), exist_ok=True)
out = os.path.join(HERE, "results", a.name + ".json")
if os.path.exists(out): print("exists", out); sys.exit(0)
t0 = time.time(); texts = {""}
for k in corpus.FAMILIES: texts.update(corpus.get(k).texts())   # "" = the empty text (--cond none)
todo = sorted(texts); print(f"[textemb] {len(todo)} texts", flush=True)
from transformers import AutoTokenizer, UMT5EncoderModel
tok = AutoTokenizer.from_pretrained(WAN21, subfolder="tokenizer")
enc = UMT5EncoderModel.from_pretrained(WAN21, subfolder="text_encoder", torch_dtype=torch.bfloat16).cuda().eval()
with torch.no_grad():
    if float(enc.encoder.embed_tokens.weight.float().abs().max()) == 0.0: enc.encoder.embed_tokens.weight = enc.shared.weight; print("[textemb] tied encoder.embed_tokens to shared", flush=True)   # some transformers versions load it untied
    assert float(enc.encoder.embed_tokens.weight.float().abs().max()) > 0 and torch.equal(enc.encoder.embed_tokens.weight, enc.shared.weight)
emb = {}; lens = {}
with torch.no_grad():
    for i in range(0, len(todo), 32):
        tb = todo[i:i + 32]; t = tok(tb, padding="max_length", max_length=a.max_len, truncation=True, add_special_tokens=True, return_attention_mask=True, return_tensors="pt")
        h = enc(t.input_ids.cuda(), t.attention_mask.cuda()).last_hidden_state
        for j, s in enumerate(tb):
            n = int(t.attention_mask[j].sum()); emb[s] = h[j, :n].to(torch.bfloat16).cpu(); lens[s] = n
bad = [s for s in todo if s and float(emb[s].float().norm()) == 0.0]; assert not bad, f"zero embeddings: {bad[:3]}"
ident = [t for t in todo if t]
same = [s for s in ident if any(t != s and emb[t].shape == emb[s].shape and torch.equal(emb[t], emb[s]) for t in ident)]
assert not same, f"embeddings identical to another text's: {same[:3]}"
torch.save({"max_len": a.max_len, "emb": emb}, os.path.join(HERE, "textemb", a.name + ".pt"))
res = {"name": a.name, "n_texts": len(emb), "max_tokens": max(lens.values()), "truncated": [s for s, n in lens.items() if n >= a.max_len],
       "family_texts": {k: corpus.get(k).texts() for k in corpus.FAMILIES}, "wall_s": time.time() - t0, "meta": runmeta.run_meta()}
json.dump(res, open(out, "w"), indent=1); print("saved", out, flush=True)
