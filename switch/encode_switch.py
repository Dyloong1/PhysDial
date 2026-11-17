"""Encode the switch-trajectory training splits (switch_data: switch_train / switch_ref) into a latent store under $PHYSDIAL_DATA_ROOT/switch/latents/<store>/.

Uses the corpus encoder (encode_corpus.main) unchanged: in this process switch_data.install_train replaces the Kolmogorov class and
encode_corpus.store_root points to the switch latent root. Tokenizer and VAE padding are taken from the main
store's kolmo/train index, so the switch latents are made exactly like the training latents. The result JSON is moved to ./results."""
import argparse, json, os, shutil, sys

HERE = os.path.dirname(os.path.abspath(__file__)); PROJ = os.path.dirname(HERE); PKG = os.path.join(PROJ, "physdial")
sys.path.insert(0, HERE); sys.path.insert(0, PROJ); sys.path.insert(0, PKG)
import switch_data   # noqa: E402


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--name", required=True); ap.add_argument("--split", required=True, choices=sorted(switch_data.TRAIN_SUBDIRS))
    ap.add_argument("--store", default="wan21"); ap.add_argument("--bs", type=int, default=64)
    a = ap.parse_args()
    out = os.path.join(HERE, "results", a.name + ".json"); os.makedirs(os.path.dirname(out), exist_ok=True)
    if os.path.exists(out): print("exists", out); return
    main_ix = json.load(open(os.path.join(os.environ["PHYSDIAL_DATA_ROOT"], "latents", a.store, "kolmo", "train", "index.json")))
    tok, wrap = main_ix["tok"], bool(main_ix.get("vae_wrap", False))
    switch_data.install_train([a.split])
    import encode_corpus
    encode_corpus.store_root = switch_data.latent_root
    argv = ["--name", a.name, "--tok", tok, "--store", a.store, "--families", "kolmo", "--splits", a.split, "--bs", str(a.bs)]
    if wrap: argv += ["--vae_wrap"]
    print(f"[{a.name}] main store {a.store}/kolmo/train: tok {tok}, vae_wrap {wrap} -> {switch_data.latent_root(a.store)}", flush=True)
    sys.argv = [os.path.join(PKG, "encode_corpus.py")] + argv; cwd = os.getcwd(); os.chdir(PKG)
    try: encode_corpus.main()
    finally:
        os.chdir(cwd); src = os.path.join(PKG, "results", a.name + ".json")
        if os.path.exists(src): shutil.move(src, out); print("moved", src, "->", out, flush=True)


if __name__ == "__main__":
    main()
