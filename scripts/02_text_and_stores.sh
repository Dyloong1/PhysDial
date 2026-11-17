#!/usr/bin/env bash
# Text embeddings and latent stores (frozen Wan2.1 VAE).
set -e
source "$(dirname "${BASH_SOURCE[0]}")/00_env.sh"

# equation-text embeddings (frozen umT5)

# textemb: 1 GPU
(cd "$REPO/physdial" && python3 textemb.py --name textemb)

# latent store wan21: frozen Wan2.1 VAE, zero padding (1 GPU per family)

# encode-wan21-kolmo: 1 GPU
(cd "$REPO/physdial" && python3 encode_corpus.py --tok wan21 --store wan21 --families kolmo --name encode-wan21-kolmo)

# encode-wan21-nsu: 1 GPU
(cd "$REPO/physdial" && python3 encode_corpus.py --tok wan21 --store wan21 --families nsu --name encode-wan21-nsu)

# encode-wan21-gs: 1 GPU
(cd "$REPO/physdial" && python3 encode_corpus.py --tok wan21 --store wan21 --families gs --name encode-wan21-gs)

# encode-wan21-rb: 1 GPU
(cd "$REPO/physdial" && python3 encode_corpus.py --tok wan21 --store wan21 --families rb --name encode-wan21-rb)

# encode-wan21-sf: 1 GPU
(cd "$REPO/physdial" && python3 encode_corpus.py --tok wan21 --store wan21 --families sf --name encode-wan21-sf)

# encode-wan21-trl: 1 GPU
(cd "$REPO/physdial" && python3 encode_corpus.py --tok wan21 --store wan21 --families trl --name encode-wan21-trl)

# encode-wan21-euler: 1 GPU
(cd "$REPO/physdial" && python3 encode_corpus.py --tok wan21 --store wan21 --families euler --name encode-wan21-euler)

# encode-wan21-acoustic: 1 GPU
(cd "$REPO/physdial" && python3 encode_corpus.py --tok wan21 --store wan21 --families acoustic --name encode-wan21-acoustic)

# encode-wan21-anchor: 1 GPU
(cd "$REPO/physdial" && python3 encode_corpus.py --tok wan21 --store wan21 --families kolmo --splits anchor --kolmo_extra anchor --name encode-wan21-anchor)

# encode-wan21-dragref: 1 GPU; in-range reference of the 6-frame-history variant (14_ablations.sh)
(cd "$REPO/physdial" && python3 encode_corpus.py --tok wan21 --store wan21 --families kolmo --splits ref_extrap_drag_lo,ref_extrap_drag_hi --kolmo_extra drag --name encode-wan21-dragref)

# encode-wan21-betaanchor: 1 GPU; beta experiment
(cd "$REPO/physdial" && python3 encode_corpus.py --tok wan21 --store wan21 --families kolmo --splits beta_anchor --kolmo_extra beta --bs 16 --name encode-wan21-betaanchor)

# encode-wan21-betaref: 1 GPU; beta experiment
(cd "$REPO/physdial" && python3 encode_corpus.py --tok wan21 --store wan21 --families kolmo --splits ref_extrap_beta --kolmo_extra beta --bs 16 --name encode-wan21-betaref)

# latent store wan21_wrap: wrap-around VAE padding on periodic axes; the non-periodic families (euler, acoustic) are shared with wan21

# encode-wan21_wrap-kolmo: 1 GPU
(cd "$REPO/physdial" && python3 encode_corpus.py --tok wan21 --store wan21_wrap --families kolmo --vae_wrap --name encode-wan21_wrap-kolmo)

# encode-wan21_wrap-nsu: 1 GPU
(cd "$REPO/physdial" && python3 encode_corpus.py --tok wan21 --store wan21_wrap --families nsu --vae_wrap --name encode-wan21_wrap-nsu)

# encode-wan21_wrap-gs: 1 GPU
(cd "$REPO/physdial" && python3 encode_corpus.py --tok wan21 --store wan21_wrap --families gs --vae_wrap --name encode-wan21_wrap-gs)

# encode-wan21_wrap-rb: 1 GPU
(cd "$REPO/physdial" && python3 encode_corpus.py --tok wan21 --store wan21_wrap --families rb --vae_wrap --name encode-wan21_wrap-rb)

# encode-wan21_wrap-sf: 1 GPU
(cd "$REPO/physdial" && python3 encode_corpus.py --tok wan21 --store wan21_wrap --families sf --vae_wrap --name encode-wan21_wrap-sf)

# encode-wan21_wrap-trl: 1 GPU
(cd "$REPO/physdial" && python3 encode_corpus.py --tok wan21 --store wan21_wrap --families trl --vae_wrap --name encode-wan21_wrap-trl)

# encode-wan21_wrap-anchor: 1 GPU
(cd "$REPO/physdial" && python3 encode_corpus.py --tok wan21 --store wan21_wrap --families kolmo --splits anchor --kolmo_extra anchor --vae_wrap --name encode-wan21_wrap-anchor)

for f in euler acoustic; do ln -sfn "$PHYSDIAL_DATA_ROOT/latents/wan21/$f" "$PHYSDIAL_DATA_ROOT/latents/wan21_wrap/$f"; done

# switch trajectories into the latent stores

# encode-switch-train-wan21: 1 GPU
(cd "$REPO/switch" && python3 encode_switch.py --split switch_train --store wan21 --bs 16 --name encode-switch-train-wan21)

# encode-switch-ref-wan21: 1 GPU
(cd "$REPO/switch" && python3 encode_switch.py --split switch_ref --store wan21 --bs 16 --name encode-switch-ref-wan21)

# encode-switch-train-wan21_wrap: 1 GPU
(cd "$REPO/switch" && python3 encode_switch.py --split switch_train --store wan21_wrap --bs 16 --name encode-switch-train-wan21_wrap)
