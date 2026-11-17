#!/usr/bin/env bash
# Decoder fine-tuning (VAE encoder frozen; spectral loss weight 0.1). Each run ends with the reconstruction check (physdial/tok_gate.py with the fine-tuned decoder: 24 frames of every evaluation split of the eight families; every channel must reach relative L2 <= 0.05 and gradient-energy ratio in [0.9, 1.1]); results in physdial/results/<run>.json, keys gate and summary.
set -e
source "$(dirname "${BASH_SOURCE[0]}")/00_env.sh"

# decoder for the wrap-padded store (every PhysDial evaluation)

# decoder-wrap: 1 GPU
(cd "$REPO/physdial" && python3 train_tok2d.py --steps 20000 --ref_run physdial-s0 --lam_spec 0.1 --vae_wrap --name decoder-wrap)

# decoder for the zero-padded store (evaluations on wan21: controlled switch study, beta experiment, ablations)

# decoder: 1 GPU
(cd "$REPO/physdial" && python3 train_tok2d.py --steps 20000 --ref_run physdial-stage1-s0 --lam_spec 0.1 --name decoder)
