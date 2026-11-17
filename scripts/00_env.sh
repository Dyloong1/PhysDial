#!/usr/bin/env bash
# Environment shared by every script (sourced). Override any variable before running; relative paths are taken from the current directory.
REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"; export REPO
abspath() { case "$1" in /*) printf '%s\n' "$1" ;; *) printf '%s\n' "$PWD/$1" ;; esac; }
export PHYSDIAL_DATA_ROOT="$(abspath "${PHYSDIAL_DATA_ROOT:-$REPO/data}")"                     # own data, latent stores, Walrus-format exports
export PHYSDIAL_EXT_DATA="$(abspath "${PHYSDIAL_EXT_DATA:-$REPO/data/ext}")"                   # public data: The Well
export WALRUS_SRC="$(abspath "${WALRUS_SRC:-$REPO/ext/walrus}")"                               # Walrus source checkout
export WALRUS_CKPT_DIR="$(abspath "${WALRUS_CKPT_DIR:-$REPO/ext_models/walrus}")"              # released Walrus checkpoint
export DPOT_SRC="$(abspath "${DPOT_SRC:-$REPO/ext/DPOT}")"                                     # DPOT source checkout
export DPOT_WEIGHTS="$(abspath "${DPOT_WEIGHTS:-$REPO/ext/dpot_weights/model_H.pth}")"         # released DPOT-H weights
export HF_HOME="$(abspath "${HF_HOME:-$REPO/cache/huggingface}")"                              # Hugging Face cache (Wan2.1-T2V-1.3B, umT5)
export PYTHONUNBUFFERED=1
mkdir -p "$PHYSDIAL_DATA_ROOT" "$PHYSDIAL_EXT_DATA"
cd "$REPO"
