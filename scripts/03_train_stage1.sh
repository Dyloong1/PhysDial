#!/usr/bin/env bash
# Stage 1: multi-family training of the eight families from the pretrained Wan2.1-T2V-1.3B (60 GPU-h each).
set -e
source "$(dirname "${BASH_SOURCE[0]}")/00_env.sh"

# stage 1 model

# physdial-stage1-s0: 4 GPUs; --budget_gpu_h 60
(cd "$REPO/physdial" && torchrun --standalone --nproc_per_node=4 -- train_gen.py --store wan21 --ens 8 --ens_n 16 --ens_splits all --budget_gpu_h 60 --name physdial-stage1-s0)

# in-range reference: same training, held-out-condition trajectories added (--reference)

# physdial-stage1-ref-s0: 4 GPUs; --budget_gpu_h 60
(cd "$REPO/physdial" && torchrun --standalone --nproc_per_node=4 -- train_gen.py --store wan21 --ens 8 --ens_n 16 --ens_splits all --reference --budget_gpu_h 60 --name physdial-stage1-ref-s0)
