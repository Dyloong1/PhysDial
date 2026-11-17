#!/usr/bin/env bash
# Stage 2: + Kolmogorov amplitude anchors + periodic handling = PhysDial (20 GPU-h each).
set -e
source "$(dirname "${BASH_SOURCE[0]}")/00_env.sh"

# PhysDial (store wan21_wrap, --bc_pos, --rand_shift 1, anchors), seeds 0 and 1 (both from the stage-1 model)

# physdial-s0: 4 GPUs; --budget_gpu_h 20
(cd "$REPO/physdial" && torchrun --standalone --nproc_per_node=4 -- train_gen.py --store wan21_wrap --budget_gpu_h 20 --lr 3e-5 --init_weights physdial-stage1-s0 --equiv_eval --kolmo_extra anchor,drag --bc_pos --rand_shift 1 --name physdial-s0)

# physdial-s1: 4 GPUs; --budget_gpu_h 20
(cd "$REPO/physdial" && torchrun --standalone --nproc_per_node=4 -- train_gen.py --store wan21_wrap --budget_gpu_h 20 --lr 3e-5 --init_weights physdial-stage1-s0 --equiv_eval --kolmo_extra anchor,drag --bc_pos --rand_shift 1 --seed 1 --name physdial-s1)

# in-range reference of PhysDial

# physdial-ref-s0: 4 GPUs; --budget_gpu_h 20
(cd "$REPO/physdial" && torchrun --standalone --nproc_per_node=4 -- train_gen.py --store wan21_wrap --budget_gpu_h 20 --lr 3e-5 --init_weights physdial-stage1-ref-s0 --reference --kolmo_extra anchor --bc_pos --rand_shift 1 --name physdial-ref-s0)

# stage 2 without the periodic handling (store wan21): the starting point of the controlled switch study and of the beta experiment,
# and the control of the periodic handling (seeds 0 and 1)

# stage2-noperiodic-s0: 4 GPUs; --budget_gpu_h 20
(cd "$REPO/physdial" && torchrun --standalone --nproc_per_node=4 -- train_gen.py --store wan21 --budget_gpu_h 20 --lr 3e-5 --init_weights physdial-stage1-s0 --equiv_eval --kolmo_extra anchor,drag --name stage2-noperiodic-s0)

# stage2-noperiodic-s1: 4 GPUs; --budget_gpu_h 20
(cd "$REPO/physdial" && torchrun --standalone --nproc_per_node=4 -- train_gen.py --store wan21 --budget_gpu_h 20 --lr 3e-5 --init_weights physdial-stage1-s0 --equiv_eval --kolmo_extra anchor,drag --seed 1 --name stage2-noperiodic-s1)

# stage2-noperiodic-ref-s0: 4 GPUs; in-range reference, --budget_gpu_h 20
(cd "$REPO/physdial" && torchrun --standalone --nproc_per_node=4 -- train_gen.py --store wan21 --budget_gpu_h 20 --lr 3e-5 --init_weights physdial-stage1-ref-s0 --reference --kolmo_extra anchor --name stage2-noperiodic-ref-s0)
