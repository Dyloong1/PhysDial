#!/usr/bin/env bash
# Switch training (20 GPU-h) and the same-compute controls without switch trajectories.
set -e
source "$(dirname "${BASH_SOURCE[0]}")/00_env.sh"

# from PhysDial (store wan21_wrap): switch training and control

# physdial-switch-s0: 4 GPUs; --budget_gpu_h 20
(cd "$REPO/switch" && torchrun --standalone --nproc_per_node=4 -- train_switch.py --switch on --share_cap 0.21 --H 16 --store wan21_wrap --name physdial-switch-s0 -- --budget_gpu_h 20 --lr 3e-5 --equiv_eval --kolmo_extra anchor,drag --bc_pos --rand_shift 1 --init_weights physdial-s0)

# physdial-switch-control-s0: 4 GPUs; --budget_gpu_h 20
(cd "$REPO/switch" && torchrun --standalone --nproc_per_node=4 -- train_switch.py --switch off --store wan21_wrap --name physdial-switch-control-s0 -- --budget_gpu_h 20 --lr 3e-5 --equiv_eval --kolmo_extra anchor,drag --bc_pos --rand_shift 1 --init_weights physdial-s0)

# controlled switch study from stage 2 without the periodic handling (store wan21), seeds 0 and 1

# study-switch-s0: 4 GPUs
(cd "$REPO/switch" && torchrun --standalone --nproc_per_node=4 -- train_switch.py --switch on --H 16 --store wan21 --name study-switch-s0 -- --budget_gpu_h 20 --lr 3e-5 --equiv_eval --kolmo_extra anchor,drag --init_weights stage2-noperiodic-s0)

# study-control-s0: 4 GPUs
(cd "$REPO/switch" && torchrun --standalone --nproc_per_node=4 -- train_switch.py --switch off --store wan21 --name study-control-s0 -- --budget_gpu_h 20 --lr 3e-5 --equiv_eval --kolmo_extra anchor,drag --init_weights stage2-noperiodic-s0)

# study-switch-s1: 4 GPUs
(cd "$REPO/switch" && torchrun --standalone --nproc_per_node=4 -- train_switch.py --switch on --H 16 --store wan21 --name study-switch-s1 -- --budget_gpu_h 20 --lr 3e-5 --equiv_eval --kolmo_extra anchor,drag --init_weights stage2-noperiodic-s0 --seed 1)

# study-control-s1: 4 GPUs
(cd "$REPO/switch" && torchrun --standalone --nproc_per_node=4 -- train_switch.py --switch off --store wan21 --name study-control-s1 -- --budget_gpu_h 20 --lr 3e-5 --equiv_eval --kolmo_extra anchor,drag --init_weights stage2-noperiodic-s0 --seed 1)

# in-range reference of the switch study (switches into held-out values in training)

# study-switch-ref-s0: 4 GPUs
(cd "$REPO/switch" && torchrun --standalone --nproc_per_node=4 -- train_switch.py --switch on --reference_switch --H 16 --store wan21 --name study-switch-ref-s0 -- --budget_gpu_h 20 --lr 3e-5 --reference --kolmo_extra anchor --init_weights stage2-noperiodic-ref-s0)
