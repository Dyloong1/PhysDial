#!/usr/bin/env bash
# Evaluation of PhysDial on the evaluation columns of the eight families.
set -e
source "$(dirname "${BASH_SOURCE[0]}")/00_env.sh"

# main table: 32 windows per column, steps 4 and 16, sampler 25 steps with shift 8, the decoder of the wrap-padded store

# physdial-s0-eval: 2 GPUs
(cd "$REPO/physdial" && torchrun --standalone --nproc_per_node=2 -- train_gen.py --eval_only physdial-s0 --store wan21_wrap --tok wan21ft:decoder-wrap.pt --sample_shift 8 --kolmo_extra drag --name physdial-s0-eval)

# physdial-s1-eval: 2 GPUs
(cd "$REPO/physdial" && torchrun --standalone --nproc_per_node=2 -- train_gen.py --eval_only physdial-s1 --store wan21_wrap --tok wan21ft:decoder-wrap.pt --sample_shift 8 --kolmo_extra drag --name physdial-s1-eval)

# 64-step rollouts (Kolmogorov, NSU) with saved predictions

# physdial-s0-eval-64step: 2 GPUs
(cd "$REPO/physdial" && torchrun --standalone --nproc_per_node=2 -- train_gen.py --eval_only physdial-s0 --store wan21_wrap --tok wan21ft:decoder-wrap.pt --sample_shift 8 --kolmo_extra drag --eval_chunks 16 --eval_splits kolmo/test,kolmo/interp_nu,kolmo/extrap_nu_low,kolmo/extrap_nu_high,kolmo/interp_kf,kolmo/extrap_kf,kolmo/extrap_amp,kolmo/extrap_joint,kolmo/nu_dense_interp,kolmo/nu_dense_ends,kolmo/multif_test,kolmo/extrap_kband,kolmo/extrap_drag_lo,kolmo/extrap_drag_hi,nsu/test,nsu/interp_test,nsu/extrap_ends --dump_pred 32 --name physdial-s0-eval-64step)

# text probes: zeroed / swapped equation text

# physdial-s0-eval-textzero: 2 GPUs
(cd "$REPO/physdial" && torchrun --standalone --nproc_per_node=2 -- train_gen.py --eval_only physdial-s0 --store wan21_wrap --tok wan21ft:decoder-wrap.pt --sample_shift 8 --kolmo_extra drag --text_probe zero --name physdial-s0-eval-textzero)

# physdial-s0-eval-textswap: 2 GPUs
(cd "$REPO/physdial" && torchrun --standalone --nproc_per_node=2 -- train_gen.py --eval_only physdial-s0 --store wan21_wrap --tok wan21ft:decoder-wrap.pt --sample_shift 8 --kolmo_extra drag --text_probe swap --name physdial-s0-eval-textswap)

# extrapolation cost of PhysDial: the model and its in-range reference with the training sampling shift (3)

# physdial-s0-eval-shift3: 2 GPUs
(cd "$REPO/physdial" && torchrun --standalone --nproc_per_node=2 -- train_gen.py --eval_only physdial-s0 --store wan21_wrap --tok wan21ft:decoder-wrap.pt --kolmo_extra drag --name physdial-s0-eval-shift3)

# physdial-ref-s0-eval-shift3: 2 GPUs
(cd "$REPO/physdial" && torchrun --standalone --nproc_per_node=2 -- train_gen.py --eval_only physdial-ref-s0 --store wan21_wrap --tok wan21ft:decoder-wrap.pt --name physdial-ref-s0-eval-shift3)
