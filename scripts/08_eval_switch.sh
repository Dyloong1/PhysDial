#!/usr/bin/env bash
# Evaluation of switch training: response coefficient r, tracking error TE, multi-switch.
set -e
source "$(dirname "${BASH_SOURCE[0]}")/00_env.sh"

# switch-trained PhysDial and its control: main columns, switch views (response r, tracking error), multi-switch

# physdial-switch-s0-eval: 2 GPUs
(cd "$REPO/physdial" && torchrun --standalone --nproc_per_node=2 -- train_gen.py --eval_only physdial-switch-s0 --store wan21_wrap --tok wan21ft:decoder-wrap.pt --sample_shift 8 --kolmo_extra drag --name physdial-switch-s0-eval)

# physdial-switch-control-s0-eval: 2 GPUs
(cd "$REPO/physdial" && torchrun --standalone --nproc_per_node=2 -- train_gen.py --eval_only physdial-switch-control-s0 --store wan21_wrap --tok wan21ft:decoder-wrap.pt --sample_shift 8 --kolmo_extra drag --name physdial-switch-control-s0-eval)

# physdial-switch-s0-eval-switch: 2 GPUs
(cd "$REPO/switch" && torchrun --standalone --nproc_per_node=2 -- eval_switch.py --run physdial-switch-s0 --setup physdial --frames 16 --dump_pred 32 --name physdial-switch-s0-eval-switch)

# physdial-switch-s0-response: CPU
(cd "$REPO/switch" && python3 analyze_switch.py --eval physdial-switch-s0-eval-switch --name physdial-switch-s0-response)

# physdial-switch-control-s0-eval-switch: 2 GPUs
(cd "$REPO/switch" && torchrun --standalone --nproc_per_node=2 -- eval_switch.py --run physdial-switch-control-s0 --setup physdial --frames 16 --dump_pred 32 --name physdial-switch-control-s0-eval-switch)

# physdial-switch-control-s0-response: CPU
(cd "$REPO/switch" && python3 analyze_switch.py --eval physdial-switch-control-s0-eval-switch --name physdial-switch-control-s0-response)

# physdial-switch-s0-eval-multiswitch: 4 GPUs
(cd "$REPO/switch" && torchrun --standalone --nproc_per_node=4 -- eval_ms.py --run physdial-switch-s0 --setup physdial --dump_pred 8 --name physdial-switch-s0-eval-multiswitch)

# physdial-switch-control-s0-eval-multiswitch: 4 GPUs
(cd "$REPO/switch" && torchrun --standalone --nproc_per_node=4 -- eval_ms.py --run physdial-switch-control-s0 --setup physdial --name physdial-switch-control-s0-eval-multiswitch)

# physdial-switch-s0-multiswitch-vs-control: CPU
(cd "$REPO/switch" && python3 analyze_ms.py --eval physdial-switch-s0-eval-multiswitch --vs physdial-switch-control-s0-eval-multiswitch --name physdial-switch-s0-multiswitch-vs-control)

# controlled switch study: single-switch views (16 and 64 steps) and main columns

# physdial-stage1-s0-eval-switch: 4 GPUs
(cd "$REPO/switch" && torchrun --standalone --nproc_per_node=4 -- eval_switch.py --run physdial-stage1-s0 --setup study --groups amp_up_in,amp_down_out,kf_in,kf_out,nu_up_in,nu_down_in,nu_up_out,nu_down_out,shape_in,shape_out --views B,A,N,T --frames 16 --dump_pred 32 --name physdial-stage1-s0-eval-switch)

# stage2-noperiodic-s0-eval-switch: 4 GPUs
(cd "$REPO/switch" && torchrun --standalone --nproc_per_node=4 -- eval_switch.py --run stage2-noperiodic-s0 --setup study --groups amp_up_in,amp_down_out,kf_in,kf_out,nu_up_in,nu_down_in,nu_up_out,nu_down_out,shape_in,shape_out --views B,A,N,T --frames 16 --dump_pred 32 --name stage2-noperiodic-s0-eval-switch)

# study-switch-s0-eval-switch: 4 GPUs
(cd "$REPO/switch" && torchrun --standalone --nproc_per_node=4 -- eval_switch.py --run study-switch-s0 --setup study --groups amp_up_in,amp_down_out,kf_in,kf_out,nu_up_in,nu_down_in,nu_up_out,nu_down_out,shape_in,shape_out --views B,A,N,T --frames 16 --dump_pred 32 --name study-switch-s0-eval-switch)

# study-control-s0-eval-switch: 4 GPUs
(cd "$REPO/switch" && torchrun --standalone --nproc_per_node=4 -- eval_switch.py --run study-control-s0 --setup study --groups amp_up_in,amp_down_out,kf_in,kf_out,nu_up_in,nu_down_in,nu_up_out,nu_down_out,shape_in,shape_out --views B,A,N,T --frames 16 --dump_pred 32 --name study-control-s0-eval-switch)

# study-switch-s1-eval-switch: 4 GPUs
(cd "$REPO/switch" && torchrun --standalone --nproc_per_node=4 -- eval_switch.py --run study-switch-s1 --setup study --groups amp_up_in,amp_down_out,kf_in,kf_out,nu_up_in,nu_down_in,nu_up_out,nu_down_out,shape_in,shape_out --views B,A,N,T --frames 16 --dump_pred 32 --name study-switch-s1-eval-switch)

# study-control-s1-eval-switch: 4 GPUs
(cd "$REPO/switch" && torchrun --standalone --nproc_per_node=4 -- eval_switch.py --run study-control-s1 --setup study --groups amp_up_in,amp_down_out,kf_in,kf_out,nu_up_in,nu_down_in,nu_up_out,nu_down_out,shape_in,shape_out --views B,A,N,T --frames 16 --dump_pred 32 --name study-control-s1-eval-switch)

# study-switch-ref-s0-eval-switch: 4 GPUs
(cd "$REPO/switch" && torchrun --standalone --nproc_per_node=4 -- eval_switch.py --run study-switch-ref-s0 --setup study --groups amp_up_in,amp_down_out,kf_in,kf_out,nu_up_in,nu_down_in,nu_up_out,nu_down_out,shape_in,shape_out --views B,A,N,T --frames 16 --dump_pred 32 --name study-switch-ref-s0-eval-switch)

# stage2-noperiodic-s0-eval-switch-64step: 4 GPUs
(cd "$REPO/switch" && torchrun --standalone --nproc_per_node=4 -- eval_switch.py --run stage2-noperiodic-s0 --setup study --groups amp_up_in,amp_down_out,kf_in,kf_out,nu_up_in,nu_down_in,nu_up_out,nu_down_out,shape_in,shape_out --views B,A,N,T --frames 64 --dump_pred 8 --name stage2-noperiodic-s0-eval-switch-64step)

# study-switch-s0-eval-switch-64step: 4 GPUs
(cd "$REPO/switch" && torchrun --standalone --nproc_per_node=4 -- eval_switch.py --run study-switch-s0 --setup study --groups amp_up_in,amp_down_out,kf_in,kf_out,nu_up_in,nu_down_in,nu_up_out,nu_down_out,shape_in,shape_out --views B,A,N --frames 64 --dump_pred 32 --name study-switch-s0-eval-switch-64step)

# study-switch-s0-eval-all: 4 GPUs
(cd "$REPO/switch" && torchrun --standalone --nproc_per_node=4 -- eval_switch.py --run study-switch-s0 --setup study --main 1 --frames 16 --name study-switch-s0-eval-all)

# study-control-s0-eval-all: 4 GPUs
(cd "$REPO/switch" && torchrun --standalone --nproc_per_node=4 -- eval_switch.py --run study-control-s0 --setup study --main 1 --frames 16 --name study-control-s0-eval-all)

# physdial-stage1-s0-response: CPU
(cd "$REPO/switch" && python3 analyze_switch.py --eval physdial-stage1-s0-eval-switch --name physdial-stage1-s0-response)

# stage2-noperiodic-s0-response: CPU
(cd "$REPO/switch" && python3 analyze_switch.py --eval stage2-noperiodic-s0-eval-switch --name stage2-noperiodic-s0-response)

# study-switch-s0-response: CPU
(cd "$REPO/switch" && python3 analyze_switch.py --eval study-switch-s0-eval-switch --name study-switch-s0-response)

# study-control-s0-response: CPU
(cd "$REPO/switch" && python3 analyze_switch.py --eval study-control-s0-eval-switch --name study-control-s0-response)

# study-switch-s1-response: CPU
(cd "$REPO/switch" && python3 analyze_switch.py --eval study-switch-s1-eval-switch --name study-switch-s1-response)

# study-control-s1-response: CPU
(cd "$REPO/switch" && python3 analyze_switch.py --eval study-control-s1-eval-switch --name study-control-s1-response)

# study-switch-ref-s0-response: CPU
(cd "$REPO/switch" && python3 analyze_switch.py --eval study-switch-ref-s0-eval-switch --name study-switch-ref-s0-response)

# stage2-noperiodic-s0-response-64step: CPU
(cd "$REPO/switch" && python3 analyze_switch.py --eval stage2-noperiodic-s0-eval-switch-64step --name stage2-noperiodic-s0-response-64step)

# study-switch-s0-response-64step: CPU
(cd "$REPO/switch" && python3 analyze_switch.py --eval study-switch-s0-eval-switch-64step --name study-switch-s0-response-64step)

# controlled switch study: multi-switch evaluation and analysis

# physdial-stage1-s0-eval-multiswitch: 4 GPUs
(cd "$REPO/switch" && torchrun --standalone --nproc_per_node=4 -- eval_ms.py --run physdial-stage1-s0 --setup study --name physdial-stage1-s0-eval-multiswitch)

# stage2-noperiodic-s0-eval-multiswitch: 4 GPUs
(cd "$REPO/switch" && torchrun --standalone --nproc_per_node=4 -- eval_ms.py --run stage2-noperiodic-s0 --setup study --name stage2-noperiodic-s0-eval-multiswitch)

# study-switch-s0-eval-multiswitch: 4 GPUs
(cd "$REPO/switch" && torchrun --standalone --nproc_per_node=4 -- eval_ms.py --run study-switch-s0 --setup study --name study-switch-s0-eval-multiswitch)

# study-control-s0-eval-multiswitch: 4 GPUs
(cd "$REPO/switch" && torchrun --standalone --nproc_per_node=4 -- eval_ms.py --run study-control-s0 --setup study --name study-control-s0-eval-multiswitch)

# study-switch-s1-eval-multiswitch: 2 GPUs
(cd "$REPO/switch" && torchrun --standalone --nproc_per_node=2 -- eval_ms.py --run study-switch-s1 --setup study --name study-switch-s1-eval-multiswitch)

# study-control-s1-eval-multiswitch: 2 GPUs
(cd "$REPO/switch" && torchrun --standalone --nproc_per_node=2 -- eval_ms.py --run study-control-s1 --setup study --name study-control-s1-eval-multiswitch)

# physdial-stage1-s0-multiswitch: CPU
(cd "$REPO/switch" && python3 analyze_ms.py --eval physdial-stage1-s0-eval-multiswitch --name physdial-stage1-s0-multiswitch)

# stage2-noperiodic-s0-multiswitch: CPU
(cd "$REPO/switch" && python3 analyze_ms.py --eval stage2-noperiodic-s0-eval-multiswitch --name stage2-noperiodic-s0-multiswitch)

# study-switch-s0-multiswitch: CPU
(cd "$REPO/switch" && python3 analyze_ms.py --eval study-switch-s0-eval-multiswitch --name study-switch-s0-multiswitch)

# study-control-s0-multiswitch: CPU
(cd "$REPO/switch" && python3 analyze_ms.py --eval study-control-s0-eval-multiswitch --name study-control-s0-multiswitch)

# study-switch-s0-multiswitch-vs-control: CPU
(cd "$REPO/switch" && python3 analyze_ms.py --eval study-switch-s0-eval-multiswitch --vs study-control-s0-eval-multiswitch --name study-switch-s0-multiswitch-vs-control)

# study-switch-s1-multiswitch: CPU
(cd "$REPO/switch" && python3 analyze_ms.py --eval study-switch-s1-eval-multiswitch --name study-switch-s1-multiswitch)

# study-control-s1-multiswitch: CPU
(cd "$REPO/switch" && python3 analyze_ms.py --eval study-control-s1-eval-multiswitch --name study-control-s1-multiswitch)

# study-switch-s1-multiswitch-vs-control: CPU
(cd "$REPO/switch" && python3 analyze_ms.py --eval study-switch-s1-eval-multiswitch --vs study-control-s1-eval-multiswitch --name study-switch-s1-multiswitch-vs-control)
