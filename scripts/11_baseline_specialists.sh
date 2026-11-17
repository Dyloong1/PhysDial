#!/usr/bin/env bash
# Baselines: per-family conditioned U-Net / FNO specialists (+ in-range references).
set -e
source "$(dirname "${BASH_SOURCE[0]}")/00_env.sh"

# per-family conditioned U-Net and FNO specialists (4 GPU-h each)

# unet-kolmo-s0: 1 GPU
(cd "$REPO/physdial/baselines" && python3 small_expert.py --family kolmo --model unet --budget_gpu_h 4 --bs 16 --noise 0.01 --name unet-kolmo-s0)

# fno-kolmo-s0: 1 GPU
(cd "$REPO/physdial/baselines" && python3 small_expert.py --family kolmo --model fno --budget_gpu_h 4 --bs 16 --modes 32 --noise 0.01 --name fno-kolmo-s0)

# unet-nsu-s0: 1 GPU
(cd "$REPO/physdial/baselines" && python3 small_expert.py --family nsu --model unet --budget_gpu_h 4 --bs 16 --noise 0.01 --name unet-nsu-s0)

# fno-nsu-s0: 1 GPU
(cd "$REPO/physdial/baselines" && python3 small_expert.py --family nsu --model fno --budget_gpu_h 4 --bs 16 --modes 32 --noise 0.01 --name fno-nsu-s0)

# unet-gs-s0: 1 GPU
(cd "$REPO/physdial/baselines" && python3 small_expert.py --family gs --model unet --budget_gpu_h 4 --bs 16 --noise 0.01 --name unet-gs-s0)

# fno-gs-s0: 1 GPU
(cd "$REPO/physdial/baselines" && python3 small_expert.py --family gs --model fno --budget_gpu_h 4 --bs 16 --modes 32 --noise 0.01 --name fno-gs-s0)

# unet-rb-s0: 1 GPU
(cd "$REPO/physdial/baselines" && python3 small_expert.py --family rb --model unet --budget_gpu_h 4 --bs 16 --noise 0.01 --name unet-rb-s0)

# fno-rb-s0: 1 GPU
(cd "$REPO/physdial/baselines" && python3 small_expert.py --family rb --model fno --budget_gpu_h 4 --bs 16 --modes 32 --noise 0.01 --name fno-rb-s0)

# unet-sf-s0: 1 GPU
(cd "$REPO/physdial/baselines" && python3 small_expert.py --family sf --model unet --budget_gpu_h 4 --bs 16 --noise 0.01 --name unet-sf-s0)

# fno-sf-s0: 1 GPU
(cd "$REPO/physdial/baselines" && python3 small_expert.py --family sf --model fno --budget_gpu_h 4 --bs 16 --modes 32 --noise 0.01 --name fno-sf-s0)

# unet-trl-s0: 1 GPU
(cd "$REPO/physdial/baselines" && python3 small_expert.py --family trl --model unet --budget_gpu_h 4 --bs 16 --noise 0.01 --name unet-trl-s0)

# fno-trl-s0: 1 GPU
(cd "$REPO/physdial/baselines" && python3 small_expert.py --family trl --model fno --budget_gpu_h 4 --bs 16 --modes 32 --noise 0.01 --name fno-trl-s0)

# unet-euler-s0: 1 GPU
(cd "$REPO/physdial/baselines" && python3 small_expert.py --family euler --model unet --budget_gpu_h 4 --bs 16 --noise 0.01 --name unet-euler-s0)

# fno-euler-s0: 1 GPU
(cd "$REPO/physdial/baselines" && python3 small_expert.py --family euler --model fno --budget_gpu_h 4 --bs 16 --modes 32 --noise 0.01 --name fno-euler-s0)

# unet-acoustic-s0: 1 GPU
(cd "$REPO/physdial/baselines" && python3 small_expert.py --family acoustic --model unet --budget_gpu_h 4 --bs 16 --noise 0.01 --name unet-acoustic-s0)

# fno-acoustic-s0: 1 GPU
(cd "$REPO/physdial/baselines" && python3 small_expert.py --family acoustic --model fno --budget_gpu_h 4 --bs 16 --modes 32 --noise 0.01 --name fno-acoustic-s0)

# in-range references and second seeds (Kolmogorov, NSU)

# unet-kolmo-ref-s0: 1 GPU
(cd "$REPO/physdial/baselines" && python3 small_expert.py --family kolmo --model unet --reference --budget_gpu_h 4 --bs 16 --noise 0.01 --name unet-kolmo-ref-s0)

# fno-kolmo-ref-s0: 1 GPU
(cd "$REPO/physdial/baselines" && python3 small_expert.py --family kolmo --model fno --reference --budget_gpu_h 4 --bs 16 --modes 32 --noise 0.01 --name fno-kolmo-ref-s0)

# unet-kolmo-s1: 1 GPU
(cd "$REPO/physdial/baselines" && python3 small_expert.py --family kolmo --model unet --budget_gpu_h 4 --bs 16 --noise 0.01 --seed 1 --name unet-kolmo-s1)

# fno-kolmo-s1: 1 GPU
(cd "$REPO/physdial/baselines" && python3 small_expert.py --family kolmo --model fno --budget_gpu_h 4 --bs 16 --noise 0.01 --modes 32 --seed 1 --name fno-kolmo-s1)

# unet-nsu-ref-s0: 1 GPU
(cd "$REPO/physdial/baselines" && python3 small_expert.py --family nsu --model unet --reference --budget_gpu_h 4 --bs 16 --noise 0.01 --name unet-nsu-ref-s0)

# fno-nsu-ref-s0: 1 GPU
(cd "$REPO/physdial/baselines" && python3 small_expert.py --family nsu --model fno --reference --budget_gpu_h 4 --bs 16 --modes 32 --noise 0.01 --name fno-nsu-ref-s0)

# unet-nsu-s1: 1 GPU
(cd "$REPO/physdial/baselines" && python3 small_expert.py --family nsu --model unet --budget_gpu_h 4 --bs 16 --noise 0.01 --seed 1 --name unet-nsu-s1)

# fno-nsu-s1: 1 GPU
(cd "$REPO/physdial/baselines" && python3 small_expert.py --family nsu --model fno --budget_gpu_h 4 --bs 16 --noise 0.01 --modes 32 --seed 1 --name fno-nsu-s1)

# 64-step rollouts on the test split (Kolmogorov, NSU) with the fields of all 32 windows saved (spectra)

# unet-kolmo-s0-eval-64step: 1 GPU
(cd "$REPO/physdial/baselines" && python3 small_expert.py --family kolmo --eval_only unet-kolmo-s0 --frames 64 --splits test --dump_main --dump_pred 32 --name unet-kolmo-s0-eval-64step)

# fno-kolmo-s0-eval-64step: 1 GPU
(cd "$REPO/physdial/baselines" && python3 small_expert.py --family kolmo --eval_only fno-kolmo-s0 --frames 64 --splits test --dump_main --dump_pred 32 --name fno-kolmo-s0-eval-64step)

# unet-nsu-s0-eval-64step: 1 GPU
(cd "$REPO/physdial/baselines" && python3 small_expert.py --family nsu --eval_only unet-nsu-s0 --frames 64 --splits test --dump_main --dump_pred 32 --name unet-nsu-s0-eval-64step)

# fno-nsu-s0-eval-64step: 1 GPU
(cd "$REPO/physdial/baselines" && python3 small_expert.py --family nsu --eval_only fno-nsu-s0 --frames 64 --splits test --dump_main --dump_pred 32 --name fno-nsu-s0-eval-64step)

# U-Net with the switch training groups, and both U-Nets on the switch views

# unet-switch-kolmo-s0: 1 GPU
(cd "$REPO/physdial/baselines" && python3 small_expert.py --family kolmo --model unet --switch_train --H 16 --budget_gpu_h 4 --bs 16 --noise 0.01 --name unet-switch-kolmo-s0)

# unet-kolmo-s0-eval-switch: 1 GPU
(cd "$REPO/physdial/baselines" && python3 small_expert.py --family kolmo --eval_only unet-kolmo-s0 --switch_eval --dump_pred 32 --name unet-kolmo-s0-eval-switch)

# unet-switch-kolmo-s0-eval-switch: 1 GPU
(cd "$REPO/physdial/baselines" && python3 small_expert.py --family kolmo --eval_only unet-switch-kolmo-s0 --switch_eval --dump_pred 32 --name unet-switch-kolmo-s0-eval-switch)

# unet-kolmo-s0-response: CPU
(cd "$REPO/switch" && python3 analyze_switch.py --eval unet-kolmo-s0-eval-switch --name unet-kolmo-s0-response)

# unet-switch-kolmo-s0-response: CPU
(cd "$REPO/switch" && python3 analyze_switch.py --eval unet-switch-kolmo-s0-eval-switch --name unet-switch-kolmo-s0-response)
