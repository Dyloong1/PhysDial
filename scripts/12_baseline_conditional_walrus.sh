#!/usr/bin/env bash
# Baseline: Walrus with the condition inputs (Kolmogorov), with and without switch data.
set -e
source "$(dirname "${BASH_SOURCE[0]}")/00_env.sh"

# Walrus / DPOT commands run in a Python environment with the Walrus package installed (see README).

# Walrus-format exports with the condition channels, plain exports, switch and multi-switch data (CPU)

# walrus-cond-data: CPU
(cd "$REPO/switch" && python3 walrus_cond_data.py --kolmo_extra drag --switch kolmo --link_main --eval_n 32 --frames 16 --name walrus-cond-data)

# walrus-cond-data-anchor: CPU
(cd "$REPO/switch" && python3 walrus_cond_data.py --kolmo_extra anchor,drag --switch kolmo --eval_n 32 --frames 16 --name walrus-cond-data-anchor)

# walrus-plain-data-switch: CPU
(cd "$REPO/switch" && python3 walrus_cond_data.py --plain 1 --skip_train --kolmo_extra drag --switch kolmo --link_main --eval_n 32 --frames 16 --name walrus-plain-data-switch)

# walrus-cond-data-switch: CPU
(cd "$REPO/switch" && python3 walrus_sw_data.py --name walrus-cond-data-switch)

# walrus-cond-data-multiswitch: CPU
(cd "$REPO/switch" && python3 walrus_ms_data.py --subdir kolmo_ms --frames 64 --name walrus-cond-data-multiswitch)

# walrus-plain-data-multiswitch: CPU
(cd "$REPO/switch" && python3 walrus_ms_data.py --plain 1 --subdir kolmo_ms --frames 64 --name walrus-plain-data-multiswitch)

# conditioned Walrus: base fine-tune, anchor continuation, anchor + switch continuation

# walrus-cond-stage1: 4 GPUs
(cd "$REPO/switch" && torchrun --standalone --nproc_per_node=4 -- walrus_cond_ft.py --epochs 51 --max_samples 500 --name walrus-cond-stage1)

# walrus-cond: 4 GPUs
(cd "$REPO/switch" && torchrun --standalone --nproc_per_node=4 -- walrus_cond_ft.py --data physdial_cond_corpus_anchor --init_from walrus-cond-stage1 --lr 3e-5 --warmup_epochs 1 --cooldown_epochs 6 --epochs 18 --max_samples 500 --name walrus-cond)

# walrus-cond-switch: 4 GPUs
(cd "$REPO/switch" && torchrun --standalone --nproc_per_node=4 -- walrus_cond_ft.py --data physdial_cond_corpus_anchor_sw --init_from walrus-cond-stage1 --lr 3e-5 --warmup_epochs 1 --cooldown_epochs 6 --epochs 18 --max_samples 500 --name walrus-cond-switch)

# single-switch views and analysis (plain Walrus on the same windows for comparison)

# walrus-cond-eval-switch: 4 GPUs
(cd "$REPO/switch" && torchrun --standalone --nproc_per_node=4 -- walrus_cond_eval.py --run walrus-cond --ckpt last --root cond --name walrus-cond-eval-switch)

# walrus-cond-switch-eval-switch: 4 GPUs
(cd "$REPO/switch" && torchrun --standalone --nproc_per_node=4 -- walrus_cond_eval.py --run walrus-cond-switch --ckpt last --root cond --name walrus-cond-switch-eval-switch)

# walrus-ft-eval-switch: 2 GPUs
(cd "$REPO/switch" && torchrun --standalone --nproc_per_node=2 -- walrus_cond_eval.py --run walrus-ft --ckpt last --families kolmo --root plain --name walrus-ft-eval-switch)

# walrus-cond-response: CPU
(cd "$REPO/switch" && python3 analyze_switch.py --eval walrus-cond-eval-switch --name walrus-cond-response)

# walrus-cond-switch-response: CPU
(cd "$REPO/switch" && python3 analyze_switch.py --eval walrus-cond-switch-eval-switch --name walrus-cond-switch-response)

# walrus-ft-response: CPU
(cd "$REPO/switch" && python3 analyze_switch.py --eval walrus-ft-eval-switch --name walrus-ft-response)

# multi-switch evaluation and analysis

# walrus-ft-eval-multiswitch: 2 GPUs
(cd "$REPO/switch" && torchrun --standalone --nproc_per_node=2 -- walrus_ms_eval.py --run walrus-ft --ckpt last --frames 64 --root plain --name walrus-ft-eval-multiswitch)

# walrus-cond-eval-multiswitch: 2 GPUs
(cd "$REPO/switch" && torchrun --standalone --nproc_per_node=2 -- walrus_ms_eval.py --run walrus-cond --ckpt last --frames 64 --root cond --name walrus-cond-eval-multiswitch)

# walrus-cond-switch-eval-multiswitch: 2 GPUs
(cd "$REPO/switch" && torchrun --standalone --nproc_per_node=2 -- walrus_ms_eval.py --run walrus-cond-switch --ckpt last --frames 64 --root cond --name walrus-cond-switch-eval-multiswitch)

# walrus-ft-multiswitch: CPU
(cd "$REPO/switch" && python3 analyze_ms.py --eval walrus-ft-eval-multiswitch --name walrus-ft-multiswitch)

# walrus-cond-multiswitch: CPU
(cd "$REPO/switch" && python3 analyze_ms.py --eval walrus-cond-eval-multiswitch --name walrus-cond-multiswitch)

# walrus-cond-switch-multiswitch: CPU
(cd "$REPO/switch" && python3 analyze_ms.py --eval walrus-cond-switch-eval-multiswitch --name walrus-cond-switch-multiswitch)

# walrus-cond-switch-multiswitch-vs-control: CPU
(cd "$REPO/switch" && python3 analyze_ms.py --eval walrus-cond-switch-eval-multiswitch --vs walrus-cond-eval-multiswitch --name walrus-cond-switch-multiswitch-vs-control)
