#!/usr/bin/env bash
# Baseline: DPOT-H fine-tuned on the same data (80 GPU-h).
set -e
source "$(dirname "${BASH_SOURCE[0]}")/00_env.sh"

# Walrus / DPOT commands run in a Python environment with the Walrus package installed (see README).

# reads the Walrus-format export of 09_baseline_walrus.sh

# dpot-ft: 4 GPUs; --budget_gpu_h 80
(cd "$REPO/physdial/baselines/dpot" && torchrun --standalone --nproc_per_node=4 -- dpot_ft.py --data physdial_corpus_anchor --budget_gpu_h 80 --batch 8 --lr 3e-4 --val_batches 20 --val_every_min 120 --name dpot-ft)

# dpot-ft-eval: 4 GPUs
(cd "$REPO/physdial/baselines/dpot" && torchrun --standalone --nproc_per_node=4 -- dpot_eval.py --run dpot-ft --name dpot-ft-eval)

# dpot-ft-eval-64step: 2 GPUs
(cd "$REPO/physdial/baselines/dpot" && torchrun --standalone --nproc_per_node=2 -- dpot_eval.py --run dpot-ft --families kolmo,nsu --frames 64 --dump_pred 32 --dump_match physdial-s0-eval-64step --name dpot-ft-eval-64step)
