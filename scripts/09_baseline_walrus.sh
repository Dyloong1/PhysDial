#!/usr/bin/env bash
# Baseline: Walrus fine-tuned with its own training code on the same data (exported to the Walrus format).
set -e
source "$(dirname "${BASH_SOURCE[0]}")/00_env.sh"

# Walrus / DPOT commands run in a Python environment with the Walrus package installed (see README).

# export the corpus to the Walrus dataset format (CPU)

# walrus-data: CPU
(cd "$REPO/physdial" && python3 walrus_data.py --eval_n 32 --frames 16 --name walrus-data)

# walrus-data-anchor: CPU
(cd "$REPO/physdial" && python3 walrus_data.py --families kolmo --kolmo_extra anchor,drag --eval_n 32 --frames 16 --name walrus-data-anchor)

# walrus-data-64step: CPU
(cd "$REPO/physdial" && python3 walrus_data.py --families kolmo,nsu --kolmo_extra drag --eval_n 32 --frames 64 --skip_train --name walrus-data-64step)

# fine-tune from the released checkpoint, then the anchor continuation

# walrus-ft-stage1: 4 GPUs
(cd "$REPO/physdial/baselines/walrus" && torchrun --standalone --nproc_per_node=4 -- walrus_ft.py --epochs 51 --max_samples 500 --name walrus-ft-stage1)

# walrus-ft: 4 GPUs
(cd "$REPO/physdial/baselines/walrus" && torchrun --standalone --nproc_per_node=4 -- walrus_ft.py --data physdial_corpus_anchor --init_from walrus-ft-stage1 --lr 3e-5 --warmup_epochs 1 --cooldown_epochs 6 --epochs 18 --max_samples 500 --name walrus-ft)

# evaluation on the shared windows

# walrus-ft-eval: 4 GPUs
(cd "$REPO/physdial/baselines/walrus" && torchrun --standalone --nproc_per_node=4 -- walrus_eval.py --run walrus-ft --ckpt last --name walrus-ft-eval)

# walrus-ft-eval-64step: 4 GPUs
(cd "$REPO/physdial/baselines/walrus" && torchrun --standalone --nproc_per_node=4 -- walrus_eval.py --run walrus-ft --ckpt last --families kolmo,nsu --frames 64 --dump_pred 32 --dump_match physdial-s0-eval-64step --name walrus-ft-eval-64step)
