#!/usr/bin/env bash
# Ablations and further experiments: beta-plane anchors, no anchors, condition inputs, initialisation, scale, objective, decoder, periodic handling, further variants.
set -e
source "$(dirname "${BASH_SOURCE[0]}")/00_env.sh"

# Ablations and further experiments of the paper. All use the eight families and the data / stores of 01-02.

# 1. beta-plane coefficient anchors (Kolmogorov flow with a beta term; beta = 0 in training, anchors at +-2.4, held-out column beta = 4)

# data: 01_data.sh (gen-kolmo-beta-*), stores: 02_text_and_stores.sh (encode-wan21-betaanchor, encode-wan21-betaref)

# without anchors: stage 2 without periodic handling, evaluated on the beta column

# stage2-noperiodic-s0-eval-beta: 2 GPUs
(cd "$REPO/physdial" && torchrun --standalone --nproc_per_node=2 -- train_gen.py --eval_only stage2-noperiodic-s0 --store wan21 --tok wan21ft:decoder.pt --kolmo_extra drag,beta --eval_splits kolmo/extrap_beta --name stage2-noperiodic-s0-eval-beta)

# stage2-noperiodic-s1-eval-beta: 2 GPUs
(cd "$REPO/physdial" && torchrun --standalone --nproc_per_node=2 -- train_gen.py --eval_only stage2-noperiodic-s1 --store wan21 --tok wan21ft:decoder.pt --kolmo_extra drag,beta --eval_splits kolmo/extrap_beta --name stage2-noperiodic-s1-eval-beta)

# with beta anchors: 20 GPU-h continuation (adds the beta parameter slot); the in-range reference also trains on beta = 4

# beta-anchor-s0: 4 GPUs; --budget_gpu_h 20
(cd "$REPO/physdial" && torchrun --standalone --nproc_per_node=4 -- train_gen.py --store wan21 --budget_gpu_h 20 --lr 3e-5 --init_weights stage2-noperiodic-s0 --equiv_eval --kolmo_extra anchor,drag,beta --name beta-anchor-s0)

# beta-anchor-s1: 4 GPUs; --budget_gpu_h 20
(cd "$REPO/physdial" && torchrun --standalone --nproc_per_node=4 -- train_gen.py --store wan21 --budget_gpu_h 20 --lr 3e-5 --init_weights stage2-noperiodic-s1 --equiv_eval --kolmo_extra anchor,drag,beta --seed 1 --name beta-anchor-s1)

# beta-anchor-ref-s0: 4 GPUs; --budget_gpu_h 20
(cd "$REPO/physdial" && torchrun --standalone --nproc_per_node=4 -- train_gen.py --store wan21 --budget_gpu_h 20 --lr 3e-5 --init_weights stage2-noperiodic-ref-s0 --reference --kolmo_extra anchor,beta --name beta-anchor-ref-s0)

# beta-anchor-s0-eval: 2 GPUs
(cd "$REPO/physdial" && torchrun --standalone --nproc_per_node=2 -- train_gen.py --eval_only beta-anchor-s0 --store wan21 --tok wan21ft:decoder.pt --kolmo_extra drag,beta --name beta-anchor-s0-eval)

# beta-anchor-s1-eval: 2 GPUs
(cd "$REPO/physdial" && torchrun --standalone --nproc_per_node=2 -- train_gen.py --eval_only beta-anchor-s1 --store wan21 --tok wan21ft:decoder.pt --kolmo_extra drag,beta --name beta-anchor-s1-eval)

# beta-anchor-ref-s0-eval-beta: 2 GPUs
(cd "$REPO/physdial" && torchrun --standalone --nproc_per_node=2 -- train_gen.py --eval_only beta-anchor-ref-s0 --store wan21 --tok wan21ft:decoder.pt --kolmo_extra drag,beta --eval_splits kolmo/extrap_beta --name beta-anchor-ref-s0-eval-beta)

# Walrus on the beta column: Walrus-format exports (Kolmogorov with the beta groups; the reference exports hold every ref_* split),
# Walrus's in-range reference from the released checkpoint (stage 1: no anchors), the continuations with the amplitude and beta anchors
# (from walrus-ft-stage1 and from the reference), evaluation. Every Walrus cost on Kolmogorov flow divides by walrus-ft-ref-beta-eval-kolmo
# (the reference continued with the amplitude and beta anchors, the same training stages as the models). Walrus / DPOT commands run in a Python environment with the Walrus package installed (see README).

# walrus-data-anchor-beta: CPU; train split + amplitude and beta anchors (physdial_kolmo_anchor_beta_r*)
(cd "$REPO/physdial" && python3 walrus_data.py --families kolmo --kolmo_extra anchor,drag,beta --eval_n 32 --frames 16 --name walrus-data-anchor-beta)

# walrus-data-ref: CPU; train split + every ref_* split, no anchors (physdial_kolmo_ref_r*)
(cd "$REPO/physdial" && python3 walrus_data.py --families kolmo --kolmo_extra drag,beta --reference --no_anchors --eval_n 32 --frames 16 --name walrus-data-ref)

# walrus-data-anchor-beta-ref: CPU; train split + every ref_* split + amplitude and beta anchors (physdial_kolmo_anchor_beta_ref_r*)
(cd "$REPO/physdial" && python3 walrus_data.py --families kolmo --kolmo_extra anchor,drag,beta --reference --eval_n 32 --frames 16 --name walrus-data-anchor-beta-ref)

# walrus-ft-beta: 4 GPUs
(cd "$REPO/physdial/baselines/walrus" && torchrun --standalone --nproc_per_node=4 -- walrus_ft.py --data physdial_corpus_anchor_beta --init_from walrus-ft-stage1 --lr 3e-5 --warmup_epochs 1 --cooldown_epochs 6 --epochs 18 --max_samples 500 --name walrus-ft-beta)

# walrus-ft-ref-stage1: 4 GPUs
(cd "$REPO/physdial/baselines/walrus" && torchrun --standalone --nproc_per_node=4 -- walrus_ft.py --data physdial_corpus_ref --epochs 51 --max_samples 500 --name walrus-ft-ref-stage1)

# walrus-ft-ref-beta: 4 GPUs
(cd "$REPO/physdial/baselines/walrus" && torchrun --standalone --nproc_per_node=4 -- walrus_ft.py --data physdial_corpus_anchor_beta_ref --init_from walrus-ft-ref-stage1 --lr 3e-5 --warmup_epochs 1 --cooldown_epochs 6 --epochs 18 --max_samples 500 --name walrus-ft-ref-beta)

# walrus-ft-eval-kolmo: 4 GPUs
(cd "$REPO/physdial/baselines/walrus" && torchrun --standalone --nproc_per_node=4 -- walrus_eval.py --run walrus-ft --ckpt last --families kolmo --name walrus-ft-eval-kolmo)

# walrus-ft-beta-eval-kolmo: 4 GPUs
(cd "$REPO/physdial/baselines/walrus" && torchrun --standalone --nproc_per_node=4 -- walrus_eval.py --run walrus-ft-beta --ckpt last --families kolmo --name walrus-ft-beta-eval-kolmo)

# walrus-ft-ref-beta-eval-kolmo: 4 GPUs
(cd "$REPO/physdial/baselines/walrus" && torchrun --standalone --nproc_per_node=4 -- walrus_eval.py --run walrus-ft-ref-beta --ckpt last --families kolmo --name walrus-ft-ref-beta-eval-kolmo)

# 2. amplitude anchors: stage 2 without anchors (and without periodic handling), seeds 0 and 1, and its in-range reference

# stage2-noanchor-s0: 4 GPUs; --budget_gpu_h 20
(cd "$REPO/physdial" && torchrun --standalone --nproc_per_node=4 -- train_gen.py --store wan21 --budget_gpu_h 20 --lr 3e-5 --init_weights physdial-stage1-s0 --equiv_eval --name stage2-noanchor-s0)

# stage2-noanchor-s1: 4 GPUs; --budget_gpu_h 20
(cd "$REPO/physdial" && torchrun --standalone --nproc_per_node=4 -- train_gen.py --store wan21 --budget_gpu_h 20 --lr 3e-5 --init_weights physdial-stage1-s0 --equiv_eval --seed 1 --name stage2-noanchor-s1)

# stage2-noanchor-ref-s0: 4 GPUs; --budget_gpu_h 20
(cd "$REPO/physdial" && torchrun --standalone --nproc_per_node=4 -- train_gen.py --store wan21 --budget_gpu_h 20 --lr 3e-5 --init_weights physdial-stage1-ref-s0 --reference --name stage2-noanchor-ref-s0)

# 3. condition inputs removed in training (stage 1, 60 GPU-h each): no numeric vector; no condition fields

# ablation-nonum-s0: 4 GPUs; --budget_gpu_h 60
(cd "$REPO/physdial" && torchrun --standalone --nproc_per_node=4 -- train_gen.py --store wan21 --ens 8 --ens_n 16 --ens_splits all --cond nonum --budget_gpu_h 60 --name ablation-nonum-s0)

# ablation-nofield-s0: 4 GPUs; --budget_gpu_h 60
(cd "$REPO/physdial" && torchrun --standalone --nproc_per_node=4 -- train_gen.py --store wan21 --ens 8 --ens_n 16 --ens_splits all --cond nofield --budget_gpu_h 60 --name ablation-nofield-s0)

# condition fields zeroed at evaluation: the stage-1 model on the families with condition fields (Kolmogorov forcing, acoustic medium),
# the same protocol as its own evaluation at the end of training (frozen decoder)

# physdial-stage1-s0-eval-nofields: 4 GPUs
(cd "$REPO/physdial" && torchrun --standalone --nproc_per_node=4 -- train_gen.py --eval_only physdial-stage1-s0 --store wan21 --drop_fields --eval_splits kolmo/test,kolmo/interp_nu,kolmo/extrap_nu_low,kolmo/extrap_nu_high,kolmo/interp_kf,kolmo/extrap_kf,kolmo/extrap_amp,kolmo/extrap_joint,kolmo/nu_dense_interp,kolmo/nu_dense_ends,kolmo/multif_test,kolmo/extrap_kband,acoustic/test --name physdial-stage1-s0-eval-nofields)

# 4. random initialisation (same data and budget as stage 1) and its evaluation on the switch views

# ablation-random-init-s0: 4 GPUs; --budget_gpu_h 60
(cd "$REPO/physdial" && torchrun --standalone --nproc_per_node=4 -- train_gen.py --store wan21 --ens 8 --ens_n 16 --ens_splits all --init random --budget_gpu_h 60 --name ablation-random-init-s0)

# ablation-random-init-s0-eval-switch: 2 GPUs
(cd "$REPO/switch" && torchrun --standalone --nproc_per_node=2 -- eval_switch.py --run ablation-random-init-s0 --setup study --groups amp_up_in,amp_down_out,kf_in,kf_out,nu_up_in,nu_down_in,nu_up_out,nu_down_out,shape_in,shape_out --views B,A,N,T --frames 16 --dump_pred 32 --name ablation-random-init-s0-eval-switch)

# ablation-random-init-s0-response: CPU
(cd "$REPO/switch" && python3 analyze_switch.py --eval ablation-random-init-s0-eval-switch --name ablation-random-init-s0-response)

# 5. scale: Wan2.1-T2V-14B with LoRA (rank 256) at the compute of stage 1, the same LoRA on the 1.3B model, and the 14B in-range reference

# scale-14b-lora-s0: 4 GPUs; 10,400 steps = about 60 GPU-h
(cd "$REPO/physdial" && torchrun --standalone --nproc_per_node=4 -- train_gen.py --store wan21 --lora_r 256 --lora_alpha 32 --no_ema --body_wd 0 --frozen_bf16 --kolmo_extra anchor --budget_steps 10400 --warmup 1000 --decay_frac 0.15 --eval_splits none --size 14b --lr 5e-4 --grad_ckpt --name scale-14b-lora-s0)

# scale-14b-lora-ref-s0: 4 GPUs
(cd "$REPO/physdial" && torchrun --standalone --nproc_per_node=4 -- train_gen.py --store wan21 --lora_r 256 --lora_alpha 32 --no_ema --body_wd 0 --frozen_bf16 --kolmo_extra anchor --budget_steps 10400 --warmup 1000 --decay_frac 0.15 --eval_splits none --size 14b --lr 5e-4 --grad_ckpt --reference --name scale-14b-lora-ref-s0)

# scale-1.3b-lora-s0: 4 GPUs
(cd "$REPO/physdial" && torchrun --standalone --nproc_per_node=4 -- train_gen.py --store wan21 --lora_r 256 --lora_alpha 32 --no_ema --body_wd 0 --frozen_bf16 --kolmo_extra anchor --budget_steps 10400 --warmup 1000 --decay_frac 0.15 --eval_splits none --lr 1e-3 --name scale-1.3b-lora-s0)

# evaluation (4 Euler steps with sampling shift 1)

# scale-14b-lora-s0-eval: 4 GPUs
(cd "$REPO/physdial" && torchrun --standalone --nproc_per_node=4 -- train_gen.py --eval_only scale-14b-lora-s0 --store wan21 --ode_steps 4 --sample_shift 1 --ens 8 --ens_n 16 --ens_splits kolmo/test,kolmo/extrap_amp,kolmo/extrap_kf,nsu/test,gs/test,sf/test --name scale-14b-lora-s0-eval)

# scale-14b-lora-ref-s0-eval: 4 GPUs
(cd "$REPO/physdial" && torchrun --standalone --nproc_per_node=4 -- train_gen.py --eval_only scale-14b-lora-ref-s0 --store wan21 --ode_steps 4 --sample_shift 1 --ens 8 --ens_n 16 --ens_splits kolmo/test,kolmo/extrap_amp,kolmo/extrap_kf,nsu/test,gs/test,sf/test --name scale-14b-lora-ref-s0-eval)

# scale-1.3b-lora-s0-eval: 4 GPUs
(cd "$REPO/physdial" && torchrun --standalone --nproc_per_node=4 -- train_gen.py --eval_only scale-1.3b-lora-s0 --store wan21 --ode_steps 4 --sample_shift 1 --ens 8 --ens_n 16 --ens_splits kolmo/test,kolmo/extrap_amp,kolmo/extrap_kf,nsu/test,gs/test,sf/test --name scale-1.3b-lora-s0-eval)

# 6. regression instead of flow matching (stage 1, 60 GPU-h), and its 64-step rollouts

# ablation-regress-s0: 4 GPUs; --budget_gpu_h 60
(cd "$REPO/physdial" && torchrun --standalone --nproc_per_node=4 -- train_gen.py --store wan21 --ens 1 --ens_n 16 --ens_splits all --objective regress --budget_gpu_h 60 --name ablation-regress-s0)

# ablation-regress-s0-eval-64step: 2 GPUs
(cd "$REPO/physdial" && torchrun --standalone --nproc_per_node=2 -- train_gen.py --eval_only ablation-regress-s0 --store wan21 --tok wan21ft:decoder.pt --kolmo_extra drag --eval_chunks 16 --eval_splits kolmo/test,kolmo/interp_nu,kolmo/extrap_nu_low,kolmo/extrap_nu_high,kolmo/interp_kf,kolmo/extrap_kf,kolmo/extrap_amp,kolmo/extrap_joint,kolmo/nu_dense_interp,kolmo/nu_dense_ends,kolmo/multif_test,kolmo/extrap_kband,kolmo/extrap_drag_lo,kolmo/extrap_drag_hi,nsu/test,nsu/interp_test,nsu/extrap_ends --name ablation-regress-s0-eval-64step)

# 7. Walrus architecture trained from scratch (same data, settings and budget as walrus-ft-stage1). Walrus / DPOT commands run in a Python environment with the Walrus package installed (see README).

# walrus-scratch: 4 GPUs
(cd "$REPO/physdial/baselines/walrus" && torchrun --standalone --nproc_per_node=4 -- walrus_ft.py --init random --epochs 51 --max_samples 500 --name walrus-scratch)

# walrus-scratch-eval: 4 GPUs
(cd "$REPO/physdial/baselines/walrus" && torchrun --standalone --nproc_per_node=4 -- walrus_eval.py --run walrus-scratch --ckpt last --name walrus-scratch-eval)

# 8. decoder fine-tuning: the same weights evaluated with the fine-tuned decoder (the training runs' own evaluations use the frozen decoder)

# physdial-stage1-s0-eval-decoder: 4 GPUs
(cd "$REPO/physdial" && torchrun --standalone --nproc_per_node=4 -- train_gen.py --eval_only physdial-stage1-s0 --store wan21 --tok wan21ft:decoder.pt --name physdial-stage1-s0-eval-decoder)

# physdial-stage1-ref-s0-eval-decoder: 4 GPUs
(cd "$REPO/physdial" && torchrun --standalone --nproc_per_node=4 -- train_gen.py --eval_only physdial-stage1-ref-s0 --store wan21 --tok wan21ft:decoder.pt --name physdial-stage1-ref-s0-eval-decoder)

# stage2-noperiodic-s0-eval-decoder: 2 GPUs
(cd "$REPO/physdial" && torchrun --standalone --nproc_per_node=2 -- train_gen.py --eval_only stage2-noperiodic-s0 --store wan21 --tok wan21ft:decoder.pt --kolmo_extra drag --name stage2-noperiodic-s0-eval-decoder)

# stage2-noperiodic-s1-eval-decoder: 2 GPUs
(cd "$REPO/physdial" && torchrun --standalone --nproc_per_node=2 -- train_gen.py --eval_only stage2-noperiodic-s1 --store wan21 --tok wan21ft:decoder.pt --kolmo_extra drag --name stage2-noperiodic-s1-eval-decoder)

# 9. periodic handling: stage 2 with (physdial-s0 / -s1) and without (stage2-noperiodic-s0 / -s1, 04_train_stage2.sh) it, both seeds,
# evaluated with the training sampling shift (stage2-noperiodic-s*-eval-decoder above; physdial-s0-eval-shift3 in 07_eval_physdial.sh)

# physdial-s1-eval-shift3: 2 GPUs
(cd "$REPO/physdial" && torchrun --standalone --nproc_per_node=2 -- train_gen.py --eval_only physdial-s1 --store wan21_wrap --tok wan21ft:decoder-wrap.pt --kolmo_extra drag --name physdial-s1-eval-shift3)

# 10. linear probes of the conditions from the history windows (Kolmogorov evaluation windows of the Walrus export with the drag and beta
# columns, walrus-data-anchor-beta above): physics statistics of the frames (CPU) and Walrus's internal features (fine-tuned and released)

# probe-physics: CPU
(cd "$REPO/analysis" && python3 probe_conditions.py --no_walrus --name probe-physics)

# probe-walrus-ft: 1 GPU; Walrus environment
(cd "$REPO/analysis" && python3 probe_conditions.py --run walrus-ft --ckpt last --name probe-walrus-ft)

# probe-walrus-released: 1 GPU; Walrus environment
(cd "$REPO/analysis" && python3 probe_conditions.py --ckpt coalesced --name probe-walrus-released)

# 11. further variants (from stage 1, 20 GPU-h, without periodic handling, as stage2-noperiodic-s0): training on the model's own rollouts,
# stride-doubling augmentation, a 6-frame history (+ its in-range reference); projection onto physical constraints (below)

# variant-pushforward-s0: 4 GPUs
(cd "$REPO/physdial" && torchrun --standalone --nproc_per_node=4 -- train_gen.py --store wan21 --budget_gpu_h 20 --lr 3e-5 --init_weights physdial-stage1-s0 --equiv_eval --kolmo_extra anchor,drag --pushforward 0.5 --pf_ode_steps 4 --name variant-pushforward-s0)

# variant-stride-aug-s0: 4 GPUs
(cd "$REPO/physdial" && torchrun --standalone --nproc_per_node=4 -- train_gen.py --store wan21 --budget_gpu_h 20 --lr 3e-5 --init_weights physdial-stage1-s0 --equiv_eval --kolmo_extra anchor,drag --stride_aug 1,2 --name variant-stride-aug-s0)

# variant-stride-aug-s0-eval-stride2: 2 GPUs
(cd "$REPO/physdial" && torchrun --standalone --nproc_per_node=2 -- train_gen.py --eval_only variant-stride-aug-s0 --store wan21 --tok wan21ft:decoder.pt --kolmo_extra drag --eval_stride_mult 2 --name variant-stride-aug-s0-eval-stride2)

# stage2-noperiodic-s0-eval-stride2: 2 GPUs
(cd "$REPO/physdial" && torchrun --standalone --nproc_per_node=2 -- train_gen.py --eval_only stage2-noperiodic-s0 --store wan21 --tok wan21ft:decoder.pt --kolmo_extra drag --eval_stride_mult 2 --name stage2-noperiodic-s0-eval-stride2)

# variant-history6-s0: 4 GPUs
(cd "$REPO/physdial" && torchrun --standalone --nproc_per_node=4 -- train_gen.py --store wan21 --budget_gpu_h 20 --lr 3e-5 --init_weights physdial-stage1-s0 --kolmo_extra anchor,drag --k_in 6 --allow_diff k_in --equiv_eval --name variant-history6-s0)

# variant-history6-ref-s0: 4 GPUs
(cd "$REPO/physdial" && torchrun --standalone --nproc_per_node=4 -- train_gen.py --store wan21 --budget_gpu_h 20 --lr 3e-5 --init_weights physdial-stage1-ref-s0 --reference --kolmo_extra anchor,drag --k_in 6 --allow_diff k_in --name variant-history6-ref-s0)

# projection onto physical constraints (zero mean, divergence-free velocity, non-negativity) of the saved predictions of stage2-noperiodic-s0
# and of its in-range reference

# stage2-noperiodic-s0-eval-dump: 2 GPUs
(cd "$REPO/physdial" && torchrun --standalone --nproc_per_node=2 -- train_gen.py --eval_only stage2-noperiodic-s0 --store wan21 --tok wan21ft:decoder.pt --kolmo_extra drag --dump_pred 32 --eval_splits acoustic/test,euler/extrap_gamma_high,euler/extrap_gamma_low,euler/test,gs/extrap_bubbles,gs/interp_maze,gs/test,kolmo/extrap_amp,kolmo/extrap_drag_hi,kolmo/extrap_drag_lo,kolmo/extrap_joint,kolmo/extrap_kband,kolmo/extrap_kf,kolmo/extrap_nu_high,kolmo/extrap_nu_low,kolmo/interp_kf,kolmo/interp_nu,kolmo/multif_test,kolmo/nu_dense_ends,kolmo/nu_dense_interp,kolmo/test,nsu/extrap_ends,nsu/interp_test,nsu/test,rb/extrap_Ra,rb/interp_Pr,rb/test,sf/extrap_Re,sf/interp_Sc,sf/test,trl/extrap_tcool,trl/interp_tcool,trl/test --name stage2-noperiodic-s0-eval-dump)

# stage2-noperiodic-ref-s0-eval-dump: 2 GPUs
(cd "$REPO/physdial" && torchrun --standalone --nproc_per_node=2 -- train_gen.py --eval_only stage2-noperiodic-ref-s0 --store wan21 --tok wan21ft:decoder.pt --dump_pred 32 --eval_splits acoustic/test,euler/extrap_gamma_high,euler/extrap_gamma_low,euler/test,gs/extrap_bubbles,gs/interp_maze,gs/test,kolmo/extrap_amp,kolmo/extrap_joint,kolmo/extrap_kband,kolmo/extrap_kf,kolmo/extrap_nu_high,kolmo/extrap_nu_low,kolmo/interp_kf,kolmo/interp_nu,kolmo/multif_test,kolmo/nu_dense_ends,kolmo/nu_dense_interp,kolmo/test,nsu/extrap_ends,nsu/interp_test,nsu/test,rb/extrap_Ra,rb/interp_Pr,rb/test,sf/extrap_Re,sf/interp_Sc,sf/test,trl/extrap_tcool,trl/interp_tcool,trl/test --name stage2-noperiodic-ref-s0-eval-dump)

# CPU
(cd "$REPO/analysis" && python3 projection.py --run stage2-noperiodic-s0-eval-dump --ref stage2-noperiodic-ref-s0-eval-dump --out results/projection.json)

# 12. repeat of the last Walrus stage (same command as walrus-ft, 09_baseline_walrus.sh). Walrus / DPOT commands run in a Python environment with the Walrus package installed (see README).

# walrus-ft-repeat: 4 GPUs
(cd "$REPO/physdial/baselines/walrus" && torchrun --standalone --nproc_per_node=4 -- walrus_ft.py --data physdial_corpus_anchor --init_from walrus-ft-stage1 --lr 3e-5 --warmup_epochs 1 --cooldown_epochs 6 --epochs 18 --max_samples 500 --name walrus-ft-repeat)

# walrus-ft-repeat-eval: 4 GPUs
(cd "$REPO/physdial/baselines/walrus" && torchrun --standalone --nproc_per_node=4 -- walrus_eval.py --run walrus-ft-repeat --ckpt last --name walrus-ft-repeat-eval)
