#!/usr/bin/env bash
# Data: public downloads, own Kolmogorov / NSU simulations, Kolmogorov switch trajectories.
set -e
source "$(dirname "${BASH_SOURCE[0]}")/00_env.sh"

# The Well (six families) into $PHYSDIAL_EXT_DATA/the_well (CPU)

# fetch-well-trl-gs: CPU
(cd "$REPO/data_gen/the_well" && python3 fetch_well.py --datasets turbulent_radiative_layer_2D,gray_scott_reaction_diffusion --name fetch-well-trl-gs)

# fetch-well-rb: CPU
(cd "$REPO/data_gen/the_well" && python3 fetch_well.py --datasets rayleigh_benard --name fetch-well-rb)

# fetch-well-sf: CPU
(cd "$REPO/data_gen/the_well" && python3 fetch_well.py --datasets shear_flow --name fetch-well-sf)

# fetch-well-acoustic: CPU
(cd "$REPO/data_gen/the_well" && python3 fetch_well.py --datasets acoustic_scattering_inclusions --name fetch-well-acoustic)

# fetch-well-euler: CPU
(cd "$REPO/data_gen/the_well" && python3 fetch_well.py --datasets euler_multi_quadrants_openBC --include train:_chunk_0.hdf5,test:_chunk_40.hdf5 --name fetch-well-euler)

# fetch-wan21-t2v-1.3b: CPU; pre-downloads the backbone into $HF_HOME (optional: from_pretrained also downloads on first use)
(cd "$REPO/physdial" && python3 fetch_hf.py --repo Wan-AI/Wan2.1-T2V-1.3B-Diffusers --subfolders transformer,vae,text_encoder,tokenizer,scheduler --name fetch-wan21-t2v-1.3b)

# fetch-wan21-t2v-14b: CPU; only for the 14B comparison (14_ablations.sh)
(cd "$REPO/physdial" && python3 fetch_hf.py --repo Wan-AI/Wan2.1-T2V-14B-Diffusers --subfolders transformer --name fetch-wan21-t2v-14b)

# Walrus: download the released checkpoint (https://huggingface.co/polymathic-ai/walrus) to $WALRUS_CKPT_DIR (walrus.pt, extended_config.yaml)
# and clone https://github.com/PolymathicAI/walrus to $WALRUS_SRC. DPOT: clone https://github.com/HaoZhongkai/DPOT to $DPOT_SRC and place
# the released DPOT-H weights (model_H.pth, https://huggingface.co/hzk17/DPOT) at $DPOT_WEIGHTS.

# Kolmogorov flow and decaying turbulence (own pseudo-spectral solver, 1 GPU per job) into $PHYSDIAL_DATA_ROOT/2d: frames on the solver grid every
# 0.1 eddy-turnover times, 300 per trajectory

# main (k_f, amplitude) sweep, every nu: evaluation and training files (12 trajectories each)

# gen-kolmo-sweep-kf2-amp0.5: 1 GPU
(cd "$REPO/data_gen/kolmo2d" && python3 gen_job.py --set all --seed_offset 0 --only 2,0.5 --name gen-kolmo-sweep-kf2-amp0.5)

# gen-kolmo-sweep-kf2-amp1.0: 1 GPU
(cd "$REPO/data_gen/kolmo2d" && python3 gen_job.py --set all --seed_offset 0 --only 2,1.0 --name gen-kolmo-sweep-kf2-amp1.0)

# gen-kolmo-sweep-kf3-amp0.5: 1 GPU
(cd "$REPO/data_gen/kolmo2d" && python3 gen_job.py --set all --seed_offset 0 --only 3,0.5 --name gen-kolmo-sweep-kf3-amp0.5)

# gen-kolmo-sweep-kf3-amp1.0: 1 GPU
(cd "$REPO/data_gen/kolmo2d" && python3 gen_job.py --set all --seed_offset 0 --only 3,1.0 --name gen-kolmo-sweep-kf3-amp1.0)

# gen-kolmo-sweep-kf4-amp0.5: 1 GPU
(cd "$REPO/data_gen/kolmo2d" && python3 gen_job.py --set all --seed_offset 0 --only 4,0.5 --name gen-kolmo-sweep-kf4-amp0.5)

# gen-kolmo-sweep-kf4-amp1.0: 1 GPU
(cd "$REPO/data_gen/kolmo2d" && python3 gen_job.py --set all --seed_offset 0 --only 4,1.0 --name gen-kolmo-sweep-kf4-amp1.0)

# gen-kolmo-sweep-kf6-amp0.5: 1 GPU
(cd "$REPO/data_gen/kolmo2d" && python3 gen_job.py --set all --seed_offset 0 --only 6,0.5 --name gen-kolmo-sweep-kf6-amp0.5)

# gen-kolmo-sweep-kf6-amp1.0: 1 GPU
(cd "$REPO/data_gen/kolmo2d" && python3 gen_job.py --set all --seed_offset 0 --only 6,1.0 --name gen-kolmo-sweep-kf6-amp1.0)

# training files at the training (k_f, nu) points (162 trajectories each; the other sweep files are linked)

# gen-kolmo-train-kf2-i3: 1 GPU
(cd "$REPO/data_gen/kolmo2d" && python3 gen_job.py --set train --ntraj 162 --seed_offset 777 --only 2,1.0,3 --name gen-kolmo-train-kf2-i3)

# gen-kolmo-train-kf2-i4: 1 GPU
(cd "$REPO/data_gen/kolmo2d" && python3 gen_job.py --set train --ntraj 162 --seed_offset 777 --only 2,1.0,4 --name gen-kolmo-train-kf2-i4)

# gen-kolmo-train-kf2-i6: 1 GPU
(cd "$REPO/data_gen/kolmo2d" && python3 gen_job.py --set train --ntraj 162 --seed_offset 777 --only 2,1.0,6 --name gen-kolmo-train-kf2-i6)

# gen-kolmo-train-kf2-i7: 1 GPU
(cd "$REPO/data_gen/kolmo2d" && python3 gen_job.py --set train --ntraj 162 --seed_offset 777 --only 2,1.0,7 --name gen-kolmo-train-kf2-i7)

# gen-kolmo-train-kf4-i3: 1 GPU
(cd "$REPO/data_gen/kolmo2d" && python3 gen_job.py --set train --ntraj 162 --seed_offset 777 --only 4,1.0,3 --name gen-kolmo-train-kf4-i3)

# gen-kolmo-train-kf4-i4: 1 GPU
(cd "$REPO/data_gen/kolmo2d" && python3 gen_job.py --set train --ntraj 162 --seed_offset 777 --only 4,1.0,4 --name gen-kolmo-train-kf4-i4)

# gen-kolmo-train-kf4-i6: 1 GPU
(cd "$REPO/data_gen/kolmo2d" && python3 gen_job.py --set train --ntraj 162 --seed_offset 777 --only 4,1.0,6 --name gen-kolmo-train-kf4-i6)

# gen-kolmo-train-kf4-i7: 1 GPU
(cd "$REPO/data_gen/kolmo2d" && python3 gen_job.py --set train --ntraj 162 --seed_offset 777 --only 4,1.0,7 --name gen-kolmo-train-kf4-i7)

# dense nu axis (k_f 2, amplitude 1; 100 nu) and decaying turbulence (NSU; 101 nu), 16 trajectories per nu

# gen-kolmo-dense-0: 1 GPU
(cd "$REPO/data_gen/kolmo2d" && python3 gen_job.py --set dense --ntraj 16 --seed_offset 0 --only 0,25 --name gen-kolmo-dense-0)

# gen-kolmo-dense-1: 1 GPU
(cd "$REPO/data_gen/kolmo2d" && python3 gen_job.py --set dense --ntraj 16 --seed_offset 0 --only 25,50 --name gen-kolmo-dense-1)

# gen-kolmo-dense-2: 1 GPU
(cd "$REPO/data_gen/kolmo2d" && python3 gen_job.py --set dense --ntraj 16 --seed_offset 0 --only 50,75 --name gen-kolmo-dense-2)

# gen-kolmo-dense-3: 1 GPU
(cd "$REPO/data_gen/kolmo2d" && python3 gen_job.py --set dense --ntraj 16 --seed_offset 0 --only 75,100 --name gen-kolmo-dense-3)

# gen-nsu-0: 1 GPU
(cd "$REPO/data_gen/kolmo2d" && python3 gen_job.py --set decay --ntraj 16 --seed_offset 0 --only 0,26 --ic_wrms 5 --name gen-nsu-0)

# gen-nsu-1: 1 GPU
(cd "$REPO/data_gen/kolmo2d" && python3 gen_job.py --set decay --ntraj 16 --seed_offset 0 --only 26,51 --ic_wrms 5 --name gen-nsu-1)

# gen-nsu-2: 1 GPU
(cd "$REPO/data_gen/kolmo2d" && python3 gen_job.py --set decay --ntraj 16 --seed_offset 0 --only 51,76 --ic_wrms 5 --name gen-nsu-2)

# gen-nsu-3: 1 GPU
(cd "$REPO/data_gen/kolmo2d" && python3 gen_job.py --set decay --ntraj 16 --seed_offset 0 --only 76,101 --ic_wrms 5 --name gen-nsu-3)

# random multi-mode forcing (training band k 2-5, extrapolation band k 6-7)

# gen-kolmo-multif-nu5e-4-k25: 1 GPU
(cd "$REPO/data_gen/kolmo2d" && python3 gen_job.py --set multif --nu 5.3e-4 --kband 2,5 --nmodes 4 --ntraj 180 --seed_offset 0 --name gen-kolmo-multif-nu5e-4-k25)

# gen-kolmo-multif-nu1e-3-k25: 1 GPU
(cd "$REPO/data_gen/kolmo2d" && python3 gen_job.py --set multif --nu 1.2e-3 --kband 2,5 --nmodes 4 --ntraj 180 --seed_offset 0 --name gen-kolmo-multif-nu1e-3-k25)

# gen-kolmo-multif-nu5e-4-k67: 1 GPU
(cd "$REPO/data_gen/kolmo2d" && python3 gen_job.py --set multif --nu 5.3e-4 --kband 6,7 --nmodes 4 --ntraj 24 --seed_offset 0 --name gen-kolmo-multif-nu5e-4-k67)

# gen-kolmo-multif-nu1e-3-k67: 1 GPU
(cd "$REPO/data_gen/kolmo2d" && python3 gen_job.py --set multif --nu 1.2e-3 --kband 6,7 --nmodes 4 --ntraj 24 --seed_offset 0 --name gen-kolmo-multif-nu1e-3-k67)

# amplitude anchors (forcing amplitude 0.7 and 1.4) and the linear-drag evaluation columns

# gen-kolmo-anchor: 1 GPU
(cd "$REPO/data_gen/kolmo2d" && python3 gen_job.py --set anchor --ntraj 24 --seed_offset 0 --name gen-kolmo-anchor)

# gen-kolmo-drag-a0.05: 1 GPU
(cd "$REPO/data_gen/kolmo2d" && python3 gen_job.py --set drag --alpha 0.05 --ntraj 16 --seed_offset 0 --name gen-kolmo-drag-a0.05)

# gen-kolmo-drag-a0.2: 1 GPU
(cd "$REPO/data_gen/kolmo2d" && python3 gen_job.py --set drag --alpha 0.2 --ntraj 16 --seed_offset 0 --name gen-kolmo-drag-a0.2)

# beta-plane coefficient: anchors beta = +-2.4 at the training (k_f, nu), held-out column beta = 4 (14_ablations.sh)

# gen-kolmo-beta-anchor-p2.4: 1 GPU
(cd "$REPO/data_gen/kolmo2d" && python3 gen_job.py --set beta --beta 2.4 --cols anchor --ntraj 24 --seed_offset 0 --name gen-kolmo-beta-anchor-p2.4)

# gen-kolmo-beta-anchor-m2.4: 1 GPU
(cd "$REPO/data_gen/kolmo2d" && python3 gen_job.py --set beta --beta -2.4 --cols anchor --ntraj 24 --seed_offset 0 --name gen-kolmo-beta-anchor-m2.4)

# gen-kolmo-beta-column-4: 1 GPU
(cd "$REPO/data_gen/kolmo2d" && python3 gen_job.py --set beta --beta 4 --cols column --ntraj 16 --seed_offset 0 --name gen-kolmo-beta-column-4)

# Kolmogorov switch trajectories (a condition changes mid-trajectory) into $PHYSDIAL_DATA_ROOT/switch/<subdir>

# gen-switch-eval: 1 GPU; evaluation groups (groups_kolmo.json)
(cd "$REPO/switch" && python3 gen_switch.py --n 32 --subdir kolmo --name gen-switch-eval)

# gen-switch-train: 1 GPU; training switch trajectories
(cd "$REPO/switch" && python3 gen_switch.py --groups groups_kolmo_train.json --n 8 --after 48 --seed 1 --subdir kolmo_train --name gen-switch-train)

# gen-switch-ref: 1 GPU; switches into held-out values, for the in-range reference of the switch study
(cd "$REPO/switch" && python3 gen_switch.py --groups groups_kolmo_ref.json --n 8 --after 48 --seed 2 --subdir kolmo_ref --name gen-switch-ref)

# gen-multiswitch: 1 GPU; multi-switch evaluation trajectories
(cd "$REPO/switch" && python3 gen_ms.py --n 32 --subdir kolmo_ms --name gen-multiswitch)
