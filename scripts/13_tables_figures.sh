#!/usr/bin/env bash
# Tables and case-study figures.
set -e
source "$(dirname "${BASH_SOURCE[0]}")/00_env.sh"

# main table, paired robustness statistics and physics report (CPU; read the result files of the runs above)

# CPU; writes physdial/results/tables/ (main table, markdown and JSON)
(cd "$REPO/tables" && python3 main_table.py)

# CPU; paired statistics
(cd "$REPO/tables" && python3 robust.py --out "$REPO/physdial/results/tables/robust.md" --json "$REPO/physdial/results/tables/robust.json")

# CPU; physics report
(cd "$REPO/tables" && python3 physics_report.py)

# CPU; per-column errors at steps 4 / 16 (/ 64) of every model and the extrapolation cost: column_table.md, .json
(cd "$REPO/tables" && python3 column_table.py)

# physical-field diagnostics of the 64-step rollouts (CPU)

# physics-fields-64step: CPU
(cd "$REPO/tables" && python3 physics_fields.py --ours physdial-s0-eval-64step --walrus walrus-ft-eval-64step --dpot dpot-ft-eval-64step --families kolmo,nsu --name physics-fields-64step)

# plot data of the figures (CPU): switch curves, controlled switch study, 64-step rollouts, spectra -> physdial/results/plot_data/plot-data/

# plot-data: CPU
(cd "$REPO/tables" && python3 plot_data.py --name plot-data)

# case-study figures (CPU)

# case-switch-frames: CPU
(cd "$REPO/switch" && python3 case_study.py --eval physdial-switch-s0-eval-switch --family kolmo --groups shape_in,kf_in,shape_out,kf_out,amp_down_out --top 3 --export --name case-switch-frames)

# case-switch-compare: CPU
(cd "$REPO/switch" && python3 case_study.py --eval physdial-switch-s0-eval-switch --family kolmo --groups shape_in,kf_in,shape_out,kf_out,amp_down_out --top 3 --compare physdial-switch-control-s0-eval-switch:same_model_without_switch_training,walrus-cond-switch-eval-switch:Walrus_with_condition_input_and_switch_data,unet-switch-kolmo-s0-eval-switch:U-Net_specialist_with_switch_data --ours_label PhysDial_switch-trained --mosaic --gif --name case-switch-compare)

# case-multiswitch: CPU
(cd "$REPO/switch" && python3 ms_case.py --eval physdial-switch-s0-eval-multiswitch --top 2 --after 4 --gif --name case-multiswitch)
