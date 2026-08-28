#!/bin/bash
# v21_noise（鲁棒化训练）感知噪声评估（2026-08-28）
set -euo pipefail
cd "$(dirname "$0")/.."
export MUJOCO_GL=egl
OUT=results/perception
V21=models/final_model_stage2_d2_v21_p3i_noise.zip
N=30
run_eval() {
    local name=$1 noise=$2 dropout=$3 scene=$4
    echo "[$(date +%H:%M:%S)] v21 $name (noise=$noise dropout=$dropout) ..."
    python3 evaluate.py --model_path "$V21" --n_episodes "$N" --action-mode residual \
        --perception-noise-std "$noise" --perception-dropout "$dropout" \
        --scenario-mix "$scene" --target-vel 0.05 --obstacle-vel 0.05 --obstacle-count 1 \
        --save_results "$OUT/v21_$name.json" --save_plot "$OUT/v21_$name.png" --log_level WARNING
    echo "[$(date +%H:%M:%S)] v21 $name done"
}
run_eval dyn_oracle    0.00 0.0 "0,0,1"
run_eval dyn_n002      0.02 0.0 "0,0,1"
run_eval dyn_n002_d01  0.02 0.1 "0,0,1"
run_eval dyn_n002_d02  0.02 0.2 "0,0,1"
run_eval static_oracle 0.00 0.0 "1,0,0"
run_eval static_n002   0.02 0.0 "1,0,0"
echo "[$(date +%H:%M:%S)] v21 评估完成"
