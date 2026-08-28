#!/bin/bash
# ============================================================
# D6 感知噪声鲁棒性评估（2026-08-28）
#   模型：v18_p3f（oracle 训练），评估时观测通道目标位置加噪 + 漏检
#   场景：动态目标+动态障碍（0,0,1）为主 + 静态抽查
#   验收：带感知噪声成功率相对 oracle 下降 <10%
#   产出：results/perception/*.json
# ============================================================
set -euo pipefail
cd "$(dirname "$0")/.."
export MUJOCO_GL=egl

OUT=results/perception
mkdir -p "$OUT"
V18=models/final_model_stage2_d2_v18_p3f.zip
N=30

run_eval() {
    local name=$1 noise=$2 dropout=$3 scene=$4 extra=$5
    echo "[$(date +%H:%M:%S)] $name (noise=$noise dropout=$dropout) ..."
    python3 evaluate.py --model_path "$V18" --n_episodes "$N" \
        --action-mode residual \
        --perception-noise-std "$noise" --perception-dropout "$dropout" \
        --scenario-mix "$scene" --target-vel 0.05 --obstacle-vel 0.05 --obstacle-count 1 \
        --save_results "$OUT/$name.json" --save_plot "$OUT/$name.png" \
        --log_level WARNING $extra
    echo "[$(date +%H:%M:%S)] $name done"
}

# 动态目标+动态障碍场景：噪声/漏检扫描
run_eval dyn_oracle       0.00 0.0 "0,0,1" ""
run_eval dyn_n001         0.01 0.0 "0,0,1" ""
run_eval dyn_n002         0.02 0.0 "0,0,1" ""
run_eval dyn_n003         0.03 0.0 "0,0,1" ""
run_eval dyn_n002_d01     0.02 0.1 "0,0,1" ""
run_eval dyn_n002_d02     0.02 0.2 "0,0,1" ""
# 静态场景抽查（无动态目标）
run_eval static_oracle    0.00 0.0 "1,0,0" ""
run_eval static_n002      0.02 0.0 "1,0,0" ""

echo "[$(date +%H:%M:%S)] 全部完成 -> $OUT/"
