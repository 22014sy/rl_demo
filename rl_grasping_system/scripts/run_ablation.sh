#!/bin/bash
# ============================================================
# 三路消融对比评估：端到端 vs 残差 vs 纯标称（2026-08-28）
#   产出：results/ablation/{end2end,residual,nominal}_{static,dyn_target,dyn_both}.json
#   统一场景（每路 × 每场景 n=30，固定物体位置，保证三路公平）：
#     static     静态无障抓取
#     dyn_target 动态目标（0.05 m/s 往返）
#     dyn_both   动态目标 + 动态障碍（0.05 m/s 随机游走）
#   论证目标：标称解决静态、残差解决动态、端到端均不如残差。
# ============================================================
set -euo pipefail
cd "$(dirname "$0")/.."
export MUJOCO_GL=egl

N=30
OUT=results/ablation
mkdir -p "$OUT"
LOG=/tmp/ablation_logs

V11=models/final_model_stage2_d2_v11_500k.zip   # 端到端（delta，无标称）
V18=models/final_model_stage2_d2_v18_p3f.zip    # 残差（residual，达标产品模型）

run_eval() {
    local algo=$1 model=$2 mode=$3 zr=$4 scene=$5 extra=$6
    echo "[$(date +%H:%M:%S)] $algo / $scene ..."
    python3 evaluate.py \
        --model_path "$model" \
        --n_episodes "$N" \
        --action-mode "$mode" \
        --save_results "$OUT/${algo}_${scene}.json" \
        --save_plot "$OUT/${algo}_${scene}.png" \
        --log_level WARNING \
        $zr $extra
    echo "[$(date +%H:%M:%S)] $algo / $scene done"
}

# ---- 场景公共参数 ----
SCENE_STATIC="--scenario-mix 1,0,0"
SCENE_DYN_T="--scenario-mix 0,1,0 --target-vel 0.05"
SCENE_DYN_B="--scenario-mix 0,0,1 --target-vel 0.05 --obstacle-vel 0.05 --obstacle-count 1"

# ---- 三路 × 三场景 ----
for scene in "static:$SCENE_STATIC" "dyn_target:$SCENE_DYN_T" "dyn_both:$SCENE_DYN_B"; do
    name="${scene%%:*}"; params="${scene#*:}"
    run_eval end2end "$V11" delta ""     "$name" "$params"
    run_eval residual "$V18" residual "" "$name" "$params"
    run_eval nominal  "$V18" residual "--zero-residual" "$name" "$params"
done

echo "[$(date +%H:%M:%S)] 全部 9 个评估完成 → $OUT/"
