#!/usr/bin/env bash
# 恒偏置感知误差扫掠（2026-09-16）——白噪声测不出东西，换成有偏的误差再测。
#
# 为什么加这一轮（结论来自 results/perception_sweep/）：
#   白噪声（perception_noise_std，每步重抽、均值为零）在 σ∈{0,0.02,0.04} 上**打不动成功率**：
#     纯标称 63.3/70.0/63.3%，v18 63.3/56.7/63.3%，v21 73.3/60.0/60.0% —— 非单调，是 n=30 波动带。
#   机制：速度场标称是 P 控制器，每步重新瞄准，130 步的零均值抖动自己平均掉，再加 α=0.7 低通
#     → 闭环对高频白噪声**天生免疫**。所以"让残差去补感知噪声"在这个设定下没有空间可补。
#   但真实感知误差不是白噪声，是**有偏的、慢变的**（标定偏了 / 分割 mask 系统性外扩）
#     → 目标估计恒定偏几厘米。恒偏**不会**被平均掉，直接把末端送错位置；
#     而抓取容差 grasp_align_xy_tol=0.03 m、立方体半宽 0.02 m → 偏 2~3cm 就该掉。
#
# 口径：`--perception-bias` 每集抽一次方向（单位向量×幅值）、集内恒定，方向取自 _perc_rng
#       （reset 里按 seed 重播种）→ 同 seed 的 A/B 两臂偏置方向**一致**，配对成立。
#       与噪声轮一样开 `--perception-affects-nominal`（标称层也吃这份带偏估计）。
#
# BIASES 里 **0 是基线**（bias=0 时开关自动关闭，与噪声轮的 σ=0 逐位等价）；
#   **0.05 是正对照**——如果连 5cm 偏置都打不动成功率，说明这个任务/判据对目标误差不敏感，
#   那"感知不是薄弱点"这个结论就不能只靠这组数据说话，得先查判据。
#
# ⚠️ 同 [[project-eval-config-divergence]]：这些开关晚于 cadd00e，只能在 HEAD 默认口径跑，
#    v18/v21 的残差量程被压到训练的约 40% → 噪声/偏置档之间比较自洽，绝对水平不可当成绩读。
#
# 成本：约 20 s/集（MPC 主导，纯标称更快）→ N=30 每格约 10 min。
# 小验证：N=30 bash scripts/run_perception_bias_sweep.sh          （默认 3 档 × 3 臂 = 9 格 ≈ 1.5h）
# 加细：BIASES="0 0.02 0.03 0.05" N=30 bash scripts/run_perception_bias_sweep.sh
set -u
cd "$(dirname "$0")/.."

OUT=results/perception_bias_sweep
N=${N:-30}
SCENE=dyn_both_train
mkdir -p "$OUT"

declare -A ARGS=(
  [vf]="--zero-residual --residual-gate off"
  [v18]="--model models/final_model_stage2_d2_v18_p3f.zip --residual-gate off"
  [v21]="--model models/final_model_stage2_d2_v21_p3i_noise.zip --residual-gate off"
)
for m in v18 v21; do
  p="models/final_model_stage2_d2_$( [ $m = v18 ] && echo v18_p3f || echo v21_p3i_noise )_vecnormalize.pkl"
  [ -f "$p" ] || { echo "[FAIL] 缺 $p（观测统计缺失会静默失配）"; exit 1; }
done

BIASES=${BIASES:-"0 0.02 0.05"}
for B in $BIASES; do
  for arm in vf v18 v21; do
    f="$OUT/${SCENE}_${arm}_b${B}.json"
    if [ -f "$f" ] && python3 -c "
import json,sys
try: sys.exit(0 if json.load(open('$f')).get('n_episodes')==$N else 1)
except Exception: sys.exit(1)" 2>/dev/null; then
      echo "[skip] $(basename "$f")"; continue
    fi
    echo "[run ] $SCENE $arm bias=$B n=$N  $(date +%H:%M:%S)"
    python3 scripts/mpc_plus_rl_eval.py --scene "$SCENE" --nominal-mode velocity_field ${ARGS[$arm]} \
      --seed 12345 --seed-per-episode --n_episodes "$N" --max-steps 200 \
      --perception-bias "$B" --perception-affects-nominal \
      --save-results "$f" 2>&1 | grep -E "^scene=|saved ->" || echo "[FAIL] $arm bias=$B"
  done
done

echo "ALL DONE $(date +%H:%M:%S)"
echo "画图（两臂一组，跑两次）："
for R in v18 v21; do
  echo "  python3 scripts/plot_ab_sweep.py --dir $OUT \\"
  echo "    --pattern '(?P<scene>.+)_(?P<arm>[^_]+)_b(?P<x>[0-9.]+)\\.json' \\"
  echo "    --baseline-tag vf --residual-tag $R \\"
  echo "    --baseline-name 'velocity field only' --residual-name 'velocity field + residual ($R)' \\"
  echo "    --xlabel 'Perception bias magnitude (m)' --out $OUT/perception_bias_${R}_vs_nominal.png"
done
