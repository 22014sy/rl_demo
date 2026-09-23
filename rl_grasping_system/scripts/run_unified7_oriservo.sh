#!/usr/bin/env bash
# 统一口径 7 路 × 4 场景重测（2026-09-15）
#
# 相对旧 unified5 口径的两处改动：
#   1. --ori-servo on：补上速度模式缺失的角速度通道（旧口径下姿态完全不受控——接近轴倾角
#      从 reset 原样保留到最后，倾角 >12° 就把 pad 中点推出 grasp_align_xy_tol=0.03，
#      静态场景 4/30 失败全源于此，且与标称层选谁无关）。
#   2. 复合判据 success_nc_rate =「零臂身碰撞 且 抓取成功」——单看 success_rate 会把
#      「撞着障碍硬穿过去抓到了」也算成功（velocity_field 标称在 static3 上正是如此）。
#
# 残差门控按各模型训练语义给：v18 训练于门控之前 → off；v25b2 带门控训练 → on。
# 幂等：已存在且满 30 集的格跳过，可反复重跑续传。
set -u
cd "$(dirname "$0")/.."        # -> rl_grasping_system
OUT=results/unified7os
mkdir -p "$OUT"

SCENES=(static dyn_target dyn_both static3)
ORDER=(vf_nominal vf_res18 mpc_nominal mpc_res25b2 mpc_armaware mpc_armaware_res25b2 e2e_v11)
declare -A ARGS=(
  [vf_nominal]="--nominal-mode velocity_field --zero-residual"
  [vf_res18]="--nominal-mode velocity_field --model models/final_model_stage2_d2_v18_p3f.zip --residual-gate off"
  [mpc_nominal]="--nominal-mode mpc --zero-residual"
  [mpc_res25b2]="--nominal-mode mpc --model models/final_model_stage2_d2_v25b2_gate.zip --residual-gate on"
  [mpc_armaware]="--nominal-mode mpc --zero-residual --arm-aware"
  [mpc_armaware_res25b2]="--nominal-mode mpc --model models/final_model_stage2_d2_v25b2_gate.zip --arm-aware --residual-gate on"
  [e2e_v11]="--action-mode delta --model models/final_model_stage2_d2_v11_500k.zip"
)

for sc in "${SCENES[@]}"; do
  for tag in "${ORDER[@]}"; do
    f="$OUT/${tag}_${sc}_dsafe005_oriservo_n30.json"
    if [ -f "$f" ] && python3 -c "
import json,sys
try: sys.exit(0 if json.load(open('$f')).get('n_episodes')==30 else 1)
except Exception: sys.exit(1)" 2>/dev/null; then
      echo "[skip] $(basename "$f")"; continue
    fi
    echo "[run ] $tag / $sc  $(date +%H:%M:%S)"
    python3 scripts/mpc_plus_rl_eval.py --scene "$sc" ${ARGS[$tag]} \
      --d-safe 0.05 --seed 12345 --seed-per-episode --n_episodes 30 --max-steps 200 \
      --ori-servo on --save-results "$f" 2>&1 | grep -E "^scene=|saved ->" \
      || echo "[FAIL] $tag/$sc"
  done
done
echo "ALL DONE $(date +%H:%M:%S)"
