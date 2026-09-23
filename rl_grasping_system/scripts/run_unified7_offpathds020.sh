#!/usr/bin/env bash
# 第三轮：off-path 障碍 + d_safe=0.20 + servo=off（2026-09-15）
#
# 目的：把「障碍摆放」这一项单独隔离出来。现有对照的 d_safe 都不一致
#   v1(unified7os) = off-path, d_safe 0.05, servo on
#   v2(unified7v2) = 抬高球,  d_safe 0.10, servo on
#   本轮          = off-path, d_safe 0.20, servo off  <- 与 run_unified7_trainmatch.sh
#                    (on-path, d_safe 0.20, servo off) **只差障碍摆放**一项
# → 两轮相减 = 障碍摆放的净效应；与 trainmatch 的 servo=on 轮一起构成
#   (摆放 off/on) × (姿态伺服 off/on) 在 d_safe=0.20 上的 2×2。
#   若「速度场标称最好」在 off-path 下复现、在 on-path 下翻转，则摆放假说成立。
#
# 其余口径与 trainmatch 完全一致（不覆盖 d_safe/arm-aware，max_steps=200，seed 12345 逐集）。
set -u
cd "$(dirname "$0")/.."        # -> rl_grasping_system
OUT=results/unified7_trainmatch
mkdir -p "$OUT"

SCENE=dyn_both                 # count=1 → obstacle_on_nominal_path=False → off-path 固定位
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

SUF="_offpath_ds020_servooff_n30.json"
for tag in "${ORDER[@]}"; do
  f="$OUT/${tag}_${SCENE}${SUF}"
  if [ -f "$f" ] && python3 -c "
import json,sys
try: sys.exit(0 if json.load(open('$f')).get('n_episodes')==30 else 1)
except Exception: sys.exit(1)" 2>/dev/null; then
    echo "[skip] $(basename "$f")"; continue
  fi
  echo "[run ] offpath/ds020/servo=off  $tag / $SCENE  $(date +%H:%M:%S)"
  python3 scripts/mpc_plus_rl_eval.py --scene "$SCENE" ${ARGS[$tag]} \
    --d-safe 0.20 --seed 12345 --seed-per-episode --n_episodes 30 \
    --max-steps 200 --ori-servo off --save-results "$f" 2>&1 \
    | grep -E "^scene=|saved ->" \
    || echo "[FAIL] $tag/$SCENE"
done
echo "ALL DONE $(date +%H:%M:%S)"
