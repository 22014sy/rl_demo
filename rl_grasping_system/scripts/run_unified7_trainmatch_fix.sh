#!/usr/bin/env bash
# 第 23 集初始穿透修复后的 trainmatch 全量重跑（2026-09-16）。
#
# 背景：`dyn_both_train` 上四个 MPC 臂在两轮 servo 口径下都是「1/30 集碰撞，且都是第 23 集」
#   （avg 0.07），唯一成因是 reset 里「先摆 on-path 障碍、后把臂瞬移到 safe_config」——
#   摆放时的接触校验用的是瞬移前的臂位形，瞬移后障碍0 变成 −2.4cm 穿透，第 1 步即记 2 次碰撞，
#   四个 MPC 臂全中且都抓取成功，与策略无关。
#   修法见 environment.py `_ensure_obstacle_clearance`（臂位形定稿后沿路径法向外推障碍墙）。
#   已验证：只改第 23 集，其余 29 集逐位一致。
#
# 本脚本 = run_unified7_trainmatch.sh 的全量重跑，输出到**新目录**（旧结果保留不动），
# 口径逐项相同：SCENE=dyn_both_train / 不传 --d-safe（=0.20）/ 不传 --arm-aware /
# 两轮 --ori-servo off|on / seed 12345 逐集 / n=30 / max_steps=200。
set -u
cd "$(dirname "$0")/.."        # -> rl_grasping_system
OUT=results/unified7_trainmatch_fix
mkdir -p "$OUT"

SCENE=dyn_both_train
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

for servo in off on; do
  SUF="_trainmatch_servo${servo}_n30.json"
  for tag in "${ORDER[@]}"; do
    f="$OUT/${tag}_${SCENE}${SUF}"
    if [ -f "$f" ] && python3 -c "
import json,sys
try: sys.exit(0 if json.load(open('$f')).get('n_episodes')==30 else 1)
except Exception: sys.exit(1)" 2>/dev/null; then
      echo "[skip] $(basename "$f")"; continue
    fi
    echo "[run ] servo=$servo  $tag / $SCENE  $(date +%H:%M:%S)"
    python3 scripts/mpc_plus_rl_eval.py --scene "$SCENE" ${ARGS[$tag]} \
      --seed 12345 --seed-per-episode --n_episodes 30 \
      --max-steps 200 --ori-servo "$servo" --save-results "$f" 2>&1 \
      | grep -E "^scene=|saved ->" \
      || echo "[FAIL] servo=$servo $tag/$SCENE"
  done
done
echo "ALL DONE $(date +%H:%M:%S)"
