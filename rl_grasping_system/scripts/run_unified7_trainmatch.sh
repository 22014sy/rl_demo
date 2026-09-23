#!/usr/bin/env bash
# 训练一致口径重跑（2026-09-15）：含障场景回到 v18/v24/v25 真正训练过的那个场景。
#
# 为什么：v1(unified7os) / v2(unified7v2) 的含障场景用的是 `dyn_both` =
#   `("legacy", 1, 0.05)` → mpc_plus_rl_eval.py:236 `obstacle_on_nominal_path=(count>1)=False`
#   → 障碍摆到 off-path 固定位 (0.06,0.40,0.35)，球顶 0.40 而夹爪穿越带在 0.45 上下，
#   **挡不住**。实测整集最小臂-球表面距：评测口径中位 +0.0391 m（仅 1/10 集 <1cm），
#   训练口径（--obstacle-on-path --obstacle-count 2）中位 **−0.0002 m**（7/10 集 <1cm）。
#   「不避障的速度场标称反而最好」的根因就是评测摆错了障碍，不是策略问题。
#
# 本脚本用 `dyn_both_train`（count=2 → on_path=True，等价训练摆法）重跑 7 臂。
#
# 口径（刻意对齐训练，不做任何覆盖）：
#   * 不传 --d-safe        → config 默认 mpc_nominal_d_safe=0.20
#                            （train_with_monitor.py 无 d_safe CLI，训练即用 0.20）
#   * 不传 --arm-aware     → 默认关（训练时不存在该特性）
#   * 两轮 --ori-servo     → off = 完全复刻训练（orientation_servo 2026-09-15 才加，
#                            训练时不存在）；on = 隔离「姿态伺服」这一项的影响
#
# 不覆盖 dyn_both/static3：那两列已作废（见上），本目录是唯一可用的含障数字。
set -u
cd "$(dirname "$0")/.."        # -> rl_grasping_system
OUT=results/unified7_trainmatch
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
