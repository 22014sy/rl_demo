#!/usr/bin/env bash
# 5 方法一张表（2026-09-17）：纯跟踪伺服 / 跟踪伺服+RL / RL / MPC / MPC+RL。
#
# 口径 = scripts/run_canonical_eval.sh 的六条硬约束，逐项对齐：
#   单 harness（mpc_plus_rl_eval.py）/ --max-steps 200 / --seed 12345 --seed-per-episode
#   每臂 native 残差裁剪几何 / 训练一致场景 dyn_both_train / 结果自描述 config。
# 本脚本相对 canonical 的两点差异（都是用户本轮明确要求）：
#   ① n=200（canonical 默认 30）；
#   ② MPC 两臂开臂身碰撞检测 --arm-aware。
#      ⚠️ 该开关只在 mpc_nominal.py 的代价里生效，速度场/端到端臂**无论传不传都一样**，
#      故只给 MPC 两臂传（日志自证：config.arm_aware=true）。
#      ⚠️ 另注意 config 默认 d_safe=0.20 下 arm-aware 历史上是**空操作**（§8.6/§9.5 实测：
#      与不开逐臂 p=1.0）。若本轮 n=200 仍是空操作，表里必须写明——收益只在 d_safe=0.05 出现。
#
# 必须从 rl_grasping_system/ 运行（agent.py:378 模型相对路径不存在时静默跳过加载）。
set -u
cd "$(dirname "$0")/.."        # -> rl_grasping_system

N=${N:-200}
SEED=${SEED:-12345}
STEPS=${MAX_STEPS:-200}
SCENE=${SCENE:-dyn_both_train}
SERVO=${SERVO:-off}
OUT=${OUT:-results/table5_n200_armaware}
mkdir -p "$OUT"

M_V18=models/final_model_stage2_d2_v18_p3f.zip
M_V11=models/final_model_stage2_d2_v11_500k.zip
M_V25B2=models/final_model_stage2_d2_v25b2_gate.zip

ORDER=(vf_nominal vf_res18 e2e_v11 mpc_nominal mpc_res25b2)
declare -A ARGS=(
  [vf_nominal]="--nominal-mode velocity_field --zero-residual"
  [vf_res18]="--nominal-mode velocity_field --model $M_V18 --residual-gate off --residual-clip-mode per_axis"
  [e2e_v11]="--action-mode delta --model $M_V11"
  [mpc_nominal]="--nominal-mode mpc --zero-residual --arm-aware"
  [mpc_res25b2]="--nominal-mode mpc --model $M_V25B2 --residual-gate on --residual-clip-mode modulus --arm-aware"
)

echo "=== 5 方法表  n=$N seed=$SEED max_steps=$STEPS scene=$SCENE servo=$SERVO out=$OUT ==="
for tag in "${ORDER[@]}"; do
  f="$OUT/${tag}_${SCENE}_n${N}.json"
  if [ -f "$f" ] && python3 -c "
import json,sys
try: sys.exit(0 if json.load(open('$f')).get('n_episodes')==$N else 1)
except Exception: sys.exit(1)" 2>/dev/null; then
    echo "[skip] $(basename "$f")"; continue
  fi
  echo "[run ] $(date +%H:%M:%S)  $tag / $SCENE"
  python3 scripts/mpc_plus_rl_eval.py --scene "$SCENE" ${ARGS[$tag]} \
    --seed "$SEED" --seed-per-episode --n_episodes "$N" \
    --max-steps "$STEPS" --ori-servo "$SERVO" --save-results "$f" 2>&1 \
    | grep -E "^scene=|saved ->" | sed 's/^/       /' \
    || echo "[FAIL] $tag / $SCENE"
done
echo "ALL DONE $(date +%H:%M:%S)"
