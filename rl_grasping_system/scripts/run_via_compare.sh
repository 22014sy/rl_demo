#!/usr/bin/env bash
# 绕行路点（via_point）标称层对照实验（2026-09-19）。
#
# 回答的问题：给标称层补一层「绕行路点」选点（MoveIt2 角色的极简替身），
# 能不能救回 MPC 在 static3 上的局部极小？残差 RL 的生态位还在不在？
#
# 四臂 × 三场景 × 两档 d_safe：
#   mpc_nominal      纯 MPC（Δv=0）                  —— 基线
#   mpc_via          MPC + 绕行路点（Δv=0）           —— 加全局规划，不加 RL
#   mpc_res25b2      MPC + v25b2 残差                 —— 有 RL，无全局规划
#   mpc_via_res25b2  MPC + 绕行路点 + v25b2 残差      —— 两者都有
#
# d_safe 两档都跑，理由见方案「已与用户敲定」1：0.20 是 config 默认档，
# 该档下 MPC 自身的障碍代价历史上接近**空操作**（§8.6/§9.5 实测）；
# 0.05 是 arm-aware 真正生效的档。结论只对各自档位成立，**不得跨档外推**。
#
# 口径 = scripts/run_canonical_eval.sh 六条硬约束：
#   单 harness（mpc_plus_rl_eval.py）/ --max-steps 200 / --seed 12345 --seed-per-episode
#   每臂 native 残差裁剪几何 / 结果自描述 config（含 tag）。
#   MPC 臂一律 --arm-aware（与 run_table5_n200_armaware.sh 对齐）。
#
# ⚠️ 必须从 rl_grasping_system/ 运行（agent.py:378 模型相对路径不存在时静默跳过加载）。
set -u
cd "$(dirname "$0")/.."        # -> rl_grasping_system

N=${N:-30}
SEED=${SEED:-12345}
STEPS=${MAX_STEPS:-200}
OUT=${OUT:-results/via_compare}
SCENES=${SCENES:-"static static3 mixed3"}
DSAFES=${DSAFES:-"0.05 0.20"}
mkdir -p "$OUT"

M_V25B2=models/final_model_stage2_d2_v25b2_gate.zip

ORDER=(mpc_nominal mpc_via mpc_res25b2 mpc_via_res25b2)
declare -A ARGS=(
  [mpc_nominal]="--nominal-mode mpc --zero-residual --arm-aware"
  [mpc_via]="--nominal-mode mpc --zero-residual --arm-aware --global-planner via_point"
  [mpc_res25b2]="--nominal-mode mpc --model $M_V25B2 --residual-gate on --residual-clip-mode modulus --arm-aware"
  [mpc_via_res25b2]="--nominal-mode mpc --model $M_V25B2 --residual-gate on --residual-clip-mode modulus --arm-aware --global-planner via_point"
)

# d_safe 写进文件名（0.05 -> d05，0.20 -> d20）——同一场景两档不能互相覆盖。
dsafe_tag() { python3 -c "print('d%02d' % round(float('$1')*100))"; }

echo "=== 绕行路点对照  n=$N seed=$SEED max_steps=$STEPS out=$OUT ==="
echo "    场景: $SCENES | d_safe: $DSAFES"
for scene in $SCENES; do
  for ds in $DSAFES; do
    dt=$(dsafe_tag "$ds")
    for tag in "${ORDER[@]}"; do
      f="$OUT/${tag}_${scene}_${dt}_n${N}.json"
      if [ -f "$f" ] && python3 -c "
import json,sys
try: sys.exit(0 if json.load(open('$f')).get('n_episodes')==$N else 1)
except Exception: sys.exit(1)" 2>/dev/null; then
        echo "[skip] $(basename "$f")"; continue
      fi
      echo "[run ] $(date +%H:%M:%S)  $tag / $scene / d_safe=$ds"
      python3 scripts/mpc_plus_rl_eval.py --scene "$scene" ${ARGS[$tag]} \
        --d-safe "$ds" --tag "$tag" \
        --seed "$SEED" --seed-per-episode --n_episodes "$N" \
        --max-steps "$STEPS" --save-results "$f" 2>&1 \
        | grep -E "^scene=|saved ->|绕行路点|规划:" | sed 's/^/       /' \
        || echo "[FAIL] $tag / $scene / d_safe=$ds"
    done
  done
done
echo "ALL DONE $(date +%H:%M:%S)"
