#!/usr/bin/env bash
# 统一口径 v2：障碍抬高 + MPC 避障生效（2026-09-15）
#
# 相对 v1(unified7os) 的两处改动，都是针对「不避障的速度场标称反而最好」这个反常结果：
#   1. --obs-raise 0.13：障碍球心 z 0.35→0.48（球顶 0.40→0.53）。
#      旧摆法下障碍球顶 0.40，而夹爪穿越高度带在 0.45 上下 —— 障碍**根本挡不住**，
#      速度场标称 9/10 集零碰撞，"不避障"没有代价。抬高后速度场掉到 4/10。
#      副带修复：旧摆法下球心(0.06,0.40,0.35)距物体(0.1,0.42,0.32)过近，
#      mj_geomDistance = −0.0276，**球戳进物体 2.8cm**（两者 contype=1，每集开局
#      MuJoCo 相互弹开，污染障碍运动）。抬高后 z 向分开，穿透归零。
#   2. --d-safe 0.10（v1 用 0.05）：MPC 避障代价是 ‖夹爪参考点−球心‖ 直接比 d_safe，
#      **不减障碍半径**，所以 0.05 等于"钻进半径 0.05 的球里才罚"= 避障失效。
#      config 默认 0.20；实测 0.20 在旧摆法 dyn_both 上 90.0%/零碰撞（速度场 86.7%）。
#
# 与 v1 不可混读：障碍几何变了，残差模型(v18/v25b2)是在低障碍下训练的 → 分布外，如实标注。
set -u
cd "$(dirname "$0")/.."        # -> rl_grasping_system
OUT=results/unified7v2
mkdir -p "$OUT"

SCENES=(static dyn_target dyn_both)   # static3 已按用户要求移出（2026-09-15）：d_safe 加大后它在 0.20 档完全卡死，单独讨论
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
SUF="_dsafe010_raise013_oriservo_n30.json"

for sc in "${SCENES[@]}"; do
  for tag in "${ORDER[@]}"; do
    f="$OUT/${tag}_${sc}${SUF}"
    if [ -f "$f" ] && python3 -c "
import json,sys
try: sys.exit(0 if json.load(open('$f')).get('n_episodes')==30 else 1)
except Exception: sys.exit(1)" 2>/dev/null; then
      echo "[skip] $(basename "$f")"; continue
    fi
    echo "[run ] $tag / $sc  $(date +%H:%M:%S)"
    python3 scripts/mpc_plus_rl_eval.py --scene "$sc" ${ARGS[$tag]} \
      --d-safe 0.10 --obs-raise 0.13 --seed 12345 --seed-per-episode --n_episodes 30 \
      --max-steps 200 --ori-servo on --save-results "$f" 2>&1 | grep -E "^scene=|saved ->" \
      || echo "[FAIL] $tag/$sc"
  done
done
echo "ALL DONE $(date +%H:%M:%S)"
