#!/usr/bin/env bash
# n=30 → n=200：把「残差带来一致性收敛加速」这个配对结论放到更大样本上（2026-09-16）。
#
# 为什么要加大样本：n=30 时四组配置的加速方向全部一致、但成功率变化全部不显著
#   （McNemar p=0.625/1.000/1.000/1.000）。加速结论本身在 n=30 已经很稳
#   （16/16、18/21、16/17、24/24 更快，符号检验 p≤2.8e-4），加大样本主要是
#   ①把平均 Δ 步数的置信区间收紧，②给成功率一个**有统计功效**的判定——
#   要么显著、要么能说「n=200 下仍无差异」，两者都比 n=30 的「测不出来」有用。
#
# 口径与 run_unified7_trainmatch_fix.sh **逐项相同**（连同第 23 集初始穿透修复）：
#   SCENE=dyn_both_train / 不传 --d-safe（=config 默认 0.20）/ 不传 --arm-aware /
#   两轮 --ori-servo off|on / --seed 12345 --seed-per-episode / max_steps=200。
#   唯一变化：--n_episodes 30→200（即 seed 覆盖 12345..12544）。
# 注意：只跑主张需要的 4 个臂。mpc_armaware* 在 d_safe=0.20 下已验证是空操作（逐臂 p=1.0），
#       e2e_v11 不参与这个对比，省掉的算力留给样本量。
# 必须从 rl_grasping_system/ 运行（agent.py:378 模型相对路径不存在时静默跳过加载）。
set -u
cd "$(dirname "$0")/.."        # -> rl_grasping_system
OUT=results/unified7_trainmatch_n200
N=200
mkdir -p "$OUT"

SCENE=dyn_both_train
# 顺序 = 先出主张核心对（mpc_nominal / mpc_res25b2），约 1.5h 后即有第一轮完整配对；
# vf 两臂单集 ~1.5s，放后面几乎不占时间。
ORDER=(mpc_nominal mpc_res25b2 vf_nominal vf_res18)
declare -A ARGS=(
  [mpc_nominal]="--nominal-mode mpc --zero-residual"
  [mpc_res25b2]="--nominal-mode mpc --model models/final_model_stage2_d2_v25b2_gate.zip --residual-gate on"
  [vf_nominal]="--nominal-mode velocity_field --zero-residual"
  [vf_res18]="--nominal-mode velocity_field --model models/final_model_stage2_d2_v18_p3f.zip --residual-gate off"
)

for servo in off on; do
  SUF="_trainmatch_servo${servo}_n${N}.json"
  for tag in "${ORDER[@]}"; do
    f="$OUT/${tag}_${SCENE}${SUF}"
    if [ -f "$f" ] && python3 -c "
import json,sys
try: sys.exit(0 if json.load(open('$f')).get('n_episodes')==$N else 1)
except Exception: sys.exit(1)" 2>/dev/null; then
      echo "[skip] $(basename "$f")"; continue
    fi
    echo "[run ] servo=$servo  $tag / $SCENE  n=$N  $(date +%H:%M:%S)"
    python3 scripts/mpc_plus_rl_eval.py --scene "$SCENE" ${ARGS[$tag]} \
      --seed 12345 --seed-per-episode --n_episodes "$N" \
      --max-steps 200 --ori-servo "$servo" --save-results "$f" 2>&1 \
      | grep -E "^scene=|saved ->" \
      || echo "[FAIL] servo=$servo $tag/$SCENE"
  done
done
echo "ALL DONE $(date +%H:%M:%S)"
