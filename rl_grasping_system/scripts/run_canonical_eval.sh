#!/usr/bin/env bash
# 规范评测 v1（2026-09-17）——单一 harness、单一超时、严格配对、每臂 native 口径显式声明。
#
# 存在理由：此前的数字散在 5 个 harness / 4 套口径里（evaluate.py max_steps=500 vs
#   mpc_plus_rl_eval.py max_steps=200；cadd00e 逐轴裁剪 vs HEAD 模长裁剪；不固定 seed vs
#   --seed-per-episode），跨表比较**结构性无效**。本脚本把「可比较」变成一组硬约束。
#
# 六条硬约束（每条对应一个历史坑）：
#   1. 单 harness —— 只用 scripts/mpc_plus_rl_eval.py。它的 success_nc_rate / collision_count
#      语义与 evaluate.py 不同（后者 max_steps=500），跨 harness 成功率不可比。
#   2. 单超时 —— --max-steps 200（=config.max_steps=训练时 episode 上限）。
#      用 500 会把「超时失败」改判成「慢成功」，直接改变成功率定义。
#   3. 严格配对 —— --seed 12345 --seed-per-episode：第 k 集 reset(seed=12345+k)，
#      该 RNG 同时决定初始构型 / 障碍布置 / 目标基座 → A/B 两臂同 k 走同一条场景序列，
#      可用 McNemar 精确检验。不固定 seed 时同配置重跑有 ±8~17pp 波动（§0）。
#   4. 每臂 native 口径显式 —— pre-gating 模型（v18_p3f / v11_500k，checkpoint 里动作空间
#      是米制 Box(±0.005)）必须配 --residual-clip-mode per_axis；v25/v25b2 是 HEAD 模长
#      口径下训练的，配 modulus。用错口径 v18 掉 33pp 以上（§9.3）。
#   5. 训练一致场景 —— 含障用 dyn_both_train（count=2 → obstacle_on_nominal_path=True，
#      等价训练摆法）；dyn_both（count=1）障碍摆在路径外、**挡不住**，不是同一个任务。
#   6. 结果自描述 —— 每个 JSON 的 config 段含 residual_clip_mode / delta_cap / max_ee_delta /
#      seed / max_steps，使单份文件可自证口径，不必回查脚本。
#
# 另附一条**对照臂**（matched-budget control）：vf_res18_headcap 与 vf_res18_native 用同一个
#   策略、同一个场景、同一个 seed，只换裁剪几何。它的作用是分开
#   「残差架构有用/没用」与「残差预算大/小」——否则「v18 比 v25b2 好」永远无法归因。
set -u
cd "$(dirname "$0")/.."        # -> rl_grasping_system

N=${N:-30}
SEED=${SEED:-12345}
STEPS=${MAX_STEPS:-200}
SCENES=${SCENES:-"dyn_both_train static"}
OUT=${OUT:-results/canonical_v1}
mkdir -p "$OUT"

M_V18=models/final_model_stage2_d2_v18_p3f.zip
M_V11=models/final_model_stage2_d2_v11_500k.zip
M_V25B2=models/final_model_stage2_d2_v25b2_gate.zip

# tag -> 附加参数（场景由外层循环给）
declare -A ARGS=(
  # ---- 无 RL 的标称基线（"去掉 RL 只留传统控制"）----
  [vf_nominal]="--nominal-mode velocity_field --zero-residual"
  [mpc_nominal]="--nominal-mode mpc --zero-residual"
  # ---- 无标称的端到端 RL（"去掉传统控制只留 RL"）----
  [e2e_v11]="--action-mode delta --model $M_V11"
  # ---- 残差架构，各按训练时的 native 口径 ----
  [vf_res18_native]="--nominal-mode velocity_field --model $M_V18 --residual-gate off --residual-clip-mode per_axis"
  [mpc_res25b2_native]="--nominal-mode mpc --model $M_V25B2 --residual-gate on --residual-clip-mode modulus"
  # ---- 对照臂：同一策略换 HEAD 模长几何（把预算几何从架构里剥离出来）----
  [vf_res18_headcap]="--nominal-mode velocity_field --model $M_V18 --residual-gate off --residual-clip-mode modulus"
)
ORDER=(vf_nominal mpc_nominal e2e_v11 vf_res18_native mpc_res25b2_native vf_res18_headcap)

echo "=== 规范评测 v1  n=$N seed=$SEED max_steps=$STEPS scenes='$SCENES' out=$OUT ==="
for scene in $SCENES; do
  for tag in "${ORDER[@]}"; do
    f="$OUT/${tag}_${scene}_n${N}.json"
    if [ -f "$f" ] && python3 -c "
import json,sys
try: sys.exit(0 if json.load(open('$f')).get('n_episodes')==$N else 1)
except Exception: sys.exit(1)" 2>/dev/null; then
      echo "[skip] $(basename "$f")"; continue
    fi
    echo "[run ] $(date +%H:%M:%S)  $tag / $scene"
    python3 scripts/mpc_plus_rl_eval.py --scene "$scene" ${ARGS[$tag]} \
      --seed "$SEED" --seed-per-episode --n_episodes "$N" \
      --max-steps "$STEPS" --ori-servo off --save-results "$f" 2>&1 \
      | grep -E "^scene=|saved ->" | sed 's/^/       /' \
      || echo "[FAIL] $tag / $scene"
  done
done
echo "ALL DONE $(date +%H:%M:%S)"
