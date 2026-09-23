#!/usr/bin/env bash
# 接触时刻诊断（2026-09-20）：判「避障还有没有救」。
#
# 背景：static3 已被放弃——它的 `success_nc` 恒为 0（碰撞率 100%，逐集碰撞步数最小 9 步，
# 90 集零碰撞 0 集），因为障碍墙静止且贴臂、hover 又落在墙内，最后一段必须穿透。
# 那么换场景时，第一个要问的不是「哪条臂更好」，而是：**这些碰撞发生在整集的哪个阶段？**
#   · 若集中在收尾段（`first_contact_frac` → 1）→ 和 static3 同病（抓取/下降必碰），
#     规划层与标称层都救不了，换场景也没用；
#   · 若散布在中段（明显 < 1）→ 运输途中就碰上了，**存在时间维度的余量**（可绕、可等窗口），
#     这才值得投入。
#
# 为什么这件事在 mixed3/mixed_z 上**可能**不一样：它们的障碍是**运动**的
# （一个 random 游走、一个沿 x 往返、周期 4.0 s；整集 8.0 s → 约 2 个周期），
# mixed_z 还叠了 z 向振荡。运动障碍意味着「等窗口」是一种真实可用的策略，
# 而现有标称层（MPC 跟踪一个点 + 绕行路点一次筛选）**完全没有时间维度**。
# dyn_both 作为对照（它 15–18/30 集本就零碰撞，nc 40%，是「已有一半干净」的参照）。
#
# 口径：seed 12345 逐集配对、max_steps 200、`--arm-aware`、`--zero-residual`（纯标称）、
# d_safe=0.05（唯一让 arm-aware 生效的档）。**只加诊断键，不改任何既有指标。**
#
# ⚠️ 必须从 rl_grasping_system/ 运行。⚠️ 本机不要与其它 MuJoCo 评测并发。
set -u
cd "$(dirname "$0")/.."        # -> rl_grasping_system

N=${N:-30}
SEED=${SEED:-12345}
STEPS=${MAX_STEPS:-200}
OUT=${OUT:-results/contact_trace}
SCENES=${SCENES:-"mixed3 mixed_z dyn_both"}
DSAFE=${DSAFE:-0.05}
mkdir -p "$OUT"

echo "=== 接触时刻诊断  n=$N seed=$SEED max_steps=$STEPS ==="
echo "    场景: $SCENES | d_safe: $DSAFE | 臂: 纯 MPC（Δv=0, arm-aware）"
echo "    输出: $OUT"
for scene in $SCENES; do
  f="$OUT/trace_${scene}_d$(python3 -c "print('%02d' % round(float('$DSAFE')*100))")_n${N}.json"
  if [ -f "$f" ] && python3 -c "
import json,sys
try: sys.exit(0 if json.load(open('$f')).get('n_episodes')==$N else 1)
except Exception: sys.exit(1)" 2>/dev/null; then
    echo "[skip] $(basename "$f")"; continue
  fi
  echo "[run ] $(date +%H:%M:%S)  $scene / d_safe=$DSAFE"
  python3 scripts/mpc_plus_rl_eval.py --scene "$scene" \
    --nominal-mode mpc --zero-residual --arm-aware \
    --d-safe "$DSAFE" --tag "trace_${scene}" --contact-trace \
    --seed "$SEED" --seed-per-episode --n_episodes "$N" \
    --max-steps "$STEPS" --save-results "$f" 2>&1 \
    | grep -E "succ_nc=|接触时刻|十分位|saved ->" | sed 's/^/       /' \
    || echo "[FAIL] $scene"
done
echo "ALL DONE $(date +%H:%M:%S)"
