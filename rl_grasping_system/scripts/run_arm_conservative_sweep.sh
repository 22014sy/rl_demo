#!/usr/bin/env bash
# 臂身保守度扫描（2026-09-20）：回答「把臂身目标留量调大，避障表现能提到多高」。
#
# 背景（用户的决定）：原目标「绕行路点给出 ≥1 cm 安全余量」经实测判定在本场景族
# **几何不可行**（见 docs/数字真值表_20260910.md §10 失败链条 / §11），用户已明确
# 取消该约束，改为「**成功避障率尽量高**」。
#
# ⚠️⚠️ 关键的度量修正（2026-09-20 实测，改写了本实验的形态）：
#   **在 static3 上 `success_nc_rate` 恒等于 0，与臂无关。** 实测 results/arm_aware/
#   三个 static3 文件（含最好的 arm-aware 臂）：`collision_rate = 100%`，
#   逐集碰撞步数**最小 9 步**、没有任何一集接近 0（90 集里 0 集零碰撞）。
#   机制：那个障碍墙天生贴臂摆放（起始余量 3.6 mm），而 hover 点在 2/3 的集里
#   **本身就落在墙内** → 最后一段必须穿透 → 「零碰撞」这个二元量在该场景被场景
#   设计**钉死为 0**，不能用来排序（`_check_arm_obstacle_contact` 是臂/夹爪 vs
#   障碍的窄相接触，已排除桌面，不是计数器故障）。
#   ⇒ 所以本扫描在 static3 上**用连续量排序**：`平均碰撞步数`（越低越好）
#     + `成功率`（不能掉）；二元「成功避障率」只在 dyn_both 这类零碰撞可达的场景才有分辨率。
#
# 为什么扫 `--arm-margin` 而不是 `--d-safe`：
#   d_safe 已在主跑批（run_via_compare.sh）跑了两档；而 arm-aware 的**臂身项**里，
#   真正编码「要求臂身离障碍多远」的是 `mpc_nominal_arm_margin`（默认 0.05 m）——
#   代价里的铰链项 `viol = max(0, margin - d)` 只在臂身表面距 d < margin 时才罚。
#   它此前**没在 CLI 暴露**，是这次新开的旋钮（另见 `--arm-pad`：把障碍几何放大，
#   属模型留量而非偏好）。先只动 margin：若各档几乎不动，说明**限制项是权重**
#   （`--w-arm`，默认 120）或 pad，而不是目标距离，下一步才去扫它们。
#
# d_safe 固定 0.05（唯一让 arm-aware 真正生效的档；0.20 档下 MPC 自身避障接近空操作）。
# 臂 × margin 矩阵：
#   mpc_nominal  纯 MPC（Δv=0）
#   mpc_via      MPC + 绕行路点（Δv=0）—— 主跑批显示「加绕行」是本轮动 success_nc 的改动
# 场景/margin 由环境变量给，便于按场景配不同档位：
#   SCENE=static3   MARGINS="0.05 0.10 0.20"   # 连续量，要趋势 → 三点
#   SCENE=dyn_both  MARGINS="0.05 0.20"        # 二元量，只需端点
#
# ⚠️ 必须从 rl_grasping_system/ 运行（agent.py:378 模型相对路径不存在时静默跳过加载）。
# ⚠️ 本机**不要**与其它 MuJoCo 评测并发（单集 9 s 曾被记成 18536 s）。
set -u
cd "$(dirname "$0")/.."        # -> rl_grasping_system

N=${N:-30}
SEED=${SEED:-12345}
STEPS=${MAX_STEPS:-200}
OUT=${OUT:-results/arm_conservative}
SCENE=${SCENE:-static3}
DSAFE=${DSAFE:-0.05}
MARGINS=${MARGINS:-"0.05 0.10 0.20"}
ARMS_ORDER=${ARMS_ORDER:-"mpc_nominal mpc_via"}
mkdir -p "$OUT"

declare -A ARGSOF=(
  [mpc_nominal]="--nominal-mode mpc --zero-residual --arm-aware"
  [mpc_via]="--nominal-mode mpc --zero-residual --arm-aware --global-planner via_point"
)

# margin 写进 tag（0.10 -> m10）——同一臂不同 margin 不能互相覆盖。
# 出表脚本**不解析** tag，而是读结果文件自描述的 config.arm_margin / config.global_planner；
# tag 只用于肉眼区分与幂等跳过。
mtag() { python3 -c "print('m%02d' % round(float('$1')*100))"; }

echo "=== 臂身保守度扫描  n=$N seed=$SEED max_steps=$STEPS ==="
echo "    场景: $SCENE | d_safe: $DSAFE | margin: $MARGINS | 臂: $ARMS_ORDER"
echo "    输出: $OUT"
for m in $MARGINS; do
  for arm in $ARMS_ORDER; do
    tag="${arm}_$(mtag "$m")"
    f="$OUT/${tag}_${SCENE}_n${N}.json"
    if [ -f "$f" ] && python3 -c "
import json,sys
try: sys.exit(0 if json.load(open('$f')).get('n_episodes')==$N else 1)
except Exception: sys.exit(1)" 2>/dev/null; then
      echo "[skip] $(basename "$f")"; continue
    fi
    echo "[run ] $(date +%H:%M:%S)  $arm / margin=$m / $SCENE / d_safe=$DSAFE"
    # 输出行里 `succ_nc=` 与 `avg_coll=` 都要抓——static3 上排序看的是后者（前者恒 0）。
    python3 scripts/mpc_plus_rl_eval.py --scene "$SCENE" ${ARGSOF[$arm]} \
      --d-safe "$DSAFE" --arm-margin "$m" --tag "$tag" \
      --seed "$SEED" --seed-per-episode --n_episodes "$N" \
      --max-steps "$STEPS" --save-results "$f" 2>&1 \
      | grep -E "succ_nc=|saved ->|绕行路点|规划:" | sed 's/^/       /' \
      || echo "[FAIL] $arm / margin=$m"
  done
done
echo "ALL DONE $(date +%H:%M:%S)"
