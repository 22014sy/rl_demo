#!/usr/bin/env bash
# 执行延迟扫掠（2026-09-16）——把「残差补偿模型失配」从 n=30 的"方向一致但不显著"做成曲线。
#
# 为什么要重跑（旧 results/delay_ab 的三个口径缺陷）：
#   ① 旧脚本只有 `--seed 12345`、**没有** `--seed-per-episode` → 两臂步数不同 → np_random 消耗
#      不同 → RNG 流分叉 → 按 episode 序号对齐**不是严格配对**（这很可能也是旧 p 偏大的原因）。
#   ② 旧脚本用场景 `dyn_both`（`("legacy", 1, 0.05)` → count=1 → off-path，障碍根本不挡路），
#      与训练口径 `dyn_both_train` 不符。
#   ③ 只有 D∈{0,4} 两点，看不出趋势形状。
#
# 本次口径：SCENE=dyn_both_train / --seed 12345 --seed-per-episode / 不传 --d-safe / 不传 --ori-servo
#          （=训练口径）/ D∈{0,2,4} / 两臂 = 纯标称(--zero-residual) vs v26 残差。
# v26 是**唯一带延迟域随机化训练**的模型（--ctrl-delay-max 4），所以它在 D>0 是"分布内"的。
# 指标以**步数**为主（旧数据里 n=18 就有 p=7.6e-06），二元成功率是参考（要 n≈300 才可能显著）。
#
# ⚠️ 第四个口径：residual-gate（2026-09-16 第 2 次修）。v26 是 `--residual-gate-enabled False`
#    训练的（见 project_v26_delay_experiment），而 config 默认 residual_gate_enabled=True
#    → 不传开关就是用"开着的门控"评"关着门控训的模型"（旧 run_delay_ab.sh 同病）。
#    后果是延迟下 MPC 解陈旧 → 触发 unstuck 分支 → 门控放大残差预算 → 而 v26 没在这种预算下
#    学过 → 第一版结果里 D=4 反而比纯 MPC 差 10pp。修正后必须重跑 v26 三格。
#    mpc 臂零残差，门控无所谓。
#
# 成本：约 20 s/集（MPC 主导）→ N=60 时约 40 min/个 D → 3 个 D 约 2h。
# 用法：N=60 bash scripts/run_delay_sweep.sh
set -u
cd "$(dirname "$0")/.."

OUT=results/delay_sweep
N=${N:-60}
SCENE=dyn_both_train
MODEL=models/final_model_stage2_d2_v26_delayrobust.zip
mkdir -p "$OUT"

[ -f "$MODEL" ] || { echo "[FAIL] 缺模型 $MODEL"; exit 1; }
[ -f "${MODEL%.zip}_vecnormalize.pkl" ] || { echo "[FAIL] 缺 ${MODEL%.zip}_vecnormalize.pkl（观测统计缺失会静默失配）"; exit 1; }

for D in 0 2 4; do
  for arm in mpc v26; do
    f="$OUT/${SCENE}_${arm}_d${D}.json"
    if [ -f "$f" ] && python3 -c "
import json,sys
try: sys.exit(0 if json.load(open('$f')).get('n_episodes')==$N else 1)
except Exception: sys.exit(1)" 2>/dev/null; then
      echo "[skip] $(basename "$f")"; continue
    fi
    if [ "$arm" = mpc ]; then EXTRA="--zero-residual"; else EXTRA="--model $MODEL --residual-gate off"; fi
    echo "[run ] $SCENE $arm D=$D n=$N  $(date +%H:%M:%S)"
    python3 scripts/mpc_plus_rl_eval.py --scene "$SCENE" --nominal-mode mpc $EXTRA \
      --seed 12345 --seed-per-episode --n_episodes "$N" --max-steps 200 \
      --ctrl-delay-steps "$D" --save-results "$f" 2>&1 \
      | grep -E "^scene=|saved ->" || echo "[FAIL] $arm D=$D"
  done
done

echo "ALL DONE $(date +%H:%M:%S)"
echo "画图："
echo "  python3 scripts/plot_ab_sweep.py --dir $OUT \\"
echo "    --pattern '(?P<scene>.+)_(?P<arm>[^_]+)_d(?P<x>\\d+)\\.json' \\"
echo "    --xlabel 'Execution delay D (decision steps)' \\"
echo "    --baseline-name 'pure MPC (no residual)' --residual-name 'MPC + residual (v26)' \\"
echo "    --out $OUT/delay_sweep.png"
