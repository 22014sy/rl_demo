#!/usr/bin/env bash
# 感知噪声扫掠（2026-09-16）——把感知误差从"只坑策略"升级成"坑整条控制链"。
#
# 与旧 D6 实验（docs/2026-08-28_D6感知噪声鲁棒性实验.md）的区别，也是本脚本存在的理由：
#   旧口径：噪声只进观测（`_get_observation` 覆盖 state['target_position']），而**标称层读真值**
#           （environment.py 的 reference_velocity 传 self.target_pos）→ 测的是"策略被喂垃圾时掉多少分"。
#           该口径下出现过"加噪反而 +16.7pp"（文档 §2.1）。但标称是 oracle 时噪声不可能因果地提升
#           性能 —— 那是 n=30 波动带（文档 §4 自己记了 oracle 两次重跑差 13.4pp）。结论站不住。
#   本口径：加 `--perception-affects-nominal` → 标称层消费**策略当步实际看到的那份估计**
#           （self._perceived_pos）→ 测的是"感知误差经控制器传播后，残差能不能补回来"。这才有意义。
#
# 三个臂（为什么要三个）：
#   vf  : 纯标称（velocity_field + --zero-residual）—— 基线，感知误差直接进控制器
#   v18 : 残差，**未加噪训练**（oracle 训练）→ 零样本受噪，代表"没为感知误差训练过"
#   v21 : 残差，**加噪训练**（v21_p3i_noise）→ 代表"为感知误差训练过"
#   三者对比才能分开"残差架构本身有用"与"为噪声训练过才有用"。
#
# ⚠️ 标称层必须用 velocity_field，**不是 mpc**（2026-09-16 复核纠错）：v18/v21 是速度场残差模型
#    （config.py:249 `nominal_mode` 默认就是 "velocity_field"；所有评 v18 的脚本一律
#    `--nominal-mode velocity_field`）。若按 mpc 评，就是把「没训过的标称层」和「感知误差」两个
#    变量混在一起，残差掉的分分不清是谁的。基线臂因此是 `vf` 而不是 `mpc_nominal`。
#   同理，两臂都要 `--residual-gate off`：v18/v21 都是 2026-08-28 的模型，早于门控（v25 时代）
#    存在，训练时残差是无条件叠加的。
#
# ⚠️ 分布说明：标称改成吃估计后，**没有任何现有模型是分布内的**——它们训练时标称都是 oracle，
#    且观测里的 v_nominal 槽位现在也变带噪了。所以本轮是**零样本**结果，口径上必须这么标。
#    要拿"学到了补偿"的主张，得用新口径补训一轮（见文末提示）。
#
# 口径：dyn_both_train / --seed 12345 --seed-per-episode / 不传 --d-safe / 不传 --ori-servo。
#      感知噪声 RNG 在 reset 里按 seed 重播种（environment.py:412）→ 两臂噪声序列逐集一致，配对成立。
# 成本：约 20 s/集 → 每格 N=60 约 20 min → 4σ × 3臂 = 12 格 约 4h。要省就减 σ 档或减臂。
# 小验证（先跑这个）：N=30 SIGMAS="0 0.02 0.04" bash scripts/run_perception_sweep.sh → 9 格 约 1.5h
# 完整轮：N=60 bash scripts/run_perception_sweep.sh
set -u
cd "$(dirname "$0")/.."

OUT=results/perception_sweep
N=${N:-60}
SCENE=dyn_both_train
mkdir -p "$OUT"

declare -A ARGS=(
  [vf]="--zero-residual --residual-gate off"
  [v18]="--model models/final_model_stage2_d2_v18_p3f.zip --residual-gate off"
  [v21]="--model models/final_model_stage2_d2_v21_p3i_noise.zip --residual-gate off"
)
for m in v18 v21; do
  p="models/final_model_stage2_d2_$( [ $m = v18 ] && echo v18_p3f || echo v21_p3i_noise )_vecnormalize.pkl"
  [ -f "$p" ] || { echo "[FAIL] 缺 $p（观测统计缺失会静默失配）"; exit 1; }
done

# SIGMAS 可用环境变量覆盖（小验证先跑 0/0.02/0.04；完整轮再放开 0.01）
SIGMAS=${SIGMAS:-"0 0.01 0.02 0.04"}
for S in $SIGMAS; do
  for arm in vf v18 v21; do
    f="$OUT/${SCENE}_${arm}_s${S}.json"
    if [ -f "$f" ] && python3 -c "
import json,sys
try: sys.exit(0 if json.load(open('$f')).get('n_episodes')==$N else 1)
except Exception: sys.exit(1)" 2>/dev/null; then
      echo "[skip] $(basename "$f")"; continue
    fi
    echo "[run ] $SCENE $arm sigma=$S n=$N  $(date +%H:%M:%S)"
    python3 scripts/mpc_plus_rl_eval.py --scene "$SCENE" --nominal-mode velocity_field ${ARGS[$arm]} \
      --seed 12345 --seed-per-episode --n_episodes "$N" --max-steps 200 \
      --perception-noise-std "$S" --perception-affects-nominal \
      --save-results "$f" 2>&1 | grep -E "^scene=|saved ->" || echo "[FAIL] $arm sigma=$S"
  done
done

echo "ALL DONE $(date +%H:%M:%S)"
echo "画图（两臂一组，跑两次）："
for R in v18 v21; do
  echo "  python3 scripts/plot_ab_sweep.py --dir $OUT \\"
  echo "    --pattern '(?P<scene>.+)_(?P<arm>[^_]+)_s(?P<x>[0-9.]+)\\.json' \\"
  echo "    --baseline-tag vf --residual-tag $R \\"
  echo "    --baseline-name 'velocity field only' --residual-name 'velocity field + residual ($R)' \\"
  echo "    --xlabel 'Perception noise sigma (m)' --out $OUT/perception_${R}_vs_nominal.png"
done
