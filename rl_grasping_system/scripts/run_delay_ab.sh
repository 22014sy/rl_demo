#!/usr/bin/env bash
# 等 v26 延迟鲁棒补训结束 → 配对同种子 A/B：纯 MPC vs v26 残差，D=0 与 D=4，n=30。
# 判定：只有 v26 在 D>0 显著优于纯 MPC，才能把「失配域鲁棒」写进简历；否则如实写「未观测到优势」。
set -u
cd "$(dirname "$0")/.."
export MUJOCO_GL=egl

SEED=12345          # 固定随机源（跨 cell 同一 RNG 流，尽量配对）
N=30
MODEL=models/final_model_stage2_d2_v26_delayrobust.zip
OUT=results/delay_ab
mkdir -p "$OUT"

echo "[ab] 等待训练进程退出... $(date)"
while pgrep -f "train_with_monitor.py --action-mode residual" >/dev/null 2>&1; do sleep 60; done
echo "[ab] 训练进程已退出 $(date)"

if [ ! -f "$MODEL" ]; then
  echo "[ab] ERROR: 未找到最终模型 $MODEL —— 训练可能失败或保存路径变了"; exit 1
fi
if [ ! -f "${MODEL%.zip}_vecnormalize.pkl" ]; then
  echo "[ab] WARN: 模型同目录缺 _vecnormalize.pkl，评估会因观测统计缺失而失配"
fi

run() {  # $1=scene  $2=extra args  $3=tag
  echo "[ab] === $1 / $3 === $(date)"
  python3 scripts/mpc_plus_rl_eval.py --scene "$1" --n_episodes "$N" \
    --seed "$SEED" --nominal-mode mpc $2 \
    --save-results "$OUT/$1_$3.json"
}

for scene in dyn_both dyn_target; do
  run "$scene" "--zero-residual"                        "mpc_d0"   # 纯 MPC，无延迟
  run "$scene" "--zero-residual --ctrl-delay-steps 4"   "mpc_d4"   # 纯 MPC，延迟 4 步
  run "$scene" "--model $MODEL"                         "v26_d0"   # v26 残差，无延迟
  run "$scene" "--model $MODEL --ctrl-delay-steps 4"    "v26_d4"   # v26 残差，延迟 4 步
done

echo "[ab] 全部完成 $(date)"
python3 - <<'PY'
import json, glob, os
print("\n===== DELAY A/B SUMMARY (seed=12345, n=30) =====")
rows = {}
for f in sorted(glob.glob("results/delay_ab/*.json")):
    r = json.load(open(f))
    k = os.path.basename(f)[:-5]
    rows[k] = r
    print(f"{k:22s} succ={r['success_rate']*100:5.1f}%  coll={r['collision_rate']*100:5.1f}%  "
          f"avg_coll={r['avg_collision_count']:.2f}  len={r['avg_episode_length']:.1f}")
print("\n--- 判定 ---")
for scene in ("dyn_both", "dyn_target"):
    for d in ("d0", "d4"):
        m = rows.get(f"{scene}_mpc_{d}", {}).get("success_rate")
        v = rows.get(f"{scene}_v26_{d}", {}).get("success_rate")
        if m is None or v is None:
            print(f"{scene} {d}: 数据不全"); continue
        delta = (v - m) * 100
        verdict = "残差占优" if delta > 5 else ("残差更差" if delta < -5 else "无显著差异")
        print(f"{scene} {d}: v26 {v*100:.1f}% vs 纯MPC {m*100:.1f}%  (Δ={delta:+.1f}pp)  → {verdict}")
PY
