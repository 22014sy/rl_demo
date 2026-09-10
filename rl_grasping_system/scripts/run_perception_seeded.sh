#!/usr/bin/env bash
# ============================================================
# D6 感知噪声鲁棒性 · 固定种子重跑（2026-09-10）
#   动机：原 D6（2026-08-28）n=30 无种控，其 oracle 对照重跑得 73.3%，反而低于带噪的
#   86.7%(σ=0.01) / 90.0%(σ=0.02) —— 差 13.4pp，落在文档自定的 ±8~17pp 噪声带内。
#   即原结论「带噪成功率不降」无法与「测不出效应」区分。
#   本次：固定 seed + n=50，做配对四路剂量-响应，回答「噪声到底有没有影响」。
#
#   模型：v18_p3f（headline 模型）
#   场景：动态目标 + 动态障碍（dyn_both，最难场景）
#   四路：oracle(σ=0) / σ=0.01 / σ=0.02 / σ=0.03
#   产出：results/perception_seeded/*.json
# ============================================================
set -u
cd "$(dirname "$0")/.."
export MUJOCO_GL=egl

SEED=12345
N=50
MODEL=models/final_model_stage2_d2_v18_p3f.zip
OUT=results/perception_seeded
mkdir -p "$OUT"

run() {  # $1=tag  $2=noise_std  $3=dropout
  echo "[perc] === $1 (noise=$2 dropout=$3) === $(date)"
  python3 evaluate.py --model_path "$MODEL" --n_episodes "$N" --seed "$SEED" \
    --action-mode residual \
    --perception-noise-std "$2" --perception-dropout "$3" \
    --scenario-mix 0,0,1 --target-vel 0.05 --obstacle-vel 0.05 --obstacle-count 1 \
    --save_results "$OUT/$1.json" --save_plot "$OUT/$1.png" \
    --log_level WARNING
  echo "[perc] $1 done $(date)"
}

run oracle     0.00 0.0
run n001       0.01 0.0
run n002       0.02 0.0
run n003       0.03 0.0
run n002_d01   0.02 0.1
run n002_d02   0.02 0.2

echo "[perc] 全部完成 $(date)"
python3 - <<'PY'
import json, glob, os
print("\n===== PERCEPTION NOISE (seed=12345, n=50, dyn_both) =====")
rows = {}
for f in sorted(glob.glob("results/perception_seeded/*.json")):
    r = json.load(open(f)); k = os.path.basename(f)[:-5]; rows[k] = r
    print(f"{k:8s} succ={r['success_rate']*100:5.1f}%  coll={r['collision_rate']*100:5.1f}%  "
          f"avg_coll={r['avg_collision_count']:.2f}  len={r['avg_episode_length']:.1f}")
o = rows.get("oracle", {}).get("success_rate")
print("\n--- 相对 oracle ---")
for k in ("n001", "n002", "n003"):
    v = rows.get(k, {}).get("success_rate")
    if o is None or v is None:
        print(f"{k}: 数据不全"); continue
    d = (v - o) * 100
    tag = "未降/反升" if d >= 0 else f"降 {abs(d):.1f}pp"
    print(f"{k}: {v*100:.1f}%  (Δ={d:+.1f}pp)  → {tag}")
PY
