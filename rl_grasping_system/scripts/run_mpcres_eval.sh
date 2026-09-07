#!/usr/bin/env bash
# Stage 2 批量评估：v24_mpcres（MPC + PPO Δv）六场景，n=30，确定性。
# 回填 4 象限第 4 格（MPC × PPO Δv）；结果落 results/mpc_plus_rl/*.json（gitignored，数字记 docs）。
#
# 用法：bash scripts/run_mpcres_eval.sh [--smoke N]
#   --smoke N  每场景只跑 N 集（冒烟，默认全量 30）
set -e
cd "$(dirname "$0")/.."

MODEL=models/final_model_stage2_d2_v24_mpcres.zip
N=30
PARALLEL=1
case "$1" in
  --smoke) N=${2:-3}; PARALLEL=0 ;;
  --serial) PARALLEL=0 ;;
esac

mkdir -p results/mpc_plus_rl
rm -f results/mpc_plus_rl/static.json results/mpc_plus_rl/dyn_target.json \
      results/mpc_plus_rl/dyn_both.json results/mpc_plus_rl/static3.json \
      results/mpc_plus_rl/mixed3.json results/mpc_plus_rl/mixed_z.json

run_scene() {
  local scene=$1
  echo "==== [mpcres-eval] scene=$scene n=$N ===="
  python3 scripts/mpc_plus_rl_eval.py --scene "$scene" --n_episodes "$N" \
      --model "$MODEL" --nominal-mode mpc \
      --save-results "results/mpc_plus_rl/${scene}.json" \
      || echo "!! scene=$scene FAILED"
}

if [ "$PARALLEL" = "1" ]; then
  for scene in static dyn_target dyn_both static3 mixed3 mixed_z; do
    run_scene "$scene" &
  done
  wait
else
  for scene in static dyn_target dyn_both static3 mixed3 mixed_z; do
    run_scene "$scene"
  done
fi

echo "==== summary ===="
for scene in static dyn_target dyn_both static3 mixed3 mixed_z; do
  [ -f "results/mpc_plus_rl/$scene.json" ] || { echo "$scene: MISSING"; continue; }
  python3 -c "
import json
d = json.load(open('results/mpc_plus_rl/$scene.json'))
print(f\"{d['scene']:>10}: success {d['success_rate']*100:5.1f}%  coll {d['collision_rate']*100:5.1f}%  avg_coll {d['avg_collision_count']:.2f}  len {d['avg_episode_length']:6.1f}  res_succ {d['avg_residual_norm_success']:.4f}  res_fail {d['avg_residual_norm_fail']:.4f}\")
"
done
