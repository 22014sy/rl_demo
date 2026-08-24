"""诊断：fixed 模型对位置随机的鲁棒阈值（2026-08-24）

目的：迁移训练在 ±3cm 下 34 万步成功率恒 0%，怀疑 fixed 策略把物体位置当常数
（未学会基于观测 target_position 定位）。本脚本评估 final_model_fixed.zip 在不同
位置随机半径下的成功率/最终距离，确定策略对位置扰动的鲁棒阈值，据此调整课程半径。

用法：
    python scripts/verify_position_migration.py [--n-episodes 30]
"""
import os
import sys
import argparse
import numpy as np

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import matplotlib
matplotlib.use('Agg')

from config import get_config
from evaluate import load_model, run_episode

DEFAULT_MODEL = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                             "models", "final_model_fixed.zip")


def main():
    parser = argparse.ArgumentParser(description="fixed 模型位置随机鲁棒性诊断")
    parser.add_argument("--model", type=str, default=DEFAULT_MODEL)
    parser.add_argument("--n-episodes", type=int, default=30)
    parser.add_argument("--radii", type=str, default="0.0,0.005,0.01,0.02,0.03",
                        help="逗号分隔的半径列表(m)")
    parser.add_argument("--stochastic", action="store_true",
                        help="用随机策略采样（默认 deterministic）——模拟训练探索噪声下的成功率")
    args = parser.parse_args()

    radii = [float(x) for x in args.radii.split(",") if x.strip()]

    print("=" * 72)
    print(f"fixed 模型位置随机鲁棒性诊断: {args.model}")
    print(f"每个半径 {args.n_episodes} episodes | "
          f"采样: {'stochastic(随机策略)' if args.stochastic else 'deterministic(确定性)'}\n")

    for r in radii:
        config = get_config()
        cx, cy = config.grasping.object_fixed_pos
        z = config.grasping.workspace_bounds[2]
        config.grasping.render_gui = False
        if r > 0:
            config.grasping.use_fixed_position = False
            config.grasping.workspace_bounds = ((cx - r, cx + r), (cy - r, cy + r), z)
        else:
            config.grasping.use_fixed_position = True

        agent, env = load_model(args.model, config)

        ok, total_len, total_dist = 0, 0.0, 0.0
        lens, dists = [], []
        for i in range(args.n_episodes):
            res = run_episode(agent, env, render=False, max_steps=500,
                              deterministic=not args.stochastic)
            ok += 1 if res['grasp_success'] else 0
            lens.append(res['episode_length'])
            dists.append(res['final_distance'])
        avg_len = float(np.mean(lens))
        avg_dist = float(np.mean(dists))
        print(f"radius={r:+.3f}: 成功率 {ok:2d}/{args.n_episodes} = {ok/args.n_episodes:6.1%}"
              f"   平均步数 {avg_len:6.1f}   平均最终距离 {avg_dist:5.3f}")

        del agent, env

    print("\n" + "=" * 72)
    print("解读：stochastic 下成功率为非零 → 训练才有成功样本可学（课程起步半径的前提）")



if __name__ == "__main__":
    main()
