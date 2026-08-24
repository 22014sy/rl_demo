"""对照：固定/随机位置下，不同 predict 归一化路径的成功率（2026-08-24）

复现并定位"训练 0% vs 诊断 46.7%"的机制：
  A. agent.predict(原始 obs)         —— evaluate 诊断路径（已知 ±0.03 stochastic = 46.7%）
  B. vn.normalize_obs(原始 obs)→model.predict —— 手动归一化路径
  C. model.predict(原始 obs)          —— 不归一化路径
并打印 vn.norm_obs / obs_rms 关键状态。

用法：
    cd rl_grasping_system && python scripts/compare_predict_paths.py [--stochastic] [--radius 0.03]
"""
import os
import sys
import logging
import argparse

os.environ.setdefault('MUJOCO_GL', 'glfw')
import matplotlib
matplotlib.use('Agg')

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
from config import get_config
from environment import GraspingEnv
from agent import GraspingAgent

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL = os.path.join(BASE_DIR, "models", "final_model_fixed.zip")


def run_path(agent, env, vn, mode, deterministic, n_ep=20):
    ok = 0
    lens = []
    for _ in range(n_ep):
        obs, _ = env.reset()
        steps = 0
        while steps < 500:
            if mode == "A":       # agent.predict（内部 vn.normalize_obs）
                act, _ = agent.predict(obs, deterministic=deterministic)
            elif mode == "B":     # 手动 vn.normalize_obs → model.predict
                act, _ = agent.agent.predict(vn.normalize_obs(obs), deterministic=deterministic)
            else:                 # C: model.predict 不归一化
                act, _ = agent.agent.predict(obs, deterministic=deterministic)
            obs, _r, term, trunc, info = env.step(act)
            steps += 1
            if term or trunc:
                break
        ok += 1 if info.get('grasp_success', False) else 0
        lens.append(steps)
    return ok / n_ep, float(np.mean(lens))


def main():
    logging.basicConfig(level=logging.WARNING)
    ap = argparse.ArgumentParser()
    ap.add_argument("--radius", type=float, default=0.03)
    ap.add_argument("--stochastic", action="store_true")
    args = ap.parse_args()
    det = not args.stochastic

    config = get_config()
    cx, cy = config.grasping.object_fixed_pos
    r = args.radius
    z = config.grasping.workspace_bounds[2]
    config.grasping.render_gui = False
    config.grasping.use_fixed_position = False
    config.grasping.workspace_bounds = ((cx - r, cx + r), (cy - r, cy + r), z)

    agent = GraspingAgent(config.network, config.training, model_path=MODEL)
    env = GraspingEnv(grasping_config=config.grasping, reward_config=config.reward)
    agent.set_environment(env)

    vn = agent.agent.get_vec_normalize_env()
    print(f"vn.norm_obs={vn.norm_obs}  vn.norm_reward={vn.norm_reward}  vn.training={vn.training}")
    if hasattr(vn, 'obs_rms') and vn.obs_rms is not None:
        m = vn.obs_rms.mean
        print(f"obs_rms.mean[:3]={np.round(m[:3],4)}  (原始 obs joint 维度均值)")
        print(f"obs_rms.var[:3]={np.round(vn.obs_rms.var[:3],6)}")

    print(f"\n位置随机 ±{r}，采样 {'stochastic' if not det else 'deterministic'}\n")
    for mode in ["A", "B", "C"]:
        sr, al = run_path(agent, env, vn, mode, det)
        print(f"路径 {mode}: 成功率 {sr:.1%}  平均步数 {al:.1f}")


if __name__ == "__main__":
    main()
