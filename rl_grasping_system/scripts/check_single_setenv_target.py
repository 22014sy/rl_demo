"""单机 set_environment（含 PPO.load）后：obs_rms.var target + 采样成功率（2026-08-24）

确认 compare 55% 的机制：PPO.load 的 set_env 是否 reset 更新 obs_rms（target var>0）。
"""
import os
import sys
import logging

os.environ.setdefault('MUJOCO_GL', 'glfw')
import matplotlib
matplotlib.use('Agg')

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
from config import get_config
from agent import GraspingAgent
from environment import GraspingEnv

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL = os.path.join(BASE_DIR, "models", "final_model_fixed.zip")


def main():
    logging.basicConfig(level=logging.WARNING)
    config = get_config()
    cx, cy = config.grasping.object_fixed_pos
    r = 0.03
    z = config.grasping.workspace_bounds[2]
    config.grasping.render_gui = False
    config.grasping.use_fixed_position = False
    config.grasping.workspace_bounds = ((cx - r, cx + r), (cy - r, cy + r), z)

    env = GraspingEnv(grasping_config=config.grasping, reward_config=config.reward)
    agent = GraspingAgent(config.network, config.training, model_path=MODEL)
    agent.set_environment(env)
    vn = agent.agent.get_vec_normalize_env()
    print(f"单机 set_environment 后 vn.obs_rms.var target[35:38]={np.round(vn.obs_rms.var[35:38],8)}")
    print(f"  var[:3]={np.round(vn.obs_rms.var[:3],4)}")

    # agent.predict 一次，看 target 是否饱和
    obs, _ = env.reset()
    obs_n = vn.normalize_obs(obs)
    print(f"裸 env.reset obs target={np.round(obs[35:38],3)}")
    print(f"vn.normalize_obs target={np.round(obs_n[35:38],2)}  (clip±10)")
    act, _ = agent.predict(obs, deterministic=False)

    # 完整 episode 统计成功率
    ok = 0
    for _ in range(20):
        obs, _ = env.reset()
        steps = 0
        while steps < 500:
            act, _ = agent.predict(obs, deterministic=False)
            obs, _r, term, trunc, info = env.step(act)
            steps += 1
            if term or trunc:
                break
        ok += 1 if info.get('grasp_success', False) else 0
    print(f"单机 set_environment 采样 20 episodes 成功率 {ok/20:.0%}")


if __name__ == "__main__":
    main()
