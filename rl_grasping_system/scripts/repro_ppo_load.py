"""复现 PPO.load 对 DummyVecEnv.reset 的影响（2026-08-24）

现象：新建 DummyVecEnv.reset 正常；set_environment（PPO.load）后 vn.venv.reset 返回全 0。
本脚本逐步复现，定位 PPO.load 如何破坏 VecEnv.reset。
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
from environment import GraspingEnv
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3 import PPO

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

    def make():
        return Monitor(GraspingEnv(grasping_config=config.grasping, reward_config=config.reward))

    denv = DummyVecEnv([make])
    obs = denv.reset()
    print(f"[新建 DummyVecEnv] reset obs[0,0:3]={np.round(obs[0,0:3],3)} "
          f"target[35:38]={np.round(obs[0,35:38],3)}")

    # ---- 复现 set_environment：VecNormalize.load(pkl) ----
    from stable_baselines3.common.vec_env import VecNormalize
    norm_pkl = os.path.join(BASE_DIR, "models", "final_model_fixed_vecnormalize.pkl")
    vn = VecNormalize(denv, norm_obs=True, norm_reward=True, clip_obs=10.0, clip_reward=10.0)
    if os.path.exists(norm_pkl):
        vn = VecNormalize.load(norm_pkl, vn)
        print(f"[VecNormalize.load] 完成")
    obs = vn.venv.reset()
    print(f"[VecNormalize.load 后 vn.venv.reset] obs[0,0:3]={np.round(obs[0,0:3],3)} "
          f"target[35:38]={np.round(obs[0,35:38],3)}")

    # PPO.load
    model = PPO.load(MODEL, env=vn)
    obs = vn.venv.reset()
    print(f"[PPO.load 后 vn.venv.reset] obs[0,0:3]={np.round(obs[0,0:3],3)} "
          f"target[35:38]={np.round(obs[0,35:38],3)}")
    obs = vn.reset()
    print(f"[PPO.load 后 vn.reset] obs[0,0:3]={np.round(obs[0,0:3],3)} "
          f"target[35:38]={np.round(obs[0,35:38],3)}")

    real = vn.venv.envs[0].env
    print(f"[vn.venv.envs[0].env 类型] {type(real).__name__}")
    obs2, _ = real.reset()
    print(f"[手动 real.reset] obs[0:3]={np.round(obs2[0:3],3)} target[35:38]={np.round(obs2[35:38],3)}")


if __name__ == "__main__":
    main()
