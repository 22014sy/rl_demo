"""快速：Monitor / DummyVecEnv / VecNormalize 逐层 reset，定位观测全 0 的层级（2026-08-24）
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
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


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
    obs, _ = env.reset()
    print(f"裸 env:    obs[0:3]={np.round(obs[0:3],3)} ee[21:24]={np.round(obs[21:24],3)} "
          f"target[35:38]={np.round(obs[35:38],3)}  shape={obs.shape}")

    menv = Monitor(env)
    obs2, _ = menv.reset()
    print(f"Monitor:   obs[0:3]={np.round(obs2[0:3],3)} ee[21:24]={np.round(obs2[21:24],3)} "
          f"target[35:38]={np.round(obs2[35:38],3)}")

    denv = DummyVecEnv([lambda: Monitor(GraspingEnv(grasping_config=config.grasping,
                                                    reward_config=config.reward))])
    obs3 = denv.reset()
    print(f"DummyVec:  obs[0,0:3]={np.round(obs3[0,0:3],3)} ee[21:24]={np.round(obs3[0,21:24],3)} "
          f"target[35:38]={np.round(obs3[0,35:38],3)}")

    vnenv = VecNormalize(denv, norm_obs=True, norm_reward=True, clip_obs=10.0, clip_reward=10.0)
    obs4 = vnenv.reset()
    print(f"VecNorm:   obs[0,0:3]={np.round(obs4[0,0:3],3)} ee[21:24]={np.round(obs4[0,21:24],3)} "
          f"target[35:38]={np.round(obs4[0,35:38],3)}")

    # ---- 复现 set_environment：VecNormalize.load(pkl, venv) ----
    norm_pkl = os.path.join(BASE_DIR, "models", "final_model_fixed_vecnormalize.pkl")
    denv2 = DummyVecEnv([lambda: Monitor(GraspingEnv(grasping_config=config.grasping,
                                                     reward_config=config.reward))])
    if os.path.exists(norm_pkl):
        vnenv2 = VecNormalize.load(norm_pkl, denv2)
        print(f"[load pkl] vnenv2.norm_obs={vnenv2.norm_obs} training={vnenv2.training}")
        print(f"[load pkl] vnenv2.venv.reset obs[0,0:3]={np.round(vnenv2.venv.reset()[0,0:3],3)} "
              f"target[35:38]={np.round(vnenv2.venv.reset()[0,35:38],3)}")
        print(f"[load pkl] vnenv2.reset obs[0,0:3]={np.round(vnenv2.reset()[0,0:3],3)} "
              f"target[35:38]={np.round(vnenv2.reset()[0,35:38],3)}")
        print(f"[load pkl] vnenv2.obs_rms.var target[35:38]={np.round(vnenv2.obs_rms.var[35:38],8)}")
    else:
        print(f"[load pkl] 不存在 {norm_pkl}")


if __name__ == "__main__":
    main()
