"""检查 set_environment 单机链路 reset 后内部 env 与观测（2026-08-24）

裸 env check_obs_layout target=0.088 正常；但 set_environment 单机链路 reset 后
观测 target=[0.001,0.004,0.003] 异常。定位差异。
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

    # 内部真实 env
    real = vn.venv.envs[0].env  # DummyVecEnv -> Monitor -> GraspingEnv
    print(f"内部 env 类型: {type(real).__name__}")
    print(f"use_fixed_position={real.grasping_config.use_fixed_position}")
    print(f"workspace_bounds={real.grasping_config.workspace_bounds}")

    obs = vn.venv.reset()
    print(f"obs.shape={obs.shape}")
    print(f"vn.venv.reset obs[0,35:38]={np.round(obs[0,35:38],4)}")
    print(f"real.target_pos={np.round(real.target_pos,4)}")
    # 手动调用 _get_observation 对比（同一 reset 状态）
    manual_obs = real._get_observation()
    print(f"real._get_observation() obs[35:38]={np.round(manual_obs[35:38],4)}  "
          f"obs[0:3]={np.round(manual_obs[0:3],4)}")
    print(f"vn.venv.reset obs[0,0:3]={np.round(obs[0,0:3],4)}")
    # 对比是否一致
    print(f">>> vn.venv.reset obs 与 real._get_observation 一致: "
          f"{np.allclose(obs[0], manual_obs)}")
    print(f"real._get_object_position()={np.round(real._get_object_position(),4)}")
    tp = real.target_pos.copy()
    for i in range(0, obs.shape[1] - 2):
        if np.allclose(obs[0, i:i+3], tp, atol=5e-3):
            print(f">>> 物体位置 {np.round(tp,4)} 出现在观测 index {i}")
    # 打印完整 obs 关键段
    print(f"obs[0,18:24]={np.round(obs[0,18:24],4)} (tendon/ee_pos?)")
    print(f"obs[0,24:28]={np.round(obs[0,24:28],4)} (ee_ori?)")
    print(f"obs[0,28:35]={np.round(obs[0,28:35],4)} (ee_vel/gripper?)")
    print(f"obs[0,35:45]={np.round(obs[0,35:45],4)}")
    print(f"obs[0,45:55]={np.round(obs[0,45:55],4)}")

    # 手动再 reset 一次对比
    real.reset()
    obs2 = vn.venv.reset()
    print(f"再次 reset: obs[0,35:38]={np.round(obs2[0,35:38],4)}  "
          f"real.target_pos={np.round(real.target_pos,4)}")


if __name__ == "__main__":
    main()
