"""快速检查：位置随机 env reset 后，观测 target_position 与实际物体位置的对应关系（2026-08-24）
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
    obs, info = env.reset()
    print(f"obs.shape={obs.shape}")
    print(f"use_fixed_position={env.grasping_config.use_fixed_position}")
    print(f"workspace_bounds={env.grasping_config.workspace_bounds}")
    print(f"object_fixed_pos={env.grasping_config.object_fixed_pos}")
    print(f"env.target_pos={np.round(env.target_pos,4)}")
    print(f"_get_object_position()={np.round(env._get_object_position(),4)}")
    print(f"task_state['object_position']={np.round(env.task_state['object_position'],4)}")
    print(f"obs[35:38] (假设 target_position)={np.round(obs[35:38],4)}")
    print(f"obs[38:42] (假设 target_orientation)={np.round(obs[38:42],4)}")
    print(f"obs[21:24] (假设 ee_position)={np.round(obs[21:24],4)}")
    print(f"obs[0:6] (joint)={np.round(obs[0:6],4)}")
    # 找物体位置在所有 index 中匹配的位置
    tp = env.target_pos.copy()
    for i in range(0, obs.shape[0] - 2):
        if np.allclose(obs[i:i+3], tp, atol=1e-3):
            print(f">>> 物体位置 {np.round(tp,4)} 出现在观测 index {i}")
    # 检查 q_rel 段（末尾 4）
    print(f"obs[-5:]={np.round(obs[-5:],4)}")


if __name__ == "__main__":
    main()
