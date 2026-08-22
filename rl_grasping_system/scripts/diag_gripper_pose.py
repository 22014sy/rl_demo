#!/usr/bin/env python3
"""诊断：加载已训练模型，逐 step 打印夹爪真实姿态（手指朝向量、对齐度）与各奖励分量，
复现"可视化里夹爪不竖直向下抓取"的问题并定位根因。

用法（在 rl_grasping_system/ 目录下）：
    python3 scripts/diag_gripper_pose.py [--model models/final_model.zip] [--steps 300]

输出每 step：
  step | hand_z(手指世界Z) | z_axis_z | align(cosθ) | r_orient | r_xy | r_z | gripper_width
  * 用 hand 局部 +Z 旋转到世界系得到手指指向（quat_rotate_world_z）；
  * align 是 hand 与目标抓取姿态 Q_APPROACH_DOWN=(绕X轴180°) 的四元数对齐 cos(θ)；
    目标=手指朝下（z_axis_z≈-1, align≈+1）。
"""
import os
import sys
import argparse

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
PKG = os.path.dirname(HERE)
sys.path.insert(0, PKG)

from config import get_config  # noqa: E402
from environment import GraspingEnv  # noqa: E402
from agent import GraspingAgent  # noqa: E402
from state import get_proprioceptive_state  # noqa: E402
from reward import reward_breakdown, Q_APPROACH_DOWN  # noqa: E402
from reward import quat_rotate_world_z, orientation_align  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default=os.path.join(PKG, "models", "final_model.zip"))
    ap.add_argument("--steps", type=int, default=300)
    ap.add_argument("--episodes", type=int, default=3)
    args = ap.parse_args()

    cfg = get_config()
    cfg.grasping.render_gui = False
    cfg.grasping.use_fixed_position = True

    env = GraspingEnv(cfg.grasping, cfg.reward)
    agent = GraspingAgent(cfg.network, cfg.training)
    agent.set_environment(env)
    agent.load(args.model)
    print(f"模型已加载: {args.model}")

    # 目标四元数（物体 identity → 手指朝下）
    q_target = Q_APPROACH_DOWN
    print(f"目标抓取姿态 q_target = {q_target}  （手指世界Z应 ≈ -1）")

    for ep in range(args.episodes):
        obs, _ = env.reset(seed=42 + ep)
        print(f"\n===== Episode {ep} =====")
        print(f"{'step':>4} | {'hand_z[0]':>8} {'hand_z[1]':>8} {'hand_z[2]':>8} | "
              f"{'z_axis_z':>8} | {'align':>6} | {'r_orient':>8} | "
              f"{'r_xy':>7} | {'r_z':>7} | {'width':>6} | {'grasp':>5}")
        for i in range(args.steps):
            action, _ = agent.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, info = env.step(action)
            st = get_proprioceptive_state(env.data, env.model, env)
            q = np.asarray(st['ee_orientation'], dtype=float)
            z_w = quat_rotate_world_z(q)          # hand 局部 +Z(手指) 在世界系
            align = orientation_align(q, q_target)
            zz = float(z_w[2])
            b = reward_breakdown(st, None,
                                 {'is_grasped': info.get('is_grasped', False),
                                  'grasp_success': info.get('grasp_success', False),
                                  'contact_force': env._get_grasp_contact_force()},
                                 cfg.reward)
            w = env._get_gripper_width()
            print(f"{i:4d} | {z_w[0]:8.3f} {z_w[1]:8.3f} {z_w[2]:8.3f} | {zz:8.3f} | "
                  f"{align:6.3f} | {b['r_orient']:8.3f} | {b['r_dist_xy']:7.2f} | "
                  f"{b['r_dist_z']:7.2f} | {w:6.3f} | {str(info.get('grasp_success', False)):>5}")
            if i % 50 == 0:
                print(f"  [mid] qpos={np.round(env.data.qpos[env.arm_joint_ids], 3).tolist()} "
                      f"hand_pos={np.round(env._get_end_effector_position(), 3).tolist()}")
            if terminated or truncated:
                break
        print(f"[ep{ep}] grasp_success={info.get('grasp_success')}  "
              f"final_align={align:.3f}  final_z_axis_z={zz:.3f}  width={w:.3f}")


if __name__ == '__main__':
    main()
