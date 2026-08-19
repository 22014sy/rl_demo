#!/usr/bin/env python3
"""
诊断：腕部关节 (joint5/6/7) 是否能响应动作增量。

无头运行，逐个关节下发 +max_joint_delta 增量并持续 30 步，
检查对应 qpos 是否真的移动，同时打印奇异检测结果。
运行（在 rl_grasping_system/ 下）：
    python scripts/diag_wrist_movement.py
"""
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
PKG = os.path.dirname(HERE)
sys.path.insert(0, PKG)

from config import get_config  # noqa: E402
from environment import PandaGraspingEnv  # noqa: E402
from state import get_proprioceptive_state  # noqa: E402


def main():
    cfg = get_config()
    cfg.grasping.render_gui = False
    cfg.grasping.use_fixed_position = True
    env = PandaGraspingEnv(cfg.grasping, cfg.reward)
    delta = cfg.grasping.max_joint_delta
    names = [f"joint{i}" for i in range(1, 8)]

    print("=== 初始 qpos（reset seed=42）===")
    env.reset(seed=42)
    q0 = env.data.qpos[:7].copy()
    print("  " + "  ".join(f"{n}={v:+.4f}" for n, v in zip(names, q0)))

    print("\n=== 逐关节下发 +delta 持续 30 步，观察 qpos 变化 ===")
    print(f"{'joint':<8}{'q0':>10}{'q30':>10}{'delta':>10}{'奇异?':>8}")
    for i in range(7):
        env.reset(seed=42)
        start = env.data.qpos[i]
        for _ in range(30):
            act = np.zeros(8, dtype=np.float32)
            act[i] = delta
            env.step(act)
        end = env.data.qpos[i]
        is_sing, stype, score = env.singularity_handler.detect_singularity(env.data.qpos[:7])
        flag = f"{stype}:{score:.2f}" if is_sing else "-"
        print(f"{names[i]:<8}{start:>10.4f}{end:>10.4f}{end - start:>10.4f}{flag:>12}")

    print("\n=== 直接对比：5/6/7 关节同时旋转（增量式位置控制）===")
    env.reset(seed=42)
    print("  wrist qpos 初始: ", np.round(env.data.qpos[4:7], 4))
    act = np.zeros(8, dtype=np.float32)
    act[4:7] = delta  # joint5/6/7 一起 +0.05
    for _ in range(50):
        env.step(act)
    print("  wrist qpos 50步后: ", np.round(env.data.qpos[4:7], 4))
    print("  末端位置变化:     ", np.round(env._get_end_effector_position(), 4))

    print("\n=== 从 safe_config 出发单步手腕旋转（检查是否被奇异覆盖）===")
    env.reset(seed=42)
    # 手动把 wrist 关节调到离奇异点更远的正常位形
    q = env.data.qpos[:7].copy()
    q[4] = 0.2   # joint5 远离 ±π/2
    q[5] = 1.57  # joint6
    q[6] = 0.2   # joint7
    env.data.qpos[:7] = q
    mujoco_forward(env)
    print("  ctrl 施加前 qpos[4:7]:", np.round(env.data.qpos[4:7], 4))
    st = get_proprioceptive_state(env.data, env.model, env)
    safe = env.action_wrapper.apply(np.array([0, 0, 0, 0, 0.05, 0.05, 0.05, 0.5], dtype=np.float32), st)
    print("  wrapper 输出 ctrl[4:7]:", np.round(safe['joint_commands'][4:7], 4))
    is_sing, stype, score = env.singularity_handler.detect_singularity(env.data.qpos[:7])
    print(f"  奇异检测: singular={is_sing} type={stype} score={score:.3f}")


def mujoco_forward(env):
    import mujoco
    mujoco.mj_forward(env.model, env.data)


if __name__ == '__main__':
    main()
