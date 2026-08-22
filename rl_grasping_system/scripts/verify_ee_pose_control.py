#!/usr/bin/env python3
"""
P3 末端位姿增量控制验证：RL 动作 = 末端位姿增量 -> DLS IK -> 关节位置伺服目标。

验证点（P3 重构的 动作空间=末端位姿增量 落地）：
  1. 动作空间维度/边界：action_space_dim=6 -> Box(6)，边界 = ±max_ee_delta/±max_orient_delta；
  2. 位置增量：动作 [dx,0,0,0,0,0] 使末端世界 X 坐标累计前进（位置伺服跟踪，从 home 位形出发）；
  3. 姿态增量：动作 [0,0,0,0,day,0]（绕世界 Y 旋转，避开"朝下=绕 X 旋转"导致测不到的退化轴）
     使末端朝向（手指方向）累计旋转 > 8°（30 步 × 0.05 rad，伺服跟踪 ~20%/步）；
  4. 零动作保持：IK 目标 = 当前位姿，末端漂移很小；
  5. 观测维度 58、reset(seed) 可复现；
  6. 退化方案：action_space_dim=3（仅位置，姿态固定朝下）——从 home 位形上移时
     位置累计前进且朝向保持朝下。

说明：位置/姿态伺服对单步指令有稳态滞后（实测位置 ~27%/步、姿态 ~20%/步），
这是位置伺服有限带宽的正常特性；策略会学输出持续的增量来积分到位（见 P3 设计文档）。

运行（rl_grasping_system/ 下）：python scripts/verify_ee_pose_control.py
退出码 0 = 通过；非 0 = 失败。
"""
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
PKG = os.path.dirname(HERE)
sys.path.insert(0, PKG)

import mujoco  # noqa: E402

from config import get_config  # noqa: E402
from environment import GraspingEnv  # noqa: E402
from reward import quat_rotate_world_z  # noqa: E402

# Task3 (UR5e home：指尖朝下悬于 cube 上方；开合沿 x)
HOME = np.array([-1.5708, -1.5708, 1.5708, -1.5708, -1.5708, 0.0])


def _goto_home(env, settle=150):
    """把机械臂摆到确定性的 home 位形（UR5e 6 关节）作为可复现起点。
    Task3：arm 为 motor（力矩），不做位置伺服 settle；速度模式零速度即保持。"""
    env.reset(seed=42)
    env.data.qpos[env.arm_joint_ids] = HOME
    mujoco.mj_forward(env.model, env.data)


def _ee_pos(env):
    return env._get_end_effector_position()


def _finger_down_z(env):
    q = env.data.xquat[env.end_effector_id].copy()
    return float(quat_rotate_world_z(q)[2])


def _quat_angle(qa, qb):
    qa = qa / np.linalg.norm(qa)
    qb = qb / np.linalg.norm(qb)
    return float(2 * np.arccos(np.clip(abs(float(np.dot(qa, qb))), 0.0, 1.0)))


def main():
    cfg = get_config()
    cfg.grasping.render_gui = False
    cfg.grasping.use_fixed_position = True
    act_dim = int(cfg.grasping.action_space_dim)  # 并行改造：支持 3（仅位置）/ 6（位置+姿态）

    env = GraspingEnv(cfg.grasping, cfg.reward)
    results = {}
    ok = True

    # ---- 1. 动作空间维度/边界 ----
    lo, hi = env.action_space.low, env.action_space.high
    results['action_dim'] = int(env.action_space.shape[0])
    results['action_space_ok'] = bool(
        results['action_dim'] == act_dim
        and np.allclose(lo[:3], -cfg.grasping.max_ee_delta)
        and np.allclose(hi[:3], cfg.grasping.max_ee_delta)
        and (act_dim < 6
             or (np.allclose(lo[3:], -cfg.grasping.max_orient_delta)
                 and np.allclose(hi[3:], cfg.grasping.max_orient_delta)))
    )
    results['obs_dim'] = int(env.observation_space.shape[0])
    results['obs_dim_ok'] = bool(results['obs_dim'] == 55)  # Task3: 6 关节（58→55）
    ok &= results['action_space_ok'] and results['obs_dim_ok']

    # ---- 2. reset(seed) 可复现 ----
    env.reset(seed=42)
    p1 = _ee_pos(env).copy()
    q1 = env.data.qpos[env.arm_joint_ids].copy()
    env.reset(seed=42)
    p2 = _ee_pos(env).copy()
    q2 = env.data.qpos[env.arm_joint_ids].copy()
    results['reset_reproducible'] = bool(np.allclose(p1, p2) and np.allclose(q1, q2))
    ok &= results['reset_reproducible']

    # ---- 3. 位置增量：+dx 使末端 X 前进（从 home 出发；速度模式实现率 ~96%） ----
    _goto_home(env)
    x0 = _ee_pos(env)[0]
    act_dx = np.zeros(act_dim, dtype=np.float32)
    act_dx[0] = cfg.grasping.max_ee_delta
    for _ in range(30):
        env.step(act_dx)
    dx_moved = _ee_pos(env)[0] - x0
    results['dx_moved_30steps'] = float(dx_moved)
    results['dx_track_ok'] = bool(dx_moved > 0.03)  # 30 步持续 +0.02，速度模式累计应 > 0.5m 量级
    ok &= results['dx_track_ok']

    # ---- 4. 姿态增量（仅 6D 动作空间）：绕世界 Y 旋转，累计转动 > 8° ----
    if act_dim >= 6:
        _goto_home(env)
        q_or0 = env.data.xquat[env.end_effector_id].copy()
        act_ry = np.zeros(act_dim, dtype=np.float32)
        act_ry[4] = cfg.grasping.max_orient_delta
        for _ in range(30):
            env.step(act_ry)
        q_or1 = env.data.xquat[env.end_effector_id].copy()
        rot_deg = np.degrees(_quat_angle(q_or0, q_or1))
        results['ori_rot_deg_30steps'] = round(rot_deg, 1)
        results['ori_track_ok'] = bool(rot_deg > 8.0)
        ok &= results['ori_track_ok']
    else:
        # 3D 动作空间：姿态固定朝下，姿态增量测试不适用（跳过）
        results['ori_rot_deg_30steps'] = 0.0
        results['ori_track_ok'] = True  # 语义：不要求姿态可调（固定朝下为预期行为）
        results['ori_skipped_3d'] = True

    # ---- 5. 零动作保持 ----
    _goto_home(env)
    p_hold = _ee_pos(env).copy()
    for _ in range(10):
        env.step(np.zeros(act_dim, dtype=np.float32))
    hold_drift = float(np.linalg.norm(_ee_pos(env) - p_hold))
    results['zero_action_drift'] = hold_drift
    results['zero_action_ok'] = bool(hold_drift < 0.02)
    ok &= results['zero_action_ok']

    env.close()

    # ---- 6. 退化方案：action_space_dim=3（仅位置，姿态保持朝下）----
    cfg3 = get_config()
    cfg3.grasping.render_gui = False
    cfg3.grasping.use_fixed_position = True
    cfg3.grasping.action_space_dim = 3
    env3 = GraspingEnv(cfg3.grasping, cfg3.reward)
    results['dim3_action_dim'] = int(env3.action_space.shape[0])
    _goto_home(env3)
    p3_0 = _ee_pos(env3).copy()
    zz3_0 = _finger_down_z(env3)
    act_z = np.array([0.0, 0.0, cfg3.grasping.max_ee_delta], dtype=np.float32)  # 上升
    for _ in range(30):
        env3.step(act_z)
    z_moved = _ee_pos(env3)[2] - p3_0[2]
    zz3_1 = _finger_down_z(env3)
    results['dim3_z_moved'] = round(float(z_moved), 4)
    results['dim3_z_track_ok'] = bool(z_moved > 0.03)
    results['dim3_orient_held'] = bool(zz3_0 < -0.9 and zz3_1 < -0.9)  # 固定朝下保持
    results['dim3_ok'] = bool(results['dim3_action_dim'] == 3
                              and results['dim3_z_track_ok']
                              and results['dim3_orient_held'])
    ok &= results['dim3_ok']
    env3.close()

    out = {'results': results, 'all_pass': ok}
    print(out)
    print("EE-POSE-CONTROL PASS" if ok else "EE-POSE-CONTROL FAIL")
    sys.exit(0 if ok else 1)


if __name__ == '__main__':
    main()
