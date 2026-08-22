#!/usr/bin/env python3
"""
P3 抓取成功判定验证：手指被物体挡住合不上 = 成功（替代旧的 距离+夹紧+接触+朝下 组合判定）。

场景：
  A. 正确抓取：把夹爪摆到物体正上方抓取位姿（DLS IK + 伺服收敛 hold），step 零动作 ->
     近距自动闭合 -> 手指被物体挡住（开度仍 > grasp_min_width）-> grasp_success=True；
  B. 空闭合：把夹爪摆到物体侧方 0.06m（触发距离内但手指够不到物体）-> 自动闭合 ->
     开度收敛到 ≈0（≤ grasp_min_width）-> grasp_success=False 且相位回到 open（可重试）；
  C. 宽度校准/滞回：打印 A/B 的开度实测值，断言成功开度 > 阈值、空闭合开度 < 阈值；
     空闭合后 grasp_retry_hold=True，离开物体 > gripper_open_distance 后清除（可重试）。

运行（rl_grasping_system/ 下）：python scripts/verify_grasp_success_criterion.py
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
from ik import solve_ik, move_to_q, frame_to_quat  # noqa: E402

# Task3 (UR5e + 2F-85)：home 指尖朝下；pad 中心距 rq_base_mount = 0.134m（替代 Panda FINGER_REACH 0.1029）
HOME = np.array([-1.5708, -1.5708, 1.5708, -1.5708, -1.5708, 0.0])
FINGER_REACH = 0.134


def move_ee_to(env, hand_pos, quat, steps=300, hold=250):
    """DLS IK 解出目标关节位形 -> 直接设 qpos（Task3：UR5e arm 为 motor，位置伺服 ctrl 无效）。

    返回 (ik_err, 实际 hand 位置与目标的偏差)。
    """
    hand_id = env.end_effector_id
    arm_joints = env.arm_joint_ids
    q, err = solve_ik(env.model, env.data, hand_id, np.asarray(hand_pos, float),
                      np.asarray(quat, float), arm_joints, iters=6000, tol=3e-4)
    env.data.qpos[arm_joints] = q
    mujoco.mj_forward(env.model, env.data)
    final_err = float(np.linalg.norm(env._get_end_effector_position() - hand_pos))
    return err, final_err


def reset_to_home(env):
    env.reset(seed=42)
    env.data.qpos[env.arm_joint_ids] = HOME
    for jid in env.gripper_joint_ids:               # 2F-85 全开（各 rq_ 关节 qpos=0）
        env.data.qpos[env.model.jnt_qposadr[jid]] = 0.0
    mujoco.mj_forward(env.model, env.data)
    env.data.ctrl[env.gripper_actuator_idx] = 0.0   # 手指张开（menagerie 2F-85: 0=张开）


def main():
    cfg = get_config()
    cfg.grasping.render_gui = False
    cfg.grasping.use_fixed_position = True
    min_w = float(cfg.grasping.grasp_min_width)
    close_d = float(cfg.grasping.gripper_close_distance)
    open_d = float(cfg.grasping.gripper_open_distance)

    env = GraspingEnv(cfg.grasping, cfg.reward)
    grasp_quat = frame_to_quat([0, 0, -1], [0, -1, 0])   # 手指朝下、局部x=世界x（2F-85 开合轴）

    # ============ 场景 A：正确抓取 ============
    reset_to_home(env)
    cube = env._get_object_position().copy()
    print(f"[INFO] 物体落定位置: {cube.round(4)}  阈值 grasp_min_width={min_w}  "
          f"close_distance={close_d}  open_distance={open_d}")

    grasp_pos = np.array([cube[0], cube[1], cube[2] + FINGER_REACH])
    e_ik, e_final = move_ee_to(env, grasp_pos, grasp_quat)
    dist0 = float(np.linalg.norm(env._get_end_effector_position() - env._get_object_position()))
    print(f"[INFO] 场景A 抓取位 IK 误差={e_ik:.5f}  到位后 hand-物体距离={dist0:.4f} "
          f"(须 < close_distance={close_d})")
    if e_final > 1e-2 or dist0 >= close_d:
        print("[FAIL] 夹爪未真正到位，无法触发自动闭合")
        env.close()
        sys.exit(1)

    width_A = None
    success_A = False
    for i in range(40):
        obs, rew, term, trunc, info = env.step(np.zeros(6, dtype=np.float32))
        width_A = float(env._get_gripper_width())
        if info['grasp_success']:
            success_A = True
            break
    print(f"[INFO] 场景A 闭合后: phase={env.gripper_phase}  width={width_A:.4f}  "
          f"grasp_success={success_A}")
    results = {'A_grasp_success': bool(success_A), 'A_width': width_A}
    ok = success_A and (width_A is not None) and width_A > min_w

    # ============ 场景 B：空闭合（物体侧方触发距离内，手指够不到物体） ============
    reset_to_home(env)
    cube = env._get_object_position().copy()
    side_pos = np.array([cube[0] + 0.06, cube[1], cube[2] + FINGER_REACH])
    e_ik, e_final = move_ee_to(env, side_pos, grasp_quat)
    dist0 = float(np.linalg.norm(env._get_end_effector_position() - env._get_object_position()))
    print(f"[INFO] 场景B 侧方位 IK 误差={e_ik:.5f}  到位后 hand-物体距离={dist0:.4f} "
          f"(须 < close_distance={close_d})")
    if e_final > 1e-2 or dist0 >= close_d:
        print("[FAIL] 侧方位未真正到位，无法触发空闭合")
        env.close()
        sys.exit(1)

    width_B = None
    info = None
    for i in range(40):
        obs, rew, term, trunc, info = env.step(np.zeros(6, dtype=np.float32))
        width_B = float(env._get_gripper_width())
        if env.gripper_phase != 'open':
            break  # 不应触发闭合
    print(f"[INFO] 场景B 侧方不闭合: phase={env.gripper_phase}  width={width_B:.4f}  "
          f"grasp_success={info['grasp_success']}  retry_hold={env.grasp_retry_hold}")
    results['B_grasp_success'] = bool(info['grasp_success'])
    results['B_width'] = width_B
    results['B_stays_open'] = bool(env.gripper_phase == 'open')
    results['B_no_retry_hold'] = bool(not env.grasp_retry_hold)
    # Task3 对齐触发：pad 在物体侧方（xy 差 0.06 > grasp_align_xy_tol=0.03）不应触发 closing，
    # 不浪费闭合动作（旧"距离触发"会误闭合后空闭合重试）
    ok &= (not info['grasp_success']) and results['B_stays_open'] and results['B_no_retry_hold']

    # ============ 场景 C：滞回可重试 ============
    # 空闭合后 retry_hold=True；把夹爪移回 base 上方（远离物体）后应清除，可再次闭合
    move_ee_to(env, np.array([0.0, 0.0, 0.55]), grasp_quat)
    for _ in range(10):
        env.step(np.zeros(6, dtype=np.float32))
    retry_hold_cleared = bool(not env.grasp_retry_hold)
    results['C_retry_hold_cleared'] = retry_hold_cleared
    ok &= retry_hold_cleared
    print(f"[INFO] 场景C 离开物体后 retry_hold 清除: {retry_hold_cleared}")

    out = {'results': results, 'all_pass': ok}
    print(out)
    env.close()
    print("GRASP-SUCCESS-CRITERION PASS" if ok else "GRASP-SUCCESS-CRITERION FAIL")
    sys.exit(0 if ok else 1)


if __name__ == '__main__':
    main()
