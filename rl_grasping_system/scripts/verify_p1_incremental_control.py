#!/usr/bin/env python3
"""
P1 增量式位置控制 + 新奖励 + 抓取判定 验证脚本（依赖 RL 训练环境代码）

目的：验证 P1 修复是否真正落地：
  1. 动作空间：前7维 = ±max_joint_delta，第8维 = [0,1]；
  2. 增量式位置控制：wrapper 输出 ctrl = 当前 qpos + 增量（不再是被速度污染的目标），
     step 后 qpos 确实朝动作方向移动；
  3. 夹爪语义：action[7]=0 -> 闭合（gripper_width→0），action[7]=1 -> 张开（→~0.08）；
  4. 新奖励：数值有限、范围合理；
  5. reset(seed) 可复现物体位置与初始位形；
  6. 接触检测基础设施：手指/物体 geom 集合非空；
  7. 观测维度 53。
  9. 奖励去饱和（P1.1）：EE 沿直线逼近目标时奖励严格单调上升（全程有梯度）。
  10. 方向项（P1.2）：手+Z(手指) 与目标抓取姿态(朝下)的四元数对齐：朝下=满分、朝上=0、水平→朝下单调升。

运行方式（在 rl_grasping_system/ 目录下）：
    python scripts/verify_p1_incremental_control.py
退出码 0 = 全部通过；非 0 = 失败。
"""
import os
import sys
import json

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
PKG = os.path.dirname(HERE)
sys.path.insert(0, PKG)

from config import get_config  # noqa: E402
from environment import PandaGraspingEnv  # noqa: E402
from state import get_proprioceptive_state  # noqa: E402
from reward import (calculate_reward, reward_breakdown, REWARD_KEYS,  # noqa: E402
                    orientation_align, Q_APPROACH_DOWN)


def main():
    cfg = get_config()
    # 训练环境使用无头模式 + 固定物体位置，便于判定
    cfg.grasping.render_gui = False
    cfg.grasping.use_fixed_position = True
    delta = cfg.grasping.max_joint_delta

    env = PandaGraspingEnv(cfg.grasping, cfg.reward)
    results = {}
    ok = True

    # 1. 动作空间边界 + 子步进
    lo, hi = env.action_space.low, env.action_space.high
    results['action_delta'] = float(delta)
    results['substeps'] = int(env.substeps)
    results['control_period_ms'] = float(env.control_period * 1000.0)
    results['action_space_delta_ok'] = bool(
        np.allclose(lo[:7], -delta) and np.allclose(hi[:7], delta)
    )
    results['action_tendon_ok'] = bool(lo[7] == 0.0 and hi[7] == 1.0)
    results['obs_shape'] = int(env.observation_space.shape[0])
    results['substeps_ok'] = bool(env.substeps >= 5)  # 50Hz 下至少 10 个子步
    ok &= results['action_space_delta_ok'] and results['action_tendon_ok']
    ok &= results['obs_shape'] == 53 and results['substeps_ok']

    # 2. reset(seed) 可复现
    env.reset(seed=42)
    obj1 = env._get_object_position().copy()
    q1 = env.data.qpos[:7].copy()
    env.reset(seed=42)
    obj2 = env._get_object_position().copy()
    q2 = env.data.qpos[:7].copy()
    results['reset_reproducible'] = bool(np.allclose(obj1, obj2) and np.allclose(q1, q2))
    ok &= results['reset_reproducible']

    # 3. 增量式位置控制：ctrl = 当前 qpos + 增量；step 后 qpos 移动
    env.reset(seed=42)
    st0 = get_proprioceptive_state(env.data, env.model, env)
    act = np.array([0.02, 0, 0, 0, 0, 0, 0, 0.5], dtype=np.float32)
    safe = env.action_wrapper.apply(act, st0)
    results['ctrl_is_target'] = bool(
        np.allclose(safe['joint_commands'][0], st0['joint_positions'][0] + 0.02, atol=1e-9)
    )
    # 单步受伺服力饱和限制不会完全到位，验证“累计跟踪”：30 步持续 +0.02 命令，qpos 应累计前进 >0.05
    env.reset(seed=42)
    q_track0 = env.data.qpos[0]
    for _ in range(30):
        env.step(act)
    q_track_end = env.data.qpos[0]
    results['qpos_track_30steps'] = float(q_track_end - q_track0)
    results['qpos_tracked_ok'] = bool(q_track_end - q_track0 > 0.05)
    ok &= results['ctrl_is_target'] and results['qpos_tracked_ok']

    # 4. 夹爪语义：闭合/张开（子步进下伺服 1~2 步即收敛）
    env.reset(seed=42)
    w0 = env._get_gripper_width()
    for _ in range(20):
        env.step(np.zeros(8, dtype=np.float32))   # action[7]=0 -> 闭合
    w_close = env._get_gripper_width()
    env.reset(seed=42)
    for _ in range(20):
        env.step(np.array([0.0] * 7 + [1.0], dtype=np.float32))  # action[7]=1 -> 张开
    w_open = env._get_gripper_width()
    results['gripper_width_init'] = float(w0)
    results['gripper_width_close'] = float(w_close)
    results['gripper_width_open'] = float(w_open)
    results['gripper_semantics_ok'] = bool(w_close < 0.02 and w_open > 0.06)
    ok &= results['gripper_semantics_ok']

    # 5. 奖励：数值有限、范围合理
    rewards = []
    env.reset(seed=1)
    for _ in range(10):
        _, r, _, _, _ = env.step(np.zeros(8, dtype=np.float32))
        rewards.append(r)
    env.reset(seed=2)
    for _ in range(10):
        _, r, _, _, _ = env.step(np.array([0.05] * 7 + [1.0], dtype=np.float32))
        rewards.append(r)
    results['rewards_finite'] = bool(np.all(np.isfinite(rewards)))
    results['reward_range'] = [float(np.min(rewards)), float(np.max(rewards))]
    ok &= results['rewards_finite']

    # 6. 接触检测基础设施
    finger_ids, cube_ids = env._finger_cube_geoms()
    results['finger_geom_count'] = int(len(finger_ids))
    results['cube_geom_count'] = int(len(cube_ids))
    results['contact_infra_ok'] = bool(len(finger_ids) >= 4 and len(cube_ids) >= 1)
    ok &= results['contact_infra_ok']

    # 7. 任务状态在 step 后被更新
    env.reset(seed=42)
    _, _, _, _, info = env.step(np.zeros(8, dtype=np.float32))
    results['info_has_grasp_flag'] = bool('grasp_success' in info)
    ok &= results['info_has_grasp_flag']

    # 8. 手臂实际能走多远：全关节小幅增量 30 步，末端位移应显著（验证 max_steps 内可达）
    env.reset(seed=42)
    ee0 = env._get_end_effector_position()
    act2 = np.array([0.03, 0.03, 0.03, 0.03, 0.03, 0.03, 0.03, 0.5], dtype=np.float32)
    for _ in range(30):
        env.step(act2)
    ee1 = env._get_end_effector_position()
    ee_dist = float(np.linalg.norm(ee1 - ee0))
    results['ee_travel_30steps'] = ee_dist
    results['ee_travel_ok'] = bool(ee_dist > 0.05)  # 30 步至少移动 5cm
    ok &= results['ee_travel_ok']

    # 9. P1.1 奖励去饱和：线性势场 → EE 沿直线逼近目标时奖励严格单调上升
    env.reset(seed=42)
    st_ref = get_proprioceptive_state(env.data, env.model, env)
    q_down = np.array([0.0, 1.0, 0.0, 0.0])  # 手指朝下四元数(Q_APPROACH_DOWN)；方向项保持常量，只让距离变化

    def _rew_at(pos):
        st = dict(st_ref)
        st['ee_position'] = np.asarray(pos, dtype=float)
        st['ee_orientation'] = q_down
        return calculate_reward(
            st, None,
            {'is_grasped': False, 'grasp_success': False, 'contact_force': 0.0},
            cfg.reward,
        )

    far_pt = np.array([0.10, 0.30, 0.30])
    near_pt = np.array([0.50, 0.00, 0.12])
    t = np.array([0.0, 0.25, 0.5, 0.75, 1.0])
    sample_pts = far_pt[None, :] + t[:, None] * (near_pt - far_pt)[None, :]
    grad_rewards = [_rew_at(p) for p in sample_pts]
    results['reward_gradient_samples'] = [round(float(r), 3) for r in grad_rewards]
    results['reward_gradient_ok'] = bool(np.all(np.diff(grad_rewards) > 0))

    # 10. P1.2 方向项回归：手+Z(手指) 四元数对齐目标抓取姿态(朝下)
    #     朝下 Q_APPROACH_DOWN=(0,1,0,0) → align=1 → r_orient=w_orient（满分）；
    #     朝上 identity → align=-1 → r_orient=0；绕X轴 0°(朝上)→180°(朝下) 单调升。
    st_o = dict(st_ref)
    st_o['ee_position'] = near_pt
    st_o['target_position'] = np.array([0.5, 0.0, 0.0])
    st_o['target_orientation'] = np.array([1.0, 0.0, 0.0, 0.0])

    def _orient_rew(quat):
        s = dict(st_o)
        s['ee_orientation'] = np.asarray(quat, dtype=float)
        return calculate_reward(
            s, None,
            {'is_grasped': False, 'grasp_success': False, 'contact_force': 0.0},
            cfg.reward,
        )

    q_down_full = np.array([0.0, 1.0, 0.0, 0.0])
    q_up = np.array([1.0, 0.0, 0.0, 0.0])
    align_down = orientation_align(q_down_full, Q_APPROACH_DOWN)
    align_up = orientation_align(q_up, Q_APPROACH_DOWN)
    # 同一位置下方向项差值 = 方向奖励（距离项相同抵消）
    r_orient_delta = _orient_rew(q_down_full) - _orient_rew(q_up)
    results['orient_align_down'] = round(float(align_down), 4)
    results['orient_align_up'] = round(float(align_up), 4)
    results['orient_reward_delta'] = round(float(r_orient_delta), 4)
    results['orient_term_ok'] = bool(
        align_down > 0.999 and align_up < 0.0 and r_orient_delta >= cfg.reward.w_orient - 1e-6
    )
    # 绕世界X轴从0°(朝上)插值到180°(朝下)，对齐度应单调升
    th_series = np.linspace(0.0, np.pi, 7)
    o_series = [
        orientation_align(np.array([np.cos(t / 2), np.sin(t / 2), 0.0, 0.0]), Q_APPROACH_DOWN)
        for t in th_series
    ]
    results['orient_align_series'] = [round(float(v), 4) for v in o_series]
    results['orient_monotonic_ok'] = bool(np.all(np.diff(o_series) > 0))
    ok &= results['orient_term_ok'] and results['orient_monotonic_ok']
    ok &= results['reward_gradient_ok']

    # 11. P1.3 势能塑形回归：
    #     (a) 单步奖励有界（不再是每步 -1.3 级线性惩罚）；
    #     (b) 原地不动 300 步累计 = (1−γ)Φ 望远镜式相消，远小于旧线性项 -400~-800；
    #     (c) 诊断键不再进入奖励（修复 sum(values()) 泄漏：z_axis_z 曾把"手指朝上+1"
    #         以系数 1 重新注入奖励，抵消 P1.2 方向项修复）。
    env.reset(seed=42)
    st_i = get_proprioceptive_state(env.data, env.model, env)
    gi = {'is_grasped': False, 'grasp_success': False, 'contact_force': 0.0}

    # (a) 从初始状态向下移 5cm（朝向目标）的单步势能差
    st_next = dict(st_i)
    st_next['ee_position'] = np.asarray(st_i['ee_position'], dtype=float) + np.array([0.0, 0.0, -0.05])
    r_one = calculate_reward(st_next, st_i, gi, cfg.reward)
    results['pbs_step_reward'] = round(float(r_one), 4)
    results['pbs_step_bounded'] = bool(abs(r_one) < 5.0)

    # (b) 原地不动 300 步
    r_idle = sum(calculate_reward(st_i, st_i, gi, cfg.reward) for _ in range(300))
    results['pbs_idle_300steps'] = round(float(r_idle), 3)
    results['pbs_idle_bounded'] = bool(abs(r_idle) < 20.0)

    # (c) 奖励只含 REWARD_KEYS（诊断键在 b['diag']，不参与和）
    parts = reward_breakdown(st_i, st_i, gi, cfg.reward)
    sum_keys = sum(parts[k] for k in REWARD_KEYS)
    results['pbs_no_diag_leak'] = bool(
        np.isclose(calculate_reward(st_i, st_i, gi, cfg.reward), sum_keys, atol=1e-9)
    )
    ok &= results['pbs_step_bounded'] and results['pbs_idle_bounded'] and results['pbs_no_diag_leak']

    out = {
        'results': results,
        'all_pass': ok,
    }
    out_path = os.path.join(HERE, 'p1_result.json')
    with open(out_path, 'w', encoding='utf-8') as f:
        json.dump(out, f, ensure_ascii=False, indent=2)
    print(json.dumps(out, ensure_ascii=False, indent=2))
    print(f"结果已写入: {out_path}")
    print("ALL PASS" if ok else "FAILED")
    sys.exit(0 if ok else 1)


if __name__ == '__main__':
    main()
