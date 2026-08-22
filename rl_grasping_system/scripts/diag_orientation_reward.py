#!/usr/bin/env python3
"""
诊断脚本：单独打印夹爪朝向奖励（方向项）及各奖励分量（P1.2 四元数对齐）。

背景：P1.2 修复前方向项是 `w_orient*max(0, z_axis[2])` —— 只在手指朝上时给分，
顶抓姿态（手指朝下 z_axis[2]<0）r_orient=0，导致"末端到位但夹爪不朝下、抓不住"。
修复后方向项改为手+Z(手指) 与目标抓取姿态的四元数对齐 cos(θ)：
    align    = 2*(|q_hand·q_target|)^2 - 1        # = cos(θ)，θ=总角偏差(含roll)
    r_orient = w_orient * max(0, align)           # 朝下=满分、朝上=0

运行（在 rl_grasping_system/ 目录下）：
    python scripts/diag_orientation_reward.py [steps]
输出逐 step 表：step | joint6 | z_axis_z | orient_align | r_orient | r_xy | r_z | total
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
from reward import reward_breakdown, REWARD_KEYS, Q_APPROACH_DOWN  # noqa: E402


def _show(b):
    for k in REWARD_KEYS:
        print(f"  {k:10s} = {b[k]:+.3f}")
    d = b['diag']
    print(f"  {'orient_align':10s} = {d['orient_align']:+.3f}   z_axis_z = {d['z_axis_z']:+.3f}"
          f"   potential = {d['potential']:+.3f}")
    print(f"  {'total':10s} = {sum(b[k] for k in REWARD_KEYS):+.3f}")


def _grasp_info(env, info):
    return {
        'is_grasped': info['is_grasped'],
        'grasp_success': info['grasp_success'],
        'contact_force': env._get_grasp_contact_force(),
    }


def main():
    n_steps = int(sys.argv[1]) if len(sys.argv) > 1 else 60
    cfg = get_config()
    cfg.grasping.render_gui = False
    cfg.grasping.use_fixed_position = True

    env = PandaGraspingEnv(cfg.grasping, cfg.reward)
    env.reset(seed=42)

    print("=== 目标抓取姿态 Q_APPROACH_DOWN =", Q_APPROACH_DOWN, "（hand+Z=手指 → 世界 −Z=朝下）")
    print("=== 初始位形方向项（修复后 r_orient 应 >0，修复前为 0）===")
    st = get_proprioceptive_state(env.data, env.model, env)
    b = reward_breakdown(st, None,
                         {'is_grasped': False, 'grasp_success': False, 'contact_force': 0.0},
                         cfg.reward)
    _show(b)

    print("\n=== 逐 step 分量（动作：姿态增量 [0,0,0,0.02,0.02,0.02]，观察 r_orient 随姿态变化（P3：6 维末端位姿动作））===")
    act = np.array([0.0, 0.0, 0.0, 0.02, 0.02, 0.02], dtype=np.float32)
    print(f"{'step':>4} | {'joint6':>6} | {'z_axis_z':>8} | {'align':>6} | {'r_orient':>8} | "
          f"{'r_xy':>7} | {'r_z':>7} | {'total':>8}")
    for i in range(n_steps):
        _, _, _, _, info = env.step(act)
        st = get_proprioceptive_state(env.data, env.model, env)
        b = reward_breakdown(st, None, _grasp_info(env, info), cfg.reward)
        print(f"{i:4d} | {env.data.qpos[5]:6.3f} | {b['diag']['z_axis_z']:8.3f} | {b['diag']['orient_align']:6.3f} | "
              f"{b['r_orient']:8.3f} | {b['r_dist_xy']:7.2f} | {b['r_dist_z']:7.2f} | "
              f"{sum(b[k] for k in REWARD_KEYS):8.2f}")

    print("\n=== 方向项边界对照（同一位置，仅换方向四元数）===")
    st_ref = get_proprioceptive_state(env.data, env.model, env)
    cases = [
        ("朝下(0,1,0,0)", np.array([0.0, 1.0, 0.0, 0.0])),
        ("朝上(identity)", np.array([1.0, 0.0, 0.0, 0.0])),
        ("水平(绕X90°)", np.array([np.cos(np.pi / 4), np.sin(np.pi / 4), 0.0, 0.0])),
    ]
    for name, quat in cases:
        s = dict(st_ref)
        s['ee_orientation'] = quat
        bb = reward_breakdown(s, None,
                              {'is_grasped': False, 'grasp_success': False, 'contact_force': 0.0},
                              cfg.reward)
        print(f"  {name:14s} align={bb['diag']['orient_align']:+.3f}  r_orient={bb['r_orient']:+.3f}")


if __name__ == '__main__':
    main()
