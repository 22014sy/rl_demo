"""D1 验证脚本（2026-08-24）：残差策略（nominal+residual）+ 动态化环境地基冒烟断言。

覆盖（一周冲刺方案 §4/§5 D1 交付物）：
  1) 默认 delta 模式向后兼容：obs 67 维、标称速度槽位占位 0、障碍隐藏（hidden_pos + contype=0）；
  2) residual 模式：标称参考速度 v_nominal 槽位 step 后非零、≤ v_max 契约；
  3) 障碍激活：fixed_pos + contype=1、obstacle_rel 槽位非零；可编程运动时位置沿轴移动 + vel 槽位反映速度；
  4) 动态目标：target_pos 沿轴往返移动，obs target_position / task_state['object_position'] 一致跟随。

运行：python3 scripts/check_d1.py   （退出码 0=全过；1=有失败）
"""
import os
import sys
import copy
import logging

os.environ.setdefault('MUJOCO_GL', 'glfw')
import matplotlib
matplotlib.use('Agg')

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
from config import get_config
from environment import GraspingEnv


def make_env(**kw):
    """每个场景独立 grasping_config 副本（default_config 是单例，避免跨场景属性污染）"""
    cfg = get_config()
    gc = copy.copy(cfg.grasping)
    gc.render_gui = False
    for k, v in kw.items():
        setattr(gc, k, v)
    return GraspingEnv(grasping_config=gc, reward_config=cfg.reward)


def check(name, cond, detail=''):
    status = 'PASS' if cond else 'FAIL'
    print(f"  [{status}] {name}" + (f"   ({detail})" if detail else ""))
    return cond


def main():
    logging.basicConfig(level=logging.WARNING)
    ok = True
    zero_action = np.zeros(int(get_config().grasping.action_space_dim))

    # ── 1. 默认 delta（向后兼容）──
    print("=== 1) 默认 delta 模式（向后兼容）===")
    env = make_env()
    obs, _ = env.reset()
    ok &= check("obs shape == 67", obs.shape == (67,), f"got {obs.shape}")
    ok &= check("v_nominal 槽位全 0（delta 占位）", np.allclose(obs[55:61], 0))
    ok &= check("obstacle_rel/vel 槽位全 0（隐藏）", np.allclose(obs[61:67], 0))
    if env.obstacle_body_id >= 0:
        hidden = env.grasping_config.obstacle_hidden_pos
        ok &= check("障碍 body 在 hidden_pos",
                    np.allclose(env._get_obstacle_position(), hidden, atol=1e-6),
                    f"{np.round(env._get_obstacle_position(), 3)}")
        ok &= check("障碍 geom contype==0",
                    all(env.model.geom_contype[g] == 0 for g in env.obstacle_geom_ids))
    for _ in range(5):
        obs, _, _, _, _ = env.step(zero_action)
    ok &= check("delta 模式 5 步执行正常", True)
    print()

    # ── 2. residual 模式 ──
    print("=== 2) residual 模式（标称 + 残差叠加）===")
    env = make_env(action_mode='residual')
    obs, _ = env.reset()
    ok &= check("obs shape == 67", obs.shape == (67,))
    ok &= check("reset 时 v_nominal 槽位为 0", np.allclose(obs[55:61], 0))
    obs, _, _, _, _ = env.step(zero_action)
    vn = obs[55:61]
    ok &= check("step 后 v_nominal 槽位非零", np.linalg.norm(vn) > 1e-3,
                f"||v_nominal||={np.linalg.norm(vn):.4f}")
    ok &= check("标称线速度 ≤ v_max(0.12)+容差", np.linalg.norm(vn[:3]) <= 0.121,
                f"||v_lin||={np.linalg.norm(vn[:3]):.4f}")
    for _ in range(4):
        obs, _, _, _, _ = env.step(zero_action)
    ok &= check("residual 5 步执行正常", True)
    print()

    # ── 3. 障碍物激活 ──
    print("=== 3) 障碍物激活（静态 + 可编程运动）===")
    env = make_env(obstacle_enabled=True, obstacle_vel=0.0)
    obs, _ = env.reset()
    ok &= check("obs shape == 67", obs.shape == (67,))
    if env.obstacle_body_id >= 0:
        fixed = env.grasping_config.obstacle_fixed_pos
        ok &= check("障碍 body 在 fixed_pos",
                    np.allclose(env._get_obstacle_position(), fixed, atol=1e-4),
                    f"{np.round(env._get_obstacle_position(), 3)}")
        ok &= check("障碍 geom contype==1",
                    all(env.model.geom_contype[g] == 1 for g in env.obstacle_geom_ids))
        ok &= check("obstacle_rel 槽位非零（ee−obstacle）", np.linalg.norm(obs[61:64]) > 1e-3,
                    f"{np.round(obs[61:64], 3)}")
    env2 = make_env(obstacle_enabled=True, obstacle_vel=0.05, obstacle_axis='y')
    obs, _ = env2.reset()
    p0 = env2._get_obstacle_position().copy()
    for _ in range(10):
        obs, _, _, _, _ = env2.step(zero_action)
    p1 = env2._get_obstacle_position().copy()
    moved = float(np.linalg.norm(p1 - p0))
    ok &= check("障碍 10 步沿 y 移动 >1mm", moved > 1e-3, f"moved={moved * 1000:.2f}mm")
    ok &= check("obs obstacle_vel 槽位反映速度", np.linalg.norm(obs[64:67]) > 1e-3,
                f"{np.round(obs[64:67], 3)}")
    print()

    # ── 4. 动态目标 ──
    print("=== 4) 动态目标（L2 往返三角波）===")
    env = make_env(dynamic_target_enabled=True, target_vel_xy=0.1,
                   target_motion_axis='x', target_period=6.0)
    obs, _ = env.reset()
    ok &= check("obs shape == 67", obs.shape == (67,))
    tp0 = env.target_pos.copy()
    obs, _, _, _, _ = env.step(zero_action)
    dx = env.target_pos[0] - tp0[0]
    ok &= check("step 后 target_pos x 变化（v=0.1,T=0.04 → ~4mm）", abs(dx) > 1e-3,
                f"dx={dx * 1000:.2f}mm")
    ok &= check("obs target_position 槽位跟随", np.allclose(obs[35:38], env.target_pos, atol=1e-3),
                f"{np.round(obs[35:38], 3)}")
    ok &= check("task_state object_position 一致",
                np.allclose(env.task_state['object_position'], env.target_pos, atol=1e-4))
    print()

    print("=" * 40)
    if ok:
        print("✅ D1 全部断言通过")
    else:
        print("❌ 存在失败断言，见上方 FAIL")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
