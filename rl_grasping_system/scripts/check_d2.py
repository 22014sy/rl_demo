"""D2 验证脚本（2026-08-24）：静态障碍绕障 + 机械臂-障碍碰撞检测 + 障碍接近惩罚 + 残差幅度正则。

覆盖（一周冲刺方案 §5 D2 交付物 / 架构文档 §5.3）：
  1) 奖励公式（确定性）：r_obstacle = -w·max(0, 1-d/range)、r_residual = -w·‖Δv‖²；
  2) 默认 delta 无障：r_obstacle=0、r_residual=0（不干扰旧训练）；info 含碰撞字段；
  3) 障碍放"标称必经之路"（on-path）：reset 后障碍在 home→pre-grasp 线段附近 + contype=1；
  4) residual 模式集成：末端靠近障碍时 r_obstacle<0、非零残差 → r_residual<0；
  5) 机械臂-障碍碰撞检测：末端推向障碍触发 _check_arm_obstacle_contact + 计数累计。

运行：python3 scripts/check_d2.py   （退出码 0=全过；1=有失败）
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
from reward import reward_breakdown


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
    cfg = get_config()
    dim = int(cfg.grasping.action_space_dim)

    # ── 1. 奖励公式（手动构造 grasp_info，确定性、不依赖物理）──
    print("=== 1) D2 奖励公式（障碍接近惩罚 + 残差幅度正则）===")
    env0 = make_env()
    env0.reset()
    st = env0.prev_state
    parts = reward_breakdown(st, None, {'obstacle_dist': 0.05}, cfg.reward)
    ok &= check("r_obstacle < 0（d=0.05 < range=0.15）", parts['r_obstacle'] < 0,
                f"{parts['r_obstacle']:.4f}")
    parts = reward_breakdown(st, None, {'obstacle_dist': float('inf')}, cfg.reward)
    ok &= check("r_obstacle == 0（远离/未启用）", abs(parts['r_obstacle']) < 1e-9,
                f"{parts['r_obstacle']:.4f}")
    parts = reward_breakdown(st, None, {'residual_norm': 0.1}, cfg.reward)
    expect = -cfg.reward.residual_reg_w * 0.1 ** 2
    ok &= check(f"r_residual == -w*||Δv||² = {expect:.4f}",
                abs(parts['r_residual'] - expect) < 1e-9, f"{parts['r_residual']:.4f}")
    parts = reward_breakdown(st, None, {'residual_norm': 0.0}, cfg.reward)
    ok &= check("r_residual == 0（无残差）", abs(parts['r_residual']) < 1e-9)
    print()

    # ── 2. 默认 delta 无障：向后兼容 ──
    print("=== 2) 默认 delta 无障（不干扰旧训练）===")
    env = make_env()
    obs, _ = env.reset()
    ok &= check("obs shape == 67", obs.shape == (67,))
    obs, _, _, _, info = env.step(np.zeros(dim))
    parts = reward_breakdown(env.prev_state, None, {}, cfg.reward)
    ok &= check("r_obstacle == 0（无障）", abs(parts['r_obstacle']) < 1e-9)
    ok &= check("r_residual == 0（delta 模式）", abs(parts['r_residual']) < 1e-9)
    ok &= check("info 有 obstacle_collision 字段", 'obstacle_collision' in info)
    ok &= check("碰撞计数为 0", info.get('obstacle_collision_count', -1) == 0)
    print()

    # ── 3. 障碍放"标称必经之路"（on-path）──
    print("=== 3) 静态障碍放标称必经之路（on-path）===")
    env = make_env(obstacle_enabled=True, obstacle_on_nominal_path=True)
    obs, _ = env.reset()
    ok &= check("on-path 模式 obs shape == 67", obs.shape == (67,))
    if env.obstacle_body_id >= 0:
        start = env._get_end_effector_position().copy()
        obj = env._get_object_position().copy()
        end = obj + np.array([0.0, 0.0, cfg.reward.pre_grasp_offset_z])
        obs_pos = env._get_obstacle_position()
        # 障碍到线段 (start→end) 的 XY 水平投影最近距离（z 偏移是刻意的——下移罩住夹爪结构，
        # 见 config.obstacle_path_z_offset 注释）
        v = end - start
        w = obs_pos - start
        t = float(np.clip(np.dot(w, v) / (np.dot(v, v) + 1e-12), 0.0, 1.0))
        dist_to_seg_xy = float(np.linalg.norm((w - t * v)[:2]))
        ok &= check("障碍 XY 投影在 home→pre-grasp 线段附近(<5cm)", dist_to_seg_xy < 0.05,
                    f"dist_xy={dist_to_seg_xy:.4f}m, obstacle={np.round(obs_pos, 3)}")
        ok &= check("障碍 geom contype==1",
                    all(env.model.geom_contype[g] == 1 for g in env.obstacle_geom_ids))
        ok &= check("obstacle_rel 槽位非零", np.linalg.norm(obs[61:64]) > 1e-3,
                    f"{np.round(obs[61:64], 3)}")
    else:
        ok = False
        print("  [FAIL] 未找到 obstacle body（MJCF 缺障碍实体）")
    print()

    # ── 4. residual 模式集成：接近惩罚 + 残差正则 ──
    print("=== 4) residual 模式：障碍接近惩罚 + 残差正则集成 ===")
    env = make_env(action_mode='residual', obstacle_enabled=True, obstacle_on_nominal_path=True)
    env.reset()
    saw_obstacle_pen, saw_residual_pen = False, False
    for _ in range(60):
        ee = env._get_end_effector_position()
        obs_p = env._get_obstacle_position()
        delta = obs_p - ee
        delta[2] = 0.0
        n = float(np.linalg.norm(delta))
        act = (delta / n * env.grasping_config.max_ee_delta) if n > 1e-6 else np.zeros(dim)
        env.step(act)
        # 手动构造 grasp_info（与 env.step 内一致），验证惩罚量级
        parts = reward_breakdown(
            env.prev_state, None,
            {'obstacle_dist': env._get_obstacle_distance(),
             'residual_norm': float(np.linalg.norm(act / 0.04))},
            cfg.reward)
        if parts['r_obstacle'] < -1e-6:
            saw_obstacle_pen = True
        if parts['r_residual'] < -1e-6:
            saw_residual_pen = True
    ok &= check("末端接近障碍时 r_obstacle < 0（60 步内出现）", saw_obstacle_pen)
    ok &= check("非零残差动作 → r_residual < 0", saw_residual_pen)
    print()

    # ── 5. 机械臂-障碍碰撞检测 ──
    print("=== 5) 机械臂-障碍碰撞检测 ===")
    env = make_env(action_mode='residual', obstacle_enabled=True, obstacle_on_nominal_path=True)
    env.reset()
    hit = False
    info = {}
    for _ in range(150):
        ee = env._get_end_effector_position()
        obs_p = env._get_obstacle_position()
        delta = obs_p - ee
        delta[2] = 0.0
        n = float(np.linalg.norm(delta))
        act = (delta / n * env.grasping_config.max_ee_delta) if n > 1e-6 else np.zeros(dim)
        _, _, _, _, info = env.step(act)
        if info.get('obstacle_collision'):
            hit = True
            break
    ok &= check("末端推向障碍 150 步内触发臂-障碍碰撞", hit,
                f"collision_count={info.get('obstacle_collision_count')}" if hit else "")
    ok &= check("碰撞计数已累计", hit and info.get('obstacle_collision_count', 0) > 0)
    print()

    # ── 6. D3 §11.4 无障碍混合采样（2026-08-25）：mix_ratio 概率隐藏障碍做纯抓取训练 ──
    print("=== 6) 无障碍混合采样（obstacle_mix_ratio）===")
    env = make_env(action_mode='residual', obstacle_enabled=True, obstacle_on_nominal_path=True,
                   obstacle_mix_ratio=1.0)
    env.reset()
    ok &= check("mix=1.0 → 障碍隐藏（contype==0）",
                not env._obstacle_active
                and all(env.model.geom_contype[g] == 0 for g in env.obstacle_geom_ids))
    ok &= check("mix=1.0 → obstacle_rel 槽位全 0（check_d1 隐藏断言同款）",
                np.linalg.norm(env._get_observation()[61:64]) < 1e-6)
    ok &= check("mix=1.0 → obstacle_dist == inf（r_obstacle=0）",
                env._get_obstacle_distance() == float('inf'))
    env = make_env(action_mode='residual', obstacle_enabled=True, obstacle_on_nominal_path=True,
                   obstacle_mix_ratio=0.0)
    env.reset()
    ok &= check("mix=0.0 → 障碍激活（contype==1 + 槽位非零）",
                env._obstacle_active
                and all(env.model.geom_contype[g] == 1 for g in env.obstacle_geom_ids)
                and np.linalg.norm(env._get_observation()[61:64]) > 1e-3)
    # 统计：mix=0.5 → 100 个 reset 中约一半激活一半隐藏（宽松阈值 20%~80%）
    env = make_env(action_mode='residual', obstacle_enabled=True, obstacle_on_nominal_path=True,
                   obstacle_mix_ratio=0.5)
    n_act = 0
    for _ in range(100):
        env.reset()
        n_act += int(env._obstacle_active)
    frac_act = n_act / 100.0
    ok &= check(f"mix=0.5 → 激活比例≈0.5（100 resets，实测 {frac_act:.2f}）",
                0.2 <= frac_act <= 0.8, f"active={n_act}/100")
    print()

    print("=" * 40)
    if ok:
        print("✅ D2 全部断言通过")
    else:
        print("❌ 存在失败断言，见上方 FAIL")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
