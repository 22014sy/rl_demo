#!/usr/bin/env python3
"""探针：static3 卡死时，残差门控到底放没放行残差？放行了多少？瓶颈在哪一层？

monkeypatch 环境实例的 `_apply_residual_gate`，逐帧记录：
  terminal_err（MPC 自估末步误差）/ unstuck（门控判据）/ 门控前残差模长 / 门控后残差模长
再配 `--force-unstuck` 把 `_residual_unstuck` 恒置 True，模拟「探测器修好」，
用于回答「把门打开能不能救」。

对应结论见 docs/数字真值表_20260910.md §7.1 / §7.2。

用法：
    cd rl_grasping_system
    MUJOCO_GL=egl python3 scripts/probe_static3_gate.py
    MUJOCO_GL=egl python3 scripts/probe_static3_gate.py --force-unstuck
"""
import os
import sys
import argparse

import numpy as np

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config import get_config
from environment import GraspingEnv
from agent import GraspingAgent


def configure_static3(cfg, max_steps=200):
    """与 scripts/mpc_plus_rl_eval.py 的 SCENE_CFG['static3'] 同口径。"""
    g = cfg.grasping
    g.render_gui = False
    g.control_mode = 'velocity'
    g.action_mode = 'residual'
    g.nominal_mode = 'mpc'
    g.nominal_enabled = True
    g.action_space_dim = 3
    g.use_fixed_position = True
    g.max_steps = max_steps
    g.dynamic_target_enabled = False
    g.target_vel_xy = 0.0
    g.obstacle_enabled = True
    g.obstacle_count = 3
    g.obstacle_vel = 0.0
    g.obstacle_on_nominal_path = True      # count>1 → 压在标称必经之路上
    g.obstacle_specs = None
    g.obstacle_z_motion = False
    return g


def main():
    ap = argparse.ArgumentParser(description='static3 残差门控探针')
    ap.add_argument('--model', default='models/final_model_stage2_d2_v25b2_gate.zip')
    ap.add_argument('--force-unstuck', action='store_true',
                    help='把 _residual_unstuck 恒置 True（模拟判据修好），测「开门能否救」')
    ap.add_argument('--max-steps', type=int, default=200)
    ap.add_argument('--print-every', type=int, default=20)
    ap.add_argument('--seed', type=int, default=12345)
    ap.add_argument('--arm-aware', action='store_true', help='arm-aware MPC（臂身碰撞球进代价）')
    ap.add_argument('--w-arm', type=float, default=None, help='覆盖 mpc_nominal_w_arm')
    args = ap.parse_args()

    cfg = get_config()
    g = configure_static3(cfg, args.max_steps)
    if args.arm_aware:
        g.mpc_nominal_arm_aware = True
    if args.w_arm is not None:
        g.mpc_nominal_w_arm = args.w_arm
    env = GraspingEnv(cfg.grasping, cfg.reward)

    log = []
    orig_gate = env._apply_residual_gate

    def patched(nominal_v, residual_v, dt):
        pre = float(np.linalg.norm(np.asarray(residual_v, dtype=float)[:3]))
        out = orig_gate(nominal_v, residual_v, dt)
        post = float(np.linalg.norm(
            np.asarray(out, dtype=float)[:3] - np.asarray(nominal_v, dtype=float)[:3]))
        nt = getattr(env, 'nominal_trajectory', None)
        log.append(dict(
            pre=pre, post=post,
            term_err=float(getattr(nt, 'terminal_err', float('nan'))),
            solver_ok=bool(getattr(nt, 'solver_ok', True)),
            unstuck=bool(env._residual_unstuck()),
            d_obs=float(env._get_obstacle_distance()),
        ))
        return out

    env._apply_residual_gate = patched
    if args.force_unstuck:
        env._residual_unstuck = lambda: True

    agent = GraspingAgent(cfg.network, cfg.training, model_path=args.model)
    agent.set_environment(env, n_envs=1)

    obs, _ = env.reset(seed=args.seed)
    if env.nominal_trajectory is not None:
        env.nominal_trajectory.reset()

    info, step = {}, 0
    for step in range(1, args.max_steps + 1):
        action, _ = agent.predict(obs, deterministic=True)
        obs, _r, term, trunc, info = env.step(action)
        d_tgt = float(np.linalg.norm(
            env._get_end_effector_position() - np.asarray(env.target_pos, dtype=float)))
        if step % args.print_every == 0 or step == 1:
            r = log[-1]
            print(f"step {step:3d} d_tgt={d_tgt:.3f} d_obs={r['d_obs']:.3f} "
                  f"term_err={r['term_err']:.3f} solver_ok={r['solver_ok']} "
                  f"unstuck={r['unstuck']} |res| pre={r['pre']:.4f} post={r['post']:.4f}")
        if term or trunc:
            break
    env.close()

    pre = np.array([r['pre'] for r in log])
    post = np.array([r['post'] for r in log])
    us = np.array([r['unstuck'] for r in log])
    print("\n==== 汇总 ====")
    print(f"force_unstuck={args.force_unstuck}  步数={len(log)}  "
          f"成功={info.get('grasp_success')}  碰撞={info.get('obstacle_collision_count')}")
    print(f"unstuck 触发帧数: {int(us.sum())}/{len(us)}")
    print(f"门控前残差模长: mean={pre.mean():.4f} max={pre.max():.4f}")
    print(f"门控后残差模长: mean={post.mean():.4f} max={post.max():.4f}")
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
