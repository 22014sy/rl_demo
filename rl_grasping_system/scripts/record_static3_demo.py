#!/usr/bin/env python3
"""录制 static3（3 个静态障碍压在标称必经之路上）的 episode 并输出诊断曲线。

static3 是 MPC 标称层全版本 0% 的场景；本脚本的作用是把这个失败**可视化 + 量化**：
- GIF：成功/失败都录（--n-episodes 条），能看到臂在障碍前停住/打转
- 曲线：末端→目标距离 与 末端→最近障碍距离 随步数变化，失败时前者平台化
- 可选 --d-safe 覆盖 mpc_nominal_d_safe，用于验证「几何拒行」假设
  （调小后若物理可绕通 → 证明瓶颈是 MPC 的安全距离约束，不是残差权限）

用法：
    cd rl_grasping_system
    # 纯 MPC（Δv=0），默认 D_SAFE=0.20
    MUJOCO_GL=egl python3 scripts/record_static3_demo.py --n-episodes 2
    # 同上但把安全距离放开，看物理上能不能绕过去
    MUJOCO_GL=egl python3 scripts/record_static3_demo.py --d-safe 0.05
    # MPC + 残差策略
    MUJOCO_GL=egl python3 scripts/record_static3_demo.py --model models/final_model_stage2_d2_v25b2_gate.zip
"""
import os
import sys
import argparse

import numpy as np
from PIL import Image

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config import get_config
from environment import GraspingEnv
from show_grasp_demo import _set_oblique_camera, _set_topdown_camera


def configure_static3(cfg, d_safe=None, max_steps=200):
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
    if d_safe is not None:
        g.mpc_nominal_d_safe = d_safe

    g.dynamic_target_enabled = False
    g.target_vel_xy = 0.0
    g.obstacle_enabled = True
    g.obstacle_count = 3
    g.obstacle_vel = 0.0
    g.obstacle_on_nominal_path = True      # count>1 → on-path，压在标称必经之路上
    g.obstacle_specs = None
    g.obstacle_z_motion = False
    return g


def nearest_obstacle_dist(env):
    try:
        obs = env._get_obstacle_positions()
    except Exception:
        obs = []
    if not obs:
        return float('nan')
    ee = env._get_end_effector_position()
    return float(min(np.linalg.norm(np.asarray(p, dtype=float) - ee) for p in obs))


def main():
    ap = argparse.ArgumentParser(description='static3 失败可视化 + 诊断曲线')
    ap.add_argument('--model', default='', help='留空 = 纯 MPC（Δv=0）')
    ap.add_argument('--d-safe', type=float, default=None, help='覆盖 mpc_nominal_d_safe（默认 config 0.20）')
    ap.add_argument('--n-episodes', type=int, default=2)
    ap.add_argument('--max-steps', type=int, default=200)
    ap.add_argument('--frame-step', type=int, default=2)
    ap.add_argument('--width', type=int, default=640)
    ap.add_argument('--height', type=int, default=480)
    ap.add_argument('--topdown', action='store_true')
    ap.add_argument('--arm-aware', action='store_true', help='arm-aware MPC（臂身碰撞球进代价）')
    ap.add_argument('--w-arm', type=float, default=None, help='覆盖 mpc_nominal_w_arm')
    ap.add_argument('--outdir', default='results/demo_static3')
    args = ap.parse_args()

    import mujoco

    cfg = get_config()
    g = configure_static3(cfg, args.d_safe, args.max_steps)
    if args.arm_aware:
        g.mpc_nominal_arm_aware = True
    if args.w_arm is not None:
        g.mpc_nominal_w_arm = args.w_arm
    env = GraspingEnv(cfg.grasping, cfg.reward)

    agent = None
    if args.model:
        from agent import GraspingAgent
        agent = GraspingAgent(cfg.network, cfg.training, model_path=args.model)
        agent.set_environment(env, n_envs=1)

    renderer = mujoco.Renderer(env.model, args.height, args.width)
    os.makedirs(args.outdir, exist_ok=True)
    lookat_xy = tuple(cfg.grasping.object_fixed_pos)
    tag = os.path.splitext(os.path.basename(args.model))[0] if args.model else 'pure_mpc'
    dsafe = cfg.grasping.mpc_nominal_d_safe

    all_curves = []
    try:
        for ep in range(1, args.n_episodes + 1):
            obs, _ = env.reset()
            if env.nominal_trajectory is not None:
                env.nominal_trajectory.reset()
            frames, d_tgt, d_obs = [], [], []
            info = {}
            for step in range(1, args.max_steps + 1):
                if agent is None:
                    action = np.zeros(env.action_space.shape[0], dtype=np.float32)
                else:
                    action, _ = agent.predict(obs, deterministic=True)
                obs, _r, term, trunc, info = env.step(action)

                d_tgt.append(float(np.linalg.norm(
                    env._get_end_effector_position() - np.asarray(env.target_pos, dtype=float))))
                d_obs.append(nearest_obstacle_dist(env))

                ee = env._get_end_effector_position()
                if step % args.frame_step == 0 or term or trunc:
                    renderer.update_scene(env.data)
                    if args.topdown:
                        _set_topdown_camera(renderer, lookat_xy=lookat_xy)
                    else:
                        _set_oblique_camera(renderer, lookat_xy=lookat_xy)
                    frames.append(renderer.render().copy())
                if term or trunc:
                    break

            ok = bool(info.get('grasp_success', False))
            print(f'[static3] ep{ep} success={ok} steps={step} '
                  f'final_d_tgt={d_tgt[-1]:.3f} min_d_obs={np.nanmin(d_obs):.3f} '
                  f'collisions={info.get("obstacle_collision_count", 0)}')
            all_curves.append((d_tgt, d_obs, ok))

            out = os.path.join(args.outdir, f'{tag}_dsafe{dsafe:g}_ep{ep}_{"ok" if ok else "fail"}.gif')
            imgs = [Image.fromarray(f) for f in frames]
            imgs[0].save(out, save_all=True, append_images=imgs[1:], duration=60, loop=0)
            print(f'           -> {out} ({len(imgs)} 帧)')
    finally:
        renderer.close()
        env.close()

    # ---- 诊断曲线 ----
    try:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        fig, axes = plt.subplots(1, 2, figsize=(11, 4))
        for i, (d_tgt, d_obs, ok) in enumerate(all_curves, 1):
            lab = f'ep{i} {"success" if ok else "fail"}'
            axes[0].plot(d_tgt, label=lab)
            axes[1].plot(d_obs, label=lab)
        axes[0].set_title('EE -> target distance (m)'); axes[0].set_xlabel('decision step')
        axes[0].axhline(0.03, ls='--', c='gray', lw=1, label='success tol 0.03')
        axes[1].set_title('EE -> nearest obstacle (m)'); axes[1].set_xlabel('decision step')
        axes[1].axhline(dsafe, ls='--', c='r', lw=1, label=f'D_SAFE={dsafe:g}')
        for ax in axes:
            ax.legend(fontsize=8); ax.grid(alpha=0.3)
        fig.suptitle(f'static3 diagnostic - {tag} - D_SAFE={dsafe:g}')
        png = os.path.join(args.outdir, f'{tag}_dsafe{dsafe:g}_curves.png')
        fig.tight_layout(); fig.savefig(png, dpi=110)
        print(f'           -> {png}')
    except Exception as e:
        print(f'[static3] 曲线绘制跳过：{e}')

    return 0


if __name__ == '__main__':
    raise SystemExit(main())
