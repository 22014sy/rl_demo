#!/usr/bin/env python3
"""Record successful pure-MPC episodes as GIFs.

The environment and scene setup intentionally match mpc_nominal_verify.py:
nominal_mode='mpc', action_mode='residual', and zero residual action.
"""
import argparse
import os
import sys

import numpy as np
from PIL import Image

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config import get_config
from environment import GraspingEnv
from mpc_nominal_verify import SCENE_CFG
from show_grasp_demo import _set_oblique_camera, _set_topdown_camera, _set_track_camera


def configure_scene(cfg, scene, target_vel=0.05, max_steps=200):
    spec = SCENE_CFG[scene]
    g = cfg.grasping
    g.render_gui = False
    g.control_mode = 'velocity'
    g.action_mode = 'residual'
    g.nominal_mode = 'mpc'
    g.nominal_enabled = True
    g.action_space_dim = 3
    g.use_fixed_position = True
    g.max_steps = max_steps
    g.dynamic_target_enabled = spec['dyn_target']
    g.target_vel_xy = target_vel if spec['dyn_target'] else 0.0
    kind, count, velocity = spec['obstacle']
    if kind == 'off':
        g.obstacle_enabled = False
        g.obstacle_count = 0
        g.obstacle_specs = None
    elif kind == 'legacy':
        g.obstacle_enabled = True
        g.obstacle_count = count
        g.obstacle_vel = velocity
        g.obstacle_on_nominal_path = count > 1
        g.obstacle_specs = None
        g.obstacle_z_motion = spec['z']
    else:
        g.obstacle_enabled = True
        g.obstacle_specs = spec['specs']
        g.obstacle_on_nominal_path = True
    return g


def render_frame(renderer, env, camera, lookat_xy):
    renderer.update_scene(env.data)
    ee = env._get_end_effector_position()
    if camera == 'topdown':
        _set_topdown_camera(renderer, lookat_xy=lookat_xy)
    elif camera == 'track':
        _set_track_camera(renderer, ee)
    else:
        _set_oblique_camera(renderer, lookat_xy=lookat_xy)
    return renderer.render().copy()


def main():
    parser = argparse.ArgumentParser(description='Record pure MPC demo GIFs')
    parser.add_argument('--scene', choices=list(SCENE_CFG), default='dyn_both')
    parser.add_argument('--n-success', type=int, default=1)
    parser.add_argument('--max-episodes', type=int, default=100)
    parser.add_argument('--frame-step', type=int, default=2)
    parser.add_argument('--max-steps', type=int, default=200)
    parser.add_argument('--target-vel', type=float, default=0.05)
    parser.add_argument('--camera', choices=['oblique', 'topdown', 'track'], default='oblique')
    parser.add_argument('--width', type=int, default=640)
    parser.add_argument('--height', type=int, default=480)
    parser.add_argument('--outdir', default='results/demo_mpc')
    args = parser.parse_args()

    import mujoco

    cfg = get_config()
    configure_scene(cfg, args.scene, args.target_vel, args.max_steps)
    env = GraspingEnv(cfg.grasping, cfg.reward)
    renderer = mujoco.Renderer(env.model, args.height, args.width)
    os.makedirs(args.outdir, exist_ok=True)
    lookat_xy = tuple(cfg.grasping.object_fixed_pos)
    saved = 0

    try:
        for episode in range(1, args.max_episodes + 1):
            obs, _ = env.reset()
            if env.nominal_trajectory is not None:
                env.nominal_trajectory.reset()
            frames = []
            info = {}
            for step in range(1, args.max_steps + 1):
                action = np.zeros(env.action_space.shape[0], dtype=np.float32)
                obs, reward, terminated, truncated, info = env.step(action)
                if step % args.frame_step == 0 or terminated or truncated:
                    frames.append(render_frame(renderer, env, args.camera, lookat_xy))
                if terminated or truncated:
                    break

            success = bool(info.get('grasp_success', False))
            print(f'[mpc-demo] scene={args.scene} ep={episode}/{args.max_episodes} '
                  f'success={success} steps={step} '
                  f'collisions={info.get("obstacle_collision_count", 0)}')
            if not success or not frames:
                continue
            saved += 1
            path = os.path.join(args.outdir, f'mpc_{args.scene}_success{saved}.gif')
            images = [Image.fromarray(frame) for frame in frames]
            images[0].save(path, save_all=True, append_images=images[1:],
                           duration=60, loop=0)
            print(f'[mpc-demo] saved={path} frames={len(images)}')
            if saved >= args.n_success:
                break
    finally:
        renderer.close()
        env.close()

    if saved == 0:
        print('[mpc-demo] no successful episode recorded')
        return 1
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
