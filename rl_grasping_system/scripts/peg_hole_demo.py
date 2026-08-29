#!/usr/bin/env python3
"""
peg-in-hole RL 演示（2026-08-28）：加载训练模型，渲染插入过程为 gif。
运行：MUJOCO_GL=egl python3 scripts/peg_hole_demo.py [--lateral 0.05]
"""
import os
import sys
import argparse

os.environ.setdefault('MUJOCO_GL', 'egl')
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import mujoco
import numpy as np
from PIL import Image
from stable_baselines3 import PPO

import peg_hole_min as P

MODEL_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'models', 'peg_hole_rl.zip')
OUT = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'results', 'demo', 'peg_hole_rl.gif')


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--lateral', type=float, default=0.05, help='初始水平偏移(m)扰动展示')
    ap.add_argument('--max-frames', type=int, default=120)
    ap.add_argument('--camera', type=str, default='side_view',
                    help='相机名：side_view（侧面）/ eye_to_hand（俯视）')
    ap.add_argument('--out', type=str, default=OUT, help='gif 输出路径')
    args = ap.parse_args()

    model = PPO.load(MODEL_PATH)
    env = P.PegHoleEnv()
    env.lateral_range = args.lateral
    renderer = mujoco.Renderer(env.model, height=480, width=640)

    obs, _ = env.reset(seed=7)
    frames = []
    success = False
    for i in range(args.max_frames):
        renderer.update_scene(env.data, camera=args.camera)
        frames.append(Image.fromarray(renderer.render()))
        a, _ = model.predict(obs, deterministic=True)
        obs, _, term, trunc, info = env.step(a)
        if term or trunc:
            success = info['success']
            break
    renderer.update_scene(env.data, camera=args.camera)   # 结束帧
    frames.append(Image.fromarray(renderer.render()))

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    frames[0].save(args.out, save_all=True, append_images=frames[1:], duration=80, loop=0)
    print(f'gif saved: {args.out} ({len(frames)} 帧, camera={args.camera}, '
          f'success={success}, 插入深度 {info["insertion"]*1000:.0f}mm)')


if __name__ == '__main__':
    main()
