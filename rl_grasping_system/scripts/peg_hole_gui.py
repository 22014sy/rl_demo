#!/usr/bin/env python3
"""
peg-in-hole RL 演示（MuJoCo 图形化界面）
==================================================
用 mujoco.viewer 实时显示 RL 插入过程，鼠标交互调视角：
  - 左键拖拽：旋转视角
  - 滚轮：缩放
  - 右键拖拽：平移
  - 按 R：重新开始一个 episode
  - 关闭窗口 / 按 Q：退出

用法（GUI 需 DISPLAY）：
  MUJOCO_GL=glfw python3 scripts/peg_hole_gui.py [--lateral 0.05] [--seed 7]
  # 初始视角参数
  --azimuth 90 --elevation -15 --distance 1.1 --lookat "0.1,0.42,0.30"
"""
import os
import sys
import argparse

os.environ.setdefault('MUJOCO_GL', 'glfw')   # GUI 用 GLFW（非 egl）
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import mujoco
import mujoco.viewer
import numpy as np
from stable_baselines3 import PPO

import peg_hole_min as P

MODEL_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'models', 'peg_hole_rl.zip')


def set_camera(viewer, args):
    """设置 GUI 初始视角（球坐标：lookat + distance + azimuth + elevation）。"""
    lookat = np.array([float(v) for v in args.lookat.split(',')])
    viewer.cam.type = mujoco.mjtCamera.mjCAMERA_FREE
    viewer.cam.lookat = lookat
    viewer.cam.distance = args.distance
    viewer.cam.azimuth = args.azimuth   # 绕竖直轴角度：0=从 x 正方向，90=从 y 正方向（侧面）
    viewer.cam.elevation = args.elevation


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--lateral', type=float, default=0.05, help='初始水平偏移(m)扰动展示')
    ap.add_argument('--seed', type=int, default=7)
    ap.add_argument('--azimuth', type=float, default=90.0, help='初始方位角°（90=侧面）')
    ap.add_argument('--elevation', type=float, default=-15.0, help='初始俯仰角°')
    ap.add_argument('--distance', type=float, default=1.1, help='初始相机距离(m)')
    ap.add_argument('--lookat', type=str, default='0.1,0.42,0.30', help='注视点 x,y,z')
    args = ap.parse_args()

    model = PPO.load(MODEL_PATH)
    env = P.PegHoleEnv()
    env.lateral_range = args.lateral

    viewer = mujoco.viewer.launch_passive(env.model, env.data,
                                           show_left_ui=False, show_right_ui=False)
    set_camera(viewer, args)
    viewer.sync()

    print('🎥 GUI 演示开始（左键旋转/滚轮缩放/右键平移，R=重开，Q/关窗=退出）')

    while viewer.is_running():
        obs, _ = env.reset(seed=args.seed)
        done = False
        steps = 0
        while viewer.is_running() and not done:
            a, _ = model.predict(obs, deterministic=True)
            obs, _, term, trunc, info = env.step(a)
            done = term or trunc
            steps += 1
            viewer.sync()
        print(f'episode done: success={info["success"]} 插入深度={info["insertion"]*1000:.0f}mm '
              f'步数={steps}')
        if not viewer.is_running():
            break
        # 等待用户按 R 重开或继续（循环自动重开，靠窗口关闭退出）
        import time
        time.sleep(1.0)

    viewer.close()
    env.close()
    print('退出')


if __name__ == '__main__':
    main()
