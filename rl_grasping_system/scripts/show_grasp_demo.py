#!/usr/bin/env python3
"""展示当前模型抓取效果：离屏渲染成功案例并保存 GIF（2026-08-24）

思路：与训练/评估同口径（use_fixed_position=False + workspace_bounds=±radius），
加载模型后用 deterministic 策略跑 episode；每 N 决策步用 mujoco.Renderer 离屏渲染一帧，
采集到成功案例即保存为 GIF（不弹 GLFW 窗口，可在无头环境运行）。

用法：
    cd rl_grasping_system
    MUJOCO_GL=egl python3 scripts/show_grasp_demo.py --model models/final_model_stage2.zip --radius 0.06

产物：results/demo/final_model_stage2_success{N}.gif

录制视角：MuJoCo 默认相机已绕竖直轴旋转 180°（从对面拍摄），见 _flip_camera_180。
"""
import os
import sys
import argparse

import numpy as np
from PIL import Image

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config import get_config
from evaluate import load_model


def _flip_camera_180(renderer):
    """把渲染器当前相机绕世界竖直轴 (Z) 旋转 180°，实现\"从对面拍摄\"。

    模型 XML 未定义 camera，mujoco.Renderer 每帧用 MuJoCo 默认 fixed 相机
    （azimuth=-90°, elevation=-30°, distance=3, lookat=原点）。update_scene 每次都会
    重算相机位姿，所以必须在 update_scene 之后、render() 之前改写 scene.camera[0]，
    render() 会直接使用该位姿。

    变换：以相机焦点 lookat 为轴心、绕竖直 Z 轴转 180°。位置保持高度/焦点不变，
    朝向 forward/up 同样旋转，画面保持正立不颠倒。
    """
    cam = renderer.scene.camera[0]
    pos = np.array(cam.pos, dtype=float)
    forward = np.array(cam.forward, dtype=float)
    # 相机焦点：fixed 相机 update_scene 后会填充 lookat 字段（默认相机 lookat=原点）。
    # 注意：cam.forward 是单位向量，不能用 pos+forward 当焦点（会忽略 distance）。
    lookat = np.array(cam.lookat, dtype=float)
    rz = np.array([[-1.0, 0.0, 0.0],
                   [0.0, -1.0, 0.0],
                   [0.0, 0.0, 1.0]])           # 绕 Z 轴旋转 180°
    cam.pos[:] = lookat + rz @ (pos - lookat)
    cam.forward[:] = rz @ forward
    cam.up[:] = rz @ np.array(cam.up, dtype=float)
    # 归一化（浮点误差保护）
    cam.forward /= np.linalg.norm(cam.forward)
    cam.up /= np.linalg.norm(cam.up)


def main():
    ap = argparse.ArgumentParser(description="渲染当前模型抓取效果 GIF")
    ap.add_argument("--model", default="models/final_model_stage2.zip")
    ap.add_argument("--radius", type=float, default=0.06)
    ap.add_argument("--n-success", type=int, default=2, help="要采集的成功案例数")
    ap.add_argument("--max-episodes", type=int, default=60, help="最多尝试 episode 数")
    ap.add_argument("--frame-step", type=int, default=2, help="每 N 决策步采一帧")
    ap.add_argument("--width", type=int, default=640)
    ap.add_argument("--height", type=int, default=480)
    ap.add_argument("--outdir", default="results/demo")
    args = ap.parse_args()

    config = get_config()
    cx, cy = config.grasping.object_fixed_pos
    z = config.grasping.workspace_bounds[2]
    config.grasping.render_gui = False            # 离屏：避免 step 内部弹 GLFW 窗口
    config.grasping.use_fixed_position = False
    config.grasping.workspace_bounds = ((cx - args.radius, cx + args.radius),
                                        (cy - args.radius, cy + args.radius), z)

    agent, env = load_model(args.model, config)

    import mujoco
    renderer = mujoco.Renderer(env.model, args.height, args.width)

    os.makedirs(args.outdir, exist_ok=True)
    base = os.path.splitext(os.path.basename(args.model))[0]

    collected = 0
    for ep in range(1, args.max_episodes + 1):
        obs, _ = env.reset()
        frames = []
        done, n_steps = False, 0
        while not done and n_steps < 500:
            action, _ = agent.predict(obs, deterministic=True)
            obs, _r, term, trunc, info = env.step(action)
            n_steps += 1
            renderer.update_scene(env.data)
            _flip_camera_180(renderer)            # 录制视角绕竖直轴旋转 180°
            img = renderer.render()               # HxWx3 uint8
            if n_steps % args.frame_step == 0:
                frames.append(img.copy())
            if term or trunc:
                done = True

        ok = bool(info.get('grasp_success', False))
        obj = env._get_object_position()
        mark = "✅成功" if ok else "❌失败"
        print(f"ep {ep:3d}: {mark}  步数={n_steps:3d}  obj=({obj[0]:.3f},{obj[1]:.3f})")
        if not ok:
            continue

        collected += 1
        out = os.path.join(args.outdir, f"{base}_success{collected}.gif")
        imgs = [Image.fromarray(f) for f in frames]
        imgs[0].save(out, save_all=True, append_images=imgs[1:],
                     duration=60, loop=0)
        print(f"      -> 已保存 {out} ({len(frames)} 帧)")
        if collected >= args.n_success:
            break

    if collected == 0:
        print("未采集到成功案例——可增大 --max-episodes，或改小 --radius 试跑")

    renderer.close()


if __name__ == "__main__":
    main()
