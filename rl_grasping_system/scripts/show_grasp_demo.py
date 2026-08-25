#!/usr/bin/env python3
"""展示当前模型抓取效果：离屏渲染成功案例并保存 GIF（2026-08-24）

思路：与训练/评估同口径（use_fixed_position=False + workspace_bounds=±radius），
加载模型后用 deterministic 策略跑 episode；每 N 决策步用 mujoco.Renderer 离屏渲染一帧，
采集到成功案例即保存为 GIF（不弹 GLFW 窗口，可在无头环境运行）。

用法：
    cd rl_grasping_system
    # 默认视角：绕竖直轴转 180°（从对面拍摄）
    MUJOCO_GL=egl python3 scripts/show_grasp_demo.py --model models/final_model_stage2_d1.zip --radius 0.06
    # 俯拍视角：--topdown（相机位于工作区正上方垂直向下，展示桌面布局）
    MUJOCO_GL=egl python3 scripts/show_grasp_demo.py --model models/final_model_stage2_d1.zip --radius 0.06 --topdown
    # D2/D3 避障抓取模型（v5 训练口径：residual + 静态障碍 on-path，必须带下列参数否则语义失配）
    MUJOCO_GL=egl python3 scripts/show_grasp_demo.py --model models/final_model_stage2_d2_v5.zip --radius 0.03 \
        --action-mode residual --obstacle-on-path --obstacle-w 0.35 --residual-reg-w 0.3
    # 双视角同屏拼接（斜视+俯拍，推荐展示绕障抓取全过程）:
    MUJOCO_GL=egl python3 scripts/show_grasp_demo.py --model models/final_model_stage2_d2_v5.zip --radius 0.03 \
        --action-mode residual --obstacle-on-path --obstacle-w 0.35 --residual-reg-w 0.3 --dual
    # 跟随末端视角（--track）：相机锁定夹爪特写，突出绕障轨迹（可单独用，也可 --dual 时做左栏）
    MUJOCO_GL=egl python3 scripts/show_grasp_demo.py --model models/final_model_stage2_d2_v5.zip --radius 0.03 \
        --action-mode residual --obstacle-on-path --obstacle-w 0.35 --residual-reg-w 0.3 --track
    MUJOCO_GL=egl python3 scripts/show_grasp_demo.py --model models/final_model_stage2_d2_v5.zip --radius 0.03 \
        --action-mode residual --obstacle-on-path --obstacle-w 0.35 --residual-reg-w 0.3 --dual --track

产物：results/demo/{model名}_success{N}.gif（单视角）/ {model名}_dual{N}.gif（双视角）
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
    # MjvGLCamera 不暴露 lookat/distance；未定义模型相机时，默认相机的焦点是世界原点。
    lookat = np.zeros(3, dtype=float)
    rz = np.array([[-1.0, 0.0, 0.0],
                   [0.0, -1.0, 0.0],
                   [0.0, 0.0, 1.0]])           # 绕 Z 轴旋转 180°
    cam.pos[:] = lookat + rz @ (pos - lookat)
    cam.forward[:] = rz @ forward
    cam.up[:] = rz @ np.array(cam.up, dtype=float)
    # 归一化（浮点误差保护）
    cam.forward /= np.linalg.norm(cam.forward)
    cam.up /= np.linalg.norm(cam.up)


def _set_topdown_camera(renderer, lookat_xy=(0.0, 0.0), height=1.0):
    """把渲染器当前相机设为桌面正上方垂直俯拍。

    位置 = 桌面中心上方 height 处，forward 垂直向下 [-Z]，up 取 +Y 使画面正立不颠倒。
    lookat 指向桌面工作区中心（z=桌面物体高度 0.32，覆盖 ±radius 随机范围）。
    """
    cam = renderer.scene.camera[0]
    lookat = np.array([lookat_xy[0], lookat_xy[1], 0.32], dtype=float)
    cam.pos[:] = lookat + np.array([0.0, 0.0, height])
    cam.forward[:] = np.array([0.0, 0.0, -1.0])
    cam.up[:] = np.array([0.0, 1.0, 0.0])


def _set_oblique_camera(renderer, lookat_xy=(0.0, 0.0), distance=0.8, elevation=0.15):
    """把相机设为水平方位角 45°、接近桌面高度的低平视角观察桌面中心。

    相机相对 lookat 的方向 = (1, -1, elevation)：x/y 各 45° 水平方位角（从 y 负侧看），
    elevation 越小越接近与桌面持平（elevation=0 完全平视，取小正值保留少量俯视纵深）。
    默认 0.15 → 俯角约 6°。up 取世界 Z 在 forward 垂直平面上的投影，保证画面正立不倾斜。
    """
    cam = renderer.scene.camera[0]
    lookat = np.array([lookat_xy[0], lookat_xy[1], 0.32], dtype=float)
    direction = np.array([1.0, -1.0, elevation], dtype=float)
    direction /= np.linalg.norm(direction)
    cam.pos[:] = lookat + distance * direction
    cam.forward[:] = -direction
    up = np.array([0.0, 0.0, 1.0], dtype=float)
    up -= cam.forward * float(up @ cam.forward)   # 投影到垂直于 forward 的平面（保证画面正立）
    up /= np.linalg.norm(up)
    cam.up[:] = up


def _set_track_camera(renderer, ee_pos, distance=0.45):
    """把相机设为跟随末端：始终对准夹爪，机械臂占画面主体（tracking 特写）。

    相机相对末端的方向取固定世界方向 (-1,-1,0.8)（末端后上方），末端移动时相机随之平移，
    画面稳定不抖；up 取世界 Z 在 forward 垂直平面上的投影，保证画面正立不倾斜。
    适合突出绕障时末端偏离标称路径的轨迹。
    """
    cam = renderer.scene.camera[0]
    lookat = np.array([ee_pos[0], ee_pos[1], ee_pos[2]], dtype=float)
    direction = np.array([-1.0, -1.0, 0.8], dtype=float)
    direction /= np.linalg.norm(direction)
    cam.pos[:] = lookat + distance * direction
    cam.forward[:] = -direction
    up = np.array([0.0, 0.0, 1.0], dtype=float)
    up -= cam.forward * float(up @ cam.forward)   # 投影到垂直于 forward 的平面（保证画面正立）
    up /= np.linalg.norm(up)
    cam.up[:] = up


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
    ap.add_argument("--topdown", action="store_true",
                    help="俯拍视角（桌面正上方垂直向下；默认绕竖直轴转 180° 对面拍摄）")
    ap.add_argument("--dual", action="store_true",
                    help="双视角同屏拼接：斜视(左) + 俯拍(右) 水平拼接，一张 GIF 同时看动作与布局")
    ap.add_argument("--track", action="store_true",
                    help="跟随末端视角：相机锁定夹爪特写（机械臂始终居中，突出绕障轨迹；单视角或 --dual 左栏）")
    # D2/D3 训练口径（2026-08-25）：v5 等避障抓取模型是 residual + 静态障碍 on-path 训练的，
    # 展示时必须用与训练一致的配置，否则观测/动作语义失配、展示的是"骗人"效果。
    ap.add_argument("--action-mode", type=str, default='', choices=['', 'delta', 'residual'],
                    help='D1 动作模式（训练口径）：delta=旧增量语义（默认）；residual=标称轨迹+残差叠加 v=v_nominal+Δv/T')
    ap.add_argument("--obstacle-on-path", action="store_true",
                    help='D2 训练口径：静态障碍放"标称必经之路"（obstacle_enabled + obstacle_on_nominal_path=True）')
    ap.add_argument("--obstacle-w", type=float, default=-1.0,
                    help='D2 障碍接近惩罚权重（<0 用 config 默认；v4/v5 训练用 0.35，仅影响 reward 数值口径）')
    ap.add_argument("--residual-reg-w", type=float, default=-1.0,
                    help='D2 残差幅度正则权重（<0 用 config 默认；v4/v5 训练用 0.3，仅影响 reward 数值口径）')
    args = ap.parse_args()

    config = get_config()
    # D2/D3 训练口径：与 train_with_monitor.py 的 CLI 覆盖保持一致的开关（默认保持旧行为）
    if args.action_mode:
        config.grasping.action_mode = args.action_mode
    if args.obstacle_on_path:
        config.grasping.obstacle_enabled = True
        config.grasping.obstacle_on_nominal_path = True
    if args.obstacle_w >= 0:
        config.reward.obstacle_w = args.obstacle_w
    if args.residual_reg_w >= 0:
        config.reward.residual_reg_w = args.residual_reg_w
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
            ee = env._get_end_effector_position()
            if args.dual:
                # 双视角同屏：左特写(低平或跟随末端) + 右俯拍布局（每次 update_scene 后重设相机保证位姿生效）
                renderer.update_scene(env.data)
                if args.track:
                    _set_track_camera(renderer, ee)                     # 左：跟随末端特写
                else:
                    _set_oblique_camera(renderer, lookat_xy=(cx, cy))   # 左：低平视角
                img_left = renderer.render()
                renderer.update_scene(env.data)
                _set_topdown_camera(renderer, lookat_xy=(cx, cy))   # 右：俯拍（桌面布局与障碍位置）
                img_right = renderer.render()
                img = np.hstack([img_left, img_right])              # H x (2W) x 3
            else:
                renderer.update_scene(env.data)
                if args.topdown:
                    _set_topdown_camera(renderer, lookat_xy=(cx, cy))  # 俯拍（桌面正上方）
                elif args.track:
                    _set_track_camera(renderer, ee)                    # 跟随末端特写
                else:
                    _set_oblique_camera(renderer, lookat_xy=(cx, cy))  # 默认低平视角（接近桌面高度）
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
        tag = "dual" if args.dual else "success"
        out = os.path.join(args.outdir, f"{base}_{tag}{collected}.gif")
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
