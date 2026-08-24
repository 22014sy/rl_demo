#!/usr/bin/env python3
"""Stage 2 前置：±6cm 物体位置随机化 IK 可达性打点验证（2026-08-24）

object_fixed_pos=(0.1, 0.42) 恰为 workspace_bounds X/Y 上界；±6cm →
X∈[0.04,0.16]、Y∈[0.36,0.48] 超出标称边界。本脚本按网格对每点做抓取位姿
IK（DLS，复用 ik.solve_ik），统计可达率并列出不可达点，用于决定 Stage 2
半径取 0.06 还是降到 0.05。

用法（rl_grasping_system/ 下）：
    python scripts/verify_radius_006_reachability.py [--step 0.02] [--out json]
"""
import os
import sys
import json
import argparse
import numpy as np
import mujoco

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from verify_grasp_feasibility import (
    MODEL_PATH, FINGER_REACH, solve_ik, frame_to_quat,
)

REST_Z = 0.32          # 与 verify_position_randomization 的 REST_Z 一致
CENTER = (0.1, 0.42)   # config.grasping.object_fixed_pos
RADIUS = 0.06          # 待验证半径


def grasp_pose_for(cube_pos):
    target_quat = frame_to_quat([0, 0, -1], [0, -1, 0])
    hand_pos = np.array([cube_pos[0], cube_pos[1], cube_pos[2] + FINGER_REACH])
    return hand_pos, target_quat


def main():
    parser = argparse.ArgumentParser(description="±6cm 随机化 IK 可达性打点")
    parser.add_argument("--step", type=float, default=0.02, help="网格步长(m)")
    parser.add_argument("--out", type=str,
                        default=os.path.join(HERE, "radius_006_reachability.json"),
                        help="结果 JSON 输出路径")
    args = parser.parse_args()

    model = mujoco.MjModel.from_xml_path(MODEL_PATH)
    data = mujoco.MjData(model)

    hand_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "rq_base_mount")
    if hand_id < 0:
        print("[FAIL] 找不到 body: rq_base_mount")
        return 1
    arm_joints = [mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, n)
                  for n in ["shoulder_pan_joint", "shoulder_lift_joint", "elbow_joint",
                            "wrist_1_joint", "wrist_2_joint", "wrist_3_joint"]]
    home = np.array([-1.5708, -1.5708, 1.5708, -1.5708, -1.5708, 0.0])

    cx, cy = CENTER
    xs = np.round(np.arange(cx - RADIUS, cx + RADIUS + 1e-9, args.step), 4)
    ys = np.round(np.arange(cy - RADIUS, cy + RADIUS + 1e-9, args.step), 4)
    print(f"网格: X∈[{xs[0]},{xs[-1]}] × Y∈[{ys[0]},{ys[-1]}]  共 {len(xs)*len(ys)} 点 (step={args.step})")

    results = []   # (x, y, err)
    for x in xs:
        for y in ys:
            data.qpos[arm_joints] = home
            qadr = model.jnt_qposadr[14]
            data.qpos[qadr:qadr + 3] = [x, y, REST_Z]
            data.qpos[qadr + 3:qadr + 7] = [1, 0, 0, 0]
            mujoco.mj_forward(model, data)
            hand_pos, target_quat = grasp_pose_for(np.array([x, y, REST_Z]))
            _, err = solve_ik(model, data, hand_id, hand_pos, target_quat, arm_joints)
            results.append((float(x), float(y), float(err)))

    reachable = [r for r in results if r[2] < 5e-3]
    rate = len(reachable) / len(results)
    unreach = [r for r in results if r[2] >= 5e-3]
    print(f"[RESULT] 可达 {len(reachable)}/{len(results)} = {rate*100:.1f}%  (阈值 e<5e-3)")
    if unreach:
        print(f"[INFO] 不可达 {len(unreach)} 点 (x, y, err):")
        for x, y, e in unreach:
            print(f"   x={x:+.3f} y={y:+.3f} err={e:.5f}")
    errs = [r[2] for r in results]
    print(f"[INFO] IK 误差: min={min(errs):.5f} max={max(errs):.5f} "
          f"p90={np.percentile(errs, 90):.5f}")

    payload = {
        "model": MODEL_PATH, "radius": RADIUS, "center": list(CENTER), "rest_z": REST_Z,
        "step": args.step, "n_points": len(results),
        "reachable": len(reachable), "reachability_rate": rate,
        "unreachable": [{"x": x, "y": y, "ik_err": e} for x, y, e in unreach],
        "ik_err_min": min(errs), "ik_err_max": max(errs),
        "ik_err_p90": float(np.percentile(errs, 90)),
    }
    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)
    print(f"[INFO] 结果已写入: {args.out}")

    return 0 if rate == 1.0 else 1


if __name__ == "__main__":
    sys.exit(main())
