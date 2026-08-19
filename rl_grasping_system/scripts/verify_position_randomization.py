#!/usr/bin/env python3
"""
P0-2 物体位置随机化验证（纯 MuJoCo，不依赖 RL 训练环境代码）

目的：验证 P0-2 引入的"物体初始位置随机化"：
  1. 固定 seed 采样 N 个位置，全部落在 workspace_bounds 的 X/Y 范围内、Z=落定高度；
  2. 每个位置都能用 DLS-IK 把夹爪送到抓取位姿（可达性校验，复用 P0-1 的 solve_ik）；
  3. 抽查 3 个代表位置（距底座最近/中/最远）完整执行 张开->下降->闭合->抬升，全部 PASS。

运行方式（在 rl_grasping_system/ 目录下）：
    python scripts/verify_position_randomization.py
退出码 0 = 验证通过；非 0 = 失败。
可选参数：
    --seed   随机采样种子（默认 2026）
    --n      采样位置数量（默认 20）
    --steps  每阶段仿真步数（默认 300）
    --out    结果 JSON 输出路径
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
    MODEL_PATH, FINGER_REACH, solve_ik, move_to_q, frame_to_quat,
)

# 与 config.GraspingConfig.workspace_bounds 保持一致（X/Y 采样范围；Z 由落定高度决定，不采样）
DEFAULT_BOUNDS = ((0.3, 0.7), (-0.2, 0.2), (0.05, 0.3))
REST_Z = 0.02  # 物体落定高度（= 0.04m 立方体半边长，P0-1 实测落定 z≈0.0199）


def grasp_pose_for(cube_pos):
    """由物体位置计算抓取位姿：指尖朝下(-z)，手指沿世界 y 张开/闭合；hand_z = cube_z + FINGER_REACH"""
    target_quat = frame_to_quat([0, 0, -1], [0, 1, 0])
    hand_pos = np.array([cube_pos[0], cube_pos[1], cube_pos[2] + FINGER_REACH])
    return hand_pos, target_quat


def run_full_grasp(model, data, cube_id, hand_id, arm_joints, cube_pos0, steps):
    """完整执行 张开->下降->闭合->抬升，返回 (rise, contact_ok, errmsg)。复用 P0-1 流程。"""
    pre_pos = np.array([cube_pos0[0], cube_pos0[1],
                        cube_pos0[2] + 0.02 + FINGER_REACH + 0.01])   # 预抓取高度
    grasp_pos = np.array([cube_pos0[0], cube_pos0[1],
                          cube_pos0[2] + FINGER_REACH])               # 抓取高度
    lift_pos = np.array([cube_pos0[0], cube_pos0[1],
                         cube_pos0[2] + FINGER_REACH + 0.12])         # 抬升高度
    target_quat = frame_to_quat([0, 0, -1], [0, 1, 0])

    data.ctrl[7] = 255.0  # 手指全开
    q_pre, e_pre = solve_ik(model, data, hand_id, pre_pos, target_quat, arm_joints)
    if e_pre > 5e-3:
        return None, None, f"预抓取位不可达 e={e_pre:.5f}"
    move_to_q(model, data, arm_joints, q_pre, steps)

    q_grasp, e_grasp = solve_ik(model, data, hand_id, grasp_pos, target_quat, arm_joints)
    if e_grasp > 5e-3:
        return None, None, f"抓取位不可达 e={e_grasp:.5f}"
    move_to_q(model, data, arm_joints, q_grasp, steps)

    # 闭合手指（捏住物体）
    data.ctrl[7] = 0.0
    for _ in range(int(steps * 1.5)):
        mujoco.mj_step(model, data)

    # 抬升
    q_lift, e_lift = solve_ik(model, data, hand_id, lift_pos, target_quat, arm_joints)
    move_to_q(model, data, arm_joints, q_lift, int(steps * 1.5))

    cube_pos1 = data.xpos[cube_id].copy()
    rise = cube_pos1[2] - cube_pos0[2]

    gripper_bodies = {hand_id,
                      mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "left_finger"),
                      mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "right_finger")}
    contact_ok = False
    for c in range(data.ncon):
        b1, b2 = data.contact[c].geom1, data.contact[c].geom2
        g1b, g2b = model.geom_bodyid[b1], model.geom_bodyid[b2]
        if cube_id in (g1b, g2b) and (g1b in gripper_bodies or g2b in gripper_bodies):
            contact_ok = True
            break
    return rise, contact_ok, None


def main():
    parser = argparse.ArgumentParser(description="P0-2 物体位置随机化验证（纯 MuJoCo）")
    parser.add_argument("--seed", type=int, default=2026, help="随机采样种子")
    parser.add_argument("--n", type=int, default=20, help="采样位置数量")
    parser.add_argument("--steps", type=int, default=300, help="每阶段仿真步数")
    parser.add_argument("--out", type=str,
                        default=os.path.join(HERE, "position_randomization_result.json"),
                        help="结果 JSON 输出路径")
    args = parser.parse_args()

    rng = np.random.default_rng(args.seed)
    model = mujoco.MjModel.from_xml_path(MODEL_PATH)
    data = mujoco.MjData(model)

    hand_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "hand")
    cube_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "target_cube")
    if hand_id < 0 or cube_id < 0:
        print("[FAIL] 模型里找不到 body: hand / target_cube")
        return 1

    arm_joints = list(range(7))
    home = np.array([0.0, 0.0, 0.0, -1.5708, 0.0, 1.5708, 0.7854])
    data.qpos[arm_joints] = home
    data.qpos[7] = 0.04  # finger_joint1 全开
    data.qpos[8] = 0.04  # finger_joint2 全开
    mujoco.mj_forward(model, data)

    # ---------------- 1) 固定 seed 采样 N 个位置 ----------------
    positions = []
    for _ in range(args.n):
        x = rng.uniform(DEFAULT_BOUNDS[0][0], DEFAULT_BOUNDS[0][1])
        y = rng.uniform(DEFAULT_BOUNDS[1][0], DEFAULT_BOUNDS[1][1])
        positions.append(np.array([x, y, REST_Z]))

    # ---------------- 2) 逐个做抓取位姿 IK 可达性校验 ----------------
    reachability = []  # [(pos, ik_err)]
    for p in positions:
        data.qpos[9:12] = p          # freejoint 位置（qpos 布局：0:7 臂, 7:9 手指, 9:16 物体）
        data.qpos[12:16] = [1, 0, 0, 0]
        mujoco.mj_forward(model, data)
        hand_pos, target_quat = grasp_pose_for(p)
        _, err = solve_ik(model, data, hand_id, hand_pos, target_quat, arm_joints)
        reachability.append((p, err))

    reachable = [p for p, e in reachability if e < 5e-3]
    rate = len(reachable) / len(positions)
    print(f"[INFO] 采样 {len(positions)} 个位置（seed={args.seed}），"
          f"抓取位姿可达 {len(reachable)}/{len(positions)} = {rate*100:.1f}%")

    # ---------------- 3) 抽查 3 个代表位置（距底座最近/中/最远）完整抓取 ----------------
    dists = [float(np.linalg.norm(p[:2])) for p, _ in reachability]
    idx = np.argsort(dists)
    spot = [int(idx[0]), int(idx[len(idx) // 2]), int(idx[-1])]
    spot_results = []
    all_pass = True
    for i in spot:
        p, e = reachability[i]
        # 重新初始化：物体放回地面、机械臂回 home（每个抽查独立）
        data.qpos[arm_joints] = home
        data.qpos[7], data.qpos[8] = 0.04, 0.04
        data.qpos[9:12] = p
        data.qpos[12:16] = [1, 0, 0, 0]
        mujoco.mj_forward(model, data)
        rise, contact_ok, errmsg = run_full_grasp(model, data, cube_id, hand_id, arm_joints, p, args.steps)
        ok = (errmsg is None) and rise is not None and rise > 0.06 and contact_ok
        all_pass = all_pass and ok
        spot_results.append({
            "pos": p.tolist(), "dist": dists[i],
            "ik_err": e, "rise_m": (None if rise is None else float(rise)),
            "contact_ok": bool(contact_ok), "error": errmsg, "passed": bool(ok),
        })
        print(f"[INFO] 抽查#{i} pos={p.round(4)} dist={dists[i]:.3f} "
              f"ik_err={e:.5f} rise={None if rise is None else round(rise, 4)} "
              f"contact={contact_ok} -> {'OK' if ok else ('FAIL ' + (errmsg or ''))}")

    # ---------------- 结果汇总 ----------------
    result = {
        "model": MODEL_PATH,
        "seed": args.seed,
        "n_samples": len(positions),
        "reachable": len(reachable),
        "reachability_rate": rate,
        "bounds": {"x": list(DEFAULT_BOUNDS[0]), "y": list(DEFAULT_BOUNDS[1]), "rest_z": REST_Z},
        "unreachable_positions": [p.tolist() for p, e in reachability if e >= 5e-3],
        "spot_checks": spot_results,
    }
    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2, ensure_ascii=False)
    print(f"[INFO] 结果已写入: {args.out}")

    passed = (rate == 1.0) and all_pass
    if passed:
        print("[PASS] 全部随机位置可达且抽查抓取全部成功。")
        return 0
    print("[FAIL] 存在不可达位置或抽查失败，需缩小随机范围或检查模型。")
    return 1


if __name__ == "__main__":
    sys.exit(main())
