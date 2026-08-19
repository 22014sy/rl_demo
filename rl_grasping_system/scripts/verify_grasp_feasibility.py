#!/usr/bin/env python3
"""
P0-1 任务物理可行性验证（纯 MuJoCo，不依赖 RL 训练环境代码）

目的：在修复后的模型上验证"抓取"在物理上是否真的可行：
  1. target_cube 现在是可自由移动的物体（带 freejoint），能被夹爪带动抬升；
  2. 夹爪能张开到足以包住物体（物体 0.04m < 夹爪内表面最大开口 ≈0.056m，实测）；
  3. 用阻尼最小二乘(DLS) IK 把夹爪送到物体上方，依次执行
     张开 -> 下降 -> 闭合 -> 抬升 序列，验证物体被夹起并保持。

运行方式（在 rl_grasping_system/ 目录下）：
    python scripts/verify_grasp_feasibility.py
退出码 0 = 验证通过；非 0 = 失败。
可选参数：
    --steps 每阶段步数（默认 300，加快/放慢用）
    --out   结果 JSON 输出路径
"""
import os
import sys
import json
import argparse

import numpy as np
import mujoco

HERE = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.dirname(os.path.dirname(HERE))  # rl_grasping_system 的上两级 = 仓库根
MODEL_PATH = os.path.join(
    REPO_ROOT, "models", "franka_emika_panda", "single_cube_scene.xml"
)

# 指尖(衬垫)在 hand 局部坐标系 +z 方向的距离（实测：finger body +z0.0584 + pad +z0.0445）
FINGER_REACH = 0.1029


# ---------------------------------------------------------------- 四元数工具
def quat_mul(q1, q2):
    """四元数乘法，顺序 q1 * q2（q = w,x,y,z）"""
    w1, x1, y1, z1 = q1
    w2, x2, y2, z2 = q2
    return np.array([
        w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2,
        w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
        w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2,
        w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2,
    ])


def quat_inv(q):
    return np.array([q[0], -q[1], -q[2], -q[3]])


def quat_vec(q):
    """四元数 -> 旋转向量（2*vec，小角度近似）"""
    return 2.0 * q[1:]


def rotmat_to_quat(R):
    """旋转矩阵 -> 四元数 (w,x,y,z)，Shepperd 方法"""
    tr = np.trace(R)
    if tr > 0.0:
        S = np.sqrt(tr + 1.0) * 2.0
        qw, qx, qy, qz = (0.25 * S,
                          (R[2, 1] - R[1, 2]) / S,
                          (R[0, 2] - R[2, 0]) / S,
                          (R[1, 0] - R[0, 1]) / S)
    elif R[0, 0] > R[1, 1] and R[0, 0] > R[2, 2]:
        S = np.sqrt(1.0 + R[0, 0] - R[1, 1] - R[2, 2]) * 2.0
        qw, qx, qy, qz = ((R[2, 1] - R[1, 2]) / S,
                          0.25 * S,
                          (R[0, 1] + R[1, 0]) / S,
                          (R[0, 2] + R[2, 0]) / S)
    elif R[1, 1] > R[2, 2]:
        S = np.sqrt(1.0 + R[1, 1] - R[0, 0] - R[2, 2]) * 2.0
        qw, qx, qy, qz = ((R[0, 2] - R[2, 0]) / S,
                          (R[0, 1] + R[1, 0]) / S,
                          0.25 * S,
                          (R[1, 2] + R[2, 1]) / S)
    else:
        S = np.sqrt(1.0 + R[2, 2] - R[0, 0] - R[1, 1]) * 2.0
        qw, qx, qy, qz = ((R[1, 0] - R[0, 1]) / S,
                          (R[0, 2] + R[2, 0]) / S,
                          (R[1, 2] + R[2, 1]) / S,
                          0.25 * S)
    q = np.array([qw, qx, qy, qz])
    return q / np.linalg.norm(q)


def frame_to_quat(z_local_world, y_local_world):
    """由局部 z 轴、y 轴在世界的朝向构造四元数（R 的列 = 局部轴在世界坐标）"""
    zl = np.asarray(z_local_world, dtype=float)
    zl /= np.linalg.norm(zl)
    yl = np.asarray(y_local_world, dtype=float)
    yl -= np.dot(yl, zl) * zl
    yl /= np.linalg.norm(yl)
    xl = np.cross(yl, zl)
    R = np.column_stack([xl, yl, zl])
    return rotmat_to_quat(R)


# ---------------------------------------------------------------- 阻尼最小二乘 IK
def solve_ik(model, data, body_id, target_pos, target_quat, joint_ids,
             iters=5000, tol=3e-4, lam=0.05, lr=0.6):
    """DLS IK：把 body_id 移到 target_pos / target_quat，返回 (q, 位置误差)。

    纯函数：在临时副本上求解，不改动主 data —— 调用者随后用 move_to_q
    平滑地伺服到返回的 q，避免“瞬移”导致物体接触丢失。
    """
    scratch = mujoco.MjData(model)
    scratch.qpos[:] = data.qpos[:]
    scratch.qvel[:] = data.qvel[:]
    scratch.ctrl[:] = data.ctrl[:]
    mujoco.mj_forward(model, scratch)

    jacp = np.zeros((3, model.nv))
    jacr = np.zeros((3, model.nv))
    q = scratch.qpos[joint_ids].copy()
    for _ in range(iters):
        scratch.qpos[joint_ids] = q
        mujoco.mj_forward(model, scratch)

        pos_err = target_pos - scratch.xpos[body_id]
        q_rel = quat_mul(target_quat, quat_inv(scratch.xquat[body_id]))
        rot_err = quat_vec(q_rel)
        err = np.concatenate([pos_err, rot_err])

        if np.linalg.norm(pos_err) < tol and np.linalg.norm(rot_err) < 1e-3:
            break

        mujoco.mj_jacBody(model, scratch, jacp, jacr, body_id)
        J = np.vstack([jacp[:, joint_ids], jacr[:, joint_ids]])  # (6, n)
        A = J @ J.T + lam * lam * np.eye(6)
        dq = lr * (J.T @ np.linalg.solve(A, err))

        q = q + dq
        for k, jid in enumerate(joint_ids):
            lo, hi = model.jnt_range[jid]
            q[k] = np.clip(q[k], lo, hi)

    scratch.qpos[joint_ids] = q
    mujoco.mj_forward(model, scratch)
    return q, float(np.linalg.norm(target_pos - scratch.xpos[body_id]))


def move_to_q(model, data, arm_joints, q_target, n_steps):
    """把关节位置伺服目标从当前值线性插值到 q_target 并逐步推进仿真"""
    q0 = data.qpos[arm_joints].copy()
    for k in range(1, n_steps + 1):
        alpha = k / n_steps
        data.ctrl[arm_joints] = q0 + alpha * (q_target - q0)
        mujoco.mj_step(model, data)


def main():
    parser = argparse.ArgumentParser(description="P0-1 抓取可行性验证（纯 MuJoCo）")
    parser.add_argument("--steps", type=int, default=300,
                        help="每阶段仿真步数")
    parser.add_argument("--out", type=str,
                        default=os.path.join(HERE, "grasp_verification_result.json"),
                        help="结果 JSON 输出路径")
    args = parser.parse_args()

    model = mujoco.MjModel.from_xml_path(MODEL_PATH)
    data = mujoco.MjData(model)

    hand_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "hand")
    cube_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "target_cube")
    if hand_id < 0 or cube_id < 0:
        print("[FAIL] 模型里找不到 body: hand / target_cube")
        return 1

    # 关节布局说明：qpos[0:7] 机械臂、qpos[7:9] 手指、qpos[9:16] 物体 freejoint
    arm_joints = list(range(7))
    home = np.array([0.0, 0.0, 0.0, -1.5708, 0.0, 1.5708, 0.7854])

    data.qpos[arm_joints] = home
    data.qpos[7] = 0.04   # finger_joint1 全开
    data.qpos[8] = 0.04   # finger_joint2 全开
    mujoco.mj_forward(model, data)

    # 物体带 freejoint 后会自由落体：先让机械臂停住、物体落到地面并稳定，
    # 再以"落定后的位置"为抓取目标（否则指尖悬空，永远碰不到物体）
    data.ctrl[arm_joints] = home
    data.ctrl[7] = 255.0  # 手指保持全开
    for _ in range(600):
        mujoco.mj_step(model, data)

    cube_pos0 = data.xpos[cube_id].copy()
    print(f"[INFO] 模型: {MODEL_PATH}")
    print(f"[INFO] 物体落定后位置: {cube_pos0.round(4)}  "
          f"(freejoint qpos 索引 9-15: 位置+四元数)")
    print(f"[INFO] nq={model.nq} nv={model.nv}  "
          f"手指关节 = {data.qpos[7]:.3f}/{data.qpos[8]:.3f}")

    # 抓取朝向：夹爪(指尖)朝下 -z，手指沿世界 y 张开/闭合（捏住物体两侧）
    target_quat = frame_to_quat([0, 0, -1], [0, 1, 0])

    # 指尖在 hand 局部 +z 方向 FINGER_REACH 处，因此：
    #   fingertip_z = hand_z - FINGER_REACH
    # 物体半高 0.02，顶面 z = cube_z + 0.02。
    # 预抓取：指尖在物体顶面上方 0.01m；抓取：指尖对准物体中部；抬升：指尖抬高 0.12m
    pre_pos = np.array([cube_pos0[0], cube_pos0[1],
                        cube_pos0[2] + 0.02 + FINGER_REACH + 0.01])  # 预抓取高度
    grasp_pos = np.array([cube_pos0[0], cube_pos0[1],
                          cube_pos0[2] + FINGER_REACH])              # 抓取高度
    lift_pos = np.array([cube_pos0[0], cube_pos0[1],
                         cube_pos0[2] + FINGER_REACH + 0.12])        # 抬升高度

    # ---------------- 阶段 1：张开手指并移动到预抓取位置 ----------------
    data.ctrl[7] = 255.0  # 手指全开
    q_pre, e_pre = solve_ik(model, data, hand_id, pre_pos, target_quat, arm_joints)
    print(f"[INFO] IK(预抓取) 结束位置误差: {e_pre:.5f} m")
    if e_pre > 5e-3:
        print("[FAIL] IK 无法把夹爪送到预抓取位置，物理上不可达？")
        return 1
    move_to_q(model, data, arm_joints, q_pre, args.steps)

    # ---------------- 阶段 2：下降到抓取高度 ----------------
    q_grasp, e_grasp = solve_ik(model, data, hand_id, grasp_pos, target_quat, arm_joints)
    print(f"[INFO] IK(抓取) 结束位置误差: {e_grasp:.5f} m")
    if e_grasp > 5e-3:
        print("[FAIL] IK 无法把夹爪送到抓取位置")
        return 1
    move_to_q(model, data, arm_joints, q_grasp, args.steps)
    print(f"[INFO] 指尖位于物体中截面: hand_z={data.xpos[hand_id][2]:.4f}, "
          f"物体 z={data.xpos[cube_id][2]:.4f}")

    # ---------------- 阶段 3：闭合手指（捏住物体） ----------------
    data.ctrl[7] = 0.0  # 手指全闭
    for _ in range(int(args.steps * 1.5)):
        mujoco.mj_step(model, data)
    print(f"[INFO] 闭合后手指关节: {data.qpos[7]:.4f}/{data.qpos[8]:.4f} "
          f"(物体 0.04m 挡住，无法全闭 => 已捏住)")

    # ---------------- 阶段 4：抬升 ----------------
    q_lift, e_lift = solve_ik(model, data, hand_id, lift_pos, target_quat, arm_joints)
    print(f"[INFO] IK(抬升) 结束位置误差: {e_lift:.5f} m")
    move_to_q(model, data, arm_joints, q_lift, int(args.steps * 1.5))

    # ---------------- 结果判定 ----------------
    cube_pos1 = data.xpos[cube_id].copy()
    hand_pos1 = data.xpos[hand_id].copy()
    rise = cube_pos1[2] - cube_pos0[2]

    # 判定条件：物体被抬升且抬升末端仍与夹爪保持接触（未被甩掉）。
    # 注意：接触发生在 left_finger/right_finger 与物体之间，需把这些 body 一并算作“夹爪”。
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

    print(f"[INFO] 抬升后 hand_z={hand_pos1[2]:.4f}, 物体 z={cube_pos1[2]:.4f}, "
          f"物体被抬升高度={rise:.4f} m")
    print(f"[INFO] 抬升末端物体与夹爪有接触: {contact_ok}")

    result = {
        "model": MODEL_PATH,
        "cube_pos0": cube_pos0.tolist(),
        "cube_pos1": cube_pos1.tolist(),
        "hand_pos1": hand_pos1.tolist(),
        "cube_rise_m": rise,
        "gripper_contact_at_lift_end": contact_ok,
        "ik_err_pre": e_pre,
        "ik_err_grasp": e_grasp,
        "ik_err_lift": e_lift,
    }

    passed = (rise > 0.06) and contact_ok
    result["passed"] = bool(passed)

    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2, ensure_ascii=False)
    print(f"[INFO] 结果已写入: {args.out}")

    if passed:
        print("[PASS] 物理可行性验证通过：0.04m 物体可被夹爪抓起并抬升。")
        return 0
    print("[FAIL] 物体未能被抬升/保持，需检查模型或抓取参数。")
    return 1


if __name__ == "__main__":
    sys.exit(main())

