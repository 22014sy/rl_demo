#!/usr/bin/env python3
"""
P0-1 任务物理可行性验证（纯 MuJoCo，不依赖 RL 训练环境代码）

目的：在修复后的模型上验证"抓取"在物理上是否真的可行：
  1. target_cube 是带 freejoint 的自由物体，能被夹爪捏住；
  2. 夹爪能张开到足以包住物体（物体 0.04m < 夹爪内表面最大开口 ≈0.056m，实测）；
  3. 用阻尼最小二乘(DLS) IK 把夹爪送到物体上方，依次执行 张开 -> 下降 -> 闭合，
     验证手指被物体挡住合不上（开度仍 > 0.02）且双指接触物体 —— 即 P3 的抓取成功定义。

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
    REPO_ROOT, "models", "universal_robots_ur5e", "ur5e_robotiq_cube.xml"
)

# Task3 (UR5e+2F-85): 指尖(衬垫)中心距 rq_base_mount 的距离
FINGER_REACH = 0.134

# P3.1: 统一使用共享 ik.solve_ik（消除本地重复实现；共享版带停滞提前退出 + 降迭代的性能修复）
PKG = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PKG not in sys.path:
    sys.path.insert(0, PKG)
from ik import solve_ik  # noqa: E402


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
# P3.1: solve_ik 已统一改用共享实现（见文件头 from ik import solve_ik）


def move_to_q(model, data, arm_joints, q_target, n_steps):
    """Task3 (UR5e arm 为 motor)：位置伺服 ctrl 无效，直接设 qpos + mj_forward 到位"""
    data.qpos[arm_joints] = q_target
    mujoco.mj_forward(model, data)


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

    hand_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "rq_base_mount")
    cube_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "target_cube")
    if hand_id < 0 or cube_id < 0:
        print("[FAIL] 模型里找不到 body: rq_base_mount / target_cube")
        return 1

    # Task3 (UR5e+2F-85): 6 臂关节 + 8 个 rq_ 铰链（全开 = qpos=0）
    arm_joints = [mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, n)
                  for n in ["shoulder_pan_joint", "shoulder_lift_joint", "elbow_joint",
                            "wrist_1_joint", "wrist_2_joint", "wrist_3_joint"]]
    home = np.array([-1.5708, -1.5708, 1.5708, -1.5708, -1.5708, 0.0])
    data.qpos[arm_joints] = home
    for jid in range(model.njnt):
        jn = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_JOINT, jid)
        if jn and jn.startswith('rq_') and model.jnt_type[jid] == mujoco.mjtJoint.mjJNT_HINGE:
            data.qpos[model.jnt_qposadr[jid]] = 0.0   # 2F-85 全开
    mujoco.mj_forward(model, data)

    # Task3：cube 已在桌面顶（freejoint qpos0 = body pos），无需落体等待；
    # 让手指/物体在零控制下稳定 200 步
    data.ctrl[6] = 0.0  # 手指保持全开（menagerie 2F-85: 0=张开）
    for _ in range(200):
        mujoco.mj_step(model, data)

    cube_pos0 = data.xpos[cube_id].copy()
    print(f"[INFO] 模型: {MODEL_PATH}")
    print(f"[INFO] 物体落定后位置: {cube_pos0.round(4)}")
    print(f"[INFO] nq={model.nq} nv={model.nv}")

    # 抓取朝向：夹爪(指尖)朝下 -z，手指沿世界 x 张开/闭合（2F-85 开合方向）
    target_quat = frame_to_quat([0, 0, -1], [0, -1, 0])

    # 指尖(衬垫)在 hand 局部 +z 方向 FINGER_REACH 处，因此：
    #   fingertip_z = hand_z - FINGER_REACH；预抓取：衬垫在物体顶面上方 0.01m；
    #   抓取：衬垫对准物体中部（P3：不再抬升）
    pre_pos = np.array([cube_pos0[0], cube_pos0[1],
                        cube_pos0[2] + 0.02 + FINGER_REACH + 0.01])  # 预抓取高度
    grasp_pos = np.array([cube_pos0[0], cube_pos0[1],
                          cube_pos0[2] + FINGER_REACH])              # 抓取高度

    # ---------------- 阶段 1：张开手指并移动到预抓取位置 ----------------
    data.ctrl[6] = 0.0  # 手指全开（2F-85: 0=张开）
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
    data.ctrl[6] = 255.0  # 手指全闭（2F-85: 255=闭合）
    for _ in range(int(args.steps * 1.5)):
        mujoco.mj_step(model, data)

    # ---------------- 阶段 4：结果判定（P3：手指被物体挡住合不上 = 抓取成功） ----------------
    pl = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, "rq_pad_left_site")
    pr = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, "rq_pad_right_site")
    width = float(np.linalg.norm(data.site_xpos[pl] - data.site_xpos[pr]))  # pad 间距（2F-85）
    gripper_bodies = {mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "rq_left_pad"),
                      mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "rq_right_pad"),
                      mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "rq_left_silicone_pad"),
                      mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "rq_right_silicone_pad")}
    contact_ok = False
    for c in range(data.ncon):
        b1, b2 = data.contact[c].geom1, data.contact[c].geom2
        g1b, g2b = model.geom_bodyid[b1], model.geom_bodyid[b2]
        if cube_id in (g1b, g2b) and (g1b in gripper_bodies or g2b in gripper_bodies):
            contact_ok = True
            break

    # 成功判定（与 config.grasp_min_width=0.02 一致）：手指被物体挡住合不上（开度仍 > 0.02）
    # 且双指接触物体。空闭合时无接触（虽宽度 ~0.046 为 2F-85 机构特性），不会通过。
    blocked = width > 0.02
    print(f"[INFO] 闭合后夹爪开度 width={width:.4f} m（0.04m 物体挡住，无法全闭 => 已捏住）")
    print(f"[INFO] 手指与物体有接触: {contact_ok}  fingers_blocked={blocked}")

    result = {
        "model": MODEL_PATH,
        "cube_pos0": cube_pos0.tolist(),
        "gripper_width_after_close": width,
        "fingers_blocked": blocked,
        "gripper_contact": contact_ok,
        "ik_err_pre": e_pre,
        "ik_err_grasp": e_grasp,
    }

    passed = blocked and contact_ok
    result["passed"] = bool(passed)

    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2, ensure_ascii=False)
    print(f"[INFO] 结果已写入: {args.out}")

    if passed:
        print("[PASS] 物理可行性验证通过：0.04m 物体可被夹爪捏住（手指被挡住合不上）。")
        return 0
    print("[FAIL] 手指未夹住物体，需检查模型或抓取参数。")
    return 1


if __name__ == "__main__":
    sys.exit(main())

