"""
共享 IK / 运动工具（P2-1 提炼为单一来源，避免 scripts 与 environment 各复制一份）。

包含：四元数工具、由局部轴构造四元数、旋转向量->四元数（P3 姿态增量）、
DLS(阻尼最小二乘) IK 求解、关节插值到位。被 scripts 与环境层统一引用。
"""
import numpy as np
import mujoco
import math

# 指尖(衬垫)在 hand 局部坐标系 +z 方向的距离。Panda 实测 +z0.0584 + pad +z0.0445 = 0.1029；
# Task3 (UR5e+2F-85)：用 finger_reach() 按 pinch site 与 rq_base_mount 动态测量，此常量为 fallback。
FINGER_REACH = 0.1029


def finger_reach(model, data, pinch_site_name='pinch', base_body_name='rq_base_mount'):
    """2F-85 指尖(衬垫)相对末端根 body 的距离（动态测量；找不到时回退 FINGER_REACH）。"""
    site_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, pinch_site_name)
    body_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, base_body_name)
    if site_id >= 0 and body_id >= 0:
        return float(np.linalg.norm(data.site_xpos[site_id] - data.xpos[body_id]))
    return FINGER_REACH


def velocity_ik(model, data, body_id, v_ee, joint_ids, lam=0.05):
    """速度级 IK（Task3 核心）：末端速度 -> 关节速度（雅可比阻尼伪逆）。

    v_ee: 末端速度，长度 3（仅线速度，角速度置 0）或 6（[vx,vy,vz,wx,wy,wz]）。
    返回 dq (n,)：可直接作为 velocity actuator 的 ctrl 目标（速度伺服 f=kp*(ctrl-qvel)）。

    纯函数：在临时副本上构雅可比，不改动主 data。
    相比 solve_ik（位置级 DLS 迭代 500 次）：单步矩阵运算，快且语义正确（≈100% 实现率）。
    """
    scratch = mujoco.MjData(model)
    scratch.qpos[:] = data.qpos[:]
    scratch.qvel[:] = data.qvel[:]
    mujoco.mj_forward(model, scratch)

    v = np.asarray(v_ee, dtype=float).ravel()
    if v.size < 6:
        v = np.concatenate([v, np.zeros(6 - v.size)])
    v = v[:6]

    jacp = np.zeros((3, model.nv))
    jacr = np.zeros((3, model.nv))
    mujoco.mj_jacBody(model, scratch, jacp, jacr, body_id)
    J = np.vstack([jacp[:, joint_ids], jacr[:, joint_ids]])  # (6, n)

    A = J @ J.T + lam * lam * np.eye(6)
    dq = J.T @ np.linalg.solve(A, v)  # 阻尼伪逆 dq = Jᵀ(JJᵀ+λ²I)⁻¹v
    return dq


def axis_angle_to_quat(axis_angle):
    """旋转向量(轴 × 角, 弧度) -> 四元数 (w,x,y,z)。

    P3 用于把动作里的姿态增量 [dax,day,daz] 当作旋转向量转成增量四元数，
    再左乘到当前朝向：target_quat = dq ⊗ current_quat。
    """
    aa = np.asarray(axis_angle, dtype=float).ravel()
    ang = float(np.linalg.norm(aa))
    if ang < 1e-12:
        return np.array([1.0, 0.0, 0.0, 0.0])
    axis = aa / ang
    s, c = math.sin(ang / 2.0), math.cos(ang / 2.0)
    return np.array([c, axis[0] * s, axis[1] * s, axis[2] * s])


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


def solve_ik(model, data, body_id, target_pos, target_quat, joint_ids,
             iters=500, tol=3e-4, lam=0.05, lr=0.6, stall_patience=40):
    """DLS IK：把 body_id 移到 target_pos / target_quat，返回 (q, 位置误差)。

    纯函数：在临时副本上求解，不改动主 data —— 调用方在拿到 q 后自行写入 ctrl。

    P3.1 性能修复：原实现目标不可达/接近奇异时会把 iters 次迭代跑满
    （5000 次 mj_forward+mj_jacBody+6×6 解方程，实测 ≈0.84s/步 → 仿真卡顿）。
    现在：
      - iters 默认 5000 → 500（正常收敛实测只需几十次迭代，2~5ms）；
      - 新增停滞提前退出：连续 stall_patience 次总误差不再下降即 break；
      - 返回历史最优解 best_q（及其位置误差），避免返回末次更差的迭代解。
    """
    scratch = mujoco.MjData(model)
    scratch.qpos[:] = data.qpos[:]
    scratch.qvel[:] = data.qvel[:]
    scratch.ctrl[:] = data.ctrl[:]
    mujoco.mj_forward(model, scratch)

    jacp = np.zeros((3, model.nv))
    jacr = np.zeros((3, model.nv))
    q = scratch.qpos[joint_ids].copy()
    best_q = q.copy()
    best_err = np.inf
    stall = 0
    for _ in range(iters):
        scratch.qpos[joint_ids] = q
        mujoco.mj_forward(model, scratch)

        pos_err = target_pos - scratch.xpos[body_id]
        q_rel = quat_mul(target_quat, quat_inv(scratch.xquat[body_id]))
        rot_err = quat_vec(q_rel)
        err = np.concatenate([pos_err, rot_err])
        err_norm = float(np.linalg.norm(err))

        if np.linalg.norm(pos_err) < tol and np.linalg.norm(rot_err) < 1e-3:
            best_q = q.copy()
            break

        if err_norm < best_err:
            best_err = err_norm
            best_q = q.copy()
            stall = 0
        else:
            stall += 1
            if stall >= stall_patience:
                break

        mujoco.mj_jacBody(model, scratch, jacp, jacr, body_id)
        J = np.vstack([jacp[:, joint_ids], jacr[:, joint_ids]])  # (6, n)
        A = J @ J.T + lam * lam * np.eye(6)
        dq = lr * (J.T @ np.linalg.solve(A, err))

        q = q + dq
        for k, jid in enumerate(joint_ids):
            lo, hi = model.jnt_range[jid]
            q[k] = np.clip(q[k], lo, hi)

    scratch.qpos[joint_ids] = best_q
    mujoco.mj_forward(model, scratch)
    return best_q, float(np.linalg.norm(target_pos - scratch.xpos[body_id]))


def move_to_q(env, arm_joints, q_target, n_steps):
    """把关节位置伺服目标从当前值线性插值到 q_target 并逐步推进仿真（与 step() 同接口）"""
    q0 = env.data.qpos[arm_joints].copy()
    for k in range(1, n_steps + 1):
        alpha = k / n_steps
        env.data.ctrl[arm_joints] = q0 + alpha * (q_target - q0)
        mujoco.mj_step(env.model, env.data)
