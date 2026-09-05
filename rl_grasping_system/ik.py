"""
共享 IK / 运动工具（P2-1 提炼为单一来源，避免 scripts 与 environment 各复制一份）。

包含：四元数工具、由局部轴构造四元数、旋转向量->四元数（P3 姿态增量）、
DLS(阻尼最小二乘) IK 求解、关节插值到位。被 scripts 与环境层统一引用。
"""
import numpy as np  # 导入 NumPy，用于向量、矩阵和数值计算。
import mujoco  # 导入 MuJoCo 仿真库，用于读取模型和做雅可比/前向动力学计算。
import math  # 导入 Python 标准数学库，用于三角函数和平方根等运算。

# 指尖(衬垫)在 hand 局部坐标系 +z 方向的距离。Panda 实测 +z0.0584 + pad +z0.0445 = 0.1029；
# Task3 (UR5e+2F-85)：用 finger_reach() 按 pinch site 与 rq_base_mount 动态测量，此常量为 fallback。
# 这里的 `site` 和 `body` 分别代表：
# - body：MuJoCo 中的刚体/连杆对象，表示真实存在的机械结构部件，如手爪基座、末端壳体、连杆等；
# - site：MuJoCo 中的“标记点”，通常不参与动力学，仅用于记录位置，例如抓取点、夹持点、视觉标定点等。
# 在本函数中，`pinch` 是抓取/夹持点的 site，而 `rq_base_mount` 是末端安装底座的 body。
FINGER_REACH = 0.1029  # 默认抓取点到指尖的长度，用于没有动态测量时的回退值。


def finger_reach(model, data, pinch_site_name='pinch', base_body_name='rq_base_mount'):
    # 定义函数：动态测量 2F-85 指尖在末端根 body 相对距离。
    # 这个函数的核心思想是：
    #   1）找到抓取点 site（例如 pinch）
    #   2）找到参考刚体 body（例如 rq_base_mount）
    #   3）计算它们在世界坐标中的距离
    # 因为抓取点一般不等于 body 的原点，所以用“site 位置 - body 原点位置”的距离来代表指尖到基座的实际长度。
    """2F-85 指尖(衬垫)相对末端根 body 的距离（动态测量；找不到时回退 FINGER_REACH）。"""  # 说明函数行为：如果能找到抓取位点与基座 body，则返回实际距离，否则用常量。
    site_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, pinch_site_name)  # 通过 MuJoCo 名称查找抓取点/夹持点 site 的 ID。
    body_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, base_body_name)  # 通过 MuJoCo 名称查找末端根 body 的 ID。
    if site_id >= 0 and body_id >= 0:  # 如果两者都存在，则进行动态测量。
        # data.site_xpos[site_id]：这个 site 在世界坐标中的位置；点
        # data.xpos[body_id]：这个 body 的参考位置（通常是 body 原点）在世界坐标中的位置；坐标系原点
        # 两者之差就是“从 body 到 site 的向量”，再取范数就是长度。
        return float(np.linalg.norm(data.site_xpos[site_id] - data.xpos[body_id]))  # 返回夹持点在世界坐标中的位置与基座位置之差的欧氏距离。
    return FINGER_REACH  # 若找不到目标对象，则回退到默认指尖长度常量。


def velocity_ik(model, data, body_id, v_ee, joint_ids, lam=0.05):
    # 定义速度级 IK 函数：把末端线速度/角速度映射为关节速度。
    """速度级 IK（Task3 核心）：末端速度 -> 关节速度（雅可比阻尼伪逆）。

    v_ee: 末端速度，长度 3（仅线速度，角速度置 0）或 6（[vx,vy,vz,wx,wy,wz]）。
    返回 dq (n,)：可直接作为 velocity actuator 的 ctrl 目标（速度伺服 f=kp*(ctrl-qvel)）仅P控制？。

    纯函数：在临时副本上构雅可比，不改动主 data。
    相比 solve_ik（位置级 DLS 迭代 500 次）：单步矩阵运算，快且语义正确（≈100% 实现率）。
    """  # 说明该函数不修改原始仿真状态，只在临时副本上计算，适合高频控制。
    scratch = mujoco.MjData(model)  # 创建一个独立的 MuJoCo 数据副本，避免覆盖真实状态。
    scratch.qpos[:] = data.qpos[:]  # 复制当前关节位置到临时副本。
    scratch.qvel[:] = data.qvel[:]  # 复制当前关节速度到临时副本。
    mujoco.mj_forward(model, scratch)  # 在副本上更新 kinematics 和 body transforms。

    v = np.asarray(v_ee, dtype=float).ravel()  # 把输入速度转换成一维浮点数组。
    if v.size < 6:  # 如果速度长度不足 6，则补齐角速度为 0。
        v = np.concatenate([v, np.zeros(6 - v.size)])  # 在尾部填充 0，保证 6 维表示。
    v = v[:6]  # 截断到长度 6，保证只保留六维末端速度信息。

    jacp = np.zeros((3, model.nv))  # 线速度雅可比矩阵，大小为 3 × nv。
    jacr = np.zeros((3, model.nv))  # 角速度雅可比矩阵，大小为 3 × nv。
    mujoco.mj_jacBody(model, scratch, jacp, jacr, body_id)  # 计算给定 body 的线性和角速度雅可比。
    J = np.vstack([jacp[:, joint_ids], jacr[:, joint_ids]])  # (6, n)  # 只取与控制关节对应的列，形成末端速度雅可比矩阵。

    A = J @ J.T + lam * lam * np.eye(6)  # 形成阻尼最小二乘矩阵，避免奇异时求逆不稳定。
    dq = J.T @ np.linalg.solve(A, v)  # 阻尼伪逆 dq = Jᵀ(JJᵀ+λ²I)⁻¹v  # 用阻尼伪逆求解关节速度增量。
    return dq  # 返回关节速度目标，可直接作为 velocity actuator 的控制输入。


def axis_angle_to_quat(axis_angle):
    # 定义函数：将旋转向量（轴 * 角度）转为四元数。
    """旋转向量(轴 × 角, 弧度) -> 四元数 (w,x,y,z)。

    P3 用于把动作里的姿态增量 [dax,day,daz] 当作旋转向量转成增量四元数，
    再左乘到当前朝向：target_quat = dq ⊗ current_quat。
    """  # 说明：此函数用于将姿态增量表示为旋转向量，并转换为四元数以便组合。
    aa = np.asarray(axis_angle, dtype=float).ravel()  # 将输入统一为 1 维浮点数组。
    ang = float(np.linalg.norm(aa))  # 计算旋转向量的角度大小（弧度）。
    if ang < 1e-12:  # 若角度非常小，则视为零旋转，返回单位四元数。
        return np.array([1.0, 0.0, 0.0, 0.0])  # 返回 [w, x, y, z] = [1, 0, 0, 0]。
    axis = aa / ang  # 归一化旋转轴方向。
    s, c = math.sin(ang / 2.0), math.cos(ang / 2.0)  # 使用半角公式计算四元数分量。
    return np.array([c, axis[0] * s, axis[1] * s, axis[2] * s])  # 返回四元数 [w, x, y, z]。


def quat_mul(q1, q2):
    # 定义四元数乘法：按 q1 * q2 顺序组合两个旋转。
    """四元数乘法，顺序 q1 * q2（q = w,x,y,z）"""  # 说明乘法顺序与旋转组合顺序一致。
    w1, x1, y1, z1 = q1  # 提取第一个四元数参数。
    w2, x2, y2, z2 = q2  # 提取第二个四元数参数。
    return np.array([  # 返回两个四元数相乘后的新四元数。
        w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2,  # 计算新四元数的 w 分量。
        w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,  # 计算新四元数的 x 分量。
        w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2,  # 计算新四元数的 y 分量。
        w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2,  # 计算新四元数的 z 分量。
    ])  # 结束四元数乘法。


def quat_inv(q):
    # 定义四元数求逆：对四元数的虚部取负，保持实部不变。
    return np.array([q[0], -q[1], -q[2], -q[3]])  # 返回四元数的共轭，由于单位四元数，等价于逆。


def quat_vec(q):
    # 定义函数：把单位四元数转换为旋转向量，适用于小角度近似。
    """四元数 -> 旋转向量（2*vec，小角度近似）"""  # 说明：在小角度情况下，四元数虚部近似为旋转向量的一半。
    return 2.0 * q[1:]  # 用 2 * vec 近似得到旋转向量（x,y,z）。


def rotmat_to_quat(R):
    # 定义函数：从旋转矩阵转换成四元数，以便在 IK 中比较姿态误差。
    """旋转矩阵 -> 四元数 (w,x,y,z)，Shepperd 方法"""  # 说明使用 Shepperd 方法以保证稳定性和数值鲁棒性。
    tr = np.trace(R)  # 取旋转矩阵的迹，用于判断最大分量对应的四元数区域。
    if tr > 0.0:  # 当迹大于 0 时，使用第一种分支计算四元数。
        S = np.sqrt(tr + 1.0) * 2.0  # 计算中间变量。
        qw, qx, qy, qz = (0.25 * S,
                          (R[2, 1] - R[1, 2]) / S,
                          (R[0, 2] - R[2, 0]) / S,
                          (R[1, 0] - R[0, 1]) / S)  # 根据 Shepperd 公式计算四元数分量。
    elif R[0, 0] > R[1, 1] and R[0, 0] > R[2, 2]:  # 若 x 轴分量最大，使用第二分支。
        S = np.sqrt(1.0 + R[0, 0] - R[1, 1] - R[2, 2]) * 2.0  # 计算中间项。
        qw, qx, qy, qz = ((R[2, 1] - R[1, 2]) / S,
                          0.25 * S,
                          (R[0, 1] + R[1, 0]) / S,
                          (R[0, 2] + R[2, 0]) / S)  # 计算对应四元数分量。
    elif R[1, 1] > R[2, 2]:  # 若 y 轴分量最大，使用第三分支。
        S = np.sqrt(1.0 + R[1, 1] - R[0, 0] - R[2, 2]) * 2.0  # 计算中间项。
        qw, qx, qy, qz = ((R[0, 2] - R[2, 0]) / S,
                          (R[0, 1] + R[1, 0]) / S,
                          0.25 * S,
                          (R[1, 2] + R[2, 1]) / S)  # 计算对应四元数分量。
    else:  # 若 z 轴分量最大，使用第四分支。
        S = np.sqrt(1.0 + R[2, 2] - R[0, 0] - R[1, 1]) * 2.0  # 计算中间项。
        qw, qx, qy, qz = ((R[1, 0] - R[0, 1]) / S,
                          (R[0, 2] + R[2, 0]) / S,
                          (R[1, 2] + R[2, 1]) / S,
                          0.25 * S)  # 计算对应四元数分量。
    q = np.array([qw, qx, qy, qz])  # 将分量组成四元数向量。
    return q / np.linalg.norm(q)  # 归一化四元数，保证表示为单位四元数。


def frame_to_quat(z_local_world, y_local_world):
    # 定义函数：由局部 z 轴和 y 轴在世界坐标中的方向，重建末端朝向四元数。
    """由局部 z 轴、y 轴在世界的朝向构造四元数（R 的列 = 局部轴在世界坐标）"""  # 说明：构造旋转矩阵时，列向量分别为局部 x,y,z 轴在世界坐标中的方向。
    zl = np.asarray(z_local_world, dtype=float)  # 把局部 z 轴向量转换为 NumPy 浮点数组。
    zl /= np.linalg.norm(zl)  # 归一化局部 z 轴，保证长度为 1。
    yl = np.asarray(y_local_world, dtype=float)  # 把局部 y 轴向量转换为 NumPy 浮点数组。
    yl -= np.dot(yl, zl) * zl  # 正交化：去除 y 轴中与 z 轴重合的分量。
    yl /= np.linalg.norm(yl)  # 归一化修正后的 y 轴。
    xl = np.cross(yl, zl)  # 由 y × z 计算得到 x 轴，确保三轴形成右手坐标系。
    R = np.column_stack([xl, yl, zl])  # 将三轴按列堆叠成旋转矩阵，列为局部轴在世界坐标中的方向。
    return rotmat_to_quat(R)  # 把旋转矩阵转换成四元数并返回。


def solve_ik(model, data, body_id, target_pos, target_quat, joint_ids,
             iters=500, tol=3e-4, lam=0.05, lr=0.6, stall_patience=40):
    # 定义 DLS IK 求解器：把给定 body 位置和姿态移动到目标位置与姿态。
    """DLS IK：把 body_id 移到 target_pos / target_quat，返回 (q, 位置误差)。

    纯函数：在临时副本上求解，不改动主 data —— 调用方在拿到 q 后自行写入 ctrl。

    P3.1 性能修复：原实现目标不可达/接近奇异时会把 iters 次迭代跑满
    （5000 次 mj_forward+mj_jacBody+6×6 解方程，实测 ≈0.84s/步 → 仿真卡顿）。
    现在：
      - iters 默认 5000 → 500（正常收敛实测只需几十次迭代，2~5ms）；
      - 新增停滞提前退出：连续 stall_patience 次总误差不再下降即 break；
      - 返回历史最优解 best_q（及其位置误差），避免返回末次更差的迭代解。
    """  # 说明：这一版在接近奇异或无法到达目标时能提前停止，减少计算负担。
    scratch = mujoco.MjData(model)  # 创建独立副本，避免破坏真实仿真状态。
    scratch.qpos[:] = data.qpos[:]  # 复制当前关节位置到 scratch。
    scratch.qvel[:] = data.qvel[:]  # 复制当前关节速度到 scratch。
    scratch.ctrl[:] = data.ctrl[:]  # 复制当前控制输入到 scratch，保证状态一致。
    mujoco.mj_forward(model, scratch)  # 计算 scratch 的前向动力学和 body 变换。

    jacp = np.zeros((3, model.nv))  # 线速度雅可比矩阵的临时容器。
    jacr = np.zeros((3, model.nv))  # 角速度雅可比矩阵的临时容器。
    q = scratch.qpos[joint_ids].copy()  # 当前关节角度作为 IK 的起点。
    best_q = q.copy()  # 初始化当前最佳解，避免返回不佳结果。
    best_err = np.inf  # 初始化最优误差为无穷大，便于比较。
    stall = 0  # 记录连续误差不下降的轮数，用于提前退出。
    for _ in range(iters):  # 迭代求解直到达到最大次数或提前退出。
        scratch.qpos[joint_ids] = q  # 把当前关节位置写回副本中。
        mujoco.mj_forward(model, scratch)  # 更新当前姿态和位姿。

        pos_err = target_pos - scratch.xpos[body_id]  # 计算位置误差 = 目标位置 - 当前 body 位置。
        q_rel = quat_mul(target_quat, quat_inv(scratch.xquat[body_id]))  # 计算目标姿态相对于当前姿态的差分四元数。
        rot_err = quat_vec(q_rel)  # 把姿态误差转换成旋转向量表示。
        err = np.concatenate([pos_err, rot_err])  # 拼接 6 维的综合误差向量 [位置, 旋转]。
        err_norm = float(np.linalg.norm(err))  # 计算总误差范数，用于停滞判定。

        if np.linalg.norm(pos_err) < tol and np.linalg.norm(rot_err) < 1e-3:  # 当位置和姿态误差都足够小则视为收敛。
            best_q = q.copy()  # 记录当前收敛解。
            break  # 提前结束迭代。

        if err_norm < best_err:  # 若当前误差优于历史最优，则更新最佳解。
            best_err = err_norm  # 更新最优误差值。
            best_q = q.copy()  # 保存当前最佳关节配置。
            stall = 0  # 复位停滞计数器。
        else:  # 若误差没有继续减少，则计数停滞。
            stall += 1  # 增加连续没有改进的迭代次数。
            if stall >= stall_patience:  # 若超过允许停滞次数，则停止迭代。
                break  # 避免无意义继续计算。

        mujoco.mj_jacBody(model, scratch, jacp, jacr, body_id)  # 计算当前 body 的雅可比矩阵。
        J = np.vstack([jacp[:, joint_ids], jacr[:, joint_ids]])  # (6, n)  # 提取控制关节对应的雅可比列，形成 6×n 线性系统。
        A = J @ J.T + lam * lam * np.eye(6)  # 构造阻尼最小二乘矩阵，用于稳定求解。
        dq = lr * (J.T @ np.linalg.solve(A, err))  # 计算增量关节角度：误差映射到关节空间，并乘以学习率。

        q = q + dq  # 更新关节角度为当前值加增量。
        for k, jid in enumerate(joint_ids):  # 对每个受控关节做角度范围裁剪。
            lo, hi = model.jnt_range[jid]  # 读取该关节的角度上下限。
            q[k] = np.clip(q[k], lo, hi)  # 将关节角度限制在合法范围内。

    scratch.qpos[joint_ids] = best_q  # 将最优关节解写回 scratch。
    mujoco.mj_forward(model, scratch)  # 重新更新 scratch 的位姿，以便返回误差。
    return best_q, float(np.linalg.norm(target_pos - scratch.xpos[body_id]))  # 返回最优关节角度和最终位置误差。


def move_to_q(env, arm_joints, q_target, n_steps):
    # 定义函数：将关节目标从当前值线性插值到 q_target，并逐步推进仿真。
    """把关节位置伺服目标从当前值线性插值到 q_target 并逐步推进仿真（与 step() 同接口）"""  # 说明：此函数以指定步数进行平滑插值，配合环境 step() 逐步更新控制量。
    q0 = env.data.qpos[arm_joints].copy()  # 记录当前关节位置作为起点。
    for k in range(1, n_steps + 1):  # 按步数进行插值循环。
        alpha = k / n_steps  # 计算当前插值比例，范围(0, 1]。
        env.data.ctrl[arm_joints] = q0 + alpha * (q_target - q0)  # 线性插值控制目标到目标关节值。
        mujoco.mj_step(env.model, env.data)  # 推进仿真一步，更新状态。
