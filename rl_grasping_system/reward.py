"""奖励函数（P3 保持逻辑不变，仅成功定义随环境重构）。

P3 起抓取成功 = 手指被物体挡住合不上（见 environment._update_gripper_state），`grasp_info['is_grasped'] == grasp_info['grasp_success'] == (phase == 'closed')`，
奖励结构不变：距离/方向势能塑形 + 接触 + grasp_reward + completion_reward + 时间惩罚。
"""

import numpy as np


def quat_rotate_world_z(q):
    """将末端坐标系 Z 轴旋转到世界系的分量（即末端 Z 轴在世界系中的坐标）"""
    w, x, y, z = q
    return np.array([
        2.0 * (x * z + w * y),
        2.0 * (y * z - w * x),
        w * w - x * x - y * y + z * z,
    ])



# ---------------------------------------------------------------- 四元数工具
# 约定与 scripts/verify_grasp_feasibility.py 一致：q = (w, x, y, z)，Hamilton 积
def quat_mul(q1, q2):
    """四元数乘法，顺序 q1 * q2"""
    w1, x1, y1, z1 = q1
    w2, x2, y2, z2 = q2
    return np.array([
        w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2,
        w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
        w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2,
        w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2,
    ])


def quat_inv(q):
    """四元数共轭（单位四元数即逆）"""
    return np.array([q[0], -q[1], -q[2], -q[3]])


def quat_normalize(q):
    """归一化四元数（防御性；来自 state 的四元数应为单位四元数）"""
    q = np.asarray(q, dtype=float)
    n = float(np.linalg.norm(q))
    return q / n if n > 0.0 else np.array([1.0, 0.0, 0.0, 0.0])


# 顶抓"从上方接近"旋转：绕世界X轴180° → 把 hand+Z(手指指向) 旋到世界 −Z(朝下)。
# 目标抓取姿态 = 物体朝向(target_orientation) ⊗ 该接近旋转；
# 当前方块 target_orientation=identity，故目标即 (0,1,0,0)。
Q_APPROACH_DOWN = np.array([0.0, 1.0, 0.0, 0.0])


def orientation_align(q_hand, q_target):
    """手与目标抓取姿态的四元数对齐度 cos(θ) ∈ [−1,1]（θ=相对转角，含绕手指轴 roll）。

    dot=|q_hand·q_target|=|cos(θ/2)|（二者为单位四元数），双角公式 cos(θ)=2·dot²−1；
    dot² 消除四元数双重覆盖（q 与 −q 表示同一姿态）。
    """
    qh = quat_normalize(q_hand)
    qt = quat_normalize(q_target)
    dot = abs(float(np.dot(qh, qt)))
    return 2.0 * dot * dot - 1.0


# ------------------------------------------------------------------ 奖励分量
# calculate_reward 只对以下键求和；orient_align / z_axis_z / potential 是纯诊断键，
# 放在返回值 b['diag'] 里，**绝不进入奖励**（P1.3 修复：此前 sum(values()) 把
# orient_align、z_axis_z 也算进奖励，而 z_axis_z=“手指朝上+1/朝下−1”，等于把 P1.2
# 修复掉的反向信号以系数 1 重新注入奖励——这正是方向项修了却不见效的原因之一）。
REWARD_KEYS = ('r_dist_xy', 'r_dist_z', 'r_orient', 'r_contact', 'r_grasp', 'r_success', 'r_step')


def _anchor_boost(w, d, d_target):
    """近距离锚定势能：d < d_target 时线性渐增（d=0 处 0，d=d_target 处 w，之外封顶 w）。

    2026-08-22 混合方案：在主势能上叠加，使近距离斜率变陡
    （斜率 = w/scale + w/d_target，如 xy ×3.5、z ×2.25），每靠近一步的塑形收益
    γ·slope·Δd 显著超过 step_penalty，打破"A 训练停在够近但不精调"的收益权衡。
    注意：必须单调（d 减 → Φ 减），d=0 处 Φ=0（目标处势能最小）——
    曾误用 max(0,1−d/d_target) 使近距离非单调（靠近反被惩罚），已修正。
    仍为势能差形式，累计有界、不改变最优策略。
    """
    if d_target <= 0.0:
        return 0.0
    return w * min(d / d_target, 1.0)


def _dist_cost_xy(ee, target, cfg):
    """平面(X/Y)距离代价势能（P1.1 去饱和线性项 + 近距离锚定，低=好）。"""
    d_xy = float(np.linalg.norm(ee[:2] - target[:2]))
    base = cfg.w_xy * min(d_xy, cfg.xy_cap) / cfg.xy_scale
    return base + _anchor_boost(cfg.anchor_w_xy, d_xy, cfg.anchor_dist_xy)


def _dist_cost_z(ee, target, cfg):
    """高度(Z)距离代价势能（低=好，含近距离锚定）。"""
    dz = float(abs(ee[2] - target[2]))
    base = cfg.w_height * min(dz, cfg.height_cap) / cfg.height_scale
    return base + _anchor_boost(cfg.anchor_w_z, dz, cfg.anchor_dist_z)


def _orient_gain(state, cfg):
    """方向收益势能（手指越朝下越大，全程连续无死区）+ 诊断量。

    P2-1 修复：旧实现 `gain = w_orient * max(0, align)` 存在**死区**——
    当夹爪与“手指朝下”目标的总角偏差 θ > 90°（align = cos(θ) < 0）时，
    方向收益被 max(0,·) 截为 0，势能与塑形奖励在该区域恒为 0、无梯度。
    策略于是只学“到位”不学“转向”，表现为可视化中夹爪水平/朝上抓取。

    改为**手指朝下投影** `gain = w_orient * (-z_axis_z)`：
      z_axis_z = 手指方向(hand 局部+Z)在世界系 Z 分量 ∈ [−1,1]
      朝下=−1 → gain=+w_orient；水平=0 → 0；朝上=+1 → −w_orient。
    全程有梯度、无死区，且不惩罚绕手指轴的 roll（对称夹爪不需要）。

    Returns: (orient_gain, align, z_axis_z)
        orient_gain = w_orient * (−z_axis_z)  （进入势能，负号表示“越朝下越接近目标”）
        align        = cos(θ)，与目标抓取姿态的完整四元数对齐(含roll)，仅供诊断
        z_axis_z     = 手指方向在世界系Z分量（诊断；朝下 <0）
    """
    q = np.asarray(state.get('ee_orientation'), dtype=float)
    q_obj = np.asarray(state.get('target_orientation'), dtype=float)
    if q.size < 4:
        return 0.0, 0.0, 0.0
    q_target = quat_mul(q_obj, Q_APPROACH_DOWN) if q_obj.size >= 4 else Q_APPROACH_DOWN
    align = orientation_align(q[:4], q_target)
    z_axis_z = float(quat_rotate_world_z(q[:4])[2])
    gain = cfg.w_orient * (-z_axis_z)  # 无死区：水平=0，朝上为负（惩罚），朝下为正
    return gain, align, z_axis_z


def reward_breakdown(state, prev_state, grasp_info, reward_config=None):
    """奖励各分量明细（诊断/回归用），calculate_reward 只对 REWARD_KEYS 求和。

    P1.3 起位置/方向项改为**势能塑形（potential-based shaping）**：
        势能      Φ(s) = w_xy·min(d_xy,xy_cap)/xy_scale + w_height·min(dz,height_cap)/height_scale
                        − w_orient·max(0, align)          （低=好）
        单步塑形  r_shape = Φ(s_prev) − γ·Φ(s)，γ=shaping_gamma

    好处：单步有界（典型 ±2），episode 总奖励不再随 max_steps 膨胀到 −800，
    成功/完成奖励不再被距离惩罚淹没；回报与路径解耦，势能差可望远镜式相消。

    prev_state 为 None 时（verify/diag 单状态诊断）返回该状态的“势能价值”
    −Φ(s)（越接近目标越大，方向项保持直给便于核对），不用于训练。
    """
    if reward_config is None:
        from config import RewardConfig
        reward_config = RewardConfig()
    gamma = getattr(reward_config, 'shaping_gamma', 0.99)

    ee = np.asarray(state['ee_position'], dtype=float)
    obj = np.asarray(state['target_position'], dtype=float)
    target = obj + np.array([0.0, 0.0, reward_config.pre_grasp_offset_z])
    c_xy = _dist_cost_xy(ee, target, reward_config)
    c_z = _dist_cost_z(ee, target, reward_config)
    o_gain, align, z_axis_z = _orient_gain(state, reward_config)

    if prev_state is not None:
        # 训练路径：势能差塑形（按分量拆分，便于诊断）
        pe = np.asarray(prev_state['ee_position'], dtype=float)
        po = np.asarray(prev_state['target_position'], dtype=float)
        pt = po + np.array([0.0, 0.0, reward_config.pre_grasp_offset_z])
        p_xy = _dist_cost_xy(pe, pt, reward_config)
        p_z = _dist_cost_z(pe, pt, reward_config)
        p_o, _, _ = _orient_gain(prev_state, reward_config)
        parts = {
            'r_dist_xy': p_xy - gamma * c_xy,
            'r_dist_z': p_z - gamma * c_z,
            'r_orient': -p_o + gamma * o_gain,
        }
        potential = c_xy + c_z - o_gain
    else:
        # 单状态诊断：返回状态势能价值 −Φ(s)（越接近目标越大）
        parts = {
            'r_dist_xy': -c_xy,
            'r_dist_z': -c_z,
            'r_orient': o_gain,
        }
        potential = c_xy + c_z - o_gain

    # 接触 / 抓取 / 完成事件
    contact_force = float(grasp_info.get('contact_force', 0.0))
    rising = grasp_info.get('grasp_success_rising', None)
    lift_active = bool(grasp_info.get('lift_active', False))
    if rising is None:
        # 诊断/单状态调用：无上升沿标志 -> 电平式（每次调用独立，保持 P3 行为）
        parts['r_contact'] = reward_config.w_contact if contact_force >= reward_config.contact_force_threshold else 0.0
        parts['r_grasp'] = reward_config.grasp_reward if grasp_info.get('is_grasped', False) else 0.0
        parts['r_success'] = reward_config.completion_reward if grasp_info.get('grasp_success', False) else 0.0
    elif lift_active and not rising:
        # P4: 抬升阶段——任务已交给自动状态机，事件奖励不再重复发放；
        # 位置/方向塑形冻结（抬升由状态机执行，不该因"远离 pre-grasp 目标"被惩罚）
        parts['r_contact'] = 0.0
        parts['r_grasp'] = 0.0
        parts['r_success'] = 0.0
        if prev_state is not None:
            parts['r_dist_xy'] = 0.0
            parts['r_dist_z'] = 0.0
            parts['r_orient'] = 0.0
    else:
        # 训练普通步：接触为电平式（接近阶段鼓励碰到物体）；
        # 抓取/成功只在 grasp_success 上升沿一次性发放（成功当步 rising=True）
        parts['r_contact'] = reward_config.w_contact if contact_force >= reward_config.contact_force_threshold else 0.0
        parts['r_grasp'] = reward_config.grasp_reward if rising else 0.0
        parts['r_success'] = reward_config.completion_reward if rising else 0.0
    parts['r_step'] = -reward_config.step_penalty

    # 纯诊断键（不进入 calculate_reward 的和）
    parts['diag'] = {
        'orient_align': align,
        'z_axis_z': z_axis_z,
        'potential': potential,
    }
    return parts


def calculate_reward(state, prev_state, grasp_info, reward_config=None):
    """P1.3 奖励函数：势能塑形 + 目标抓取姿态四元数对齐。

    仅对 REWARD_KEYS 求和（诊断键 orient_align/z_axis_z/potential 在 b['diag']
    中，不参与奖励——P1.3 修复了此前 sum(values()) 的“诊断键泄漏”）。

    演进记录：
      P1.1  距离项去掉 min(d/scale,1) 饱和 → 线性势场，恢复远离目标时的梯度；
      P1.2  方向项由 `z_axis[2]`（符号反）改为四元数对齐 cos(θ)，方向梯度正确；
      P1.3  距离/方向项改为势能塑形 Φ(s_prev) − γ·Φ(s)：单步有界、回报不随
            episode 长度膨胀（此前 300 步累计 ~−800，成功 +16 被淹没，见
            docs/项目设计与结构审查.md）。
    """
    parts = reward_breakdown(state, prev_state, grasp_info, reward_config)
    return float(sum(parts[k] for k in REWARD_KEYS))