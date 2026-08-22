"""
安全动作包装器（P3 重构：末端位姿增量 -> DLS IK -> 关节位置伺服目标）

动作空间：
  action_space_dim = 6 -> [dx, dy, dz, dax, day, daz]
      位置增量 (m) + 姿态增量 (rad，旋转向量)
  action_space_dim = 3 -> [dx, dy, dz]
      仅位置，姿态固定为"朝下接近"抓取姿态

流程：把末端位姿增量叠加到当前 EE 位姿，经 ik.solve_ik 反解为目标关节位形，
返回 'joint_commands'（写 ctrl[:7]，位置伺服）。IK 误差超过 ik_error_hold 时
保持当前位置不动，避免朝不可达目标乱动。

夹爪（ctrl[7]）不再经此包装器：由环境内部的"近距自动闭合状态机"驱动（P3）。
"""

import numpy as np
import logging

from ik import quat_mul, axis_angle_to_quat, solve_ik
from reward import Q_APPROACH_DOWN


class SafeActionWrapper:
    """安全动作包装器（末端位姿增量控制）"""

    def __init__(self, env):
        self.env = env
        self.logger = logging.getLogger(__name__)

        # P3 动作空间参数（来自 GraspingConfig）
        cfg = env.grasping_config
        self.action_space_dim = int(getattr(cfg, 'action_space_dim', 6))
        self.max_ee_delta = float(getattr(cfg, 'max_ee_delta', 0.02))
        self.max_orient_delta = float(getattr(cfg, 'max_orient_delta', 0.05))
        self.ik_error_hold = float(getattr(cfg, 'ik_error_hold', 0.02))

    def apply(self, raw_action: np.ndarray, current_state) -> dict:
        """
        把原始动作（末端位姿增量）转换为安全的目标关节位置（增量式末端控制）。

        P3 重构：RL 只输出末端位姿增量，不再直接控制关节/肌腱。
        """
        # 1. 解析并限幅动作
        if isinstance(raw_action, np.ndarray):
            a = np.asarray(raw_action, dtype=np.float64).ravel()
        else:
            a = np.asarray(raw_action.get('ee_delta', np.zeros(self.action_space_dim)),
                           dtype=np.float64).ravel()
        if a.size < self.action_space_dim:
            a = np.concatenate([a, np.zeros(self.action_space_dim - a.size)])

        pos_delta = np.clip(a[:3], -self.max_ee_delta, self.max_ee_delta)
        if self.action_space_dim >= 6:
            ori_delta = np.clip(a[3:6], -self.max_orient_delta, self.max_orient_delta)
        else:
            ori_delta = np.zeros(3)

        # 2. 当前 EE 位姿 -> 目标位姿
        cur_pos = np.asarray(current_state.get('ee_position', np.zeros(3)), dtype=np.float64)
        cur_quat = np.asarray(current_state.get('ee_orientation',
                                                np.array([1.0, 0.0, 0.0, 0.0])),
                              dtype=np.float64)
        target_pos = cur_pos + pos_delta

        if self.action_space_dim >= 6:
            # 姿态增量按旋转向量叠加到当前朝向（小角度、无万向节死锁）
            dq = axis_angle_to_quat(ori_delta)
            target_quat = quat_mul(dq, cur_quat)
            target_quat = target_quat / np.linalg.norm(target_quat)
        else:
            # 3 维模式：姿态固定为目标抓取姿态（物体朝向 ⊗ 朝下接近）
            q_obj = np.asarray(current_state.get('target_orientation',
                                                 np.array([1.0, 0.0, 0.0, 0.0])),
                               dtype=np.float64)
            target_quat = quat_mul(q_obj, Q_APPROACH_DOWN) if q_obj.size >= 4 else Q_APPROACH_DOWN

        # 3. DLS IK 反解目标关节位形（纯函数，不改动主 data）
        model, data = self.env.model, self.env.data
        hand_id = self.env.end_effector_id
        arm_joints = self.env.arm_joint_ids
        q_target, err = solve_ik(model, data, hand_id, target_pos, target_quat, arm_joints)

        # 4. IK 不可达 -> 保持当前位置（不朝不可达目标乱动）
        if err > self.ik_error_hold:
            q_target = data.qpos[arm_joints].copy()
            if self.logger.isEnabledFor(logging.DEBUG):
                self.logger.debug(f"IK 位置误差 {err:.4f} m > {self.ik_error_hold}，保持当前位置")

        return {'joint_commands': q_target}

    def get_safe_initial_action(self) -> dict:
        """获取安全的初始动作（零增量 = 保持当前末端位姿）"""
        return {'joint_commands': np.zeros(7, dtype=np.float32)}
