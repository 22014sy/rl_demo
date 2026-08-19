"""
安全动作包装器
提供渐进式奇异点处理和动作安全约束，包含肌腱控制
"""

import numpy as np
import logging
from typing import Dict, Any

class SafeActionWrapper:
    """安全动作包装器"""
    
    def __init__(self, env):
        self.env = env
        self.logger = logging.getLogger(__name__)
        
        # 安全参数
        # P1: 动作前7维为每步关节增量(rad)，增量式位置控制；上限来自配置
        self.max_joint_delta = float(getattr(env.grasping_config, 'max_joint_delta', 0.05))
        self.singularity_threshold = 0.1  # 提高可操作度阈值，减少误报
        self.slowdown_factor = 0.3  # 奇异点附近降速因子
        self.max_tension = 100.0  # 最大允许张力(N)
        
        # 关节限制
        self.joint_limits = {
            'joint1': (-2.8973, 2.8973),
            'joint2': (-1.7628, 1.7628),
            'joint3': (-2.8973, 2.8973),
            'joint4': (-3.0718, -0.0698),
            'joint5': (-2.8973, 2.8973),
            'joint6': (-0.0175, 3.7525),
            'joint7': (-2.8973, 2.8973)
        }
        
        # 奇异点处理参数
        self.singularity_slowdown_factors = {
            'elbow': 0.2,      # 肘部奇异点：更慢
            'wrist': 0.3,      # 腕部奇异点：中等
            'shoulder': 0.4,   # 肩部奇异点：较快
            'jacobian': 0.1,   # 雅可比奇异：最慢
            'default': 0.3     # 默认降速
        }
    
    def apply(self, raw_action: np.ndarray, current_state: Dict[str, Any]) -> Dict[str, np.ndarray]:
        """
        将原始动作转换为安全的目标关节位置（增量式位置控制）

        P1 重构：前7维 = 每步关节增量(rad)，第8维 = 肌腱命令(0=闭合,1=张开)。
        执行器是位置伺服（ctrl=目标位置），因此输出 = 当前 qpos + 增量，
        而不是旧版错误的“当前速度±0.005 当作位置目标”（导致手臂几乎不可控）。
        """
        # 解析动作
        if isinstance(raw_action, np.ndarray):
            joint_deltas = np.asarray(raw_action[:7], dtype=np.float64)
            tendon_command = float(raw_action[7]) if len(raw_action) > 7 else 0.0
        else:
            action_dict = raw_action.copy()
            joint_deltas = np.asarray(action_dict.get('joint_commands', np.zeros(7)), dtype=np.float64)
            tendon_command = float(action_dict.get('tendon_command', 0.0))

        current_joint_pos = np.asarray(current_state.get('joint_positions', np.zeros(7)), dtype=np.float64)

        # 1. 关节增量限幅（保证增量式位置控制稳定）
        joint_deltas = np.clip(joint_deltas, -self.max_joint_delta, self.max_joint_delta)

        # 2. 奇异点降速（在增量上缩放，与 SingularityHandler 协调）
        if hasattr(self.env, 'singularity_handler'):
            is_singular, singularity_type, singularity_score = self.env.singularity_handler.detect_singularity(current_joint_pos)
            if is_singular:
                slowdown_factor = self._get_singularity_slowdown_factor(singularity_type, singularity_score)
                joint_deltas *= slowdown_factor
                if self.logger.isEnabledFor(logging.DEBUG):
                    self.logger.debug(f"奇异点处理: 类型={singularity_type}, 程度={singularity_score:.3f}, 降速因子={slowdown_factor:.2f}")

        # 3. 目标关节位置 = 当前 + 增量，裁剪到关节限位
        target_positions = current_joint_pos + joint_deltas
        for i, (joint_name, (low, high)) in enumerate(self.joint_limits.items()):
            target_positions[i] = np.clip(target_positions[i], low, high)

        # 4. 肌腱命令限幅 [0,1]（0=闭合, 1=张开）
        tendon_command = float(np.clip(tendon_command, 0.0, 1.0))

        return {
            'joint_commands': target_positions,   # ctrl[:7] = 目标位置
            'tendon_command': tendon_command,     # ctrl[7] = int(cmd*255)，0=闭合
            'gripper_command': tendon_command
        }
    
    def _get_singularity_slowdown_factor(self, singularity_type: str, singularity_score: float) -> float:
        """
        根据奇异点类型和程度获取降速因子
        
        Args:
            singularity_type: 奇异点类型
            singularity_score: 奇异程度 (0-1)
            
        Returns:
            slowdown_factor: 降速因子
        """
        # 基础降速因子
        base_factor = self.singularity_slowdown_factors.get(singularity_type, self.singularity_slowdown_factors['default'])
        
        # 根据奇异程度调整
        if singularity_score > 0.9:  # 严重奇异
            return base_factor * 0.5  # 进一步降速
        elif singularity_score > 0.7:  # 中等奇异
            return base_factor
        else:  # 轻微奇异
            return base_factor * 1.5  # 稍微放宽
        
        return base_factor
    
    def get_safe_initial_action(self) -> Dict[str, np.ndarray]:
        """获取安全的初始动作（零增量=保持当前位形，肌腱半开）"""
        return {
            'joint_commands': np.zeros(7, dtype=np.float32),
            'tendon_command': 0.5,
            'gripper_command': 0.5
        }

