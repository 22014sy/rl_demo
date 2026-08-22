"""
奇异点检测和处理模块（Task3：UR5e 6 关节版；原 Panda 7 关节版已替换）

检测：关节限位 + 已知奇异位形（肩奇异 shoulder_lift≈±π/2、肘伸直 elbow≈0、腕奇异 wrist_2≈0）。
处理：渐进移回 safe_config（home 附近）。速度模式下 velocity_ik 阻尼伪逆本身奇异鲁棒，
      奇异只触发告警 + 速度置零（防御）。
"""

import numpy as np
import mujoco
import logging
from typing import Tuple


class SingularityHandler:
    """UR5e 奇异点处理器（6 关节）"""

    def __init__(self):
        self.logger = logging.getLogger(__name__)

        # UR5e 关节限制 (rad)
        self.joint_limits = {
            'shoulder_pan': (-6.28319, 6.28319),
            'shoulder_lift': (-6.28319, 6.28319),
            'elbow': (-3.1415, 3.1415),
            'wrist_1': (-6.28319, 6.28319),
            'wrist_2': (-6.28319, 6.28319),
            'wrist_3': (-6.28319, 6.28319),
        }
        # 关节名 -> 数组索引
        self._idx = {'shoulder_pan': 0, 'shoulder_lift': 1, 'elbow': 2,
                     'wrist_1': 3, 'wrist_2': 4, 'wrist_3': 5}

        # 已知奇异位形（UR5e 运动学奇异；注意 shoulder_lift 单独取值不构成奇异，
        # 肩奇异需 elbow 伸直+wrist_2=0 组合，故不在此列——避免 home(-π/2) 误报）
        self.singularity_configs = [
            {'elbow': 0.0, 'tolerance': 0.25},        # 肘伸直
            {'wrist_2': 0.0, 'tolerance': 0.3},       # 腕奇异
        ]

        # 安全配置（home 附近：指尖朝下悬于 cube 上方，远离奇异、可达）
        self.safe_config = np.array([-1.5708, -1.5708, 1.5708, -1.5708, -1.5708, 0.0])

        self.warning_cooldown = 0
        self.last_warning_time = 0
        self.warning_interval = 30.0

    def detect_singularity(self, joint_positions) -> Tuple[bool, str, float]:
        if joint_positions is None or len(joint_positions) < 6:
            return False, "invalid", 0.0
        jp = np.asarray(joint_positions, dtype=float)

        # 关节限位
        for i, (name, (lo, hi)) in enumerate(self.joint_limits.items()):
            if jp[i] < lo or jp[i] > hi:
                return True, f"joint_limit_{name}", 1.0

        max_score, s_type = 0.0, "none"
        for cfg in self.singularity_configs:
            tol = cfg['tolerance']
            for name, target in cfg.items():
                if name == 'tolerance':
                    continue
                d = abs(jp[self._idx[name]] - target)
                if d < tol:
                    sc = 1.0 - d / tol
                    if sc > max_score:
                        max_score, s_type = sc, f"singularity_{name}"
        return max_score > 0.0, s_type, max_score

    def get_progressive_safe_action(self, current_config, step_size=0.05):
        cur = np.asarray(current_config, dtype=float)
        target = self.safe_config
        delta = target - cur
        step = np.clip(delta, -step_size, step_size)
        if np.linalg.norm(delta) < step_size:
            return target.copy()
        return cur + step

    def get_safe_config(self, current_config):
        return self.get_progressive_safe_action(current_config, step_size=0.05)

    def generate_safe_initial_config(self, rng=None):
        """UR5e 初始位形：围绕 home 小范围随机（home 时 pad 已悬于 cube 上方，利于学习）。
        并行改造(2026-08-22): spread 0.3→0.15（课程式：先学近端精确微调→对齐→闭合，
        后续训练稳定后可再放开；max_steps 已同步 300→200）。"""
        spread = 0.15
        if rng is None:
            rng = np.random
        config = self.safe_config + np.array([rng.uniform(-spread, spread) for _ in range(6)])
        is_singular, _, _ = self.detect_singularity(config)
        return self.safe_config.copy() if is_singular else config

    def check_singularity_recovery(self, joint_positions, previous_positions):
        was_singular, _, _ = self.detect_singularity(previous_positions)
        is_singular, _, _ = self.detect_singularity(joint_positions)
        return was_singular and not is_singular

    def detect_singularity_with_warning_control(self, joint_positions, current_time=None):
        import time
        if current_time is None:
            current_time = time.time()
        is_singular, s_type, score = self.detect_singularity(joint_positions)
        should_warn = False
        if is_singular:
            if current_time - self.last_warning_time > self.warning_interval:
                should_warn = True
                self.last_warning_time = current_time
                self.warning_interval = 10.0 if score > 0.9 else (30.0 if score > 0.7 else 60.0)
        return is_singular, s_type, score, should_warn

    def reset_warning_control(self):
        self.last_warning_time = 0
        self.warning_interval = 5.0
