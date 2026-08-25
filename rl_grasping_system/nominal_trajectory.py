"""MoveIt 标称轨迹的 MuJoCo 侧仿真替身（D1，一周冲刺方案 §5）。

在真实混合架构（docs/混合控制架构设计.md）中，MoveIt2 对"当前估计的目标位置"
用 OMPL 规划出一条无碰撞标称轨迹（含静态避障），执行层沿轨迹给 Servo 下发
twist 参考速度 v_nominal(t)；RL 残差在其上叠加：v_servo = v_nominal(t) + Δv。

MuJoCo 侧没有 MoveIt，用本模块生成"朝当前目标上方 pre-grasp 点的阻尼速度场"
作为 v_nominal(t)，完全对应真实 Servo 叠加语义：
    v_nominal = clamp( gain * (hover - ee_pos), v_max )
    hover     = target_pos + [0, 0, nominal_hover_z_offset]

设计依据（对齐一周冲刺方案 §2 技术澄清）：
- 标称"对着当前估计的目标位置规划/跟踪"：速度场天然支持移动目标（L2 动态目标
  时 target_pos 实时变化，标称自动跟踪），无需每步重规划；
- 静态世界（L1）：标称把末端稳定导向目标上方，残差 Δv 自然趋近 0 ——
  "残差幅度统计：静态小 / 动态大"这一架构分工成立证据的仿真侧前提；
- 近端 gain·d 线性减速（阻尼引导收敛），远端 clamp 到 nominal_approach_speed，
  全程 ≤ v_max 契约 0.125 m/s（docs/sim_to_real动力学匹配方案.md）。
"""

import numpy as np


class NominalTrajectory:
    """MoveIt 标称轨迹仿真替身：输出世界系标称参考速度 v_nominal(t)（6 维 twist）。

    无内部状态（纯速度场），reset 仅保留接口一致性；可随时重复调用。
    """

    def __init__(self, cfg):
        self.cfg = cfg
        self.v_max = float(getattr(cfg, "nominal_approach_speed", 0.12))
        self.gain = float(getattr(cfg, "nominal_gain", 3.0))
        self.hover_z_offset = float(getattr(cfg, "nominal_hover_z_offset", 0.134))
        self.reset()

    def reset(self):
        """重置（纯速度场无状态，占位保证接口一致）"""
        self.t = 0.0

    def reference_velocity(self, ee_pos, target_pos, dt=None) -> np.ndarray:
        """返回当前时刻标称参考速度 v_nominal（世界系 6 维 twist：[vx,vy,vz,0,0,0]）。

        Args:
            ee_pos: 末端位置 (3,)
            target_pos: 当前（估计的）目标位置 (3,)
            dt: 时间步长（速度场法不使用，保留为 MoveIt 轨迹接口预留）

        Returns:
            np.ndarray (6,)：线速度前 3 维（m/s），角速度后 3 维恒 0
        """
        ee = np.asarray(ee_pos, dtype=float).ravel()[:3]
        tg = np.asarray(target_pos, dtype=float).ravel()[:3]
        hover = tg + np.array([0.0, 0.0, self.hover_z_offset])

        # 竖直分量：始终朝 hover_z 微调（低处上抬、高处下压到悬停高度）
        v = np.zeros(3)
        v[2] = self.gain * (hover[2] - ee[2])
        # 水平分量：朝 hover 的 X/Y 阻尼接近（远端限幅 v_max）
        h_err = hover[:2] - ee[:2]
        h_dist = float(np.linalg.norm(h_err))
        if h_dist > 1e-6:
            v_h = min(self.v_max, self.gain * h_dist)  # 近端线性减速、远端限速
            v[:2] = h_err / h_dist * v_h

        # 整体限幅（含竖直分量）到 v_max，保证 ≤ 契约速度
        n = float(np.linalg.norm(v))
        if n > self.v_max:
            v = v * (self.v_max / n)

        return np.concatenate([v, np.zeros(3)]).astype(np.float64)


def make_nominal_trajectory(cfg) -> NominalTrajectory:
    """工厂：residual 模式返回标称轨迹实例，否则返回 None。"""
    if str(getattr(cfg, "action_mode", "delta")).lower() == "residual":
        return NominalTrajectory(cfg)
    return None
