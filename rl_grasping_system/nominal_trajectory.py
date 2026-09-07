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

2026-09-02 P2b 升级：可选 IK 可达性检查（nominal_ik_check=True，默认开）。
对标真 MoveIt 语义——真 MoveIt「IK 解算目标位姿 → 轨迹采样 → Servo 下发 twist」：
本替身对段目标（hover 位姿）用 solve_ik 从当前位形验证可达性：
  - 可达 → 正常速度场（与纯几何两段速度场行为 bit 一致，零回归）；
  - 不可达 → 返回零速度（保持），对标 MoveIt「规划失败 → 不动」，
    并为将来扩大 workspace（边界点可能越出 IK 可达域）提供兜底。
"""

import numpy as np
import mujoco
import time
import ik as _ik


class NominalTrajectory:
    """MoveIt 标称轨迹仿真替身：输出世界系标称参考速度 v_nominal(t)（6 维 twist）。

    无内部状态（纯速度场），reset 仅保留接口一致性；可随时重复调用。
    """

    def __init__(self, cfg):
        self.cfg = cfg
        self.v_max = float(getattr(cfg, "nominal_approach_speed", 0.12))
        self.gain = float(getattr(cfg, "nominal_gain", 3.0))
        self.gain_h = float(getattr(cfg, "nominal_horizontal_gain", 6.0))
        self.k_ff = float(getattr(cfg, "nominal_feedforward_gain", 1.0))
        self.hover_z_offset = float(getattr(cfg, "nominal_hover_z_offset", 0.134))
        # 2026-08-27 P2a（§8.3）：两段轨迹模式（复刻 MoveIt 无碰撞接近，速度环执行 90%）
        self.use_ik_traj = bool(getattr(cfg, "nominal_ik_two_stage", True))
        self.ik_clearance = float(getattr(cfg, "nominal_approach_clearance", 0.15))
        self.ik_switch_tol = float(getattr(cfg, "nominal_stage_switch_tol", 0.02))
        # 2026-09-02 P2b：IK 可达性检查（对齐真 MoveIt「IK 解算目标位姿」语义，见模块 docstring）
        self.use_ik_check = bool(getattr(cfg, "nominal_ik_check", True))
        self.ik_check_tol = float(getattr(cfg, "nominal_ik_check_tol", 0.01))
        self.ik_recheck = float(getattr(cfg, "nominal_ik_check_recheck", 0.005))
        self._model = None              # bind() 后非 None → 启用 IK 检查
        self._check_pos = None          # 上次检查的目标位置（缓存键）
        self._reachable = True          # 缓存的可达性结论
        self._last_warn = 0.0           # 不可达告警频率控制
        self.reset()

    def bind(self, model, data, arm_joint_ids, body_id):
        """绑定模型/数据引用（环境构造后调用）。未 bind 时 IK 检查自动跳过。"""
        self._model = model
        self._data = data
        self._arm_joint_ids = list(arm_joint_ids)
        self._body_id = int(body_id)
        # 抓取姿态：手指朝下、局部 x = 世界 x（2F-85 开合轴），与 verify_grasp_success_criterion.py 一致
        self._grasp_quat = _ik.frame_to_quat(np.array([0.0, 0.0, -1.0]),
                                             np.array([0.0, -1.0, 0.0]))
        # home keyframe 位形：防"当前位形落局部极小→假不可达"的二次确认起点
        kid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_KEY, "home")
        self._home_q = None
        if kid >= 0:
            self._home_q = np.array([model.key_qpos[kid][model.jnt_qposadr[j]]
                                     for j in self._arm_joint_ids], dtype=float)
        return self

    def _solve_err_from(self, scratch_qpos, target_pos):
        """从指定臂位形出发 DLS 解 hover 位姿，返回位置误差(m)；异常/不可达返回大数。"""
        model = self._model
        scratch = mujoco.MjData(model)
        scratch.qpos[:] = self._data.qpos[:]
        scratch.qpos[self._arm_joint_ids] = scratch_qpos
        scratch.qvel[:] = 0.0
        mujoco.mj_forward(model, scratch)
        hover = (np.asarray(target_pos, dtype=float).ravel()[:3]
                 + np.array([0.0, 0.0, self.hover_z_offset]))
        try:
            _, err = _ik.solve_ik(model, scratch, self._body_id, hover, self._grasp_quat,
                                  self._arm_joint_ids, iters=300, tol=5e-3)
        except Exception:
            return 1.0
        return float(err)

    def _target_reachable(self, target_pos):
        """hover 目标可达性检查（带缓存）。当前位形 + home 双起点防局部极小误判。"""
        tg = np.asarray(target_pos, dtype=float).ravel()[:3]
        if self._check_pos is not None and                 float(np.linalg.norm(tg - self._check_pos)) < self.ik_recheck:
            return self._reachable
        cur_q = self._data.qpos[self._arm_joint_ids].copy()
        if self._solve_err_from(cur_q, tg) < self.ik_check_tol:
            self._reachable = True
        elif self._home_q is not None:
            self._reachable = self._solve_err_from(self._home_q, tg) < self.ik_check_tol
        else:
            self._reachable = False
        self._check_pos = tg
        return self._reachable

    def reset(self):
        """重置（两段轨迹模式需清段状态）"""
        self.t = 0.0
        self.ik_phase = 0

    def reference_velocity(self, ee_pos, target_pos, dt=None, target_vel=None) -> np.ndarray:
        """返回当前时刻标称参考速度 v_nominal（世界系 6 维 twist：[vx,vy,vz,0,0,0]）。

        Args:
            ee_pos: 末端位置 (3,)
            target_pos: 当前（估计的）目标位置 (3,)
            dt: 时间步长（速度场法不使用，保留为 MoveIt 轨迹接口预留）
            target_vel: 目标速度 (3,)（动态目标 L2；2026-08-27 新增速度前馈：
                消纯 P 控制对移动目标的稳态跟踪滞后 err≈v_target/gain，见
                docs/2026-08-27_部署分工与最小验证.md §4.2；None=静态目标无前馈）

        Returns:
            np.ndarray (6,)：线速度前 3 维（m/s），角速度后 3 维恒 0
        """
        ee = np.asarray(ee_pos, dtype=float).ravel()[:3]
        tg = np.asarray(target_pos, dtype=float).ravel()[:3]
        hover = tg + np.array([0.0, 0.0, self.hover_z_offset])

        # 2026-09-02 P2b：IK 可达性检查——hover 目标不可达时保持（对标 MoveIt 规划失败→不动）。
        # 当前 workspace（IK 打点 100% 可达）下恒通过，行为与纯几何两段速度场 bit 一致；
        # 仅当 workspace 扩大/目标越出 IK 可达域时才生效，为 RL 残差保留行动空间而非朝不可达空转。
        if self.use_ik_check and self._model is not None:
            if not self._target_reachable(tg):
                _now = time.time()
                if _now - self._last_warn > 5.0:
                    self._last_warn = _now
                    print(f"[nominal] IK 不可达: target={tg.round(4)} hover={hover.round(4)} → 标称保持")
                return np.zeros(6)

        # 两段轨迹模式（2026-08-27 P2a §8.3，复刻 MoveIt 无碰撞接近）：段1 先到 cube 正上方高处
        # （水平对齐、不扫过物体），段2 垂直下降到 hover。段目标随 target_pos 更新（动态目标天然跟踪）。
        if self.use_ik_traj:
            if self.ik_phase == 0:
                t1 = hover + np.array([0.0, 0.0, self.ik_clearance])
                if float(np.linalg.norm(ee - t1)) < self.ik_switch_tol:
                    self.ik_phase = 1
            target = hover if self.ik_phase == 1 else t1
        else:
            target = hover

        # 竖直分量：始终朝目标 z 微调（低处上抬、高处下压到悬停高度）
        v = np.zeros(3)
        v[2] = self.gain * (target[2] - ee[2])
        # 水平分量：朝目标 X/Y 阻尼接近（远端限幅 v_max）
        h_err = target[:2] - ee[:2]
        h_dist = float(np.linalg.norm(h_err))
        if h_dist > 1e-6:
            v_h = min(self.v_max, self.gain_h * h_dist)  # 近端线性减速、远端限速（水平增益独立，收敛更精确）
            v[:2] = h_err / h_dist * v_h

        # 动态目标速度前馈：叠加目标速度，抵消 P 控制跟踪滞后（k_ff=1 完全补偿）
        if target_vel is not None:
            tv = np.asarray(target_vel, dtype=float).ravel()[:3]
            if float(np.linalg.norm(tv)) > 1e-9:
                v = v + self.k_ff * tv

        # 整体限幅（含竖直分量与前馈）到 v_max，保证 ≤ 契约速度
        n = float(np.linalg.norm(v))
        if n > self.v_max:
            v = v * (self.v_max / n)

        return np.concatenate([v, np.zeros(3)]).astype(np.float64)


def make_nominal_trajectory(cfg):
    """工厂：residual 模式返回标称轨迹实例（速度场或 MPC），否则返回 None。

    P2c（2026-09-07）：按 cfg.nominal_mode 分发——
      'velocity_field' → 本文件速度场替身（假 MoveIt，无状态 O(1)，训练快）
      'mpc'            → mpc_nominal.MpcNominal（末端级滚动最优控制，自带障碍避障软约束）
    两者接口一致（reference_velocity → 6 维 twist），观测槽位/环境 step 零改动。
    """
    if str(getattr(cfg, "action_mode", "delta")).lower() != "residual":
        return None
    mode = str(getattr(cfg, "nominal_mode", "velocity_field")).lower()
    if mode == "mpc":
        from mpc_nominal import MpcNominal
        return MpcNominal(cfg)
    return NominalTrajectory(cfg)
