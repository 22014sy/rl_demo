"""MPC 标称层：末端级滚动最优控制（MPC 标称 + RL 残差架构的标称侧，2026-09-07）。

在混合残差架构中，标称层输出 v_nominal(t)，RL 残差叠加：v_servo = v_nominal(t) + Δv。
本模块用 MPC 替换原速度场替身（nominal_trajectory.py 的 NominalTrajectory），
接口完全一致（reference_velocity → 6 维 twist），观测槽位/环境 step 零改动：

  - 决策变量 u ∈ R^{N×3}（未来 N 决策步末端速度）
  - 预测模型  p_{k+1} = p_k + u_k·dt         （单积分，dt = 决策周期 ~0.04s）
  - 目标      hover_k = hover + target_vel·k·dt（动态目标外推，z 分量不外推）
  - 障碍      obs_k   = obs + obs_vel·k·dt   （当前速度线性外推；含 z 方向运动障碍，
              obs_vel 由环境传入——每步重解纠偏，随机游走预测失效交给 RL 残差）
  - 成本      Σ w_p‖p_k−hover_k‖² + w_o·relu(d_safe−d_k)² + w_s‖u_k−u_{k−1}‖² + w_term·终端
  - 约束      |u| ≤ v_max = 0.125（契约速度）
  - 求解      scipy SLSQP（warm-start 上次解；失败降级为 P 控制）

与 scripts/mpc_ur5_grasp.py 基线同口径，区别：
  - 本类是"标称层"，供 residual 模式叠加 RL 残差（mpc_ur5_grasp 是 delta 完全自主基线）；
  - 障碍信息经 attach_env 从环境实时读取（世界系 pos/vel），MPC 自带避障软约束；
  - 返回 6 维 twist（后 3 维角速度恒 0），语义与速度场标称一致。
"""

import numpy as np
import scipy.optimize as opt
import time


class MpcNominal:
    """MPC 标称轨迹：输出世界系标称参考速度 v_nominal(t)（6 维 twist）。

    有状态（warm-start _u_prev），reset() 必须清除（跨 episode 污染防护）；
    障碍经 attach_env 注入环境引用（环境维护激活障碍 pos/vel），未 attach 时无避障。
    """

    def __init__(self, cfg):
        self.cfg = cfg
        self.N = int(getattr(cfg, "mpc_nominal_horizon", 10))
        self.w_p = float(getattr(cfg, "mpc_nominal_w_p", 40.0))
        self.w_o = float(getattr(cfg, "mpc_nominal_w_o", 1500.0))
        self.d_safe = float(getattr(cfg, "mpc_nominal_d_safe", 0.20))
        self.w_s = float(getattr(cfg, "mpc_nominal_w_s", 0.5))
        self.w_term = float(getattr(cfg, "mpc_nominal_w_term", 800.0))
        self.xy_align_tol = float(getattr(cfg, "mpc_nominal_xy_align_tol", 0.035))
        self.approach_z = float(getattr(cfg, "mpc_nominal_approach_z", 0.06))
        self.hover_z_offset = float(getattr(cfg, "nominal_hover_z_offset", 0.134))
        # 速度上限 = 契约速度 0.125（对齐基线 V_MAX；2026-09-07 受控对比实证：用
        # nominal_approach_speed=0.12 追击动态目标每步少 4% 速度，dyn_target 成功率 80→70，
        # 且 avg_len 121.8→111.3。nominal_approach_speed 是速度场标称语义，此处不复用）
        self.v_max = float(getattr(cfg, "mpc_nominal_v_max", 0.125))
        self.kp_p = 3.0                       # P 控制兜底增益（求解失败降级）
        self._u_prev = None                   # warm start（上一步解）
        self._env = None                      # attach_env 后非 None → 启用障碍避障
        self.solve_time = 0.0
        self.fallback_cnt = 0
        self.solve_cnt = 0
        # v25b2 门控信号（每决策步 reference_velocity 更新）：solver_ok=SLSQP 求解成功（含无异常）；
        # min_obstacle_dist=预测路径上末端参考点到最近障碍的最小距离（低裕度判据，∞=无障碍/未 attach）；
        # terminal_err=预测序列末步到 hover（含目标外推）的误差——「MPC 到不了目标」判据
        # （2026-09-09 实证：static3 下 SLSQP 报 success=True 但停在安全位到不了 hover，
        #  res.success 不能判"无解"，terminal_err 才是）。
        self.solver_ok = True
        self.min_obstacle_dist = float('inf')
        self.terminal_err = float('inf')
        self.reset()

    def bind(self, model, data, arm_joint_ids, body_id):
        """接口一致性（速度场替身的 bind）；MPC 纯末端级无需模型，保留签名。"""
        return self

    def attach_env(self, env):
        """注入环境引用（读取激活障碍 pos/vel 做避障软约束）。"""
        self._env = env

    def reset(self):
        """重置（清除 warm-start 状态，防跨 episode 污染）。"""
        self._u_prev = None

    def _read_obstacles(self):
        """读取环境当前激活障碍（世界系位置 + 速度）。无 env/未激活/无障碍 → (None, None)。"""
        env = self._env
        if env is None:
            return None, None
        try:
            poses = env._get_obstacle_positions()
        except Exception:
            return None, None
        if not poses:
            return None, None
        vels = []
        for _bid in getattr(env, '_active_obstacle_body_ids', []):
            vels.append(env._get_obstacle_velocity_by_id(_bid))
        return poses, vels

    def _predict(self, u, ee, hover, tv, obs_list, ov_list):
        """按决策序列回放预测路径，返回总成本（标量）。"""
        N, dt = self.N, self.dt
        p = np.asarray(ee, dtype=float).copy()
        cost = 0.0
        for k in range(N):
            p = p + u[k] * dt
            hk = hover + tv * (k + 1) * dt
            err = p - hk
            cost += self.w_p * float(np.dot(err, err))
            if obs_list:
                # 全部激活障碍（不止最近）：夹爪中部参考点覆盖 pad 端
                p_col = p - np.array([0.0, 0.0, 0.07])
                for obs, ov in zip(obs_list, ov_list):
                    ok = obs + ov * (k + 1) * dt
                    d = float(np.linalg.norm(p_col - ok))
                    if d < self.d_safe:
                        cost += self.w_o * (self.d_safe - d) ** 2
            if k > 0:
                cost += self.w_s * float(np.dot(u[k] - u[k - 1], u[k] - u[k - 1]))
        # 终端硬权重：末步强制到达（接近 hover 后 w_p 梯度→0，SLSQP 停住的根因）
        cost += self.w_term * float(np.dot(p - hover - tv * N * dt,
                                           p - hover - tv * N * dt))
        return cost


    def reference_velocity(self, ee_pos, target_pos, dt=None, target_vel=None) -> np.ndarray:
        """返回当前时刻标称参考速度 v_nominal（世界系 6 维 twist：[vx,vy,vz,0,0,0]）。

        Args:
            ee_pos: 末端位置 (3,)
            target_pos: 当前（估计的）目标位置 (3,)
            dt: 决策周期 (s)；None 时用 0.04（action_repeat=2 × 50Hz 默认）
            target_vel: 目标速度 (3,)（动态目标 L2，z 分量不参与外推）

        Returns:
            np.ndarray (6,)：线速度前 3 维（m/s），角速度后 3 维恒 0
        """
        ee = np.asarray(ee_pos, dtype=float).ravel()[:3]
        tg = np.asarray(target_pos, dtype=float).ravel()[:3]
        self.dt = 0.04 if dt is None else float(dt)
        hover = tg + np.array([0.0, 0.0, self.hover_z_offset])
        tv = np.zeros(3) if target_vel is None else np.asarray(target_vel, dtype=float).ravel()[:3]
        tv[2] = 0.0                      # 悬停高度不随目标外推

        # 两阶段：XY 未对准 -> 目标 z 抬高（悬高接近，避免 pad 斜向侧撞 cube 推走）；对准后 -> hover
        if float(np.linalg.norm(ee[:2] - hover[:2])) > self.xy_align_tol:
            hover = np.array([hover[0], hover[1], hover[2] + self.approach_z])

        obs_list, ov_list = self._read_obstacles()
        if obs_list is None:
            obs_list, ov_list = [], []

        N = self.N
        vmax = self.v_max
        # x0：P 方向平铺（基线 2026-09-02 诊断：warm-start 整序列平移会让 SLSQP 停在"不动"的平坦区，
        # 到不了 hover 触发 closing；P 方向给出强下降方向，末段由 SLSQP 分配减速）
        _v = np.clip(self.kp_p * (hover - ee), -vmax, vmax)
        u0 = np.tile(_v, (N, 1))

        def _obj(u_flat):
            return self._predict(u_flat.reshape(N, 3), ee, hover, tv, obs_list, ov_list)

        t0 = time.time()
        try:
            res = opt.minimize(_obj, u0.ravel(), method='SLSQP',
                               bounds=[(-vmax, vmax)] * (N * 3),
                               options={'maxiter': 80, 'ftol': 1e-6})   # 对齐基线 maxiter=80
        except Exception:
            res = None

        if res is not None:
            # 与 mpc_ur5_grasp.py 基线同口径：SLSQP 非 success（maxiter/ftol 未达标）的解仍接近最优，
            # 直接使用；仅求解器抛异常才降级 P 控制（fallback 只统计异常）。
            self._u_prev = res.x.reshape(N, 3)
            u_used = self._u_prev
            v = self._u_prev[0].copy()
            if not res.success:
                self.fallback_cnt += 1
        else:
            # 失败降级：P 控制朝 hover（限幅 v_max）
            self.fallback_cnt += 1
            v = self.kp_p * (hover - ee)
            n = float(np.linalg.norm(v))
            if n > vmax:
                v = v * (vmax / n)
            u_used = np.tile(v, (N, 1))
        # v25b2 门控信号：solver_ok=False → 环境 gate 判定「MPC 无解」→ 放大残差预算（双向解耦）
        self.solver_ok = (res is not None) and bool(getattr(res, 'success', False))
        self.min_obstacle_dist = self._min_obstacle_dist_on_path(
            u_used, ee, hover, tv, obs_list, ov_list)
        self.terminal_err = self._terminal_err_on_path(u_used, ee, hover, tv)
        self.solve_time = time.time() - t0
        self.solve_cnt += 1

        return np.concatenate([v, np.zeros(3)]).astype(np.float64)

    def _terminal_err_on_path(self, u, ee, hover, tv) -> float:
        """预测序列末步到 hover（含目标外推）的误差（v25b2「MPC 到不了」判据）。

        与 _predict 终端硬权重项同参考点（hover + tv·N·dt），保证 gate 的「到不了」
        判据与 MPC 终端代价口径一致。
        """
        N, dt = self.N, self.dt
        p = np.asarray(ee, dtype=float).ravel()[:3].copy()
        for k in range(N):
            p = p + u[k] * dt
        return float(np.linalg.norm(p - (hover + tv * N * dt)))

    def _min_obstacle_dist_on_path(self, u, ee, hover, tv, obs_list, ov_list) -> float:
        """预测路径（决策序列回放）上末端参考点到最近障碍的最小距离（v25b2 gate 低裕度判据）。

        与 _predict 同参考点（夹爪中部 p_col = p − [0,0,0.07]）与障碍外推（obs + ov·(k+1)·dt），
        保证 gate 的「接近障碍」判据与 MPC 成本中避障项口径一致。
        """
        if not obs_list:
            return float('inf')
        N, dt = self.N, self.dt
        p = np.asarray(ee, dtype=float).ravel()[:3].copy()
        best = float('inf')
        for k in range(N):
            p = p + u[k] * dt
            p_col = p - np.array([0.0, 0.0, 0.07])
            for obs, ov in zip(obs_list, ov_list):
                ok = obs + ov * (k + 1) * dt
                d = float(np.linalg.norm(p_col - ok))
                if d < best:
                    best = d
        return best


def make_mpc_nominal(cfg) -> MpcNominal:
    """工厂：residual 模式返回 MPC 标称实例，否则返回 None（由 nominal_trajectory 分发调用）。"""
    if str(getattr(cfg, "action_mode", "delta")).lower() != "residual":
        return None
    return MpcNominal(cfg)
