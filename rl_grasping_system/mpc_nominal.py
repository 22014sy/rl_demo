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
        self.solve_time_total = 0.0           # 累加量：平均规划时间 = solve_time_total / solve_cnt
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
        # arm-aware（2026-09-11）：臂身碰撞球软代价。默认关——关闭时 _arm_ctx=None，
        # _predict 逐位等价于原单点模型（回归安全）。开启见 _build_arm_context。
        self.arm_aware = bool(getattr(cfg, "mpc_nominal_arm_aware", False))
        self.w_arm = float(getattr(cfg, "mpc_nominal_w_arm", 120.0))
        self.arm_margin = float(getattr(cfg, "mpc_nominal_arm_margin", 0.05))
        self.arm_lam = float(getattr(cfg, "mpc_nominal_arm_lam", 0.05))
        self.arm_horizon = int(getattr(cfg, "mpc_nominal_arm_horizon", 5))
        # 夹爪是 MESH/BOX，无解析半径 → 用 MuJoCo 编译出的包围球 geom_rbound 覆盖（保证包住，见
        # _sample_geom_spheres）。实测 static3 的接触 100% 发生在夹爪 mesh 上、联杆 capsule 零接触，
        # 故「只取 capsule/cylinder」的旧口径等于优化一个从不接触任何东西的部位（2026-09-11 修正）。
        self.arm_pad = float(getattr(cfg, "mpc_nominal_arm_pad", 0.01))
        self._arm_ctx = None                  # 每次 solve 重建（当前构型处的线性化）
        self.arm_skip_cnt = 0                 # 奇异/不可用导致的跳过计数（诊断用）
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

    def _build_arm_context(self):
        """构建臂身碰撞球的线性化上下文（每次 solve 一次；SLSQP 内层不重算）。

        返回 None → _predict 完全不执行臂身项（退化为原单点模型）。

        一阶模型必须与**执行层同口径**：action_wrapper 把 twist 积分成目标位姿后调
        solve_ik（位置级 DLS、锁姿态），故用 6 行 DLS——`qdot = J₆ᵀ(J₆J₆ᵀ+λ²I)⁻¹·[v;0]`
        （`J₆` = `rq_base_mount` body 雅可比取臂 dof 列，λ 同 ik.py）。用 3 行 pinv(J_ee)
        会预测系统根本不产生的运动（nullspace 选择不一致）。
        """
        env = self._env
        if env is None or not self.arm_aware:
            return None
        try:
            import mujoco

            model, data = env.model, env.data
            ee_body = int(env.end_effector_id)
            arm_dofs = np.asarray(
                model.jnt_dofadr[np.asarray(env.arm_joint_ids, dtype=int)], dtype=int)
            # 临时副本上前向：读 geom_xpos/xmat 与算雅可比都不动主 data（同 ik.py）
            scratch = mujoco.MjData(model)
            scratch.qpos[:] = data.qpos[:]
            scratch.qvel[:] = data.qvel[:]
            mujoco.mj_forward(model, scratch)

            jacp = np.zeros((3, model.nv))
            jacr = np.zeros((3, model.nv))
            mujoco.mj_jacBody(model, scratch, jacp, jacr, ee_body)
            J6 = np.vstack([jacp[:, arm_dofs], jacr[:, arm_dofs]])   # (6, 6)
            if float(np.linalg.cond(J6)) > 1e3:
                self.arm_skip_cnt += 1      # 近奇异：别让坏 pinv 注入垃圾
                return None
            A6 = J6 @ J6.T + (self.arm_lam ** 2) * np.eye(6)
            M = J6.T @ np.linalg.solve(A6, np.eye(6))                # qdot = M @ [v; 0]

            r_obs = float(getattr(self.cfg, 'obstacle_radius', 0.05))
            pts, rads, jacs = [], [], []
            for g in env._arm_obstacle_geoms():
                if model.geom_contype[g] == 0:
                    continue
                for pt, r in self._sample_geom_spheres(model, scratch, int(g)):
                    jp = np.zeros((3, model.nv))
                    jr = np.zeros((3, model.nv))
                    mujoco.mj_jac(model, scratch, jp, jr, pt, int(model.geom_bodyid[g]))
                    pts.append(pt)
                    rads.append(r)
                    jacs.append(jp[:, arm_dofs])

            if not pts:
                return None
            pos0 = np.asarray(pts, dtype=float)                    # (n, 3)
            rad = np.asarray(rads, dtype=float)                    # (n,)
            Js = np.asarray(jacs, dtype=float)                     # (n, 3, 6)
            # 灵敏度：dp_s = (J_s @ M)[:, :3] @ u · dt = A_s @ u · dt
            A_s = np.einsum('nij,jk->nik', Js, M)[:, :, :3]        # (n, 3, 3)
            return dict(n=len(pts), pos0=pos0, rad=rad, A_s=A_s, r_obs=r_obs)
        except Exception:
            return None

    def _sample_geom_spheres(self, model, scratch, g):
        """把一个可碰撞 geom 覆盖成若干「世界系球心 + 半径」。

        MESH/BOX 无解析半径语义 → 用 MuJoCo 编译出的**包围球** `geom_rbound`（按定义
        = geom 局部原点到最远顶点的距离，故该球**必包住** geom；代价是偏保守，方向安全）。
        CAPSULE/CYLINDER 用轴向采样（rbound 含半长，单球会严重过覆盖细长连杆）。
        半径统一加 `arm_pad` 留量，补采样/线性化误差。
        """
        import mujoco
        c = np.asarray(scratch.geom_xpos[g], dtype=float)
        gt = int(model.geom_type[g])
        pad = self.arm_pad
        if gt == int(mujoco.mjtGeom.mjGEOM_CAPSULE):
            axis = np.asarray(scratch.geom_xmat[g], dtype=float).reshape(3, 3)[:, 2]
            r = float(model.geom_size[g][0])
            half = float(model.geom_size[g][1])
            k_n = int(max(2, np.ceil(2.0 * half / max(1.5 * r, 1e-6)))) + 1
            return [(c + axis * (half * t), r + pad) for t in np.linspace(-1.0, 1.0, k_n)]
        if gt == int(mujoco.mjtGeom.mjGEOM_CYLINDER):
            axis = np.asarray(scratch.geom_xmat[g], dtype=float).reshape(3, 3)[:, 2]
            r = float(model.geom_size[g][0])
            half = float(model.geom_size[g][1])
            k_n = int(max(2, np.ceil(2.0 * half / max(1.0 * r, 1e-6)))) + 1
            return [(c + axis * (half * t), r + pad) for t in np.linspace(-1.0, 1.0, k_n)]
        # MESH / BOX / SPHERE / ELLIPSOID：包围球（rad 由 geom_rbound 给出，必包住 geom）
        r = float(model.geom_rbound[g])
        if r <= 0.0:
            return []
        return [(c, r + pad)]

    def _arm_cost(self, u, obs_list, ov_list):
        """臂身-障碍表面距离 hinge 惩罚（按球数取均值，球数不改变量级）。"""
        ctx = self._arm_ctx
        if ctx is None or not obs_list:
            return 0.0
        A = ctx['A_s']
        p = ctx['pos0'].copy()
        rad = ctx['rad']
        r_obs = ctx['r_obs']
        margin = self.arm_margin
        k_h = min(self.arm_horizon, self.N)
        acc = 0.0
        for k in range(k_h):
            p = p + np.einsum('nij,j->ni', A, np.asarray(u[k], dtype=float)) * self.dt
            for obs, ov in zip(obs_list, ov_list):
                ok = obs + ov * (k + 1) * self.dt
                d = np.linalg.norm(p - ok, axis=1) - rad - r_obs
                viol = np.maximum(0.0, margin - d)
                acc += float(np.dot(viol, viol))
        return self.w_arm * acc / max(1, ctx['n'])

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
        # arm-aware：仅当上下文可用（关闭/奇异时为 None → 逐位等价原模型）
        if self._arm_ctx is not None:
            cost += self._arm_cost(u, obs_list, ov_list)
        return cost


    def reference_velocity(self, ee_pos, target_pos, dt=None, target_vel=None,
                           hover_override=None) -> np.ndarray:
        """返回当前时刻标称参考速度 v_nominal（世界系 6 维 twist：[vx,vy,vz,0,0,0]）。

        Args:
            ee_pos: 末端位置 (3,)
            target_pos: 当前（估计的）目标位置 (3,)
            dt: 决策周期 (s)；None 时用 0.04（action_repeat=2 × 50Hz 默认）
            target_vel: 目标速度 (3,)（动态目标 L2，z 分量不参与外推）
            hover_override: 绕行路点（世界系 3 维）；非 None 时**直接用作 hover**，
                跳过 `+hover_z_offset` 与 `xy_align` 抬 z——两阶段接近语义由调用方
                （via_point_nominal）自行负责，内层不再插手。默认 None 时逐位等同原行为。

        Returns:
            np.ndarray (6,)：线速度前 3 维（m/s），角速度后 3 维恒 0
        """
        ee = np.asarray(ee_pos, dtype=float).ravel()[:3]
        tg = np.asarray(target_pos, dtype=float).ravel()[:3]
        self.dt = 0.04 if dt is None else float(dt)
        tv = np.zeros(3) if target_vel is None else np.asarray(target_vel, dtype=float).ravel()[:3]
        tv[2] = 0.0                      # 悬停高度不随目标外推

        if hover_override is None:
            hover = tg + np.array([0.0, 0.0, self.hover_z_offset])
            # 两阶段：XY 未对准 -> 目标 z 抬高（悬高接近，避免 pad 斜向侧撞 cube 推走）；对准后 -> hover
            if float(np.linalg.norm(ee[:2] - hover[:2])) > self.xy_align_tol:
                hover = np.array([hover[0], hover[1], hover[2] + self.approach_z])
        else:
            hover = np.asarray(hover_override, dtype=float).ravel()[:3]

        obs_list, ov_list = self._read_obstacles()
        if obs_list is None:
            obs_list, ov_list = [], []

        # arm-aware 上下文：当前构型处的一次线性化（关闭/不可用/奇异时为 None）
        self._arm_ctx = self._build_arm_context()

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
        # ⚠️ solve_time 是**单次**耗时（赋值，不是累加）；评测要的「平均规划时间」必须用
        # solve_time_total / solve_cnt，直接读 solve_time 会把最后一次的耗时当成全集均值。
        _dt_solve = time.time() - t0
        self.solve_time = _dt_solve
        self.solve_time_total += _dt_solve
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
