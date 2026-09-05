#!/usr/bin/env python3
"""MPC 基线：UR5e+2F-85 动态抓取避障（末端级滚动最优控制，2026-09-02）

对比口径对齐消融 dyn_both（0,0,1：动态目标 0.05 m/s + 动态障碍 0.05 m/s ×1，固定物体位置）。

控制器设计（末端级，刻意不用关节级以保持求解实时性）：
  决策变量 u ∈ R^{N×3}（未来 N 决策步末端速度）
  预测模型  p_{k+1} = p_k + u_k·dt         （单积分，dt = 决策周期 0.04s）
  目标      hover_k = hover + target_vel·k·dt（动态目标外推）
  障碍      obs_k   = obs + obs_vel·k·dt   （当前速度线性外推；随机游走下预测失效 → 每步重解纠偏）
  成本      Σ w_p‖p_k-hover_k‖² + w_o·relu(d_safe-‖p_k-obs_k‖)² + w_s·‖u_k-u_{k-1}‖²
  约束      |u| ≤ v_max = 0.125（契约速度）
  求解      scipy SLSQP（warm-start 上次解；失败降级为 P 控制）

执行与 RL 完全同口径：env.step(delta 动作 = v·T)，夹爪自动闭合状态机复用；
closing/lift 阶段 env 自动接管（RL/MPC 都不参与）。

用法：python3 scripts/mpc_ur5_grasp.py --scene dyn_both --n_episodes 30 --save_results results/mpc/dyn_both.json
"""
import os, sys, argparse, json, time
import numpy as np
import scipy.optimize as opt

sys.path.append(os.path.dirname(os.path.abspath(__file__)) + "/..")
from config import get_config
from environment import GraspingEnv

# ---------- MPC 超参数（先在冒烟里标定） ----------
N_DEC = 10          # 预测步数（0.4s 时域）
W_P = 40.0          # 到达 hover 权重
W_O = 1500.0        # 障碍分离软约束权重
D_SAFE = 0.20       # 避障参考点→障碍球心安全距离 (m)（球 r=0.05 + margin）
W_S = 0.5           # 控制平滑权重
W_TERM = 800.0      # 终端硬权重（强制末步到位，破 warm-start 粘滞）
KP_P = 3.0          # P 控制兜底增益
XY_ALIGN_TOL = 0.035 # XY 对准阈值：对准后才允许降 z（防 pad 侧撞推走 cube）
APPROACH_Z = 0.06    # 未对准时目标悬高 (m)：先水平对准再下降（对标称两阶段语义）
HOVER_Z = 0.134     # pre_grasp_offset_z（pad 对准 cube 中心）
V_MAX = 0.125       # v_max 契约（= max_ee_delta / T）


class EndEffectorMPC:
    """末端级 MPC：输出本决策步末端速度 v (3,)。"""

    def __init__(self, N=N_DEC, dt=0.04, v_max=V_MAX):
        self.N = N
        self.dt = dt
        self.v_max = v_max
        self._u_prev = None   # warm start（上一步解）
        self.solve_time = 0.0
        self.fallback_cnt = 0

    def _predict(self, u, ee, hover, tv, obs, ov):
        """按决策序列回放预测路径，返回 (路径点列表, 成本)。"""
        N, dt = self.N, self.dt
        p = np.asarray(ee, dtype=float).copy()
        cost = 0.0
        for k in range(N):
            p = p + u[k] * dt
            hk = hover + tv * (k + 1) * dt
            err = p - hk
            cost += W_P * float(np.dot(err, err))
            if obs is not None:
                ok = obs + ov * (k + 1) * dt
                p_col = p - np.array([0.0, 0.0, 0.07])   # 夹爪中部参考点（覆盖 pad 端）
                d = float(np.linalg.norm(p_col - ok))
                if d < D_SAFE:
                    cost += W_O * (D_SAFE - d) ** 2
            if k > 0:
                cost += W_S * float(np.dot(u[k] - u[k - 1], u[k] - u[k - 1]))
        # 终端硬权重：末步强制到达（接近 hover 后 w_p 梯度→0，SLSQP 停住的根因）
        cost += W_TERM * float(np.dot(p - hover - tv * (N) * dt, p - hover - tv * (N) * dt))
        return cost

    def solve(self, ee, hover, tv=None, obs=None, ov=None):
        """滚动求解，返回本步末端速度 v (3,)。obs=None 表示无激活障碍。"""
        N, dt = self.N, self.dt
        tv = np.zeros(3) if tv is None else np.asarray(tv, dtype=float).ravel()[:3]
        tv[2] = 0.0                      # 悬停高度不随目标外推
        ov = np.zeros(3) if ov is None else np.asarray(ov, dtype=float).ravel()[:3]
        hover = np.asarray(hover, dtype=float).ravel()[:3]
        # 两阶段：XY 未对准 -> 目标 z 抬高（悬高接近，避免 pad 斜向侧撞 cube 推走）；对准后 -> hover
        _xy_err = float(np.linalg.norm(ee[:2] - hover[:2]))
        if _xy_err > XY_ALIGN_TOL:
            hover = np.array([hover[0], hover[1], hover[2] + APPROACH_Z])
        obs = None if obs is None else np.asarray(obs, dtype=float).ravel()[:3]

        # x0：P 方向平铺（诊断 2026-09-02：warm-start 整序列平移会让 SLSQP 停在"不动"的平坦区，
        # 到不了 hover 触发 closing；P 方向给出强下降方向，末段由 SLSQP 分配减速）
        _v = np.clip(KP_P * (hover - ee), -self.v_max, self.v_max)
        x0 = np.tile(_v, (N, 1))

        def _obj(u_flat):
            return self._predict(u_flat.reshape(N, 3), ee, hover, tv, obs, ov)

        bounds = [(-self.v_max, self.v_max)] * (N * 3)
        t0 = time.time()
        try:
            res = opt.minimize(_obj, x0.ravel(), method="SLSQP", bounds=bounds,
                               options={"maxiter": 80, "ftol": 1e-6})
            u = res.x.reshape(N, 3)
            if not res.success:
                self.fallback_cnt += 1   # 结果仍可用（接近最优）
        except Exception:
            u = np.clip(KP_P * (hover - ee), -self.v_max, self.v_max)[None, :].repeat(N, axis=0)
            self.fallback_cnt += 1
        self.solve_time += time.time() - t0
        self._u_prev = u
        return np.clip(u[0], -self.v_max, self.v_max)


class MpcEvaluator:
    """用 GraspingEnv 跑 MPC 闭环评估（与 evaluate.py 口径一致）。"""

    def __init__(self, env, scene="dyn_both", max_steps=200):
        self.env = env
        self.scene = scene
        self.max_steps = max_steps
        self._obs_prev = None

    def run_episode(self):
        env = self.env
        mpc = EndEffectorMPC()
        obs_pos_prev = None
        obs_vel = np.zeros(3)
        obs, _ = env.reset()
        mpc._u_prev = None
        self._obs_prev = None
        # 决策周期 T（与 env 内部一致）
        T = env.substeps * max(1, env.action_repeat) * env.model.opt.timestep

        ep_len = 0
        ep_coll = 0
        info_last = {}
        while ep_len < self.max_steps:
            # 读真值（oracle 口径，与 RL 消融评估一致）
            ee = env._get_end_effector_position().copy()
            target = env._get_object_position().copy()
            hover = target + np.array([0.0, 0.0, HOVER_Z])
            tv = getattr(env, "_dyn_target_vel", np.zeros(3)).copy()
            obs_pos = None
            obs_list = env._get_obstacle_positions()
            if obs_list:
                # 取最近激活障碍（与 RL 观测口径一致）
                obs_pos = min(obs_list, key=lambda o: float(np.linalg.norm(np.asarray(o) - ee)))
                obs_pos = np.asarray(obs_pos, dtype=float).ravel()[:3]
                if obs_pos_prev is not None:
                    obs_vel = (obs_pos - obs_pos_prev) / T
                else:
                    obs_vel = np.zeros(3)

            v_mpc = mpc.solve(ee, hover, tv, obs_pos, obs_vel if obs_pos is not None else None)
            action = np.concatenate([v_mpc * T, np.zeros(3)]).astype(np.float32)[:3]

            obs, reward, terminated, truncated, info = env.step(action)
            ep_len += 1
            obs_pos_prev = obs_pos
            info_last = info
            ep_coll = max(ep_coll, info.get("obstacle_collision_count", 0))
            if terminated or truncated:
                break

        return {
            "episode_length": ep_len,
            "grasp_success": bool(info_last.get("grasp_success", False)),
            "final_distance": float(info_last.get("distance_to_object", 0.0)),
            "collision_count": int(ep_coll),
            "avg_mpc_solve_s": round(mpc.solve_time / max(1, ep_len), 4),
            "mpc_fallback": int(mpc.fallback_cnt),
        }

    def run(self, n_episodes):
        results = []
        for i in range(n_episodes):
            r = self.run_episode()
            results.append(r)
            print(f"[mpc] ep {i + 1}/{n_episodes}: len={r['episode_length']} "
                  f"success={r['grasp_success']} coll={r['collision_count']} "
                  f"solve={r['avg_mpc_solve_s']}s fallback={r['mpc_fallback']}")
        n = len(results)
        success_rate = sum(1 for r in results if r["grasp_success"]) / n
        collision_rate = sum(1 for r in results if r["collision_count"] > 0) / n
        avg_coll = float(np.mean([r["collision_count"] for r in results]))
        avg_len = float(np.mean([r["episode_length"] for r in results]))
        return {
            "scene": self.scene,
            "n_episodes": n,
            "success_rate": success_rate,
            "collision_rate": collision_rate,
            "avg_collision_count": avg_coll,
            "avg_episode_length": avg_len,
            "avg_mpc_solve_s": float(np.mean([r["avg_mpc_solve_s"] for r in results])),
            "episode_results": results,
        }


def main():
    global D_SAFE
    ap = argparse.ArgumentParser(description="MPC 基线评估（UR5e 动态抓取避障）")
    ap.add_argument("--scene", default="dyn_both",
                    choices=["static", "dyn_target", "dyn_both"])
    ap.add_argument("--n_episodes", type=int, default=30)
    ap.add_argument("--target-vel", type=float, default=0.05)
    ap.add_argument("--obstacle-vel", type=float, default=0.05)
    ap.add_argument("--obstacle-count", type=int, default=1)
    ap.add_argument("--save-results", type=str, default="")
    ap.add_argument("--max-steps", type=int, default=200)
    ap.add_argument("--d-safe", type=float, default=D_SAFE,
                    help="安全距离裕度 (m)，Pareto 扫描用：调大更保守（碰撞↓成功率↓）")
    ap.add_argument("--verbose", action="store_true")
    args = ap.parse_args()

    D_SAFE = args.d_safe

    cfg = get_config()
    g = cfg.grasping
    g.render_gui = False
    g.action_mode = "delta"      # MPC 完全自主（对标端到端控制器），不叠加标称
    g.control_mode = "velocity"
    g.use_fixed_position = True  # 固定物体位置（对齐消融口径）
    g.action_space_dim = 3
    g.dynamic_target_enabled = args.scene in ("dyn_target", "dyn_both")
    g.target_vel_xy = args.target_vel if g.dynamic_target_enabled else 0.0
    g.obstacle_enabled = (args.scene == "dyn_both")
    g.obstacle_vel = args.obstacle_vel if g.obstacle_enabled else 0.0
    g.obstacle_count = args.obstacle_count
    g.obstacle_on_nominal_path = False   # 与消融 dyn_both 一致（非 on-path）
    g.max_steps = args.max_steps

    env = GraspingEnv(g, cfg.reward)
    ev = MpcEvaluator(env, scene=args.scene, max_steps=args.max_steps)
    res = ev.run(args.n_episodes)

    print("\n========== MPC 评估结果 ==========")
    print(f"scene={res['scene']}  success_rate={res['success_rate'] * 100:.1f}% "
          f"collision_rate={res['collision_rate'] * 100:.1f}%  avg_coll={res['avg_collision_count']:.2f}  "
          f"avg_len={res['avg_episode_length']:.1f}  avg_solve={res['avg_mpc_solve_s']}s")

    if args.save_results:
        path = args.save_results if os.path.isabs(args.save_results) else \
            os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", args.save_results)
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            json.dump(res, f, indent=2, ensure_ascii=False)
        print(f"saved -> {path}")


if __name__ == "__main__":
    main()
