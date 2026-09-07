#!/usr/bin/env python3
"""Stage 2 评估：residual 架构 4 象限对比（velocity_field|MPC × 无RL|PPO Δv）（2026-09-07）。

同 scripts/mpc_nominal_verify.py 环境口径（GraspingEnv + action_mode=residual + nominal_mode），
区别：本脚本可加载 PPO 策略（--model，含 _vecnormalize.pkl 自动恢复）输出残差动作；
--zero-residual 时动作恒 0 = 纯标称（等价 mpc_nominal_verify.py 的 action=0）。

四象限回填（2026-09-02 对比表缺失的第 4 格 = MPC + PPO Δv，即 Stage 2 目标）：
  (velocity_field, 0)      = 速度场标称纯标称
  (velocity_field, PPO)    = v23_mix（velocity_field + PPO 残差）
  (MPC, 0)                 = v_max=0.125 纯 MPC（mpc_nominal_verify.py 同口径）
  (MPC, PPO)               = v24_mpcres（MPC + PPO 残差）→ --model ... --nominal-mode mpc

判定线（Stage 2，n=30）：dyn_both success ≥80% 且 coll ≤5%；dyn_target →83.3%；
static3 >0%；mixed_z coll <10%（部分成功可接受）。残差幅度监控：成功=小残差
（correct-not-take-over）、失败=大残差。

用法：
  # MPC+RL（Stage 2 目标，回填第 4 格）
  python3 scripts/mpc_plus_rl_eval.py --scene dyn_both \
      --model models/final_model_stage2_d2_v24_mpcres.zip \
      --n_episodes 30 --nominal-mode mpc --save-results results/mpc_plus_rl/dyn_both.json
  # 纯 MPC 标称对照（等价 mpc_nominal_verify.py action=0）
  python3 scripts/mpc_plus_rl_eval.py --scene dyn_both --n_episodes 30 \
      --nominal-mode mpc --zero-residual
  # v23 对照（velocity_field + PPO）
  python3 scripts/mpc_plus_rl_eval.py --scene dyn_both \
      --model models/final_model_stage2_d2_v23_mix.zip \
      --n_episodes 30 --nominal-mode velocity_field
"""
import os, sys, argparse, json, time
import numpy as np

sys.path.append(os.path.dirname(os.path.abspath(__file__)) + "/..")
from config import get_config
from environment import GraspingEnv
from agent import GraspingAgent

SCENE_CFG = {
    "static":     dict(dyn_target=False, obstacle=("off", 0, 0.0),    specs=None, z=False),
    "dyn_target": dict(dyn_target=True,  obstacle=("off", 0, 0.0),    specs=None, z=False),
    "dyn_both":   dict(dyn_target=True,  obstacle=("legacy", 1, 0.05), specs=None, z=False),
    "static3":    dict(dyn_target=False, obstacle=("legacy", 3, 0.0), specs=None, z=False),
    "mixed3":     dict(dyn_target=False, obstacle=("specs", 3, 0.0),
                       specs=[{"type": "static",  "pos": (0.10, 0.40, 0.36)},
                              {"type": "dynamic", "vel": 0.05, "mode": "random",
                               "pos": (0.06, 0.46, 0.36)},
                              {"type": "dynamic", "vel": 0.04, "mode": "roundtrip", "axis": "x",
                               "half_range": 0.10, "period": 4.0, "pos": (0.16, 0.38, 0.36)}],
                       z=False),
    "mixed_z":    dict(dyn_target=False, obstacle=("specs", 3, 0.0),
                       specs=[{"type": "static",  "pos": (0.10, 0.40, 0.36)},
                              {"type": "dynamic", "vel": 0.05, "mode": "random",
                               "z_motion": True, "z_amp": 0.10, "z_period": 3.0,
                               "pos": (0.06, 0.46, 0.36)},
                              {"type": "dynamic", "vel": 0.04, "mode": "roundtrip", "axis": "x",
                               "half_range": 0.10, "period": 4.0,
                               "z_motion": True, "z_amp": 0.12, "z_period": 5.0,
                               "pos": (0.16, 0.38, 0.36)}],
                       z=True),
}


class ResidualEvaluator:
    """residual 架构闭环评估：--model 策略残差 / --zero-residual 纯标称（指标与 mpc_nominal_verify 同口径）。"""

    def __init__(self, env, agent, scene="dyn_both", max_steps=200, zero_residual=False):
        self.env = env
        self.agent = agent
        self.scene = scene
        self.max_steps = max_steps
        self.zero_residual = zero_residual

    def run_episode(self):
        env = self.env
        obs, _ = env.reset()
        if env.nominal_trajectory is not None:
            env.nominal_trajectory.reset()        # 清 MPC warm-start / 速度场状态（跨 episode 防护）
        ep_len = 0
        ep_coll = 0
        residual_norms = []
        info_last = {}
        while ep_len < self.max_steps:
            if self.zero_residual or self.agent is None:
                action = np.zeros(env.action_space.shape[0], dtype=np.float32)
            else:
                action, _ = self.agent.predict(obs, deterministic=True)
            obs, _, terminated, truncated, info = env.step(action)
            ep_len += 1
            ep_coll = max(ep_coll, info.get('obstacle_collision_count', 0))
            residual_norms.append(info.get('residual_norm', 0.0))
            info_last = info
            if terminated or truncated:
                break
        # MPC 标称侧求解统计（速度场标称无 solve_cnt，跳过）
        mpc = getattr(env, 'nominal_trajectory', None)
        solve_time, fallback = 0.0, 0
        if mpc is not None and hasattr(mpc, 'solve_cnt') and mpc.solve_cnt > 0:
            solve_time = float(getattr(mpc, 'solve_time', 0.0))
            fallback = int(getattr(mpc, 'fallback_cnt', 0))
        return {
            'episode_length': ep_len,
            'grasp_success': bool(info_last.get('grasp_success', False)),
            'final_distance': float(info_last.get('distance_to_object', 0.0)),
            'collision_count': int(ep_coll),
            'avg_residual_norm': float(np.mean(residual_norms)) if residual_norms else 0.0,
            'max_residual_norm': float(np.max(residual_norms)) if residual_norms else 0.0,
            'avg_mpc_solve_s': solve_time,
            'mpc_fallback': int(fallback),
        }

    def run(self, n_episodes):
        results = []
        for i in range(n_episodes):
            r = self.run_episode()
            results.append(r)
            print(f"[res-eval] ep {i + 1}/{n_episodes}: len={r['episode_length']} "
                  f"success={r['grasp_success']} coll={r['collision_count']} "
                  f"res={r['avg_residual_norm']:.4f} solve={r['avg_mpc_solve_s']}s "
                  f"fallback={r['mpc_fallback']}")
        n = len(results)
        _succ = [r for r in results if r['grasp_success']]
        _fail = [r for r in results if not r['grasp_success']]
        return {
            'scene': self.scene,
            'n_episodes': n,
            'success_rate': sum(1 for r in results if r['grasp_success']) / n,
            'collision_rate': sum(1 for r in results if r['collision_count'] > 0) / n,
            'avg_collision_count': float(np.mean([r['collision_count'] for r in results])),
            'avg_episode_length': float(np.mean([r['episode_length'] for r in results])),
            'avg_residual_norm_success': float(np.mean([r['avg_residual_norm'] for r in _succ])) if _succ else 0.0,
            'avg_residual_norm_fail': float(np.mean([r['avg_residual_norm'] for r in _fail])) if _fail else 0.0,
            'max_residual_norm_overall': float(np.max([r['max_residual_norm'] for r in results])) if results else 0.0,
            'episode_results': results,
        }


def main():
    ap = argparse.ArgumentParser(description="Stage 2 评估：residual 架构 4 象限（velocity_field|MPC × 无RL|PPO Δv）")
    ap.add_argument("--scene", default="dyn_both", choices=list(SCENE_CFG))
    ap.add_argument("--n_episodes", type=int, default=30)
    ap.add_argument("--model", type=str, default="",
                    help="PPO 模型 zip（同目录 _vecnormalize.pkl 自动恢复观测统计）")
    ap.add_argument("--nominal-mode", default="mpc", choices=["mpc", "velocity_field"])
    ap.add_argument("--zero-residual", action="store_true",
                    help="动作恒 0 → v_servo=v_nominal（纯标称对照）")
    ap.add_argument("--target-vel", type=float, default=0.05)
    ap.add_argument("--max-steps", type=int, default=200)
    ap.add_argument("--residual-delta-cap", type=float, default=None,
                    help="v25 残差预算解耦：residual 分支位置增量上限 m（默认 config.residual_delta_cap=0.002；"
                         "重跑 v24 同口径对照传 0.005 恢复旧 clip）")
    ap.add_argument("--save-results", type=str, default="")
    args = ap.parse_args()
    sc = SCENE_CFG[args.scene]

    cfg = get_config()
    g = cfg.grasping
    g.render_gui = False
    g.control_mode = "velocity"
    g.action_mode = "residual"          # 残差架构
    g.nominal_mode = args.nominal_mode  # 'mpc' 或 'velocity_field'
    g.nominal_enabled = True
    g.action_space_dim = 3
    g.use_fixed_position = True         # 固定物体位置（对齐消融口径）
    g.max_steps = args.max_steps
    if args.residual_delta_cap is not None:
        g.residual_delta_cap = args.residual_delta_cap   # v25：残差预算覆盖（默认 config 0.002）

    # ---- 目标/障碍场景（与 mpc_nominal_verify.py 同口径） ----
    g.dynamic_target_enabled = sc["dyn_target"]
    g.target_vel_xy = args.target_vel if g.dynamic_target_enabled else 0.0
    kind, count, vel = sc["obstacle"]
    if kind == "off":
        g.obstacle_enabled = False
        g.obstacle_count = 0
        g.obstacle_specs = None
    elif kind == "legacy":
        g.obstacle_enabled = True
        g.obstacle_count = count
        g.obstacle_vel = vel
        g.obstacle_on_nominal_path = (count > 1)   # 单障碍非 on-path（对齐基线 dyn_both 口径）
        g.obstacle_specs = None
        g.obstacle_z_motion = sc["z"]
    else:  # specs
        g.obstacle_enabled = True
        g.obstacle_specs = sc["specs"]
        g.obstacle_on_nominal_path = True

    env = GraspingEnv(g, cfg.reward)
    agent = None
    if args.model and not args.zero_residual:
        agent = GraspingAgent(cfg.network, cfg.training, model_path=args.model)
        agent.set_environment(env, n_envs=1)      # DummyVecEnv + VecNormalize（自动恢复 pkl 统计）
        print(f"[res-eval] 已加载策略模型: {args.model} (nominal_mode={args.nominal_mode})")

    ev = ResidualEvaluator(env, agent, scene=args.scene, max_steps=args.max_steps,
                           zero_residual=args.zero_residual)
    res = ev.run(args.n_episodes)

    mode = "纯标称(Δv=0)" if (args.zero_residual or not args.model) else "PPO Δv"
    print("\n========== Stage2 residual 评估结果 ==========")
    print(f"scene={res['scene']} nominal={args.nominal_mode} model={mode}  "
          f"success={res['success_rate'] * 100:.1f}% coll={res['collision_rate'] * 100:.1f}% "
          f"avg_coll={res['avg_collision_count']:.2f} len={res['avg_episode_length']:.1f} "
          f"res_succ={res['avg_residual_norm_success']:.4f} res_fail={res['avg_residual_norm_fail']:.4f}")

    if args.save_results:
        path = args.save_results if os.path.isabs(args.save_results) else \
            os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", args.save_results)
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            json.dump(res, f, indent=2, ensure_ascii=False)
        print(f"saved -> {path}")


if __name__ == "__main__":
    main()

