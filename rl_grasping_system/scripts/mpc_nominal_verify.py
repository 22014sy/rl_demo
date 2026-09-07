#!/usr/bin/env python3
"""阶段1验证：纯 MPC（residual 模式 + nominal_mode='mpc' + action=0）对齐 mpc_ur5_grasp.py 基线。

架构（混合残差）：v_servo = v_nominal(t) + Δv；action=0 → Δv=0 → v_servo = v_MPC(t)，
等价于完全自主 MPC 控制器（对标 scripts/mpc_ur5_grasp.py 的 delta 完全自主基线），
且走的是与 RL 训练完全相同的环境 step / 夹爪状态机 / 速度执行链路——验证 MPC 标称层
在"真实"环境闭环中的行为（对齐基线的同时，为阶段2 MPC+RL 打底）。

场景（--scene）：
  static    静态目标 + 1 静态障碍（obstacle_vel=0）        —— 对齐 mpc 基线 static 口径
  dyn_target 动态目标 + 无障碍                                —— 对齐 dyn_target 口径
  dyn_both  动态目标 + 1 动态障碍（obstacle_vel=0.05）       —— 对齐消融 dyn_both 口径
  static3   静态目标 + 3 静态障碍（obstacle_count=3）        —— 更多障碍（旧统一逻辑）
  mixed3    静态目标 + 3 障碍混合（obstacle_specs：1 静态 + 2 动态）—— 新增 per-obstacle 规格
  mixed_z   静态目标 + 3 障碍混合（含 z 方向往返运动）       —— 新增 z 方向运动验证

指标与 mpc_ur5_grasp.py 同口径（grasp_success / collision_count / episode_length / solve）。

用法：python3 scripts/mpc_nominal_verify.py --scene dyn_both --n_episodes 30
"""
import os, sys, argparse, json, time
import numpy as np

sys.path.append(os.path.dirname(os.path.abspath(__file__)) + "/..")
from config import get_config
from environment import GraspingEnv

SCENE_CFG = {
    "static":     dict(dyn_target=False, obstacle=("off", 0, 0.0),    specs=None, z=False),   # 对齐基线 static：无障碍
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


class MpcNominalEvaluator:
    """residual 模式 + action=0 → 纯 MPC 标称驱动的闭环评估（指标与 mpc_ur5_grasp.py 同口径）。"""

    def __init__(self, env, scene="dyn_both", max_steps=200):
        self.env = env
        self.scene = scene
        self.max_steps = max_steps

    def run_episode(self):
        env = self.env
        obs, _ = env.reset()
        mpc = env.nominal_trajectory          # MpcNominal 实例（已 attach_env）
        if mpc is not None:
            mpc.reset()                        # 清 warm-start（跨 episode 防护）
        solve_time = 0.0
        fallback = 0
        ep_len = 0
        ep_coll = 0
        info_last = {}
        while ep_len < self.max_steps:
            t0 = time.time()
            action = np.zeros(env.action_space.shape[0], dtype=np.float32)   # 残差=0 → 纯标称
            obs, reward, terminated, truncated, info = env.step(action)
            if mpc is not None:
                solve_time += mpc.solve_time
                fallback = mpc.fallback_cnt
            ep_len += 1
            info_last = info
            ep_coll = max(ep_coll, info.get("obstacle_collision_count", 0))
            if terminated or truncated:
                break
        return {
            "episode_length": ep_len,
            "grasp_success": bool(info_last.get("grasp_success", False)),
            "final_distance": float(info_last.get("distance_to_object", 0.0)),
            "collision_count": int(ep_coll),
            "avg_mpc_solve_s": round(solve_time / max(1, ep_len), 4),
            "mpc_fallback": int(fallback),
        }

    def run(self, n_episodes):
        results = []
        for i in range(n_episodes):
            r = self.run_episode()
            results.append(r)
            print(f"[mpc-nominal] ep {i + 1}/{n_episodes}: len={r['episode_length']} "
                  f"success={r['grasp_success']} coll={r['collision_count']} "
                  f"solve={r['avg_mpc_solve_s']}s fallback={r['mpc_fallback']}")
        n = len(results)
        return {
            "scene": self.scene,
            "n_episodes": n,
            "success_rate": sum(1 for r in results if r["grasp_success"]) / n,
            "collision_rate": sum(1 for r in results if r["collision_count"] > 0) / n,
            "avg_collision_count": float(np.mean([r["collision_count"] for r in results])),
            "avg_episode_length": float(np.mean([r["episode_length"] for r in results])),
            "avg_mpc_solve_s": float(np.mean([r["avg_mpc_solve_s"] for r in results])),
            "episode_results": results,
        }


def main():
    ap = argparse.ArgumentParser(description="纯 MPC 标称层验证（对齐 mpc_ur5_grasp.py 基线）")
    ap.add_argument("--scene", default="dyn_both", choices=list(SCENE_CFG))
    ap.add_argument("--n_episodes", type=int, default=30)
    ap.add_argument("--target-vel", type=float, default=0.05)
    ap.add_argument("--save-results", type=str, default="")
    ap.add_argument("--max-steps", type=int, default=200)
    ap.add_argument("--mpc-v-max", type=float, default=None,
                    help='覆盖 MPC 标称内部速度上限 v_max（默认用 config.mpc_nominal_v_max=0.125，'
                         '即契约速度，与基线 V_MAX 对齐；传其他值可做速度上限扫描）')
    args = ap.parse_args()
    sc = SCENE_CFG[args.scene]

    cfg = get_config()
    g = cfg.grasping
    if args.mpc_v_max is not None:
        g.mpc_nominal_v_max = args.mpc_v_max
    g.render_gui = False
    g.control_mode = "velocity"
    g.action_mode = "residual"        # 残差架构：action=0 → v_servo = v_nominal = v_MPC
    g.nominal_mode = "mpc"            # MPC 标称层
    g.nominal_enabled = True
    g.action_space_dim = 3
    g.use_fixed_position = True       # 固定物体位置（对齐消融口径）
    g.max_steps = args.max_steps

    # ---- 目标/障碍场景 ----
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
        # 多障碍（count>1）：走 _place_obstacles_on_path 的"横向障碍墙"自动布局
        # （防重叠/防初始接触；单障碍保持非 on-path 以对齐基线 dyn_both 口径）。
        g.obstacle_on_nominal_path = (count > 1)
        g.obstacle_specs = None
        g.obstacle_z_motion = sc["z"]
    else:  # specs
        g.obstacle_enabled = True
        g.obstacle_specs = sc["specs"]
        # 位置也走 on-path 自动布局（specs 只定运动行为：type/vel/mode/axis/z_motion）；
        # 避免手动 spec pos 与 cube(0.10,0.42,0.32) 重叠/堵死 hover（D_SAFE=0.20 覆盖 hover 区）。
        g.obstacle_on_nominal_path = True

    env = GraspingEnv(g, cfg.reward)
    ev = MpcNominalEvaluator(env, scene=args.scene, max_steps=args.max_steps)
    res = ev.run(args.n_episodes)

    print("\n========== 纯 MPC 标称验证结果 ==========")
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

