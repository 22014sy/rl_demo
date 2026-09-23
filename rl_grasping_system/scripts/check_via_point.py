"""绕行路点层（ViaPointNominal）预检——**接入评测循环之前**必须先跑这一步。

它回答四个问题（对应方案 §验证 1）：
  1. 候选集里有多少个能过 IK？多少个能过第一段可执行性检验？
  2. 选中的路点，**达成的最小余量**是多少？（1 cm 是软目标，如实报达成值）
  3. 单次 screening 要多久？（决定 4 臂 × 3 场景 × 2 档位的机时预算）
  4. 无障碍场景下是否**逐位退化**（回归护栏）？

口径纪律：
  - 达成余量用两条**独立**路径各算一遍：本模块的 `_path_gap_exact`（scratch 数据）与
    `env._min_arm_obstacle_gap()`（真实数据 + qpos 存还）。两条不一致就说明实现有问题，
    不能只信模块自己的数（自证）。
  - 场景构造与 `scripts/mpc_plus_rl_eval.py` 的 SCENE_CFG **完全一致**，seed 序列与评测器
    的 `--seed-per-episode` 一致（base + i）——否则量的是另一个场景。
  - 桌面不在判据内（沿用 `_arm_obstacle_geoms()`，排除 table），与既有口径一致。

用法（必须在 rl_grasping_system/ 下跑，模型是相对路径）：
    python3 scripts/check_via_point.py --scene static3 --n 30
    python3 scripts/check_via_point.py --scene static3 --grid-curve
    python3 scripts/check_via_point.py --scene static --n 3      # 回归护栏
"""
import argparse
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import mujoco  # noqa: E402
from config import GraspingConfig, RewardConfig  # noqa: E402
from environment import GraspingEnv  # noqa: E402
from via_point_nominal import _sample_seg  # noqa: E402
from mpc_plus_rl_eval import SCENE_CFG  # noqa: E402


def build_env(scene, grid=None, step=None):
    """按评测器同口径构造环境（SCENE_CFG），可选覆盖 via_grid / via_step。"""
    g = GraspingConfig()
    g.action_mode = "residual"
    g.nominal_mode = "mpc"
    g.global_planner = "via_point"
    if grid is not None:
        g.via_grid = int(grid)
    if step is not None:
        g.via_step = float(step)
    sc = SCENE_CFG[scene]
    g.dynamic_target_enabled = sc["dyn_target"]
    g.target_vel_xy = 0.0
    kind, count, vel = sc["obstacle"]
    if kind == "off":
        g.obstacle_enabled, g.obstacle_count, g.obstacle_specs = False, 0, None
    elif kind == "legacy":
        g.obstacle_enabled, g.obstacle_count, g.obstacle_vel = True, count, vel
        g.obstacle_on_nominal_path = (count > 1)
        g.obstacle_specs = None
        g.obstacle_z_motion = sc["z"]
    else:
        g.obstacle_enabled, g.obstacle_specs = True, sc["specs"]
        g.obstacle_on_nominal_path = True
    env = GraspingEnv(g, RewardConfig())
    return env, g


def independent_margin(env, q_now, q_via, q_hover, n_points=24):
    """用**环境自己的** `_min_arm_obstacle_gap()` 复算三段余量（独立于模块实现）。

    临时把关节角写进真实 data，算完立即还原（qpos 存还 + mj_forward），不留副作用。
    返回 (via, seg1, seg2)，口径与模块的 `_margins_exact` 对齐（同 geom 集合、distmax=1.0）。
    """
    save = env.data.qpos.copy()
    try:
        def gap_at(q):
            env.data.qpos[env.arm_joint_ids] = q
            mujoco.mj_forward(env.model, env.data)
            return env._min_arm_obstacle_gap()

        via = gap_at(q_via)
        seg1 = min(gap_at(q) for q in _sample_seg(q_now, q_via, n_points))
        seg2 = (None if q_hover is None
                else min(gap_at(q) for q in _sample_seg(q_via, q_hover, n_points)))
        return via, seg1, seg2
    finally:
        env.data.qpos[:] = save
        mujoco.mj_forward(env.model, env.data)


def probe_episode(env, seed):
    """reset 一集 → 走一次公开规划入口 → 读出 screening 结果 + 独立复检。"""
    env.reset(seed=seed)
    nt = env.nominal_trajectory
    nt.reset()
    ee = env._get_end_effector_position().copy()
    tgt = env._get_object_position().copy()
    # screen_time_total 是**跨集累计**（供评测器差分用），这里必须自己取增量，
    # 否则打印出来的是「从第一集到本集的累计耗时」，看着像每集越来越慢。
    scr0 = float(getattr(nt, "screen_time_total", 0.0) or 0.0)
    ikf0 = int(getattr(nt, "ik_fail_cnt", 0) or 0)
    t0 = time.time()
    nt.reference_velocity(ee, tgt, 0.04)
    dt = time.time() - t0
    r = {
        "planned": nt._via_ee is not None,
        "wall_s": dt, "screen_s": float(getattr(nt, "screen_time_total", 0.0) or 0.0) - scr0,
        "cand": nt.cand_total, "pass": nt.cand_pass,
        "ik_fail": int(getattr(nt, "ik_fail_cnt", 0) or 0) - ikf0,   # 同样取增量
        "via": nt.best_margin_m, "seg1": nt.margin_seg1, "seg2": nt.margin_seg2,
        "total": nt.margin_total, "fast": nt.margin_fast,
        "seg2_contact": nt.seg2_contact,
        "ind_via": None, "ind_seg1": None, "ind_seg2": None,
    }
    if nt._via_ee is not None and nt._cache_passers:
        q_now = env.data.qpos[env.arm_joint_ids].copy()
        q_via = None
        for c in nt._cache_passers:      # 采纳的未必是 passers[0]（缓存命中会顺延）→ 按 FK 反查
            if float(np.linalg.norm(nt._fk(c["q"]) - nt._via_ee)) < 1e-9:
                q_via = c["q"]
                break
        if q_via is not None:
            a, b, c2 = independent_margin(env, q_now, q_via, nt._cache_q_hover)
            r["ind_via"], r["ind_seg1"], r["ind_seg2"] = a, b, c2
    return r


def summarize(rows, scene, margin_target):
    ok = [r for r in rows if r.get("planned")]
    print("\n==== %s 汇总（n=%d 集）====" % (scene, len(rows)))
    print("  采纳了路点的集数        : %d/%d" % (len(ok), len(rows)))
    if not rows:
        return
    scr = [r["screen_s"] for r in rows if r.get("screen_s")]
    if scr:
        print("  screening 次数/耗时     : %d 次，中位 %.2f s，合计 %.1f s"
              % (len(scr), float(np.median(scr)), float(np.sum(scr))))
    if not ok:
        print("  ⚠️ 一条可用路点都没有 → via 层全退化为纯 MPC（这就是结论，如实上报）")
        return
    med = lambda k: float(np.median([r[k] for r in ok if r[k] is not None]))
    print("  候选数(过预筛)         : 中位 %.0f" % med("cand"))
    print("  通过第一段的候选数     : 中位 %.0f" % med("pass"))
    print("  筛选期 IK 失败         : 中位 %.0f" % med("ik_fail"))

    def line(name, key, ikey=None):
        v = np.array([r[key] for r in ok if r[key] is not None], dtype=float)
        if v.size == 0:
            print("  %-22s : 无数据" % name)
            return None
        s = ("  %-22s : min %+.4f  中位 %+.4f  max %+.4f  ≥%dmm: %d/%d"
             % (name, v.min(), float(np.median(v)), v.max(),
                margin_target * 1000, int((v >= margin_target).sum()), v.size))
        if ikey is not None:
            iv = np.array([r[ikey] for r in ok if r[ikey] is not None], dtype=float)
            if iv.size == v.size:
                s += "  | env 独立最大偏差 %.1e %s" % (
                    float(np.max(np.abs(iv - v))),
                    "OK" if float(np.max(np.abs(iv - v))) < 1e-6 else "⚠️ 不一致")
        print(s)
        return v

    line("路点位形余量 via", "via", "ind_via")
    line("第一段余量 seg1", "seg1", "ind_seg1")
    line("第二段余量 seg2", "seg2", "ind_seg2")
    line("整条代理路径 total", "total")
    s2 = [r["seg2_contact"] for r in ok]
    print("  采纳路点第二段穿透      : %d/%d（软指标，未作门控）"
          % (int(np.sum(s2)), len(s2)))
    print("  说明：seg2 的终点是内层 hover，其位形在多数集里本就泡在墙内（与绕行选点无关）；"
          "可归因于本层的是 via 与 seg1。")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--scene", default="static3", choices=list(SCENE_CFG.keys()))
    ap.add_argument("--n", type=int, default=30, help="集数（seed = 12345+i，与评测器一致）")
    ap.add_argument("--base-seed", type=int, default=12345)
    ap.add_argument("--grid", type=int, default=None, help="覆盖 via_grid")
    ap.add_argument("--step", type=float, default=None, help="覆盖 via_step")
    ap.add_argument("--margin-target", type=float, default=0.01)
    ap.add_argument("--grid-curve", action="store_true",
                    help="额外扫 via_grid ∈ {4,6,8}（每档只跑首集，看达成余量曲线）")
    ap.add_argument("--quiet", action="store_true")
    args = ap.parse_args()

    env, g = build_env(args.scene, args.grid, args.step)
    rows = []
    t0 = time.time()
    for i in range(args.n):
        r = probe_episode(env, args.base_seed + i)
        rows.append(r)
        if not args.quiet:
            if r.get("planned"):
                print("ep %2d | 候选 %3d 过 %3d IKfail %3d | via %+.4f seg1 %+.4f "
                      "seg2 %s total %+.4f | 独立 via %+.4f seg1 %+.4f | screen %.2fs"
                      % (i, r["cand"], r["pass"], r["ik_fail"], r["via"], r["seg1"],
                         ("%+.4f" % r["seg2"]) if r["seg2"] is not None else "  n/a ",
                         r["total"], r["ind_via"] if r["ind_via"] is not None else float("nan"),
                         r["ind_seg1"] if r["ind_seg1"] is not None else float("nan"),
                         r["screen_s"]))
            else:
                print("ep %2d | 无可用路点（退化内层） | screen %.2fs" % (i, r["screen_s"]))
    elapsed = time.time() - t0
    print("\n总耗时 %.1f s（%.1f s/集）" % (elapsed, elapsed / max(1, args.n)))
    summarize(rows, args.scene, args.margin_target)

    if args.grid_curve:
        print("\n==== via_grid 达成余量曲线（首集；改 grid 会改候选集大小）====")
        for gv in (4, 6, 8):
            e2, _ = build_env(args.scene, grid=gv, step=args.step)
            r = probe_episode(e2, args.base_seed)
            n_cand = (2 * gv + 1) ** 2 * 3
            print("  via_grid=%d (网格 %d 点) | 候选 %s 过 %s | via %s seg1 %s | 独立 via %s "
                  "| screen %s"
                  % (gv, n_cand, r.get("cand"), r.get("pass"),
                     ("%+.4f" % r["via"]) if r.get("planned") else "无",
                     ("%+.4f" % r["seg1"]) if r.get("planned") else "无",
                     ("%+.4f" % r["ind_via"]) if r.get("ind_via") is not None else "n/a",
                     "%.2f s" % r.get("screen_s", 0.0)))


if __name__ == "__main__":
    main()
