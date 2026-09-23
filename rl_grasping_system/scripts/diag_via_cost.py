#!/usr/bin/env python3
"""绕行层耗时定位：为什么某一集的 screening 会跑到 5 小时？

背景（2026-09-19）：static3 30 集预检里 29 集 screening 3.7–13.4 s，唯独 ep23（seed 12368）
**18536 s**（5.1 小时）。这不是「慢一点」，是量级跳变，必须定位到具体调用而不是猜。

做法：把 `ViaPointNominal` 的四个几何入口（`_seg_min_gap` / `_gap_at` / `_path_gap_fast` /
`_margins_exact`）包一层插桩，记录
  - 调用次数、累计墙钟、单次最大墙钟
  - 每次调用的输入关节距离 |dq| 与由 `n = ceil(|dq|/edge_res)` 推出的**采样点数**
  - 每次调用实际发出的 `mj_geomDistance` 次数
并给采样点数加**硬上限** `--cap-n`（默认 20000），命中即计数并截断——这样探针本身不会再挂住，
同时「谁把 n 撑爆了」会直接印出来。

用法（必须在 rl_grasping_system/ 下跑）：
    python3 scripts/diag_via_cost.py --seed 12368              # 复现 ep23
    python3 scripts/diag_via_cost.py --seed 12345 --cap-n 3000 # 正常集做对照
"""
import argparse
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import mujoco  # noqa: E402
import via_point_nominal as vp  # noqa: E402
from check_via_point import build_env  # noqa: E402


class Rec:
    def __init__(self, name):
        self.name = name
        self.n_calls = 0
        self.t_total = 0.0
        self.t_max = 0.0
        self.gd_calls = 0
        self.max_n = 0
        self.max_dq = 0.0
        self.capped = 0
        self.nan = 0
        self.worst = []          # 命中上限的 (n, |dq|, q_a[0], q_b[0])，最多留 5 条


STATS = {}


def stat(name):
    return STATS.setdefault(name, Rec(name))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--scene", default="static3")
    ap.add_argument("--seed", type=int, default=12368, help="ep23 = base 12345 + 23")
    ap.add_argument("--cap-n", type=int, default=20000,
                    help="单次 `_seg_min_gap` 的采样点数上限（防再挂住）；命中会记账")
    args = ap.parse_args()

    env, g = build_env(args.scene)
    env.reset(seed=args.seed)
    nt = env.nominal_trajectory
    print(f"[env] scene={args.scene} seed={args.seed} planner={g.global_planner} "
          f"edge_res={nt.edge_res} grid={nt.grid} step={nt.step} topk={nt.topk} "
          f"score_points={nt.score_points}")

    # ---------- 插桩：_seg_min_gap ----------
    orig_seg = vp.ViaPointNominal._seg_min_gap

    def seg(self, q_a, q_b, early_stop=0.0):
        s = stat("_seg_min_gap")
        dq = np.asarray(q_b, float) - np.asarray(q_a, float)
        nrm = float(np.linalg.norm(dq))
        s.n_calls += 1
        if not np.isfinite(nrm):
            s.nan += 1
            return float("inf")
        n = max(1, int(np.ceil(nrm / self.edge_res)))
        s.max_dq = max(s.max_dq, nrm)
        s.max_n = max(s.max_n, n)
        if n > args.cap_n:
            # 命中上限：**不再下钻**，直接返回。探针的目的是「谁把 n 撑爆了」，
            # 而不是再花 5 小时把那一集真跑完（真耗时已知 ≈18536 s）。
            s.capped += 1
            s.worst.append((n, nrm, float(q_a[0]), float(q_b[0])))
            s.worst = sorted(s.worst, key=lambda t: -t[0])[:5]
            return 0.0
        t0 = time.time()
        out = orig_seg(self, q_a, q_b, early_stop)
        dt = time.time() - t0
        s.t_total += dt
        s.t_max = max(s.t_max, dt)
        return out

    vp.ViaPointNominal._seg_min_gap = seg

    # ---------- 插桩：_gap_at ----------
    orig_gap = vp.ViaPointNominal._gap_at

    def gap(self, q, geoms, distmax):
        s = stat("_gap_at")
        s.n_calls += 1
        t0 = time.time()
        out = orig_gap(self, q, geoms, distmax)
        dt = time.time() - t0
        s.t_total += dt
        s.t_max = max(s.t_max, dt)
        return out

    vp.ViaPointNominal._gap_at = gap

    # ---------- 插桩：_path_gap_fast ----------
    orig_pgf = vp.ViaPointNominal._path_gap_fast

    def pgf(self, q_now, q_via, q_hover):
        s = stat("_path_gap_fast")
        s.n_calls += 1
        t0 = time.time()
        out = orig_pgf(self, q_now, q_via, q_hover)
        dt = time.time() - t0
        s.t_total += dt
        s.t_max = max(s.t_max, dt)
        return out

    vp.ViaPointNominal._path_gap_fast = pgf

    # ---------- 插桩：_margins_exact ----------
    orig_me = vp.ViaPointNominal._margins_exact

    def me(self, q_now, q_via, q_hover):
        s = stat("_margins_exact")
        s.n_calls += 1
        t0 = time.time()
        out = orig_me(self, q_now, q_via, q_hover)
        dt = time.time() - t0
        s.t_total += dt
        s.t_max = max(s.t_max, dt)
        return out

    vp.ViaPointNominal._margins_exact = me

    # ---------- 跑一次规划入口 ----------
    ee = env._get_end_effector_position().copy()
    tgt = env._get_object_position().copy()
    t0 = time.time()
    nt.reference_velocity(ee, tgt, 0.04)
    wall = time.time() - t0

    print(f"\n[done] 一次 screening 墙钟 {wall:.2f} s | 采纳路点={nt._via_ee is not None} "
          f"候选={nt.cand_total} 过={nt.cand_pass} IK失败={nt.ik_fail_cnt}")
    if nt._via_ee is not None:
        print(f"       余量 via={nt.best_margin_m:+.4f} seg1={nt.margin_seg1:+.4f} "
              f"seg2={nt.margin_seg2 if nt.margin_seg2 is None else '%+.4f' % nt.margin_seg2}")
    print(f"       模块自报 screening 耗时 {nt.screen_time_total:.2f} s"
          f"（cap_n={args.cap_n}，命中上限 {stat('_seg_min_gap').capped} 次）")

    print("\n==== 按累计墙钟排序 ====")
    for s in sorted(STATS.values(), key=lambda r: -r.t_total):
        print(f"  {s.name:18s} 调用 {s.n_calls:7d}  累计 {s.t_total:10.2f} s  "
              f"单次最大 {s.t_max:9.3f} s")
        if s.name == "_seg_min_gap":
            print(f"  {'':18s} |dq| 最大 {s.max_dq:.4f} rad  n 最大 {s.max_n}  "
                  f"非有限 {s.nan}  超上限 {s.capped}")
            for n, dq, a0, b0 in s.worst:
                print(f"  {'':18s}   ← n={n} (|dq|={dq:.4f})  q_a[0]={a0:.4f} q_b[0]={b0:.4f}")


if __name__ == "__main__":
    main()
