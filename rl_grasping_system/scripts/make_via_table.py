#!/usr/bin/env python3
"""绕行路点对照表（2026-09-19）：纯 MPC / MPC+绕行路点 / MPC+RL / MPC+绕行+RL。

口径 = scripts/run_via_compare.sh（单 harness、seed 12345 逐集配对、max_steps 200、
MPC 臂一律 --arm-aware、每档 d_safe 独立成表）。结果文件**自描述**：脚本只按
`config.tag / config.scene / config.d_safe` 分组，不解析文件名。

配对检验：逐集同一 seed → 同一初始构型 + 同一障碍轨迹，
  成功率类（二值）用 McNemar 精确检验；时长（连续）用配对 Wilcoxon + 配对 t。
时长换算：决策周期 T = 0.04 s（25 Hz），200 步 = 8.0 s 上限。

⚠️ 阅读纪律（写进表里，防止误读）：
  0. **头条指标 = 成功避障率（success_nc，无碰撞且抓起成功）**。2026-09-20 用户决定取消
     「1 cm 安全余量」这一要求（实测在本场景族几何不可行），目标改为「成功避障率尽量高」，
     故表中不再有任何「离目标值还差多少」的表述。达成余量降为**诊断列**，只报实测值。
  1. 「达成的最小余量」是**关节插值代理路径**上的 mj_geomDistance 值，**不是**执行轨迹上的
     实测余量。它**不是**达标判据，只用来回答「绕行层到底绕开了多少」。
  2. 障碍碰撞数只统计**窄相接触**（environment.py 的 obstacle_collision_count）。MuJoCo 窄相
     会漏掉深穿透的 mesh-sphere 对（实测有「接触表为空但 mj_geomDistance 报 −55 mm」的位形），
     故所有碰撞率都是**偏宽**的。各臂用同一个计数器，横向可比；绝对值需带此保留读。
  3. 同一场景的两档 d_safe 结论**不可互相外推**（0.20 档下 MPC 自身避障接近空操作）。
"""
import argparse
import glob
import json
import os
import sys

import numpy as np
from scipy import stats

# 复用既有统计口径，避免两套实现漂移（wilson/summarize/mcnemar 与五方法表逐字同源）
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from make_table5 import wilson, summarize, mcnemar  # noqa: E402

T_DECISION = 0.04

ARMS = [
    ("mpc_nominal",      "纯 MPC（Δv=0）"),
    ("mpc_via",          "MPC + 绕行路点（Δv=0）"),
    ("mpc_res25b2",      "MPC + RL（v25b2）"),
    ("mpc_via_res25b2",  "MPC + 绕行路点 + RL（v25b2）"),
]
LABEL = dict(ARMS)

# 关注的配对（A 相对 B 的变化），比全 6 对更贴问题：
#   ① 加绕行有没有用（无 RL）      ② 加绕行有没有用（有 RL）
#   ③ 有绕行后 RL 还有没有增量     ④ 无绕行时 RL 的增量（基线对照）
PAIRS = [
    ("mpc_via", "mpc_nominal",     "加绕行路点（无 RL 时）"),
    ("mpc_via_res25b2", "mpc_res25b2", "加绕行路点（有 RL 时）"),
    ("mpc_via_res25b2", "mpc_via",     "加 RL（有绕行时）"),
    ("mpc_res25b2", "mpc_nominal",     "加 RL（无绕行时）"),
]


def _mean_opt(eps, key):
    v = [e[key] for e in eps if e.get(key) is not None]
    return float(np.mean(v)) if v else float("nan")


def _min_opt(eps, key):
    v = [e[key] for e in eps if e.get(key) is not None]
    return float(np.min(v)) if v else float("nan")


def _median_opt(eps, key):
    v = [e[key] for e in eps if e.get(key) is not None]
    return float(np.median(v)) if v else float("nan")


def summarize_via(d):
    """在 make_table5.summarize 的基础上，补绕行路点/IK/规划时间三项。"""
    s = summarize(d)
    eps = d["episode_results"]
    s["via_rate"] = float(np.mean([bool(e.get("via_planned")) for e in eps]))
    s["via_screen_sum"] = int(np.sum([int(e.get("via_screen_cnt", 0) or 0) for e in eps]))
    s["ik_fail_sum"] = int(np.sum([int(e.get("ik_fail_count", 0) or 0) for e in eps]))
    s["ik_call_sum"] = int(np.sum([int(e.get("ik_call_count", 0) or 0) for e in eps]))
    s["via_plan_s_mean"] = float(np.mean([float(e.get("via_plan_time_s", 0.0) or 0.0) for e in eps]))
    # 规划时间用 ratio-of-sums（Σ总耗时 / Σ次数），与评测器 aggregate 同口径：
    # 逐集均值的均值会给求解次数少的集过高权重。
    st = float(np.sum([float(e.get("mpc_solve_s_total", 0.0) or 0.0) for e in eps]))
    sc = int(np.sum([int(e.get("mpc_solve_cnt", 0) or 0) for e in eps]))
    s["mpc_solve_s"] = st / sc if sc > 0 else float("nan")
    s["mpc_solve_total"] = st
    s["mpc_solve_cnt"] = sc
    # 达成余量：只对**真的规划出路点**的集统计（否则均值被 None 稀释）
    s["mv_via"] = _mean_opt(eps, "via_margin_via_m")
    s["mv_seg1"] = _mean_opt(eps, "via_margin_seg1_m")
    s["mv_total"] = _mean_opt(eps, "via_margin_total_m")
    s["mm_seg1"] = _min_opt(eps, "via_margin_seg1_m")
    s["med_seg1"] = _median_opt(eps, "via_margin_seg1_m")
    return s


def mm(x):
    """米 → '+12.3 mm'；NaN → 'n/a'（该臂没规划路点时）。"""
    return "n/a" if x is None or (isinstance(x, float) and np.isnan(x)) else "%+.1f mm" % (x * 1000)


def load_all(d):
    """扫目录，按 (scene, d_safe, tag) 建表——只信结果文件自己的 config 段。"""
    table, seen = {}, []
    for p in sorted(glob.glob(os.path.join(d, "*.json"))):
        try:
            with open(p, encoding="utf-8") as f:
                j = json.load(f)
        except Exception as e:
            print(f"[跳过] {p}: {e}", file=sys.stderr)
            continue
        c = j.get("config") or {}
        tag, scene = c.get("tag"), c.get("scene")
        if not tag or not scene:
            print(f"[跳过] {p}: config 缺 tag/scene（旧结果文件没有 tag，请用 runner 重跑）",
                  file=sys.stderr)
            continue
        ds = float(c.get("d_safe", float("nan")))
        table[(scene, ds, tag)] = summarize_via(j)
        seen.append((scene, ds, tag, p))
    return table, seen


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", default="results/via_compare")
    ap.add_argument("--md-out", default="")
    ap.add_argument("--margin-target", type=float, default=None,
                    help="已作废（1 cm 目标已取消），保留仅为兼容旧命令行，传了也不起作用")
    args = ap.parse_args()

    table, seen = load_all(args.dir)
    if not table:
        sys.exit(f"[空] {args.dir} 里没有带 config.tag 的结果文件")

    keys = sorted({(s, ds) for (s, ds, _t) in table}, key=lambda k: (k[0], k[1]))
    lines = []

    def out(s=""):
        lines.append(s)
        print(s)

    out("# 绕行路点对照（seed 12345 逐集配对，max_steps=200，T=0.04 s → 8.0 s 上限）")
    out()
    out("四臂：纯 MPC / MPC+绕行路点 / MPC+RL / MPC+绕行路点+RL。MPC 各臂一律开 `--arm-aware`。")
    out()
    out("**读表纪律**：① 头条指标是**成功避障率**（无碰撞且抓起成功）；「达成最小余量」是"
        "**关节插值代理路径**上的精确表面距，不是执行轨迹实测值，且**不作为达标判据**"
        "（1 cm 目标已于 2026-09-20 取消）——只用来回答「绕行层绕开了多少」。"
        "② 碰撞数只计**窄相接触**，MuJoCo 窄相会漏掉深穿透 mesh-sphere 对，故碰撞率**偏宽**；"
        "各臂同计数器，横向可比，绝对值须带此保留。③ 同一场景两档 `d_safe` 的结论**不可互相外推**。")
    out()

    for scene, ds in keys:
        rows = {t: table[(scene, ds, t)] for t, _ in ARMS if (scene, ds, t) in table}
        if not rows:
            continue
        n_used = next(iter(rows.values()))["n"]
        out(f"## 场景 `{scene}`，d_safe = {ds:.2f}，n = {n_used}")
        out()
        # ⚠️ 退化列检测（2026-09-20 加）：某场景若**每一臂**都碰撞率 100%，则该场景的
        # 「成功避障率」恒为 0，是场景设计钉死的常数，**不含臂间差异**，不能拿来比较。
        # 实测例 static3：障碍墙天生贴臂摆放且 hover 落在墙内 → 最后一段必须穿透，
        # 逐集碰撞步数最小 9 步、90 集里 0 集零碰撞（含最好的 arm-aware 臂）。
        # 此时该场景只能看「平均碰撞」（越低越好），否则会把「四臂并列 0%」误读成「都没用」。
        if rows and all(r["coll_rate"] >= 1.0 for r in rows.values()):
            out("> ⚠️ **本场景「成功避障率」是退化列（四臂恒为 0），不可用于比较。** "
                "所有臂、所有集的碰撞数都 > 0 —— 障碍贴臂摆放且 hover 落在墙内，"
                "最后一段必须穿透。该列在此**不反映任何臂间差异**；"
                "请改看 **平均碰撞**（越低越好）与 **成功率**。")
            out()
        out("| 方法 | **成功避障率** (k/n) | 95% CI (Wilson) | 成功率 | 碰撞率 | 平均碰撞 | "
            "平均规划时间 | IK 失败 | 绕行规划成功率 | 达成余量 via / seg1 |")
        out("|---|---|---|---|---|---|---|---|---|---|")
        for tag, label in ARMS:
            if tag not in rows:
                out(f"| {label} | *(未产出)* | | | | | | | | |")
                continue
            r = rows[tag]
            n = r["n"]
            via_col = ("%.0f%% (%d/%d)" % (r["via_rate"] * 100, int(round(r["via_rate"] * n)), n)
                       if r["via_rate"] > 0 else "—")
            out(f"| {label} | **{r['nc_rate'] * 100:.1f}%** ({int(r['nc'].sum())}/{n}) | "
                f"{r['nc_ci'][0] * 100:.1f}–{r['nc_ci'][1] * 100:.1f}% | "
                f"{r['succ_rate'] * 100:.1f}% | {r['coll_rate'] * 100:.1f}% | {r['avg_coll']:.2f} | "
                f"{r['mpc_solve_s'] * 1000:.1f} ms/次 | {r['ik_fail_sum']}/{r['ik_call_sum']} | "
                f"{via_col} | {mm(r['mv_via'])} / {mm(r['mv_seg1'])} |")

        have_via = [t for t, _ in ARMS if t in rows and rows[t]["via_rate"] > 0]
        if have_via:
            out()
            out("达成余量分布（只统计**真的规划出路点**的集；`via` = 路点位形自身，`seg1` = "
                "起点→路点路径最小）：")
            out()
            out("| 方法 | via 均值 | seg1 均值 | seg1 中位 | seg1 **最小** | 整条代理路径均值 | "
                "screening 总次数 |")
            out("|---|---|---|---|---|---|---|")
            for tag in have_via:
                r = rows[tag]
                out(f"| {LABEL[tag]} | {mm(r['mv_via'])} | {mm(r['mv_seg1'])} | "
                    f"{mm(r['med_seg1'])} | {mm(r['mm_seg1'])} | {mm(r['mv_total'])} | "
                    f"{r['via_screen_sum']} |")
            out()
            out("（该表是**诊断**，不是达标表：1 cm 目标已取消，这里只如实报出绕行层实际绕开了多少。"
                "`seg1` 是本层**硬门控**的那一段，`via` 是路点位形自身；`整条代理路径` 含第二段"
                "（其终点是内层 hover，本身就在墙里，本层只排序、未硬门控），故它偏小**不**等于"
                "「本层没绕开」。静态场景每集只筛选一次是**缓存**的结果，不是「重规划免费」。）")
        else:
            out()
            out("⚠️ 本场景**没有任何一集采纳路点**——绕行层完全退化为纯 MPC，这就是结论本身。")

        out()
        out("配对检验（逐集同 seed）：")
        out()
        out("| 对比 | Δ成功避障率 | McNemar (A 独赢 / B 独赢, p) | 时长 A→B (步) | "
            "Wilcoxon p | 配对 t p |")
        out("|---|---|---|---|---|---|")
        for a_tag, b_tag, desc in PAIRS:
            if a_tag not in rows or b_tag not in rows:
                continue
            a, b = rows[a_tag], rows[b_tag]
            # ⚠️ mcnemar 的返回序是 (b_only, a_only, p)——**先 b 后 a**（见 make_table5.py:67）。
            # 之前按 (a_only, b_only) 接，把「A 独赢/B 独赢」两列**对调**了（2026-09-20 修正）。
            b_only, a_only, p = mcnemar(a["nc"], b["nc"])
            d = (a["nc_rate"] - b["nc_rate"]) * 100
            try:
                w = stats.wilcoxon(a["length"], b["length"])
                wp = f"{w.pvalue:.2e}"
            except ValueError:
                wp = "n/a"      # 全零差（两臂逐集完全一致）
            tt = stats.ttest_rel(a["length"], b["length"])
            out(f"| {desc}<br>{LABEL[a_tag]} vs {LABEL[b_tag]} | {d:+.1f}pp | "
                f"{a_only}/{b_only}, p={p:.4f} | {a['len_mean']:.1f} → {b['len_mean']:.1f} | "
                f"{wp} | {tt.pvalue:.2e} |")
        out()

    if args.md_out:
        with open(args.md_out, "w", encoding="utf-8") as f:
            f.write("\n".join(lines) + "\n")
        print(f"\nsaved -> {args.md_out}")


if __name__ == "__main__":
    main()
