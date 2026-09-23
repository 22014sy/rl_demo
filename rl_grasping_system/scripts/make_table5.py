#!/usr/bin/env python3
"""5 方法一张表（2026-09-17）：纯跟踪伺服 / 跟踪伺服+RL / RL / MPC / MPC+RL。

指标 = 无碰撞成功抓取率（grasp_success 且整集 obstacle_collision_count==0）+ 时长（决策步数）。
口径 = scripts/run_table5_n200_armaware.sh（dyn_both_train / seed 12345 逐集配对 / max_steps 200 /
ori-servo off / MPC 两臂 --arm-aware）。

配对检验：逐集同一 seed → 同一初始构型+同一障碍轨迹，
  成功率类（二值）用 McNemar 精确检验；时长（连续）用配对 Wilcoxon 符号秩 + 配对 t。
时长换算：决策周期 T = action_repeat(2) × 0.02 s = 0.04 s（25 Hz），200 步 = 8.0 s 上限。
"""
import json, os, sys, argparse
import numpy as np
from scipy import stats

T_DECISION = 0.04     # 决策周期 s

ARMS = [
    ("纯跟踪伺服",        "vf_nominal",   "速度场标称 Δv=0"),
    ("跟踪伺服+RL",       "vf_res18",     "速度场标称 + v18_p3f 残差（native per_axis）"),
    ("RL",                "e2e_v11",      "端到端 PPO v11_500k（delta，无标称）"),
    ("MPC",               "mpc_nominal",  "MPC 标称 Δv=0 + arm-aware"),
    ("MPC+RL",            "mpc_res25b2",  "MPC 标称 + v25b2 残差（modulus）+ arm-aware"),
]
# 对照臂：同场景同 seed、无 arm-aware（= results/unified7_trainmatch_n200/ 口径，逐项相同）
CONTROLS = {
    "mpc_nominal":  ("unified7_trainmatch_n200/mpc_nominal_dyn_both_train_trainmatch_servooff_n200.json",
                     "MPC（arm-aware off）"),
    "mpc_res25b2":  ("unified7_trainmatch_n200/mpc_res25b2_dyn_both_train_trainmatch_servooff_n200.json",
                     "MPC+RL（arm-aware off）"),
}


def wilson(k, n, z=1.96):
    if n == 0:
        return (0.0, 0.0)
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return (max(0.0, c - h), min(1.0, c + h))


def load(path):
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def summarize(d):
    n = d['n_episodes']
    eps = d['episode_results']
    succ = np.array([e['grasp_success'] for e in eps], dtype=bool)
    coll = np.array([e['collision_count'] for e in eps])
    nc = succ & (coll == 0)
    ln = np.array([e['episode_length'] for e in eps], dtype=float)
    k = int(nc.sum())
    lo, hi = wilson(k, n)
    return dict(n=n, succ=succ, coll=coll, nc=nc, length=ln,
                nc_rate=k / n, nc_ci=(lo, hi),
                succ_rate=float(succ.mean()),
                coll_rate=float((coll > 0).mean()),
                avg_coll=float(coll.mean()),
                len_mean=float(ln.mean()), len_std=float(ln.std(ddof=1)),
                len_ci=float(1.96 * ln.std(ddof=1) / np.sqrt(n)))


def mcnemar(a_nc, b_nc):
    """精确 McNemar：a 相对 b 的 nc 变化。返回 (b_only, a_only, p)。"""
    b = int((~a_nc & b_nc).sum())   # 只有 b 成功
    c = int((a_nc & ~b_nc).sum())   # 只有 a 成功
    n = b + c
    if n == 0:
        return b, c, 1.0
    p = float(stats.binomtest(min(b, c), n, 0.5).pvalue) * 2
    return b, c, min(1.0, p)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", default="results/table5_n200_armaware")
    ap.add_argument("--scene", default="dyn_both_train")
    ap.add_argument("--n", type=int, default=200)
    ap.add_argument("--md-out", default="")
    ap.add_argument("--servo", default="off", choices=("off", "on"),
                    help="本表的 ori-servo 口径，只影响标题（结果文件的 config 才是权威）")
    ap.add_argument("--control", action="append", default=[], metavar="TAG=PATH=LABEL",
                    help="对照臂，可重复。不给则用内置两条（results/unified7_trainmatch_n200 的 "
                         "arm-aware off，对应 --ori-servo off 口径）")
    args = ap.parse_args()

    R, missing = {}, []
    for label, tag, _ in ARMS:
        p = os.path.join(args.dir, f"{tag}_{args.scene}_n{args.n}.json")
        if not os.path.exists(p):
            missing.append(p)
            continue
        R[tag] = summarize(load(p))
    if missing:
        print("[缺失] 以下结果文件还没产出，表不完整：", file=sys.stderr)
        for m in missing:
            print("   ", m, file=sys.stderr)
        if not R:
            sys.exit(1)

    if args.control:
        specs = []
        for s in args.control:
            parts = s.split("=", 2)
            if len(parts) != 3:
                sys.exit(f"--control 需要 MAIN_TAG=PATH=LABEL 三段，收到: {s!r}")
            specs.append(tuple(parts))
    else:
        specs = [(tag, os.path.join("results", rel), clabel)
                 for tag, (rel, clabel) in CONTROLS.items()]

    C, cspec = {}, {}
    for tag, path, clabel in specs:
        if os.path.exists(path):
            C[tag] = summarize(load(path))
            cspec[tag] = clabel
        else:
            print(f"[缺失] 对照臂 {clabel}: {path}", file=sys.stderr)

    lines = []
    def out(s=""):
        lines.append(s)
        print(s)

    n_used = next(iter(R.values()))['n']
    out(f"# 五方法对比（{args.scene}，seed 12345 逐集配对，n={n_used}，max_steps=200，ori-servo {args.servo}）")
    out()
    out("MPC 两臂开启臂身碰撞检测（`--arm-aware`）；决策周期 T=0.04 s（25 Hz），故 200 步 = 8.0 s。")
    out()
    out("| 方法 | 无碰撞成功抓取率 | 95% CI (Wilson) | 平均时长 (步) | 95% CI | 时长 (s) | 成功率 | 碰撞率 | 平均碰撞 |")
    out("|---|---|---|---|---|---|---|---|---|")
    for label, tag, _ in ARMS:
        if tag not in R:
            out(f"| {label} | *(未产出)* | | | | | | | |")
            continue
        r = R[tag]
        out(f"| {label} | **{r['nc_rate'] * 100:.1f}%** ({int(r['nc'].sum())}/{r['n']}) | "
            f"{r['nc_ci'][0] * 100:.1f}–{r['nc_ci'][1] * 100:.1f}% | "
            f"{r['len_mean']:.1f} | ±{r['len_ci']:.1f} | {r['len_mean'] * T_DECISION:.2f} | "
            f"{r['succ_rate'] * 100:.1f}% | {r['coll_rate'] * 100:.1f}% | {r['avg_coll']:.2f} |")

    if C:
        out()
        out("**arm-aware 对照（同场景同 seed，其余口径逐项相同）**：")
        out()
        out("| 臂 | 无碰撞成功率 (arm-aware on) | 无碰撞成功率 (arm-aware off) | 配对变化 | McNemar p | 平均时长 on → off |")
        out("|---|---|---|---|---|---|")
        for tag, clabel in cspec.items():
            if tag not in R or tag not in C:
                continue
            a, b = R[tag], C[tag]
            b_only, a_only, p = mcnemar(a['nc'], b['nc'])
            out(f"| {clabel} | {a['nc_rate'] * 100:.1f}% | {b['nc_rate'] * 100:.1f}% | "
                f"+{a_only}/−{b_only} | {p:.3f} | {a['len_mean']:.1f} → {b['len_mean']:.1f} |")

    out()
    out("## 逐对配对检验（无碰撞成功率 = McNemar 精确；时长 = 配对 Wilcoxon / 配对 t）")
    out()
    out("| A | B | Δ无碰撞成功率 | McNemar (A 独赢 / B 独赢, p) | 时长 A→B (步) | Wilcoxon p | 配对 t p |")
    out("|---|---|---|---|---|---|---|")
    tags = [t for _, t, _ in ARMS if t in R]
    labels = {t: l for l, t, _ in ARMS}
    for i in range(len(tags)):
        for j in range(i + 1, len(tags)):
            a, b = R[tags[i]], R[tags[j]]
            a_only, b_only, p = mcnemar(a['nc'], b['nc'])
            d = (a['nc_rate'] - b['nc_rate']) * 100
            w = stats.wilcoxon(a['length'], b['length'])
            tt = stats.ttest_rel(a['length'], b['length'])
            out(f"| {labels[tags[i]]} | {labels[tags[j]]} | {d:+.1f}pp | "
                f"{a_only}/{b_only}, p={p:.4f} | {a['len_mean']:.1f} → {b['len_mean']:.1f} | "
                f"p={w.pvalue:.2e} | p={tt.pvalue:.2e} |")

    if args.md_out:
        with open(args.md_out, "w", encoding="utf-8") as f:
            f.write("\n".join(lines) + "\n")
        print(f"\nsaved -> {args.md_out}")


if __name__ == "__main__":
    main()
