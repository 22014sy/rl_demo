#!/usr/bin/env python3
"""臂身保守度扫描出表（2026-09-20）：成功避障率 vs `arm_margin`。

回答的问题只有一个：**把臂身目标留量调大，成功避障率能提到多高？**

口径 = scripts/run_arm_conservative_sweep.sh（单 harness、seed 12345 逐集配对、
max_steps 200、每臂 --arm-aware、场景/d_safe 固定）。分组**只读结果文件自描述的 config**
（`arm_margin` / `global_planner` / `d_safe` / `scene`），**不解析文件名**。

⚠️ 阅读纪律：
  1. 头条是**成功避障率**（`success_nc` = 抓起成功且零碰撞）。1 cm 余量目标已于
     2026-09-20 取消，本表**不报**任何「离目标差多少」的量——那套框架已作废。
  2. 逐集配 McNemar 精确检验（同 seed → 同初始构型 + 同障碍轨迹）；时长用配对 Wilcoxon。
     决策周期 T = 0.04 s（25 Hz），200 步 = 8.0 s 上限。
  3. `n=30` 的 95% CI（Wilson）宽约 ±15pp —— **不要**把 3–4pp 的差异读成结论。
  4. 碰撞数只计**窄相接触**（MuJoCo 窄相会漏掉深穿透 mesh-sphere 对），故碰撞率**偏宽**；
     各臂同计数器，横向可比，绝对值须带此保留。
  5. 本表只扫了 `arm_margin`。若各档几乎不动，限制项在**权重**（`--w-arm`）或
     `--arm-pad`（障碍几何放大），而不是目标距离——换旋钮，别硬解读。
"""
import argparse
import glob
import json
import os
import sys

import numpy as np
from scipy import stats

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from make_via_table import summarize_via  # noqa: E402
from make_table5 import mcnemar  # noqa: E402

ARM_LABEL = {
    "mpc_nominal": "纯 MPC（Δv=0）",
    "mpc_via": "MPC + 绕行路点（Δv=0）",
}


def load_all(d):
    """扫目录，返回 {(scene, d_safe, arm, arm_margin, arm_pad, w_arm): summary}。"""
    out = {}
    for p in sorted(glob.glob(os.path.join(d, "*.json"))):
        try:
            with open(p, encoding="utf-8") as f:
                j = json.load(f)
        except Exception as e:
            print(f"[跳过] {p}: {e}", file=sys.stderr)
            continue
        c = j.get("config") or {}
        scene, ds = c.get("scene"), c.get("d_safe")
        m = c.get("arm_margin")
        if scene is None or ds is None or m is None:
            print(f"[跳过] {p}: config 缺 scene/d_safe/arm_margin"
                  f"（arm_margin 是 2026-09-20 才进 config 的，旧文件没有 → 请用 runner 重跑）",
                  file=sys.stderr)
            continue
        arm = "mpc_via" if str(c.get("global_planner", "off")) == "via_point" else "mpc_nominal"
        key = (scene, float(ds), arm, float(m), float(c.get("arm_pad", 0.01)),
               float(c.get("w_arm", float("nan"))))
        if key in out:
            print(f"[警告] {p}: 与已有文件同键 {key}，后者被忽略", file=sys.stderr)
            continue
        out[key] = summarize_via(j)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", default="results/arm_conservative")
    ap.add_argument("--md-out", default="")
    args = ap.parse_args()

    T = load_all(args.dir)
    if not T:
        sys.exit(f"[空] {args.dir} 里没有可用的结果文件")

    scenes = sorted({k[0] for k in T})
    dsafes = sorted({k[1] for k in T})
    arms = [a for a in ("mpc_nominal", "mpc_via") if any(k[2] == a for k in T)]
    margins = sorted({k[3] for k in T})

    lines = []

    def out(s=""):
        lines.append(s)
        print(s)

    out("# 臂身保守度扫描：成功避障率 vs `arm_margin`")
    out()
    out(f"场景 `{', '.join(scenes)}`，`d_safe ∈ {{" +
        ", ".join(f"{d:.2f}" for d in dsafes) + "}`，每臂 `--arm-aware`，"
        "seed 12345 逐集配对，`max_steps=200`（T=0.04 s → 8.0 s 上限）。")
    out()
    out("**头条 = 成功避障率（无碰撞且抓起成功）**。1 cm 余量目标已于 2026-09-20 取消，"
        "本表不报任何「离目标差多少」的量。n=30 的 95% CI 宽约 ±15pp —— "
        "**不要把 3–4pp 的差异读成结论**。碰撞数只计窄相接触，绝对值**偏宽**（各臂同计数器，横向可比）。")
    out()

    for scene, ds in [(s, d) for s in scenes for d in dsafes]:
        sel = {k: v for k, v in T.items() if k[0] == scene and k[1] == ds}
        if not sel:
            continue
        n = next(iter(sel.values()))["n"]
        out(f"## 场景 `{scene}`，d_safe = {ds:.2f}，n = {n}")
        out()
        # ⚠️ 退化检测：若本场景**每一臂**都是「碰撞率 100%」且成功避障率全 0，那么二元
        # 「成功避障率」这一列是**场景设计钉死的常数**，不含任何臂间差异，不能用来排序。
        # 实测例（static3）：障碍墙天生贴臂 + hover 落在墙内 → 最后一段必须穿透，
        # 逐集碰撞步数最小 9 步、90 集里 0 集零碰撞。此时只能看 `平均碰撞步数`（越低越好）。
        degen = bool(sel) and all(v["coll_rate"] >= 1.0 for v in sel.values())
        if degen:
            out("> ⚠️ **本场景的「成功避障率」是退化列（恒为 0），不可用于排序。** "
                "每一臂、每一集的碰撞数都 > 0（碰撞率 100%）——障碍贴臂摆放且 hover 落在墙内，"
                "最后一段必须穿透。请改看 **平均碰撞步数**（越低越好）+ **成功率**（不能掉）。")
            out()
        out("| 臂 | `arm_margin` | 成功避障率 (k/n) | 95% CI (Wilson) | 成功率 | "
            "碰撞率 | **平均碰撞步数** | 平均规划时间 | IK 失败 | 绕行采纳率 |")
        out("|---|---|---|---|---|---|---|---|---|---|")
        # 先定出「样本内最优」是哪一格，再逐格打印（避免边打印边更新 best 的自比较）。
        # 排序依据随场景切换：退化场景（碰撞率恒 100%）看**碰撞步数最少**，否则看成功避障率最高。
        if degen:
            best = min(((k[2], k[3]) for k in sel),
                       key=lambda t: next(v["avg_coll"] for k, v in sel.items()
                                          if (k[2], k[3]) == t)) if sel else None
        else:
            best = max(((k[2], k[3]) for k in sel),
                       key=lambda t: next(v["nc_rate"] for k, v in sel.items()
                                          if (k[2], k[3]) == t)) if sel else None
        for arm in arms:
            for m in margins:
                hit = [v for k, v in sel.items() if k[2] == arm and k[3] == m]
                if not hit:
                    out(f"| {ARM_LABEL.get(arm, arm)} | {m:.2f} | *(未产出)* | | | | | | | | |")
                    continue
                r = hit[0]
                star = " ⬅ **最高**" if best and (arm, m) == best[0] else ""
                k = int(r["nc"].sum())
                via = (f"{r['via_rate'] * 100:.0f}%" if r["via_rate"] > 0 else "—")
                out(f"| {ARM_LABEL.get(arm, arm)} | {m:.2f}{star} | "
                    f"**{r['nc_rate'] * 100:.1f}%** ({k}/{n}) | "
                    f"{r['nc_ci'][0] * 100:.1f}–{r['nc_ci'][1] * 100:.1f}% | "
                    f"{r['succ_rate'] * 100:.1f}% | {r['coll_rate'] * 100:.1f}% | "
                    f"{r['avg_coll']:.2f} | {r['mpc_solve_s'] * 1000:.1f} ms/次 | "
                    f"{r['ik_fail_sum']}/{r['ik_call_sum']} | {via} |")
        out()
        if best:
            if degen:
                bv = next(v for k, v in sel.items() if (k[2], k[3]) == best)
                out(f"本场景（退化列，按碰撞步数排）最少碰撞："
                    f"**{ARM_LABEL.get(best[0], best[0])}**，`arm_margin={best[1]:.2f}` → "
                    f"平均 **{bv['avg_coll']:.1f} 步**、成功率 {bv['succ_rate'] * 100:.1f}%。")
            else:
                bv = next(v for k, v in sel.items() if (k[2], k[3]) == best)
                out(f"本场景最高：**{ARM_LABEL.get(best[0], best[0])}**，"
                    f"`arm_margin={best[1]:.2f}` → 成功避障率 **{bv['nc_rate'] * 100:.1f}%**。")
            out(f"⚠️ 这是 n={n} 上的**样本内最优**，不是统计显著的「最优配置」；"
                f"次优档与本档的差落在 Wilson CI 宽度内时（n={n} 约 ±15pp），"
                f"**不要**当作真实排序。")
        out()

        # 配对检验：同臂内，各 margin 相对该臂基线（最小 margin）的逐集变化
        base_m = min(margins)
        out(f"配对检验（同臂内，`arm_margin={base_m:.2f}` 为基线；逐集同 seed）：")
        out()
        out("| 臂 | 对比 | Δ成功避障率 | Δ平均碰撞步数 | McNemar (基线独赢 / 新档独赢, p) | "
            "时长 基线→新档 (步) | Wilcoxon p |")
        out("|---|---|---|---|---|---|---|")
        for arm in arms:
            b = [v for k, v in sel.items() if k[2] == arm and k[3] == base_m]
            if not b:
                continue
            b = b[0]
            for m in margins:
                if m == base_m:
                    continue
                a = [v for k, v in sel.items() if k[2] == arm and k[3] == m]
                if not a:
                    continue
                a = a[0]
                # mcnemar 返回 (b_only, a_only, p) —— 先 b 后 a（make_table5.py:67）
                b_only, a_only, p = mcnemar(a["nc"], b["nc"])
                d = (a["nc_rate"] - b["nc_rate"]) * 100
                # 碰撞步数用**配对 Wilcoxon**报差异显著性（连续量，比二元 McNemar 更有力）；
                # 退化场景里 Δ成功避障率 恒为 0.0pp，只有这一列有信息。
                dc = a["avg_coll"] - b["avg_coll"]
                try:
                    wc = stats.wilcoxon(a["coll"], b["coll"])
                    cpt = f"{wc.pvalue:.2e}"
                except ValueError:
                    cpt = "n/a"     # 全零差
                try:
                    wp = f"{stats.wilcoxon(a['length'], b['length']).pvalue:.2e}"
                except ValueError:
                    wp = "n/a"      # 全零差（逐集完全一致）
                out(f"| {ARM_LABEL.get(arm, arm)} | {base_m:.2f} → {m:.2f} | {d:+.1f}pp | "
                    f"{dc:+.1f} 步 (p={cpt}) | "
                    f"{b_only}/{a_only}, p={p:.4f} | "
                    f"{b['len_mean']:.1f} → {a['len_mean']:.1f} | {wp} |")
        out()

    if args.md_out:
        os.makedirs(os.path.dirname(os.path.abspath(args.md_out)), exist_ok=True)
        with open(args.md_out, "w", encoding="utf-8") as f:
            f.write("\n".join(lines) + "\n")
        print(f"\nsaved -> {args.md_out}")


if __name__ == "__main__":
    main()
