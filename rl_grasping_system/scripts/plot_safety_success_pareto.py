#!/usr/bin/env python3
"""安全-成功 Pareto 前沿图：dyn_both 场景下各控制器 (碰撞率, 成功率)。

读 results/mpc/dyn_both_1obs_dsafe{016,020,024}.json（MPC 不同 D_SAFE 裕度），
叠加 RL（历史消融 1 障 / v23 2 障）与纯标称数据点，验证
「RL 位于 MPC 安全-成功权衡曲线之上」的支配论证。

用法：python3 scripts/plot_safety_success_pareto.py
"""
import json, os, glob
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

BASE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "results")

# ---------- MPC Pareto 点（dyn_both 1 障，n=30） ----------
mpc_pts = []   # (collision_rate, success_rate, d_safe)
for f in sorted(glob.glob(os.path.join(BASE, "mpc", "dyn_both_1obs_dsafe*.json"))):
    try:
        d = json.load(open(f))
        mpc_pts.append((d["collision_rate"], d["success_rate"],
                        float(os.path.basename(f).split("dsafe")[1].split(".")[0]) / 100.0))
    except Exception as e:
        print("skip", f, e)

# 补 D_SAFE=0.20 点（dyn_both_v2.json，早期 1 障结果）
_f20 = os.path.join(BASE, "mpc", "dyn_both_v2.json")
if os.path.exists(_f20):
    try:
        _d = json.load(open(_f20))
        mpc_pts.append((_d["collision_rate"], _d["success_rate"], 0.20))
    except Exception:
        pass

# ---------- 其他控制器数据点（固定 1 障 dyn_both 口径） ----------
others = [
    ("MPC (D_SAFE sweep)", "tab:blue", "o", mpc_pts),
    ("RL residual (1 obs, historical)", "tab:red", "s",
     [(0.2333, 0.8667, 0.0)]),
    ("RL v23 (1 obs)", "tab:red", "D", [(0.1667, 0.8667, 0.0)]),
    ("RL v23 (2 obs)", "tab:red", "v", [(0.3667, 0.80, 0.0)]),
    ("Nominal (1 obs)", "tab:gray", "^", [(0.20, 0.70, 0.0)]),
]

fig, ax = plt.subplots(figsize=(7.5, 5.5))
for label, color, marker, pts in others:
    if not pts:
        continue
    xs = [p[0] * 100 for p in pts]
    ys = [p[1] * 100 for p in pts]
    if "D_SAFE" in label and len(pts) >= 2:
        order = np.argsort(xs)
        xs = [xs[i] for i in order]
        ys = [ys[i] for i in order]
        ax.plot(xs, ys, "--", color=color, alpha=0.5, zorder=1)
    ax.scatter(xs, ys, color=color, marker=marker, s=110, zorder=3, label=label)
    for p, x, y in zip(pts, xs, ys):
        ann = f"D_SAFE={p[2]:.2f}" if "D_SAFE" in label else ""
        if ann:
            ax.annotate(ann, (x, y), textcoords="offset points", xytext=(6, -4),
                        fontsize=8, color=color)
        elif "RL 残差" in label:
            ax.annotate("RL 1 obs", (x, y), textcoords="offset points", xytext=(6, -4), fontsize=8, color=color)
        elif "v23" in label and "2 障" in label:
            ax.annotate("RL v23 2 obs", (x, y), textcoords="offset points", xytext=(8, 6), fontsize=8, color=color)
        elif "v23" in label:
            ax.annotate("RL v23 1 obs", (x, y), textcoords="offset points", xytext=(8, 6), fontsize=8, color=color)
        elif "纯标称" in label:
            ax.annotate("Nominal 1 obs", (x, y), textcoords="offset points", xytext=(6, -4), fontsize=8, color=color)

ax.axhline(70, color="gray", ls=":", lw=0.8, alpha=0.6)
ax.set_xlabel("Collision rate (%)  →  safer")
ax.set_ylabel("Success rate (%)  →  task completion")
ax.set_title("Safety-Success Pareto trade-off (dyn_both: dynamic target + obstacle)")
ax.set_xlim(-3, 70)
ax.set_ylim(50, 105)
ax.legend(loc="lower right", fontsize=9)
ax.grid(alpha=0.3)
fig.tight_layout()

out = os.path.join(BASE, "safety_success_pareto.png")
fig.savefig(out, dpi=150)
print(f"saved -> {out}")
print("MPC Pareto pts (coll%, succ%, d_safe):",
      [(round(c * 100, 1), round(s * 100, 1), d) for c, s, d in mpc_pts])
