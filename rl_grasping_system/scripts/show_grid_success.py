#!/usr/bin/env python3
"""位置成功率网格热力图：展示当前模型在 ±radius 工作区内的空间泛化（2026-08-24）

思路：把 X/Y 工作区均分成 grid×grid 小格，每格把 workspace_bounds 收窄到该小格、
use_fixed_position=False（物体在小格内均匀随机采样，观测与物理一致），跑 N 个 episode
统计 deterministic 成功率；最后输出表格 + 保存 PNG 热力图。

用法：
    cd rl_grasping_system
    python3 scripts/show_grid_success.py --model models/final_model_stage2.zip --radius 0.06 \
        --grid 4 --n-episodes 12 --save results/demo/stage2_grid.png

产物：results/demo/stage2_grid.png + 控制台表格
"""
import os
import sys
import argparse

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config import get_config
from evaluate import load_model, run_episode


def main():
    ap = argparse.ArgumentParser(description="位置成功率网格热力图")
    ap.add_argument("--model", default="models/final_model_stage2.zip")
    ap.add_argument("--radius", type=float, default=0.06)
    ap.add_argument("--grid", type=int, default=4, help="每维分割数")
    ap.add_argument("--n-episodes", type=int, default=12, help="每格 episode 数")
    ap.add_argument("--save", default="results/demo/stage2_grid.png")
    ap.add_argument("--deterministic", action="store_true",
                    help="确定性采样（默认；配 --stochastic 用随机策略）")
    ap.add_argument("--stochastic", action="store_true", help="随机策略采样")
    args = ap.parse_args()

    config = get_config()
    cx, cy = config.grasping.object_fixed_pos
    z = config.grasping.workspace_bounds[2]
    config.grasping.render_gui = False
    config.grasping.use_fixed_position = False
    r = args.radius
    xs = np.linspace(cx - r, cx + r, args.grid + 1)   # 分界
    ys = np.linspace(cy - r, cy + r, args.grid + 1)
    deterministic = not args.stochastic

    rate = np.zeros((args.grid, args.grid))
    for j in range(args.grid):        # Y（表格行，Y 增大方向）
        for i in range(args.grid):    # X（列）
            cfg = get_config()
            cfg.grasping.render_gui = False
            cfg.grasping.use_fixed_position = False
            cfg.grasping.workspace_bounds = ((xs[i], xs[i + 1]), (ys[j], ys[j + 1]), z)
            agent, env = load_model(args.model, cfg)
            ok = 0
            for _ in range(args.n_episodes):
                res = run_episode(agent, env, render=False, max_steps=500,
                                  deterministic=deterministic)
                ok += 1 if res['grasp_success'] else 0
            rate[j, i] = ok / args.n_episodes
            del agent, env
            print(f"  X[{xs[i]:+.3f},{xs[i+1]:+.3f}] × Y[{ys[j]:+.3f},{ys[j+1]:+.3f}]"
                  f": {ok:2d}/{args.n_episodes} = {rate[j,i]:6.1%}")

    print("\n成功率矩阵（行=Y 增，列=X 增）")
    hdr = "          " + "".join(f"X{i+1:>8}" for i in range(args.grid))
    print(hdr)
    for j in range(args.grid - 1, -1, -1):
        row = f"Y{j+1} ({ys[j]:+.3f})" + "".join(f"{rate[j,i]*100:7.1f}%" for i in range(args.grid))
        print(row)

    # 热力图
    fig, ax = plt.subplots(figsize=(6, 5))
    im = ax.imshow(rate, cmap='RdYlGn', vmin=0, vmax=1, origin='lower')
    ax.set_xticks(range(args.grid))
    ax.set_yticks(range(args.grid))
    ax.set_xticklabels([f"{xs[i]+0.5*(xs[1]-xs[0]):.3f}" for i in range(args.grid)])
    ax.set_yticklabels([f"{ys[j]+0.5*(ys[1]-ys[0]):.3f}" for j in range(args.grid)])
    ax.set_xlabel("X (m)")
    ax.set_ylabel("Y (m)")
    ax.set_title(f"{os.path.basename(args.model)} ±{r}m deterministic 成功率网格 "
                 f"({args.n_episodes}ep/格)")
    for j in range(args.grid):
        for i in range(args.grid):
            ax.text(i, j, f"{rate[j, i]*100:.0f}%", ha='center', va='center',
                    color='black', fontsize=11, fontweight='bold')
    fig.colorbar(im, ax=ax, label='成功率')
    plt.tight_layout()
    os.makedirs(os.path.dirname(args.save) or ".", exist_ok=True)
    plt.savefig(args.save, dpi=200, bbox_inches='tight')
    print(f"\n热力图已保存: {args.save}")


if __name__ == "__main__":
    main()
