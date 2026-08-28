#!/usr/bin/env python3
"""
三路消融对比绘图（2026-08-28）
  输入：results/ablation/{end2end,residual,nominal}_{static,dyn_target,dyn_both}.json
  输出：results/ablation_compare.png（1×3：成功率 / 碰撞率 / 平均步数）
        results/ablation/summary_table.txt（汇总表，供 README/简历引用）
论证：静态下纯标称即够用（83.3%）、动态场景残差大幅领先（90%/86.7% vs 66.7%/70%）、
      端到端全场景最差（26.7%/63.3%/43.3%）——"标称解决静态、残差解决动态、端到端均不如残差"。
注：三路均在当前环境（扩大桌面）下评估、均未重训——端到端对域变化最敏感、残差最鲁棒。
"""
import json
import os

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

BASE = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'results', 'ablation')
SCENES = ['static', 'dyn_target', 'dyn_both']
SCENE_LABEL = {'static': 'Static', 'dyn_target': 'Dynamic target', 'dyn_both': 'Dynamic target\n+ obstacle'}
ALGOS = ['end2end', 'residual', 'nominal']
ALGO_LABEL = {'end2end': 'End-to-end PPO', 'residual': 'Residual PPO', 'nominal': 'Nominal only\n(zero residual)'}
COLOR = {'end2end': '#d9534f', 'residual': '#337ab7', 'nominal': '#999999'}


def load():
    data = {}
    for algo in ALGOS:
        data[algo] = {}
        for scene in SCENES:
            p = os.path.join(BASE, f'{algo}_{scene}.json')
            with open(p) as f:
                data[algo][scene] = json.load(f)
    return data


def plot(data, out_path):
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5))
    x = np.arange(len(SCENES))
    width = 0.26

    # 1) 成功率
    ax = axes[0]
    for i, algo in enumerate(ALGOS):
        vals = [data[algo][s]['success_rate'] * 100 for s in SCENES]
        ax.bar(x + (i - 1) * width, vals, width, label=ALGO_LABEL[algo],
               color=COLOR[algo], alpha=0.9)
        for xi, v in zip(x + (i - 1) * width, vals):
            ax.text(xi, v + 1.5, f'{v:.0f}', ha='center', fontsize=9)
    ax.axhline(70, ls='--', c='green', lw=1, label='70% target')
    ax.set_xticks(x); ax.set_xticklabels([SCENE_LABEL[s] for s in SCENES], fontsize=10)
    ax.set_ylabel('Success rate (%)'); ax.set_title('Success rate (n=30/scenario)', fontsize=12)
    ax.legend(fontsize=9); ax.set_ylim(0, 110); ax.grid(axis='y', alpha=0.3)

    # 2) 碰撞率
    ax = axes[1]
    for i, algo in enumerate(ALGOS):
        vals = [data[algo][s].get('collision_rate', 0.0) * 100 for s in SCENES]
        ax.bar(x + (i - 1) * width, vals, width, label=ALGO_LABEL[algo],
               color=COLOR[algo], alpha=0.9)
        for xi, v in zip(x + (i - 1) * width, vals):
            ax.text(xi, v + 1.0, f'{v:.0f}', ha='center', fontsize=9)
    ax.set_xticks(x); ax.set_xticklabels([SCENE_LABEL[s] for s in SCENES], fontsize=10)
    ax.set_ylabel('Collision rate (%)'); ax.set_title('Collision episode ratio', fontsize=12)
    ax.legend(fontsize=9); ax.set_ylim(0, 110); ax.grid(axis='y', alpha=0.3)

    # 3) 平均步数（效率）
    ax = axes[2]
    for i, algo in enumerate(ALGOS):
        vals = [data[algo][s].get('avg_episode_length', 0) for s in SCENES]
        ax.bar(x + (i - 1) * width, vals, width, label=ALGO_LABEL[algo],
               color=COLOR[algo], alpha=0.9)
        for xi, v in zip(x + (i - 1) * width, vals):
            ax.text(xi, v + 2, f'{v:.0f}', ha='center', fontsize=9)
    ax.set_xticks(x); ax.set_xticklabels([SCENE_LABEL[s] for s in SCENES], fontsize=10)
    ax.set_ylabel('Avg episode steps'); ax.set_title('Task efficiency (lower is better)', fontsize=12)
    ax.legend(fontsize=9); ax.grid(axis='y', alpha=0.3)

    fig.suptitle('Ablation: End-to-end vs Residual vs Nominal-only (zero residual)\n'
                 'UR5e + Robotiq 2F-85 · MuJoCo · velocity-level IK + gripper state machine · PPO residual (67-dim obs)',
                 fontsize=13, y=1.03)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches='tight')
    print(f'图已保存: {out_path}')


def write_table(data, out_path):
    lines = []
    lines.append('| Scenario | End-to-end SR | Residual SR | Nominal SR | E2E collision | Resid collision | Nom collision |')
    lines.append('|---|---|---|---|---|---|---|')
    for scene in SCENES:
        def g(a, k):
            v = data[a][scene].get(k)
            return '—' if v is None else f'{v*100:.1f}%'
        lines.append(f'| {SCENE_LABEL[scene].replace(chr(10), " ")} '
                     f'| {g("end2end","success_rate")} | {g("residual","success_rate")} | {g("nominal","success_rate")} '
                     f'| {g("end2end","collision_rate")} | {g("residual","collision_rate")} | {g("nominal","collision_rate")} |')
    lines.append('')
    lines.append('Models: end2end = v11_500k (delta, no nominal); residual = v18_p3f (residual, product model); '
                 'nominal = v18_p3f + zero residual (nominal-only).')
    lines.append('Setup: n=30 per scenario, fixed object pose, same current env (enlarged table 0.70m), '
                 'none retrained in current env.')
    lines.append('Conclusion: nominal suffices for static (83.3%), residual dominates dynamic scenarios, '
                 'end-to-end worst everywhere.')
    with open(out_path, 'w') as f:
        f.write('\n'.join(lines) + '\n')
    print(f'Summary saved: {out_path}')


def main():
    data = load()
    out_png = os.path.join(os.path.dirname(BASE), 'ablation_compare.png')
    plot(data, out_png)
    write_table(data, os.path.join(BASE, 'summary_table.txt'))


if __name__ == '__main__':
    main()
