"""
Task3: 训练"每个奖励项"事后回放图。

读训练日志 JSON（每 episode 含 breakdown 七项：r_dist_xy/r_dist_z/r_orient/r_contact/r_grasp/r_success/r_step），
绘制 2x4 面板：7 个奖励分项 + 总奖励，各自随 episode 的原始值 + 10-episode 移动平均。

用法：
    python3 plot_reward_breakdown.py [日志.json] [输出.png]
"""
import os
import sys
import json
import argparse

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

KEYS = ['r_dist_xy', 'r_dist_z', 'r_orient', 'r_contact', 'r_grasp', 'r_success', 'r_step']
TITLES = {
    'r_dist_xy': 'r_dist_xy (xy potential)',
    'r_dist_z': 'r_dist_z (height potential)',
    'r_orient': 'r_orient (orientation)',
    'r_contact': 'r_contact (contact)',
    'r_grasp': 'r_grasp (grasp)',
    'r_success': 'r_success (completion)',
    'r_step': 'r_step (time penalty)',
}


def moving_avg(x, w=10):
    return np.array([np.mean(x[max(0, i - w):i + 1]) for i in range(len(x))])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('log', nargs='?',
                    default='/home/zrq/githubprojects/robotic_arm_control/rl_grasping_system/logs/training_log_20260821_180428.json')
    ap.add_argument('--out', default='/home/zrq/githubprojects/robotic_arm_control/rl_grasping_system/results/reward_breakdown_150k.png')
    args = ap.parse_args()

    data = json.load(open(args.log, encoding='utf-8'))
    eps = data.get('episodes', [])
    if not eps:
        print('no episodes')
        return 1

    n = len(eps)
    x = np.arange(1, n + 1)
    series = {k: [] for k in KEYS + ['total']}
    for e in eps:
        bd = e.get('breakdown') or {}
        for k in KEYS:
            series[k].append(float(bd.get(k, 0.0)))
        series['total'].append(float(e.get('reward', 0.0)))

    fig, axes = plt.subplots(2, 4, figsize=(20, 9))
    fig.suptitle(f'Reward Breakdown per Episode (N={n}) — {os.path.basename(args.log)}', fontsize=15)
    for ax, k in zip(axes.ravel(), KEYS + ['total']):
        y = np.asarray(series[k])
        ax.plot(x, y, '-', alpha=0.35, color='tab:blue')
        ax.plot(x, moving_avg(y), '-', linewidth=2, color='tab:red', label='10-ep avg')
        ax.axhline(0, color='gray', lw=0.8, ls='--')
        ax.set_title(TITLES.get(k, k))
        ax.set_xlabel('Episode')
        ax.grid(True, alpha=0.3)
        ax.legend(loc='best', fontsize=8)
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    plt.savefig(args.out, dpi=150, bbox_inches='tight')
    plt.close()
    print(f'[plot] 保存: {args.out}')
    print(f'[plot] 末 20 episode 均值: ' + ', '.join(
        f'{k}={np.mean(series[k][-20:]):+.1f}' for k in KEYS + ['total']))
    return 0


if __name__ == '__main__':
    sys.exit(main())
