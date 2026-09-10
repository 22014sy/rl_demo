"""
Task3: 训练"每个奖励项"事后回放图。

读训练日志 JSON（每 episode 含 breakdown 分项：r_dist_xy/r_dist_z/r_contact/r_close/r_grasp/
r_obstacle/r_residual/r_residual_step/r_avoid/r_step），
绘制 2x4 面板：10 个奖励分项 + 总奖励，各自随 episode 的原始值 + 10-episode 移动平均。

用法：
    python3 plot_reward_breakdown.py [日志.json] [输出.png]
    python3 plot_reward_breakdown.py --all --log-dir ../logs --out-dir ../results/reward_breakdown
"""
import os
import sys
import json
import argparse

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# 2026-09-08 去冗余：r_orient/r_success 恒 0 已从 REWARD_KEYS 移除，故从回放图列表剔除；
# 补齐当时缺失的 r_close，并按当前 REWARD_KEYS 扩展（对旧日志 bd.get(k,0.0) 容错）。
KEYS = ['r_dist_xy', 'r_dist_z', 'r_contact', 'r_close', 'r_grasp',
        'r_obstacle', 'r_residual', 'r_residual_step', 'r_avoid', 'r_step']
TITLES = {
    'r_dist_xy': 'r_dist_xy (xy potential)',
    'r_dist_z': 'r_dist_z (height potential)',
    'r_contact': 'r_contact (contact)',
    'r_close': 'r_close (closing milestone)',
    'r_grasp': 'r_grasp (grasp)',
    'r_obstacle': 'r_obstacle (obstacle penalty)',
    'r_residual': 'r_residual (residual reg)',
    'r_residual_step': 'r_residual_step (activation penalty)',
    'r_avoid': 'r_avoid (clean success bonus)',
    'r_step': 'r_step (time penalty)',
}


def moving_avg(x, w=10):
    return np.array([np.mean(x[max(0, i - w + 1):i + 1]) for i in range(len(x))])


def load_series(log_path):
    with open(log_path, encoding='utf-8') as f:
        data = json.load(f)
    eps = data.get('episodes', [])
    if not eps:
        raise ValueError('no episodes')

    series = {k: [] for k in KEYS + ['total']}
    for episode in eps:
        breakdown = episode.get('breakdown') or {}
        for key in KEYS:
            series[key].append(float(breakdown.get(key, 0.0)))
        series['total'].append(float(episode.get('reward', 0.0)))
    return series


def plot_log(log_path, out_path, window):
    series = load_series(log_path)
    n = len(series['total'])
    x = np.arange(1, n + 1)

    fig, axes = plt.subplots(3, 4, figsize=(20, 13))
    fig.suptitle(f'Reward Breakdown per Episode (N={n}) - {os.path.basename(log_path)}', fontsize=15)
    for ax, key in zip(axes.ravel(), KEYS + ['total']):
        y = np.asarray(series[key])
        ax.plot(x, y, '-', alpha=0.25, color='tab:blue')
        ax.plot(x, moving_avg(y, window), '-', linewidth=2, color='tab:red',
                label=f'{window}-episode avg')
        ax.axhline(0, color='gray', lw=0.8, ls='--')
        ax.set_title(TITLES.get(key, key))
        ax.set_xlabel('Episode')
        ax.grid(True, alpha=0.3)
        ax.legend(loc='best', fontsize=8)
    for ax in axes.ravel()[len(KEYS) + 1:]:
        ax.axis('off')
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    plt.savefig(out_path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    return series


def plot_all_summary(log_paths, out_path, window):
    fig, axes = plt.subplots(3, 4, figsize=(20, 13))
    for log_path in log_paths:
        series = load_series(log_path)
        x = np.arange(1, len(series['total']) + 1)
        label = os.path.basename(log_path).replace('training_log_', '').replace('.json', '')
        for ax, key in zip(axes.ravel(), KEYS + ['total']):
            ax.plot(x, moving_avg(np.asarray(series[key]), window), linewidth=1.2, label=label)
    for ax, key in zip(axes.ravel(), KEYS + ['total']):
        ax.axhline(0, color='gray', lw=0.8, ls='--')
        ax.set_title(TITLES.get(key, key))
        ax.set_xlabel('Episode')
        ax.grid(True, alpha=0.3)
        ax.legend(loc='best', fontsize=7)
    for ax in axes.ravel()[len(KEYS) + 1:]:
        ax.axis('off')
    fig.suptitle(f'Reward Breakdown Comparison ({len(log_paths)} training runs)', fontsize=15)
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    plt.savefig(out_path, dpi=150, bbox_inches='tight')
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('log', nargs='?',
                    default='/home/zrq/githubprojects/robotic_arm_control/rl_grasping_system/logs/training_log_20260821_180428.json')
    ap.add_argument('--out', default='/home/zrq/githubprojects/robotic_arm_control/rl_grasping_system/results/reward_breakdown_150k.png')
    ap.add_argument('--all', action='store_true', help='批量绘制日志目录中的所有训练')
    ap.add_argument('--log-dir', default='/home/zrq/githubprojects/robotic_arm_control/rl_grasping_system/logs')
    ap.add_argument('--out-dir', default='/home/zrq/githubprojects/robotic_arm_control/rl_grasping_system/results/reward_breakdown')
    ap.add_argument('--window', type=int, default=20, help='滑动平均窗口')
    args = ap.parse_args()

    if args.window < 1:
        raise SystemExit('--window 必须大于 0')

    if args.all:
        logs = sorted(
            os.path.join(args.log_dir, name)
            for name in os.listdir(args.log_dir)
            if name.startswith('training_log_') and name.endswith('.json'))
        valid_logs = []
        for log_path in logs:
            try:
                load_series(log_path)
                valid_logs.append(log_path)
                output = os.path.join(args.out_dir, os.path.basename(log_path).replace('.json', '.png'))
                plot_log(log_path, output, args.window)
                print(f'[plot] 保存: {output}')
            except (ValueError, json.JSONDecodeError) as exc:
                print(f'[plot] 跳过 {log_path}: {exc}')
        if valid_logs:
            summary = os.path.join(args.out_dir, 'reward_breakdown_comparison.png')
            plot_all_summary(valid_logs, summary, args.window)
            print(f'[plot] 汇总: {summary}')
        return 0

    series = plot_log(args.log, args.out, args.window)
    print(f'[plot] 保存: {args.out}')
    print(f'[plot] 末 20 episode 均值: ' + ', '.join(
        f'{k}={np.mean(series[k][-20:]):+.1f}' for k in KEYS + ['total']))
    return 0


if __name__ == '__main__':
    sys.exit(main())
