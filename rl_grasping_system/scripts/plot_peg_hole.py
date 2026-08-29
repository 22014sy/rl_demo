#!/usr/bin/env python3
"""peg-in-hole 对比图：阻抗基线 vs RL（理想/±5cm 扰动，12mm 孔，n=30）"""
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import os

base = np.array([73.3, 76.7])   # 基线：理想 / ±5cm 扰动
rl = np.array([100.0, 100.0])   # RL
labels = ['No perturbation', '±5cm lateral']

x = np.arange(2)
w = 0.3
fig, ax = plt.subplots(figsize=(8, 5))
ax.bar(x - w/2, base, w, label='Impedance baseline (PID align + press)', color='#999999', alpha=0.9)
ax.bar(x + w/2, rl, w, label='RL (PPO contact correction)', color='#337ab7', alpha=0.9)
for xi, v in zip(x - w/2, base): ax.text(xi, v + 1.5, f'{v:.0f}%', ha='center', fontsize=11)
for xi, v in zip(x + w/2, rl): ax.text(xi, v + 1.5, f'{v:.0f}%', ha='center', fontsize=11)
ax.set_xticks(x); ax.set_xticklabels(labels, fontsize=11)
ax.set_ylabel('Success rate (%)')
ax.set_ylim(0, 115)
ax.set_title('Peg-in-hole: RL vs impedance baseline (12mm hole, 1mm clearance, n=30)')
ax.legend(fontsize=10)
ax.grid(axis='y', alpha=0.3)
fig.tight_layout()
out = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'results', 'peg_hole_compare.png')
fig.savefig(out, dpi=150)
print('saved:', out)
