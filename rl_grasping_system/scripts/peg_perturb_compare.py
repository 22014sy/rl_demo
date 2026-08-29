#!/usr/bin/env python3
"""
peg-in-hole 扰动对比（2026-08-28，步 2 核心卖点）
==================================================
对比阻抗基线 vs RL 在初始水平偏移扰动下的成功率：
  - 理想情况：lateral_range = ±2cm（课程分布）
  - 扰动：    lateral_range = ±5cm（分布外）
预期：基线（开环 PID 对齐 + 垂直下压）在 ±5cm 下失败（对齐容差 2mm 击穿/超时），
      RL（学过的修正）保持高成功率 → "接触插入 RL 相对基线 +X%，扰动下优势扩大"。

运行：MUJOCO_GL=egl python3 scripts/peg_perturb_compare.py
"""
import os
import sys

os.environ.setdefault('MUJOCO_GL', 'egl')
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np
from stable_baselines3 import PPO

import peg_hole_min as P
from peg_baseline import run_baseline_episode

MODEL_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'models', 'peg_hole_rl.zip')
N = 30


def eval_baseline(lateral_range, n=N):
    env = P.PegHoleEnv()
    env.lateral_range = lateral_range
    succ, lens, ph = 0, [], {}
    for i in range(n):
        ok, steps, info = run_baseline_episode(env, seed=i)
        succ += int(ok)
        lens.append(steps)
        ph[info['phase']] = ph.get(info['phase'], 0) + 1
    env.close()
    return succ / n, float(np.mean(lens)), ph


def eval_rl(model, lateral_range, n=N):
    env = P.PegHoleEnv()
    env.lateral_range = lateral_range
    succ, lens = 0, []
    for i in range(n):
        obs, _ = env.reset(seed=i)
        done, steps = False, 0
        while not done:
            a, _ = model.predict(obs, deterministic=True)
            obs, _, term, trunc, info = env.step(a)
            done = term or trunc
            steps += 1
            if steps > 200:
                break
        succ += int(info['success'])
        lens.append(steps)
    env.close()
    return succ / n, float(np.mean(lens))


def main():
    model = PPO.load(MODEL_PATH)
    print('=== 阻抗基线 ===')
    sr_base_ok, len_base_ok, _ = eval_baseline(0.02)
    sr_base_pt, len_base_pt, ph_pt = eval_baseline(0.05)
    print(f'  理想(±2cm): {sr_base_ok*100:5.1f}%  len={len_base_ok:.1f}')
    print(f'  扰动(±5cm): {sr_base_pt*100:5.1f}%  len={len_base_pt:.1f}  失败阶段={ph_pt}')
    print('=== RL（加载 peg_hole_rl.zip） ===')
    sr_rl_ok, len_rl_ok = eval_rl(model, 0.02)
    sr_rl_pt, len_rl_pt = eval_rl(model, 0.05)
    print(f'  理想(±2cm): {sr_rl_ok*100:5.1f}%  len={len_rl_ok:.1f}')
    print(f'  扰动(±5cm): {sr_rl_pt*100:5.1f}%  len={len_rl_pt:.1f}')
    print('\n=== 扰动下对比（核心卖点） ===')
    print(f'  基线 ±5cm: {sr_base_pt*100:.1f}%   RL ±5cm: {sr_rl_pt*100:.1f}%  → RL 优势 +{(sr_rl_pt-sr_base_pt)*100:.1f}pp')


if __name__ == '__main__':
    main()
