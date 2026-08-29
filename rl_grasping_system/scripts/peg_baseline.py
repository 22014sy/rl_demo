#!/usr/bin/env python3
"""
peg-in-hole 阻抗控制基线（2026-08-28，步 1）
==================================================
基线策略（对比锚点）：阶段1 水平 PID 对齐（peg 尖端→孔中心，保持高度）；
阶段2 垂直下压（恒速），直到插入深度 ≥ 12mm 或超时/触底。

与 RL 对比：理想情况（无扰动）基线应 ≈ RL 成功率（验证环境任务可被传统控制完成）；
后续步 2 加扰动（位置/姿态/摩擦）时，基线失败而 RL 鲁棒 → "接触修正相对基线优势"。

运行：MUJOCO_GL=egl python3 scripts/peg_baseline.py
"""
import os
import sys

os.environ.setdefault('MUJOCO_GL', 'egl')
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np

import peg_hole_min as P

MAX_DELTA = P.MAX_EE_DELTA          # 5mm/步
ALIGN_TOL = 0.002                   # 水平对准容差 2mm（孔半宽 7mm - peg 半径 5mm）
KP_ALIGN = 2.0                      # 对齐 PID 增益
INSERT_SPEED = -MAX_DELTA           # 下压步进（-5mm）
MAX_STEPS = 200


def run_baseline_episode(env, seed=None, verbose=False):
    """基线：对齐（水平 PID，保持高度）→ 垂直下压。返回 (success, steps, info)。"""
    env.reset(seed=seed)
    tip = env.data.site_xpos[env.tip_id]
    d_xy0 = float(np.linalg.norm(tip[:2] - P.HOLE_CENTER[:2]))
    steps = 0

    # 阶段 1：水平对齐（peg 尖端 XY → 孔中心；z 保持）
    for _ in range(MAX_STEPS):
        tip = env.data.site_xpos[env.tip_id]
        d_xy = P.HOLE_CENTER[:2] - tip[:2]
        dist = float(np.linalg.norm(d_xy))
        if dist < ALIGN_TOL:
            break
        a = np.zeros(3)
        if dist > 1e-6:
            a[:2] = np.clip(KP_ALIGN * d_xy, -MAX_DELTA, MAX_DELTA)
        obs, r, term, trunc, info = env.step(a)
        steps += 1
        if term or trunc:
            return info['success'], steps, {'phase': 'align', 'd_xy0': d_xy0,
                                             'insertion': info['insertion']}
    if verbose:
        print(f'  align done: {steps} 步')

    # 阶段 2：垂直下压（恒速 -5mm/步），直到插入或触底/超时
    for _ in range(MAX_STEPS):
        a = np.array([0.0, 0.0, INSERT_SPEED])
        obs, r, term, trunc, info = env.step(a)
        steps += 1
        if info['success']:
            return True, steps, {'phase': 'insert', 'd_xy0': d_xy0, 'insertion': info['insertion']}
        tip = env.data.site_xpos[env.tip_id]
        if tip[2] < P.HOLE_CENTER[2] - 0.02:   # peg 尖端低于孔底 20mm = 插到底未成功
            return False, steps, {'phase': 'bottom', 'd_xy0': d_xy0, 'insertion': info['insertion']}
        if term or trunc:
            return info['success'], steps, {'phase': 'timeout', 'd_xy0': d_xy0,
                                            'insertion': info['insertion']}
    return False, MAX_STEPS, {'phase': 'timeout', 'd_xy0': d_xy0, 'insertion': 0.0}


def main():
    env = P.PegHoleEnv()
    n = 30
    succ = 0
    steps_list = []
    phases = {}
    d0_list = []
    for i in range(n):
        ok, steps, info = run_baseline_episode(env, seed=i)
        succ += int(ok)
        steps_list.append(steps)
        d0_list.append(info['d_xy0'] * 100)
        phases[info['phase']] = phases.get(info['phase'], 0) + 1
        print(f'ep{i}: {"OK " if ok else "FAIL"} len={steps:3d} phase={info["phase"]} '
              f'insert={info["insertion"]*1000:.1f}mm d_xy0={info["d_xy0"]*100:.1f}cm')
    env.close()
    print(f'\n=== 阻抗基线结果（n={n}）===')
    print(f'成功率: {succ/n*100:.1f}%  平均步数: {np.mean(steps_list):.1f}')
    print(f'失败阶段分布: {phases}')
    print(f'初始水平偏差: mean={np.mean(d0_list):.1f}cm max={np.max(d0_list):.1f}cm')


if __name__ == '__main__':
    main()
