#!/usr/bin/env python3
"""扰动扫掠的 A/B 对比图：横轴是扰动强度（执行延迟 D / 感知噪声 σ），两条线是
「纯标称」与「标称+残差」，第三张面板画残差带来的收益随扰动强度的变化。

存在的理由：`scripts/plot_residual_vs_nominal.py` 画的是**单集轨迹**对比（同口径下的
逐集配对），回答的是「这一步残差在干什么」；本脚本画的是**扫掠趋势**，回答的是
「残差的价值随扰动强度怎么变」——后者才是「残差补偿模型失配/感知误差」这个主张的形状。
两者互补，不是替代。

图内文字一律英文：本机 matplotlib 无 CJK 字体（ttflist 里 0 个），中文会渲染成 tofu。

用法（延迟扫掠）：
  python3 scripts/plot_ab_sweep.py --dir results/delay_sweep \
      --pattern '(?P<scene>.+?)_(?P<arm>.+?)_d(?P<x>\\d+)\\.json' \
      --xlabel 'Execution delay D (decision steps)' \
      --out results/delay_sweep/delay_ab_sweep.png
用法（感知扫掠）：
  ... --pattern '(?P<scene>.+?)_(?P<arm>.+?)_s(?P<x>[0-9.]+)\\.json' \
      --xlabel 'Perception noise sigma (m)'
"""
import os
import re
import glob
import json
import math
import argparse

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


def _binom_two_sided(k, n):
    """精确二项双侧 p（同 plot_residual_vs_nominal.py，不引 scipy）。"""
    if n == 0:
        return 1.0
    k = min(k, n - k)
    return min(1.0, 2.0 * sum(math.comb(n, i) for i in range(k + 1)) / 2.0 ** n)


def load_sweep(d, pattern, baseline_tag, residual_tag):
    """-> {(scene, role, x): dict}  role ∈ {'baseline','residual'}"""
    rx = re.compile(pattern)
    out = {}
    for f in sorted(glob.glob(os.path.join(d, '*.json'))):
        m = rx.search(os.path.basename(f))
        if not m:
            continue
        g = m.groupdict()
        arm = g['arm']
        if residual_tag and residual_tag in arm:
            role = 'residual'
        elif baseline_tag and baseline_tag in arm:
            role = 'baseline'
        else:
            continue
        x = float(g['x'])
        J = json.load(open(f))
        E = J['episode_results']
        ok = np.array([bool(e['grasp_success']) for e in E])
        st = np.array([e['episode_length'] for e in E], dtype=float)
        coll = np.array([e['collision_count'] for e in E], dtype=float)
        out[(g['scene'], role, x)] = dict(
            n=len(E), ok=ok, st=st, coll=coll,
            succ=float(ok.mean()),
            nc=float((ok & (coll == 0)).mean()),
            mean_st_succ=float(st[ok].mean()) if ok.any() else float('nan'))
    return out


def paired(saved, scene, x):
    """同一 scene/x 下、逐集对齐的配对统计。返回 None 表示数据不全。
    注意：仅当两侧用同一 seed 序列、且逐 episode 重播种时对齐才是严格配对。"""
    a = saved.get((scene, 'baseline', x))
    b = saved.get((scene, 'residual', x))
    if a is None or b is None:
        return None
    n = min(a['n'], b['n'])
    both = np.flatnonzero(a['ok'][:n] & b['ok'][:n])
    d = a['st'][:n][both] - b['st'][:n][both]      # >0 表示残差更快
    faster = int(np.sum(d > 0))
    slower = int(np.sum(d < 0))
    up = int(np.sum(~a['ok'][:n] & b['ok'][:n]))   # 残差把失败变成功
    dn = int(np.sum(a['ok'][:n] & ~b['ok'][:n]))
    return dict(n_pair=len(both), mean_d=float(d.mean()) if len(d) else float('nan'),
                faster=faster, slower=slower, p_step=_binom_two_sided(faster, faster + slower),
                up=up, dn=dn, p_succ=_binom_two_sided(up, up + dn),
                d_succ=(b['succ'] - a['succ']) * 100.0,
                d_nc=(b['nc'] - a['nc']) * 100.0)


C_BASE, C_RES, C_GAIN = '#444444', '#c0392b', '#2471a3'


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--dir', required=True)
    ap.add_argument('--pattern', required=True,
                    help="文件名正则，须含命名组 scene / arm / x")
    ap.add_argument('--baseline-tag', default='mpc')
    ap.add_argument('--residual-tag', default='v26')
    ap.add_argument('--xlabel', default='Perturbation level')
    ap.add_argument('--baseline-name', default='nominal only')
    ap.add_argument('--residual-name', default='nominal + residual RL')
    ap.add_argument('--title', default='')
    ap.add_argument('--out', required=True)
    args = ap.parse_args()

    S = load_sweep(args.dir, args.pattern, args.baseline_tag, args.residual_tag)
    if not S:
        raise SystemExit(f"没有匹配的文件：{args.dir} / {args.pattern}")
    scenes = sorted({k[0] for k in S})
    xs = sorted({k[2] for k in S})

    ncol = len(scenes) + 1
    fig, axes = plt.subplots(1, ncol, figsize=(3.9 * ncol, 3.9), squeeze=False)
    axes = axes[0]

    for i, sc in enumerate(scenes):
        ax = axes[i]
        ax2 = ax.twinx()
        for role, col, ls, nm in (('baseline', C_BASE, '-', args.baseline_name),
                                  ('residual', C_RES, '-', args.residual_name)):
            ys, ss = [], []
            for x in xs:
                r = S.get((sc, role, x))
                ys.append(r['mean_st_succ'] if r else np.nan)
                ss.append(r['succ'] * 100.0 if r else np.nan)
            ax.plot(xs, ys, ls, color=col, lw=1.9, marker='o', ms=4.5, label=nm)
            ax2.plot(xs, ss, '--', color=col, lw=1.2, marker='s', ms=3.5, alpha=0.65)
        ax.set_xlabel(args.xlabel, fontsize=8.5)
        ax.set_ylabel('Mean steps to success', fontsize=8.5)
        ax2.set_ylabel('Success rate (%)', fontsize=8.5, color='#666666')
        ax2.tick_params(axis='y', colors='#666666')
        ax2.set_ylim(0, 105)
        ax.grid(alpha=0.25, lw=0.6)
        ax.set_xticks(xs)
        ax.tick_params(labelsize=8)
        ax2.tick_params(labelsize=8)
        ax.set_title(sc, fontsize=10)
        ax.legend(loc='upper left', fontsize=7.5, frameon=False)
        # 配对统计随 x 标注在底部（步数符号检验 = 主主张；成功率 = 参考）
        lines = []
        for x in xs:
            p = paired(S, sc, x)
            if p is None:
                continue
            lines.append(f"D={x:g}: faster {p['faster']}/{p['n_pair']} "
                         f"p={p['p_step']:.1e} | dsucc {p['d_succ']:+.0f}pp")
        if lines:
            ax.text(0.02, -0.30, '\n'.join(lines), transform=ax.transAxes,
                    fontsize=6.6, va='top', family='monospace')

    # 最后一张：残差收益 vs 扰动强度
    ax = axes[-1]
    nsc = len(scenes)
    w = 0.8 / max(1, nsc)
    for j, sc in enumerate(scenes):
        ds = []
        for x in xs:
            p = paired(S, sc, x)
            ds.append(p['mean_d'] if p else np.nan)
        ax.bar(np.arange(len(xs)) + (j - (nsc - 1) / 2.0) * w, ds, width=w,
               color=C_GAIN, alpha=0.55 + 0.35 * (j / max(1, nsc - 1)) if nsc > 1 else 0.8,
               edgecolor='k', lw=0.5, label=sc)
    ax.axhline(0, color='k', lw=0.8)
    ax.set_xticks(np.arange(len(xs)))
    ax.set_xticklabels([f'{x:g}' for x in xs])
    ax.set_xlabel(args.xlabel, fontsize=8.5)
    ax.set_ylabel('Steps saved by residual\n(baseline - residual, paired successes)', fontsize=8.5)
    ax.grid(alpha=0.25, lw=0.6, axis='y')
    ax.tick_params(labelsize=8)
    ax.set_title('Residual benefit vs perturbation', fontsize=10)
    if nsc > 1:
        ax.legend(fontsize=7.5, frameon=False)
    ax.text(0.02, -0.30,
            'positive = residual reaches the goal in fewer steps.\n'
            'Pairing is exact only if both arms were run with the same\n'
            'seed sequence and --seed-per-episode.',
            transform=ax.transAxes, fontsize=6.6, va='top')

    if args.title:
        fig.suptitle(args.title, fontsize=11)
    fig.tight_layout(rect=[0, 0.02, 1, 0.97] if args.title else None)
    os.makedirs(os.path.dirname(os.path.abspath(args.out)) or '.', exist_ok=True)
    fig.savefig(args.out, dpi=170, bbox_inches='tight')
    print(f"saved -> {args.out}")

    # 控制台表格（便于核对图上数字）
    print(f"\n{'scene':<14}{'role':<10}" + ''.join(f'{x:>10g}' for x in xs))
    for sc in scenes:
        for role in ('baseline', 'residual'):
            vals = []
            for x in xs:
                r = S.get((sc, role, x))
                vals.append(f"{100*r['succ']:>9.1f}%" if r else f"{'-':>10}")
            print(f"{sc:<14}{role:<10}" + ''.join(vals))
    print(f"\n{'scene':<14}{'metric':<16}" + ''.join(f'{x:>12g}' for x in xs))
    for sc in scenes:
        for key, lbl, fmt in (('mean_d', 'steps saved', '{:>12.1f}'),
                              ('p_step', 'sign p', '{:>12.1e}'),
                              ('d_succ', 'dsucc pp', '{:>12.1f}'),
                              ('d_nc', 'dnc pp', '{:>12.1f}')):
            vals = []
            for x in xs:
                p = paired(S, sc, x)
                vals.append(fmt.format(p[key]) if p else f"{'-':>12}")
            print(f"{sc:<14}{lbl:<16}" + ''.join(vals))


if __name__ == '__main__':
    main()
