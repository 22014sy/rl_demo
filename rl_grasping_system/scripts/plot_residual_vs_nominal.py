#!/usr/bin/env python3
"""把「标称速度」与「RL 残差 Δv」分层可视化（一个 episode 一张诊断图）。

背景：residual 架构下真正下发给执行层的是 `v_cmd = v_nominal + Δv`，两者叠在一起看不出
谁在起作用。本脚本把 step() 内部的三层量分开画：
    v_nominal  —— 标称层输出（速度场 / MPC），env._v_nominal[:3]
    Δv         —— 残差层实际生效量（**过门控之后**），env._v_raw[:3] − env._v_nominal[:3]
    v_cmd      —— 平滑/限速/覆盖之后的实际命令，env._vel_target[:3]
（门控：closing/成功时 Δv 强制 0；MPC 无解时放宽 cap；近障碍时切平面投影——见
 `_apply_residual_gate`。所以 Δv 是"门控后"的量，正是真正起作用的残差。）

图 6 格：速度范数 / 标称三分量 / Δv 三分量 / XY 轨迹+箭头 / XZ 轨迹+箭头 / 残差占比。

下排两格的速度箭头**长度正比于速度大小**：v_nom 与 Δv 共用同一尺度，所以同格内灰箭头
与红箭头的长度比 == 真实速度比。尺度默认自动（`--arrow-frac`，让最长箭头 = 视野边长的
该比例），避免每换一集/一臂都要手调。
动态障碍会移动，下排用**渐变色轨迹**画出它整集的路径（红=起始 → 黄=结束，空心圈=起点，
半透明圆=终点），避免只画终点位置造成的误导。

用法：
    cd rl_grasping_system
    # MPC + 残差，并与纯 MPC 同集对照（虚线 = 纯标称轨迹）
    python3 scripts/plot_residual_vs_nominal.py --arm mpc_res25b2 --episode 4 --vs-nominal
    # 速度场 + 残差（v18），开姿态伺服
    python3 scripts/plot_residual_vs_nominal.py --arm vf_res18 --episode 4 --ori-servo on --vs-nominal
"""
import os
import sys
import argparse

os.environ.setdefault('MUJOCO_GL', 'egl')

import numpy as np

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from config import get_config
from record_armcompare_demo import build_env, obstacle_positions, ARMS

# 每个残差臂对应的「纯标称」孪生臂（做 --vs-nominal 对照用）；端到端没有标称层
NOMINAL_TWIN = {
    'vf_res18': 'vf_nominal',
    'vf_res25b2': 'vf_nominal',
    'mpc_res25b2': 'mpc_nominal',
    'mpc_armaware_res25b2': 'mpc_armaware',
    'vf_nominal': None,
    'mpc_nominal': None,
    'mpc_armaware': None,
    'e2e_v11': None,
}


def run_episode(cfg, scene, arm, seed, max_steps):
    """跑一集并记录逐决策步的标称/残差/命令三层量 + 末端轨迹。"""
    env, agent = build_env(cfg, scene, arm)
    try:
        obs, _ = env.reset(seed=seed)
        if env.nominal_trajectory is not None:
            env.nominal_trajectory.reset()
            # 与官方跑批对齐「第 1 步的门控状态」。MpcNominal.reset() 只清 warm-start
            # （mpc_nominal.py:86-88），solver_ok/terminal_err 会留着上一集终局的陈旧值。
            # 本脚本每集新建 env → terminal_err 是初始的 inf，被 _residual_unstuck()
            # （environment.py:1480）读成「MPC 无解」→ 残差预算翻倍（0.002→0.004 m/步，
            # environment.py:615-617），于是画出的这一集与跑批表里的同集步数对不上
            # （实测 ep17 200 vs 81、ep9 98 vs 99，2026-09-16）。官方跑批里第 1 步读到的是
            # 上一集终局的小值 → unstuck=False。这里按「未求解 = 不主张无解」置干净，
            # 使本图与表同口径。根治要改 MpcNominal.reset()，但那会动到已在跑的跑批。
            for _k, _clean in (('solver_ok', True), ('terminal_err', 0.0)):
                if hasattr(env.nominal_trajectory, _k):
                    setattr(env.nominal_trajectory, _k, _clean)
        rec = {k: [] for k in ('ee', 'v_nom', 'v_raw', 'v_cmd', 'obs')}
        info = {}
        for _ in range(max_steps):
            if agent is None:
                action = np.zeros(env.action_space.shape[0], dtype=np.float32)
            else:
                action, _ = agent.predict(obs, deterministic=True)
            obs, _r, term, trunc, info = env.step(action)
            rec['ee'].append(env._get_end_effector_position().copy())
            rec['v_nom'].append(np.asarray(getattr(env, '_v_nominal', np.zeros(6))[:3], dtype=float))
            rec['v_raw'].append(np.asarray(getattr(env, '_v_raw', np.zeros(6))[:3], dtype=float))
            rec['v_cmd'].append(np.asarray(env._vel_target[:3], dtype=float))
            _op = env._get_obstacle_positions()          # 动态障碍逐物理子步在动，须逐步记录
            rec['obs'].append(np.asarray(_op, dtype=float).reshape(-1, 3) if len(_op)
                              else np.zeros((0, 3)))
            if term or trunc:
                break
        obs_seq = rec.pop('obs')
        out = {k: np.asarray(v) for k, v in rec.items()}
        n_obs = obs_seq[0].shape[0] if obs_seq else 0
        if n_obs and all(o.shape[0] == n_obs for o in obs_seq):
            out['obs'] = np.asarray(obs_seq)             # (T, n_obs, 3)
        else:                                            # 障碍集合中途变了：放弃轨迹，退回终点
            out['obs'] = np.zeros((len(obs_seq), 0, 3))
        out['info'] = dict(success=bool(info.get('grasp_success', False)),
                           collisions=int(info.get('obstacle_collision_count', 0)))
        out['obstacles'] = obs_seq[-1] if obs_seq else obstacle_positions(env)
        out['object'] = env._get_object_position().copy()
        # 真实障碍半径取自 model（球 geom 的 size[0]），不从 config 猜——避免画出的球与仿真不符
        _og = getattr(env, 'obstacle_geom_ids_all', [])
        out['obs_radius'] = float(env.model.geom_size[_og[0]][0]) if _og else 0.05
        return out
    finally:
        env.close()


def _lim(ax, pts, margin=0.06):
    """按点集锁定视野（等比例），避免速度箭头把坐标范围撑爆。返回视野边长（数据单位）。"""
    pts = np.asarray(pts, dtype=float).reshape(-1, 2)
    lo, hi = pts.min(axis=0) - margin, pts.max(axis=0) + margin
    c, half = (lo + hi) / 2.0, np.max(hi - lo) / 2.0
    ax.set_xlim(c[0] - half, c[0] + half)
    ax.set_ylim(c[1] - half, c[1] + half)
    return 2.0 * half


def arrow_scale(vecs, span, frac):
    """求「最长箭头 = 视野边长 × frac」所需的缩放系数（数据单位 / (m/s)）。

    必须把同格内所有速度集合（v_nom 与 Δv）**合并后**取 vmax 再用同一个系数，
    否则两种箭头的长度比不等于真实速度比——而"相对长度可比"正是这张图的意义。
    """
    vmax = max((float(np.linalg.norm(v, axis=1).max()) for v in vecs if len(v)), default=0.0)
    return (frac * span / vmax) if vmax > 1e-9 else 0.0


OBS_CMAPS = ('Blues', 'Oranges', 'Greens', 'Purples')   # 每个障碍一个色系（同系内浅=早、深=晚）


def time_path(ax, xy, cmap='Greys', lw=1.6, ls='-', label=None, lo=0.30, hi=1.0, zorder=5):
    """按时间着色的轨迹：浅=起始 → 深=结束（与障碍扫掠带同一套时间语言）。

    用 LineCollection 逐段着色（普通 plot 只能整条一个颜色）；`lo` 抬一点是为了起点不白到看不见。
    """
    from matplotlib.collections import LineCollection
    from matplotlib.colors import Normalize
    from matplotlib.lines import Line2D
    import matplotlib.pyplot as plt
    xy = np.asarray(xy, dtype=float).reshape(-1, 2)
    n = len(xy)
    if n < 2:
        return
    cm = plt.get_cmap(cmap)
    segs = np.stack([xy[:-1], xy[1:]], axis=1)
    vals = lo + (hi - lo) * (np.arange(n - 1, dtype=float) / max(n - 2, 1))
    ax.add_collection(LineCollection(segs, cmap=cm, norm=Normalize(lo, hi), array=vals,
                                     linewidths=lw, linestyles=ls, zorder=zorder))
    if label:
        ax.add_line(Line2D([], [], color=cm(hi), lw=lw, ls=ls, label=label))


def obs_trail(ax, obs_seq, radius=None, first_label=True):
    """动态障碍的整集轨迹：沿轨迹按**真实半径平移画球**，颜色由浅到深表示时间推进。

    为什么这么画：细线看不出"球有多大、扫过哪里"。按真实半径（model 里 0.05 m）沿轨迹铺一串
    半透明圆，重叠处自然加深——等于把**扫掠带**画出来，同时球径与仿真相符。
    ⚠️ 不能用散点/标记代替：`ms`/`s` 的单位是**点**，与数据坐标无关，在 0.5 m 视野里
    12pt 标记 ≈ 0.02 m，比真球小一半以上，会严重失真。
    多个球各用一种色系（blues/oranges/…），便于分辨是哪个球被推到了哪。
    """
    from matplotlib.lines import Line2D
    from matplotlib.patches import Circle, Patch
    import matplotlib.pyplot as plt
    if obs_seq.size == 0:
        return None
    n = obs_seq.shape[0]
    if n < 2:
        return obs_seq[:, :, [0, 1]].reshape(-1, 2)
    r = float(radius) if radius and radius > 0 else 0.05
    for j in range(obs_seq.shape[1]):
        p = obs_seq[:, j]
        cmap = plt.get_cmap(OBS_CMAPS[j % len(OBS_CMAPS)])
        # 按**弧长每 r 米铺一个球**（刚好相接），而不是每决策步一个：几十个圆叠在一起时
        # alpha 会按 1-(1-a)^N 迅速饱和到底，"调透明度"根本调不动，糊成一坨色块。
        # 相接铺法下，直线段只有 2~3 个圆重叠（浅），折返/停留处重叠多（深）——深浅自带信息。
        seg = np.linalg.norm(np.diff(p[:, [0, 1]], axis=0), axis=1)
        s = np.concatenate([[0.0], np.cumsum(seg)])
        if s[-1] > 1e-9:
            idx = np.unique(np.searchsorted(s, np.arange(0.0, s[-1], r)))
            idx = np.append(idx, n - 1)
        else:
            idx = np.array([0, n - 1])
        for k in idx:
            c = cmap(0.25 + 0.75 * (s[k] / max(s[-1], 1e-9)))   # 起始最浅 → 结束最深
            ax.add_patch(Circle((p[k, 0], p[k, 1]), r, facecolor=c, edgecolor='none',
                                alpha=0.30, zorder=3))
        ax.add_patch(Circle((p[-1, 0], p[-1, 1]), r, facecolor='none', edgecolor=cmap(1.0),
                            lw=1.6, zorder=4))              # 终点描边：加深色边 = 当前位置
    if first_label:                                          # Patch 不进图例，显式给代理句柄
        for j in range(obs_seq.shape[1]):
            cmap = plt.get_cmap(OBS_CMAPS[j % len(OBS_CMAPS)])
            ax.add_line(Line2D([], [], color=cmap(0.95), lw=0, marker='o', ms=9, alpha=0.75,
                               label=f'obstacle {j}: light=t0 → dark=tEnd (r={r:.3f} m)'))
    return obs_seq[:, :, [0, 1]].reshape(-1, 2)               # 供 _lim 纳入视野


def quiver2d(ax, xy, vec, idx, scale, color, label):
    """在 (x, y) 平面画速度矢量（只取给定下标的步）；箭头长 = 速度 × scale。"""
    if len(idx) == 0 or scale <= 0:
        return
    p = xy[idx]
    v = vec[idx][:, :2] * scale
    ax.quiver(p[:, 0], p[:, 1], v[:, 0], v[:, 1], angles='xy', scale_units='xy',
              scale=1.0, color=color, width=0.005, label=label)


def quiver_xz(ax, xyz, vec, idx, scale, color, label):
    if len(idx) == 0 or scale <= 0:
        return
    p = xyz[idx]
    v = vec[idx][:, [0, 2]] * scale
    ax.quiver(p[:, 0], p[:, 2], v[:, 0], v[:, 1], angles='xy', scale_units='xy',
              scale=1.0, color=color, width=0.005, label=label)


def legend_below(ax, **kw):
    """图例放到坐标轴**外面**（贴在下方、x 轴标签之后），不压任何数据。"""
    kw.setdefault('fontsize', 6.5)
    kw.setdefault('ncol', 2)
    ax.legend(loc='upper center', bbox_to_anchor=(0.5, -0.20), frameon=False,
              borderpad=0.2, handletextpad=0.5, columnspacing=1.0, **kw)


def load_eval_table(table_dir, arm, twin, scene, servo, n=30):
    """读两个臂的**整轮评测**聚合指标 + 逐集记录。

    逐集记录是画配对对比的前提：两个臂逐集用同一 seed（`--seed-per-episode`），
    所以第 k 集可以直接对齐做**配对检验**；拿两个独立平均值相减会把方差平均掉，
    什么都"显著"。任一文件缺失则返回 None（表整块跳过）。
    """
    import json
    rows = []
    for tag, name in ((twin, f'WITHOUT RL   ({twin})'), (arm, f'WITH RL   ({arm})')):
        if not tag:
            print('[table] 该臂无纯标称孪生臂（端到端），跳过对比表')
            return None
        p = os.path.join(table_dir, f'{tag}_{scene}_trainmatch_servo{servo}_n{n}.json')
        if not os.path.exists(p):
            print(f'[table] 缺 {p} → 跳过对比表')
            return None
        d = json.load(open(p))
        eps = d.get('episode_results') or []
        rows.append(dict(name=name, nc=float(d['success_nc_rate']), cr=float(d['collision_rate']),
                         ac=float(d['avg_collision_count']), ln=float(d['avg_episode_length']),
                         sr=float(d['success_rate']), n=int(d['n_episodes']),
                         steps=np.array([e['episode_length'] for e in eps], dtype=float),
                         ok=np.array([bool(e['grasp_success']) for e in eps], dtype=bool),
                         # 复合判据的逐集真值：抓取成功**且**零碰撞。表里 nc 那一列的配对检验
                         # 必须用它——否则列头写 nc、检验的却是抓取成功（vf 臂上两者差 20pp+，
                         # 会得出"nc 差异不显著"这种列头与检验不匹配的结论）。
                         ncok=np.array([bool(e['grasp_success']) and e['collision_count'] == 0
                                        for e in eps], dtype=bool)))
    return rows


def _binom_two_sided(k, n, p=0.5):
    """精确二项检验（双侧）。自己算，不为一张图引入 scipy 依赖。"""
    if n == 0:
        return 1.0
    from math import comb
    pmf = [comb(n, i) * p ** i * (1 - p) ** (n - i) for i in range(n + 1)]
    return float(min(1.0, 2 * min(sum(pmf[:k + 1]), sum(pmf[k:]))))


def wilson_ci(k, n, z=1.96):
    """成功率的 Wilson 95% 区间——小样本比正态近似稳，且不会越出 [0,1]。"""
    if n == 0:
        return 0.0, 0.0
    ph = k / n; den = 1 + z * z / n
    c = (ph + z * z / (2 * n)) / den
    h = z * np.sqrt(ph * (1 - ph) / n + z * z / (4 * n * n)) / den
    return max(0.0, c - h), min(1.0, c + h)


def paired_stats(rows):
    """逐集配对统计。成功率用 McNemar 精确检验；步数只在**两臂都成功**的子集上比。

    步数为什么取交集：一集失败 = 超时跑满 max_steps，把超时混进平均会把差异放大成
    「谁失败得少谁就快」的同义反复，那不是残差的功劳。交集内比较才是纯粹的
    「同一集、同一标称，加不加残差谁先抓到」。
    """
    a, b = rows
    oka, okb = a['ok'], b['ok']
    up = int(np.sum(~oka & okb))            # 无 RL 失败 → 加 RL 成功
    dn = int(np.sum(oka & ~okb))            # 反向
    # 复合判据（抓取且零碰撞）的另一套配对计数：表里 nc 那一列用它。
    nca, ncb = a['ncok'], b['ncok']
    up_nc = int(np.sum(~nca & ncb)); dn_nc = int(np.sum(nca & ~ncb))
    both = oka & okb
    d = a['steps'][both] - b['steps'][both]   # >0 = 加 RL 更快
    faster = int(np.sum(d > 0)); slower = int(np.sum(d < 0))
    return dict(n=len(oka), up=up, dn=dn, p_succ=_binom_two_sided(up, up + dn),
                up_nc=up_nc, dn_nc=dn_nc, p_nc=_binom_two_sided(up_nc, up_nc + dn_nc),
                both=int(both.sum()), both_fail=int(np.sum(~oka & ~okb)),
                n_pair=int(both.sum()), mean_d=float(d.mean()) if len(d) else 0.0,
                med_d=float(np.median(d)) if len(d) else 0.0,
                faster=faster, slower=slower,
                p_step=_binom_two_sided(faster, faster + slower),
                pair_idx=np.flatnonzero(both), d=d)


def pick_median_episode(rows, stats=None):
    """按**声明在先的规则**选集：Δ 步数最接近中位的那一集（避免挑展示效果最好的一集）。"""
    st = stats or paired_stats(rows)
    if not st['n_pair']:
        return None, st
    i = int(st['pair_idx'][int(np.argmin(np.abs(st['d'] - st['med_d'])))])
    return i, st


def draw_eval_table(ax_t, rows, servo, stats=None):
    """在给定 axes 上画聚合对比表（Δ 行 + 配对检验行）。"""
    # 表内文字一律用英文：本机 matplotlib 无 CJK 字体（ttflist 里 0 个），
    # 中文字形会渲染成 tofu/?，而整张图其余文字本就全英文。
    hdr = ['method', 'no-collision success', 'collision rate', 'mean collisions',
           'mean steps (time)', 'grasp success']
    a, b = rows[0], rows[1]
    body = [
        [a['name'], f"{100*a['nc']:.1f}%", f"{100*a['cr']:.1f}%", f"{a['ac']:.2f}",
         f"{a['ln']:.1f}", f"{100*a['sr']:.1f}%"],
        [b['name'], f"{100*b['nc']:.1f}%", f"{100*b['cr']:.1f}%", f"{b['ac']:.2f}",
         f"{b['ln']:.1f}", f"{100*b['sr']:.1f}%"],
        [f"Δ  (WITH RL − WITHOUT)", f"{100*(b['nc']-a['nc']):+.1f} pp",
         f"{100*(b['cr']-a['cr']):+.1f} pp", f"{b['ac']-a['ac']:+.2f}",
         f"{b['ln']-a['ln']:+.1f}", f"{100*(b['sr']-a['sr']):+.1f} pp"],
    ]
    extra = 0
    if stats is not None:
        body.append([f"paired test (same seed per ep, n={stats['n']})",
                     f"McNemar p={stats['p_nc']:.2f}"
                     f"  ({stats['up_nc']}↑ / {stats['dn_nc']}↓)",
                     '—', '—',
                     f"sign test p={stats['p_step']:.1e}"
                     f"  ({stats['faster']}/{stats['n_pair']} faster)",
                     f"McNemar p={stats['p_succ']:.2f}"
                     f"  ({stats['up']}↑ / {stats['dn']}↓)"])
        extra = 1
    ax_t.axis('off')
    t = ax_t.table(cellText=body, colLabels=hdr, loc='center', cellLoc='center')
    t.auto_set_font_size(False); t.set_fontsize(8.5); t.scale(1.0, 1.20)
    for (r, c), cell in t.get_celld().items():
        cell.set_edgecolor('#999999')
        if r == 0:
            cell.set_facecolor('#dcdcdc'); cell.set_text_props(weight='bold', fontsize=7.5)
        elif r == 3:
            cell.set_facecolor('#fff3cd')
        elif extra and r == 4:
            cell.set_facecolor('#dfe9f5'); cell.set_text_props(fontsize=7.5)
        if c == 0 and not (extra and r == 4):
            cell.set_text_props(ha='left', fontsize=8)
    ax_t.set_title('controlled A/B — aggregate over the whole eval run  '
                   f'(n={rows[0]["n"]} episodes, seeded 12345+ep, servo={servo})',
                   fontsize=9, pad=10)


def draw_paired_panel(ax, rows, stats, picked=None):
    """配对斜率图：每集一条竖线（左=无 RL，右=加 RL），只画两臂都成功的集。

    比两根柱子强在哪：柱子只给均值，看不出「16 集里 16 集都更快」这种一致性；
    配对线一眼能看出方向和离散度，也不给挑集的自由度。
    """
    a, b = rows
    i = stats['pair_idx']
    for j in i:
        hot = (picked is not None and j == picked)
        ax.plot([0, 1], [a['steps'][j], b['steps'][j]], '-',
                color=('tab:red' if hot else '0.55'), lw=(2.4 if hot else 1.1),
                alpha=(1.0 if hot else 0.75), zorder=(4 if hot else 2))
    ax.plot(np.zeros(len(i)), a['steps'][i], 'o', color='0.35', ms=4, zorder=3)
    ax.plot(np.ones(len(i)), b['steps'][i], 'o', color='tab:red', ms=4, zorder=3)
    ax.plot([0, 1], [a['steps'][i].mean(), b['steps'][i].mean()], '-', color='k', lw=2.6, zorder=5,
            label=f"mean {a['steps'][i].mean():.1f} → {b['steps'][i].mean():.1f} steps")
    ax.plot([0], [a['steps'][i].mean()], 's', color='k', ms=8, zorder=6)
    ax.plot([1], [b['steps'][i].mean()], 's', color='k', ms=8, zorder=6)
    if picked is not None:
        ax.plot([0, 1], [a['steps'][picked], b['steps'][picked]], '-', color='tab:red', lw=2.4,
                zorder=4, label=f'episode {picked} = the one drawn above')
    ax.set_xticks([0, 1])
    ax.set_xticklabels([f"WITHOUT RL\n({a['name'].split('(')[1][:-1]})",
                        f"WITH RL\n({b['name'].split('(')[1][:-1]})"], fontsize=8)
    ax.set_xlim(-0.35, 1.35)
    ax.set_ylabel('episode length (decision steps)')
    ax.set_title(f"Paired per-episode time-to-grasp — the {stats['n_pair']} episodes both arms grasp\n"
                 f"{stats['faster']}/{stats['n_pair']} faster with RL, sign test p={stats['p_step']:.1e}"
                 f"   |   mean Δ={stats['mean_d']:+.1f} steps (median {stats['med_d']:+.1f})",
                 fontsize=10)
    ax.grid(alpha=0.3, axis='y')
    ax.legend(loc='upper right', fontsize=7.5, frameon=False)


def draw_success_panel(ax, rows, stats):
    """成功率 + Wilson 95% 区间：把「+6.7pp 不显著」这件事直接画出来。"""
    a, b = rows
    ks = [int(a['ok'].sum()), int(b['ok'].sum())]
    ns = [int(a['ok'].size), int(b['ok'].size)]
    rates = [100 * k / nn for k, nn in zip(ks, ns)]
    lo, hi = zip(*[wilson_ci(k, nn) for k, nn in zip(ks, ns)])
    err = [[rates[i] - 100 * lo[i] for i in range(2)], [100 * hi[i] - rates[i] for i in range(2)]]
    ax.bar([0, 1], rates, width=0.55, color=['tab:gray', 'tab:red'], alpha=0.75,
           yerr=err, capsize=7, error_kw=dict(ecolor='k', lw=1.2))
    for x, r, k, nn, h in zip([0, 1], rates, ks, ns, hi):
        ax.text(x, max(r, 100 * h) + 6, f'{k}/{nn}  {r:.1f}%', ha='center', fontsize=8.5)
    ax.set_xticks([0, 1]); ax.set_xticklabels(['WITHOUT RL', 'WITH RL'], fontsize=8.5)
    ax.set_xlim(-0.6, 1.6); ax.set_ylim(0, 100)
    ax.set_ylabel('grasp success (%)')
    ax.set_title(f"Success with Wilson 95% CI — overlap means not significant\n"
                 f"{stats['up']} episodes ↑ / {stats['dn']} ↓, McNemar p={stats['p_succ']:.2f}"
                 f"   |   both-fail {stats['both_fail']} / both-grasp {stats['both']}",
                 fontsize=10)
    ax.grid(alpha=0.3, axis='y')


def main():
    ap = argparse.ArgumentParser(description='标称速度 vs RL 残差 Δv 分层可视化')
    ap.add_argument('--scene', default='dyn_both_train')
    ap.add_argument('--arm', default='mpc_res25b2', choices=list(ARMS))
    # 不给 --episode 时：按「Δ 步数最接近中位」的规则自动选集（规则声明在先，避免挑好看的集）
    ap.add_argument('--episode', type=int, default=None,
                    help='不给则按中位 Δ 步数自动选集（避免挑樱桃）；给了就用给定集')
    ap.add_argument('--no-pick', action='store_true',
                    help='关掉自动选集，退回第 0 集')
    ap.add_argument('--seed', type=int, default=12345)
    ap.add_argument('--d-safe', type=float, default=None)
    ap.add_argument('--ori-servo', default=None, choices=['on', 'off'])
    ap.add_argument('--max-steps', type=int, default=200)
    ap.add_argument('--arrow-every', type=int, default=12, help='每 N 决策步画一个速度箭头')
    ap.add_argument('--arrow-frac', type=float, default=0.14,
                    help='自动定标：最长箭头 = 视野边长的该比例（v_nom 与 Δv 共用，长度比=速度比）')
    ap.add_argument('--arrow-scale', type=float, default=None,
                    help='手动覆盖：箭头长 = 速度 × 该值（单位秒）。不给则用 --arrow-frac 自动定标')
    # 默认开：这张图的意义就是「同标称 ± RL」的受控对照，跨标称比较不是本图目的。
    # 端到端臂（e2e_v11）没有标称层，NOMINAL_TWIN 为 None，自动跳过。
    ap.add_argument('--vs-nominal', dest='vs_nominal', action='store_true', default=True,
                    help='（默认开）同集再跑一次**同一个标称**的纯标称臂做对照')
    ap.add_argument('--no-vs-nominal', dest='vs_nominal', action='store_false',
                    help='只跑单个臂，不做对照（快一倍）')
    ap.add_argument('--table-dir', default='results/unified7_trainmatch_fix',
                    help='整轮评测 JSON 所在目录（对比表的数据源）')
    ap.add_argument('--table-n', type=int, default=30, help='对比表对应的集数（须与文件名一致）')
    ap.add_argument('--no-table', action='store_true', help='不画对比表')
    ap.add_argument('--outdir', default='results/plot_residual')
    args = ap.parse_args()

    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    cfg = get_config()
    cfg.grasping.max_steps = args.max_steps
    if args.d_safe is not None:
        cfg.grasping.mpc_nominal_d_safe = args.d_safe
    if args.ori_servo is not None:
        cfg.grasping.orientation_servo_enabled = (args.ori_servo == 'on')

    twin = NOMINAL_TWIN.get(args.arm)
    servo = 'on' if cfg.grasping.orientation_servo_enabled else 'off'
    rows = None if args.no_table else load_eval_table(
        args.table_dir, args.arm, twin, args.scene, servo, args.table_n)
    stats = paired_stats(rows) if rows is not None else None

    if args.episode is None:                     # 选集规则声明在先：Δ 步数最接近中位者
        ep = 0
        if args.no_pick:
            print('[pick] --no-pick → 用第 0 集')
        elif stats is not None:
            ep, _ = pick_median_episode(rows, stats)
            if ep is not None:
                print(f"[pick] 未指定 --episode → 取 Δ 步数最接近中位的第 {ep} 集："
                      f"Δ={(rows[0]['steps'][ep]-rows[1]['steps'][ep]):+.0f} 步，"
                      f"中位 Δ={stats['med_d']:+.1f}，全集 Δ 范围 "
                      f"[{stats['d'].min():+.0f}, {stats['d'].max():+.0f}]")
            else:
                ep = 0
                print('[pick] 两臂无共同成功集 → 退回第 0 集')
        args.episode = ep
    print(f"[pick] 本图用第 {args.episode} 集（seed {args.seed + args.episode}）")

    seed = args.seed + args.episode
    main = run_episode(cfg, args.scene, args.arm, seed, args.max_steps)
    ref = None
    if args.vs_nominal and twin:
        ref = run_episode(cfg, args.scene, twin, seed, args.max_steps)
    elif args.vs_nominal:
        print(f'[plot] {args.arm} 无纯标称孪生臂（端到端），跳过 --vs-nominal')

    dv = main['v_raw'] - main['v_nom']          # 门控后的残差 Δv
    n = len(main['ee'])
    t = np.arange(n)
    idx = np.arange(0, n, args.arrow_every)

    # 4 行 GridSpec：① ② 机制面板（单集）③ 配对聚合面板 ④ 聚合对比表。
    # hspace 给得大是为了把图例塞到每个坐标轴**外面**（贴在各图下方），不再压数据。
    fig = plt.figure(figsize=(17.5, 14.2))
    gs = fig.add_gridspec(4, 3, height_ratios=[1.0, 1.0, 0.62, 0.42],
                          hspace=0.85, wspace=0.26)
    ax = [fig.add_subplot(gs[0, i]) for i in range(3)] + \
         [fig.add_subplot(gs[1, i]) for i in range(3)] + \
         [fig.add_subplot(gs[2, :2]), fig.add_subplot(gs[2, 2]), fig.add_subplot(gs[3, :])]

    # 对照臂（同标称、不加 RL）的时间轴可能与本臂步数不同（先结束/先超时），各自用自己的 x
    rt = np.arange(len(ref['ee'])) if ref is not None else None
    NOM_C = ('gray', 'olive', 'brown')
    DV_C = ('red', 'orange', 'purple')

    # (1) 范数
    ax[0].plot(t, np.linalg.norm(main['v_nom'], axis=1), label=r'$\|v_{nom}\|$', color='tab:gray')
    if ref is not None:
        ax[0].plot(rt, np.linalg.norm(ref['v_nom'], axis=1), '--', color='tab:gray', lw=1.3,
                   alpha=0.85, label=r'$\|v_{nom}\|$ of same nominal WITHOUT RL')
    ax[0].plot(t, np.linalg.norm(dv, axis=1), label=r'$\|\Delta v\|$ (post-gate residual)', color='tab:red')
    ax[0].plot(t, np.linalg.norm(main['v_cmd'], axis=1), label=r'$\|v_{cmd}\|$ (executed cmd)', color='tab:blue', alpha=0.7)
    ax[0].set_title('Speed norms (m/s)'); ax[0].set_xlabel('decision step'); ax[0].set_ylabel('m/s')
    legend_below(ax[0], ncol=2)

    # (2) 标称三分量（实线=加 RL，虚线=同标称不加 RL；标称层本身被 RL 改变正是看点）
    for i, c in enumerate(('x', 'y', 'z')):
        ax[1].plot(t, main['v_nom'][:, i], label=f'v_nom.{c}', color=f'tab:{NOM_C[i]}')
        if ref is not None:
            ax[1].plot(rt, ref['v_nom'][:, i], '--', color=f'tab:{NOM_C[i]}', lw=1.2,
                       alpha=0.8, label=f'v_nom.{c} (no RL)')
    ax[1].set_title('Nominal velocity $v_{nom}$ (solid=with RL, dashed=same nominal without)')
    ax[1].set_xlabel('decision step'); ax[1].set_ylabel('m/s')
    legend_below(ax[1], ncol=3, fontsize=6.5)

    # (3) Δv 三分量（对照臂按定义 Δv≡0，无需画）
    for i, c in enumerate(('x', 'y', 'z')):
        ax[2].plot(t, dv[:, i], label=f'Δv.{c}', color=f'tab:{DV_C[i]}')
    ax[2].set_title(r'Residual $\Delta v$ components (post-gate)'); ax[2].set_xlabel('decision step'); ax[2].set_ylabel('m/s')
    legend_below(ax[2], ncol=3)

    # (4) XY 俯视
    ee = main['ee']
    time_path(ax[3], ee[:, :2], label='EE path (this arm: light=t0 → dark=tEnd)')
    if ref is not None:
        time_path(ax[3], ref['ee'][:, :2], cmap='Blues', ls='--',
                  label=f'EE path ({twin})')
    tr4 = obs_trail(ax[3], main['obs'], radius=main.get('obs_radius'))
    ax[3].plot(main['object'][0], main['object'][1], 'g*', ms=14, label='object')
    pts4 = [ee[:, :2], main['object'][None, :2]] + ([tr4] if tr4 is not None else [])
    span4 = _lim(ax[3], np.vstack(pts4), margin=0.06)
    sc4 = args.arrow_scale if args.arrow_scale else arrow_scale(
        [main['v_nom'][idx][:, :2], dv[idx][:, :2]], span4, args.arrow_frac)
    quiver2d(ax[3], ee, main['v_nom'], idx, sc4, 'tab:gray', r'$v_{nom}$')
    quiver2d(ax[3], ee, dv, idx, sc4, 'tab:red', r'$\Delta v$')
    ax[3].set_title(f'Top view XY: path + velocity vectors  (longest arrow = {args.arrow_frac:.0%} of span)')
    ax[3].set_xlabel('x (m)'); ax[3].set_ylabel('y (m)')
    ax[3].set_aspect('equal'); ax[3].grid(alpha=0.3)
    legend_below(ax[3], ncol=2, fontsize=6.2)

    # (5) XZ 侧视（障碍按 x-z 投影，y 被丢掉 → 不同 y 的球会重叠）
    time_path(ax[4], ee[:, [0, 2]], label='EE path (this arm: light=t0 → dark=tEnd)')
    if ref is not None:
        time_path(ax[4], ref['ee'][:, [0, 2]], cmap='Blues', ls='--',
                  label=f'EE path ({twin})')
    tr5 = obs_trail(ax[4], main['obs'][:, :, [0, 2]], radius=main.get('obs_radius'),
                    first_label=False)
    ax[4].plot(main['object'][0], main['object'][2], 'g*', ms=14, label='object')
    pts5 = [ee[:, [0, 2]], main['object'][None, [0, 2]]] + ([tr5] if tr5 is not None else [])
    span5 = _lim(ax[4], np.vstack(pts5), margin=0.06)
    sc5 = args.arrow_scale if args.arrow_scale else arrow_scale(
        [main['v_nom'][idx][:, [0, 2]], dv[idx][:, [0, 2]]], span5, args.arrow_frac)
    quiver_xz(ax[4], ee, main['v_nom'], idx, sc5, 'tab:gray', r'$v_{nom}$')
    quiver_xz(ax[4], ee, dv, idx, sc5, 'tab:red', r'$\Delta v$')
    ax[4].set_title('Side view XZ (y projected): path + velocity vectors')
    ax[4].set_xlabel('x (m)'); ax[4].set_ylabel('z (m)')
    ax[4].set_aspect('equal'); ax[4].grid(alpha=0.3)
    legend_below(ax[4], ncol=2, fontsize=6.2)

    # (6) 残差占比
    nv, dvn = np.linalg.norm(main['v_nom'], axis=1), np.linalg.norm(dv, axis=1)
    share = dvn / np.maximum(nv + dvn, 1e-9)
    ax[5].plot(t, 100.0 * share, color='tab:red', label='this arm (with RL)')
    ax[5].set_ylim(0, 100)
    ax[5].set_title(r'Residual share $\|\Delta v\|/(\|v_{nom}\|+\|\Delta v\|)$')
    ax[5].set_xlabel('decision step'); ax[5].set_ylabel('%')
    ax[5].axhline(100 * float(np.mean(share)), ls='--', c='gray', lw=1,
                  label=f'mean {100*float(np.mean(share)):.1f}%')
    if ref is not None:                    # 对照臂按定义 Δv≡0 → 占比恒 0
        ax[5].axhline(0.0, ls=':', c='tab:gray', lw=1.6, label='same nominal WITHOUT RL (≡0%)')
    ax[5].grid(alpha=0.3); legend_below(ax[5], ncol=2)

    supt = (f"{args.arm} @ {args.scene} ep{args.episode} (seed {seed}) | servo={servo} | "
            f"d_safe={cfg.grasping.mpc_nominal_d_safe:g} | "
            f"{'SUCCESS' if main['info']['success'] else 'FAIL'} "
            f"collisions={main['info']['collisions']} steps={n}")
    if ref is not None:                    # 说明这是「同标称 ± RL」的受控 A/B，不是跨标称比较
        supt += (f"\ncontrolled A/B — SAME nominal ({twin}):  solid = +RL,  dashed = no RL"
                 f"   |   without RL: collisions={ref['info']['collisions']} "
                 f"steps={len(ref['ee'])}"
                 f"\nepisode chosen by rule, not by hand:  "
                 f"Δ steps closest to the median of the {stats['n_pair'] if stats else 0} "
                 f"episodes both arms grasp")
    fig.suptitle(supt, fontsize=12, y=0.985, va='top')

    # 第 3 行：配对聚合（斜率图 + 成功率 CI）；第 4 行：聚合对比表。
    # 为什么必须有这一行：单集只有 0%/100%，而两根柱子（均值）看不出「16 集里 16 集都更快」
    # 这种一致性，也给挑集的自由度。配对检验是这张图唯一撑得起结论的东西。
    if rows is not None:
        draw_paired_panel(ax[6], rows, stats, picked=args.episode)
        draw_success_panel(ax[7], rows, stats)
        draw_eval_table(ax[8], rows, servo, stats)
        a, b = rows
        print(f"[table] n={a['n']} 无碰撞成功率 {100*a['nc']:.1f}% → {100*b['nc']:.1f}% "
              f"({100*(b['nc']-a['nc']):+.1f}pp) | 碰撞率 {100*a['cr']:.1f}% → {100*b['cr']:.1f}% "
              f"| 平均耗时 {a['ln']:.1f} → {b['ln']:.1f} 步")
        print(f"[paired] 成功率 {stats['up']}↑/{stats['dn']}↓ McNemar p={stats['p_succ']:.3f} | "
              f"都成功的 {stats['n_pair']} 集步数 {a['steps'][stats['pair_idx']].mean():.1f} → "
              f"{b['steps'][stats['pair_idx']].mean():.1f}，{stats['faster']}/{stats['n_pair']} 更快，"
              f"符号检验 p={stats['p_step']:.2e}")
    else:
        for k in (6, 7, 8):
            ax[k].axis('off')
        ax[6].text(0.5, 0.5, 'aggregate panels unavailable (missing eval JSONs)',
                   ha='center', va='center', fontsize=9, color='gray')

    fig.subplots_adjust(left=0.055, right=0.985, top=0.90, bottom=0.02)

    os.makedirs(args.outdir, exist_ok=True)
    tag = f"{args.arm}_{args.scene}_servo{servo}_ep{args.episode}"
    png = os.path.join(args.outdir, f'{tag}.png')
    fig.savefig(png, dpi=115)
    print(f'[plot] {supt}')
    print(f'       -> {png}')
    print(f"       mean |dv|={dvn.mean():.4f} m/s  mean |v_nom|={nv.mean():.4f} m/s  residual share={100*share.mean():.1f}%")
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
