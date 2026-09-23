#!/usr/bin/env python3
"""画两种图（2026-09-17，英文标签）：

  fig1_reward_composition : 奖励分项构成对比（左=每集堆叠构成，右=剔除 r_obstacle 后的分组对比）
  fig2_reward_profile     : 每步奖励剖面（归一化时间轴 0→1，小多图，每项一格，5 方法各一条线）

数据来源 = mpc_plus_rl_eval.py 产出的 JSON（需 2026-09-17 之后带 reward_profile_* 的版本）。
用法：python3 scripts/plot_reward_breakdown.py --dir results/table5_n200_breakdown
"""
import json, os, sys, argparse
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ARMS = [
    ("Pure tracking servo", "vf_nominal",   "#1f77b4"),
    ("Tracking servo + RL", "vf_res18",     "#ff7f0e"),
    ("RL (end-to-end)",     "e2e_v11",      "#2ca02c"),
    ("MPC",                 "mpc_nominal",  "#d62728"),
    ("MPC + RL",            "mpc_res25b2",  "#9467bd"),
]
# 键 -> (英文标签, 是否事件项)   绘图顺序按"信息量"排：先事件/区分项，后近常数项
TERMS = [
    ("r_grasp",         "grasp success",        True),
    ("r_avoid",         "clean-success bonus",  True),
    ("r_close",         "closing trigger",      True),
    ("r_contact",       "contact (level)",      False),
    ("r_dist_z",        "Z distance shaping",   False),
    ("r_dist_xy",       "XY distance shaping",  False),
    ("r_obstacle",      "obstacle proximity",   False),
    ("r_residual",      "residual magnitude",   False),
    ("r_residual_step", "residual active step", False),
    ("r_collision",     "collision (event pen.)", False),
    ("r_singularity",   "singularity",          False),
    ("r_step",          "time step",            False),
]


def per_step_means(d):
    """每步均值 = 总奖励 / 总步数，从 episode_results 现算（口径见 print_reward_breakdown.py）。"""
    eps = d['episode_results']
    tot = sum(max(1, e['episode_length']) for e in eps)
    keys = {k for e in eps for k in e.get('reward_breakdown', {})}
    return {k: sum(float(e.get('reward_breakdown', {}).get(k, 0.0)) for e in eps) / tot
            for k in keys}


def load_all(d, scene, n, require_profile=False):
    out, missing, stale = {}, [], []
    for label, tag, color in ARMS:
        p = os.path.join(d, f"{tag}_{scene}_n{n}.json")
        if not os.path.exists(p):
            missing.append(label)
            continue
        with open(p, encoding="utf-8") as f:
            j = json.load(f)
        if 'reward_breakdown_per_episode' not in j or \
                (require_profile and 'reward_stepcum_mean' not in j):
            stale.append(label)
            continue
        out[label] = dict(color=color, tag=tag, json=j,
                          per_step=per_step_means(j),
                          has_profile='reward_stepcum_mean' in j)
    for m in missing:
        print(f"[缺失] {m}", file=sys.stderr)
    for s in stale:
        print(f"[旧口径] {s}: 缺 reward_breakdown_*/reward_profile_* —— 需重跑", file=sys.stderr)
    return out


def fig_composition(D, scene, n, outdir):
    labels = list(D)
    order = [t for t, _, _ in TERMS]
    pos = [k for k in order if not k.startswith(('r_obstacle',)) and not all(
        abs(D[l]['json']['reward_breakdown_per_episode'].get(k, 0.0)) < 1e-9 for l in labels)]
    neg = [k for k in order if k not in pos and not all(
        abs(D[l]['json']['reward_breakdown_per_episode'].get(k, 0.0)) < 1e-9 for l in labels)]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6.2))

    # 左：每集堆叠构成（正项向上、负项向下分开堆，避免正负互相抵消看不出量级）
    x = np.arange(len(labels))
    bot_p = np.zeros(len(labels)); bot_n = np.zeros(len(labels))
    for k in order:
        vals = np.array([D[l]['json']['reward_breakdown_per_episode'].get(k, 0.0) for l in labels])
        if np.all(np.abs(vals) < 1e-9):
            continue
        name = dict((a, b) for a, b, _ in TERMS)[k]
        if vals.mean() >= 0:
            ax1.bar(x, vals, 0.6, bottom=bot_p, label=name, alpha=0.9)
            bot_p += vals
        else:
            ax1.bar(x, vals, 0.6, bottom=bot_n, label=name, alpha=0.9)
            bot_n += vals
    ax1.axhline(0, color='k', lw=0.8)
    ax1.set_xticks(x); ax1.set_xticklabels(labels, rotation=20, ha='right')
    ax1.set_ylabel("reward per episode")
    ax1.set_title(f"(a) Episode return composition\n({scene}, n={n})")
    ax1.legend(fontsize=8, ncol=2, loc='upper center', bbox_to_anchor=(0.5, -0.16))
    ax1.grid(axis='y', alpha=0.25)
    for i, l in enumerate(labels):
        net = D[l]['json']['avg_episode_reward']
        # 净回报落在堆叠内部（正负两条堆叠各自延展），直接画会被柱色吃掉 —— 加白底
        ax1.annotate(f"{net:+.1f}", (i, net),
                     textcoords="offset points", xytext=(0, 6 if net >= 0 else -14),
                     ha='center', fontsize=8.5, fontweight='bold',
                     bbox=dict(boxstyle='round,pad=0.18', fc='white', ec='0.6', lw=0.5, alpha=0.9))

    # 右：剔掉 r_obstacle 的分组对比（它一项就把其余全压平了）
    pos2 = [k for k in pos + neg if k != 'r_obstacle']
    y = np.arange(len(pos2)); w = 0.8 / len(labels)
    for i, l in enumerate(labels):
        vals = [D[l]['per_step'].get(k, 0.0) for k in pos2]
        ax2.barh(y + i * w - 0.4 + w / 2, vals, w, label=l, color=D[l]['color'], alpha=0.9)
    ax2.axvline(0, color='k', lw=0.8)
    ax2.set_yticks(y); ax2.set_yticklabels([dict((a, b) for a, b, _ in TERMS)[k] for k in pos2])
    ax2.set_xlabel("reward per decision step")
    ax2.set_title("(b) Per-step means, obstacle term excluded\n(its -0.8/step swamps everything else)")
    ax2.legend(fontsize=8)
    ax2.grid(axis='x', alpha=0.25)
    fig.tight_layout()
    p = os.path.join(outdir, "fig1_reward_composition.png")
    fig.savefig(p, dpi=150)
    plt.close(fig)
    print("saved ->", p)


def fig_profile(D, scene, n, outdir, success_only=False):
    """事件项画累计曲线（单调、末点 = 该项一集总和），电平项画每步 rate。

    事件项（grasp/close/avoid）是单步发放的冲击，画成 rate 是一根细针、看不出趋势；
    累计曲线直接读作"到这个阶段为止拿到了多少"，且末点就等于成功率×100。
    """
    sfx = '_success' if success_only else ''
    rate_key = 'reward_profile_mean' + sfx
    stepcum_key = 'reward_stepcum_mean' + sfx
    labels = list(D)
    bins = next(iter(D.values()))['json']['reward_profile_bins']
    t = np.linspace(0.0, 1.0, bins)
    is_event = dict((a, c) for a, _, c in TERMS)

    def _peak(arr):
        a = np.asarray(arr, dtype=float)
        return float(np.abs(a).max()) if a.size else 0.0   # 无成功集的臂给 {}，不能直接 .max()

    def _active(key):
        return [k for k, _, _ in TERMS if any(_peak(D[l]['json'][key].get(k)) > 1e-9 for l in labels)]
    # 事件项用**绝对步号**累计曲线：episode 成功即终止，按长度归一化后每次成功都落在 t=1，
    # 累计曲线退化成一根末尾竖线；放绝对步号上才能看出"各方法在第几步抓到"。
    active = [(k, True) for k in _active(stepcum_key) if is_event[k]] + \
             [(k, False) for k in _active(rate_key) if not is_event[k]]
    ncol = 3
    if not active:
        print(f"[跳过] 无可画项 ({rate_key})", file=sys.stderr)
        return
    nrow = int(np.ceil(len(active) / ncol))
    fig, axes = plt.subplots(nrow, ncol, figsize=(4.4 * ncol, 2.7 * nrow), squeeze=False)
    for i, (k, ev) in enumerate(active):
        ax = axes[i // ncol][i % ncol]
        name = dict((a, b) for a, b, _ in TERMS)[k]
        if ev:
            for l in labels:
                prof = D[l]['json'][stepcum_key].get(k)
                if prof is None or not len(prof):
                    continue
                ax.plot(np.arange(len(prof)), prof, label=l, color=D[l]['color'], lw=1.8)
            ax.set_title(f"{name}  <{k}>  [cumulative vs step]", fontsize=10)
            ax.set_xlabel("decision step within episode (episode ends on success)")
        else:
            # rate 的首末格点覆盖不满一个完整步宽（网格端点截断），会画出一对假尖峰，裁掉
            for l in labels:
                prof = D[l]['json'][rate_key].get(k)
                if prof is None or len(prof) != bins:
                    continue
                ax.plot(t[2:-2], np.asarray(prof)[2:-2], label=l, color=D[l]['color'], lw=1.8)
            ax.set_title(f"{name}  <{k}>  [per step vs progress]", fontsize=10)
            ax.axhline(0, color='k', lw=0.6, alpha=0.5)
            ax.set_xlabel("normalized time within episode")
        ax.grid(alpha=0.25)
        if i % ncol == 0:
            ax.set_ylabel("cumulative reward" if ev else "reward / decision step")
    for j in range(len(active), nrow * ncol):
        axes[j // ncol][j % ncol].axis('off')
    h, lab = axes[0][0].get_legend_handles_labels()
    fig.legend(h, lab, loc='lower center', ncol=len(labels), fontsize=9.5,
               bbox_to_anchor=(0.5, -0.01))
    sub = "successful episodes only" if success_only else "all episodes"
    fig.suptitle(f"Reward profile over episode progress ({sub}) — {scene}, n={n}",
                 fontsize=13.5, y=0.995)
    fig.text(0.5, 0.960, "event terms: cumulative reward vs decision step (episode ends on success); "
                         "level terms: per-step rate vs normalized progress",
             ha='center', va='top', fontsize=9.5, color='0.35')
    fig.tight_layout(rect=[0, 0.045, 1, 0.945])
    p = os.path.join(outdir, f"fig2_reward_profile{sfx}.png")
    fig.savefig(p, dpi=150)
    plt.close(fig)
    print("saved ->", p)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", default="results/table5_n200_breakdown")
    ap.add_argument("--scene", default="dyn_both_train")
    ap.add_argument("--n", type=int, default=200)
    ap.add_argument("--outdir", default="")
    args = ap.parse_args()
    outdir = args.outdir or args.dir
    D = load_all(args.dir, args.scene, args.n)
    if not D:
        sys.exit(1)
    print(f"arms: {list(D)}")
    fig_composition(D, args.scene, args.n, outdir)
    P = {l: v for l, v in D.items() if v['has_profile']}
    if len(P) < len(D):
        print(f"[提示] {len(D) - len(P)} 条臂缺 reward_profile_*，剖面图只含 {list(P)}", file=sys.stderr)
    if not P:
        print("[跳过] 无臂带剖面数据，不出 fig2", file=sys.stderr)
        return
    fig_profile(P, args.scene, args.n, outdir)
    fig_profile(P, args.scene, args.n, outdir, success_only=True)


if __name__ == "__main__":
    main()
