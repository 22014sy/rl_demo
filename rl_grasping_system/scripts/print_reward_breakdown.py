#!/usr/bin/env python3
"""打印各方法的奖励分项均值（每集 / 每步两列）。

读 mpc_plus_rl_eval.py 产出的 JSON（需 2026-09-17 之后带 reward_breakdown_* 的版本）。
若 JSON 缺该字段（旧结果），脚本会明确报错而不是静默打印空表。

口径提醒：事件项（r_grasp / r_close / r_avoid）一集只发一次，看「每集」列；
电平项（r_obstacle / r_residual / r_step / r_dist_*）每步都发，看「每步」列。
两列的关系是 每集 = 每步 × 平均 episode_length。
"""
import json, os, sys, argparse

LABELS = [
    ("r_dist_xy",      "XY 距离塑形",   "step"),
    ("r_dist_z",       "Z 距离塑形",    "step"),
    ("r_contact",      "接触（电平）",   "step"),
    ("r_close",        "闭合相位触发",   "event"),
    ("r_grasp",        "抓取成功",      "event"),
    ("r_obstacle",     "障碍接近惩罚",   "step"),
    ("r_residual",     "残差幅度正则",   "step"),
    ("r_residual_step", "残差激活步罚",  "step"),
    ("r_avoid",        "干净成功 bonus", "event"),
    ("r_step",         "时间步罚",      "step"),
    ("r_singularity",  "奇异惩罚",      "step"),
    ("r_collision",    "碰撞当步惩罚",   "step"),
]

DEFAULT_ARMS = [
    ("纯跟踪伺服",  "vf_nominal"),
    ("跟踪伺服+RL", "vf_res18"),
    ("RL",          "e2e_v11"),
    ("MPC",         "mpc_nominal"),
    ("MPC+RL",      "mpc_res25b2"),
]


def load(path):
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def per_step_means(d):
    """每步均值 = 总奖励 / 总步数，从 episode_results 现算。

    不直接用 JSON 里的 reward_breakdown_per_step：早期版本存的是"每集比值的均值"，
    会给短 episode（成功提前结束的）过高权重。现算保证任何一版 JSON 口径一致。
    """
    eps = d['episode_results']
    tot_steps = sum(max(1, e['episode_length']) for e in eps)
    keys = {k for e in eps for k in e.get('reward_breakdown', {})}
    return {k: sum(float(e.get('reward_breakdown', {}).get(k, 0.0)) for e in eps) / tot_steps
            for k in keys}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", default="results/table5_n200_armaware")
    ap.add_argument("--scene", default="dyn_both_train")
    ap.add_argument("--n", type=int, default=200)
    ap.add_argument("--md-out", default="")
    args = ap.parse_args()

    data, missing, stale = {}, [], []
    for label, tag in DEFAULT_ARMS:
        p = os.path.join(args.dir, f"{tag}_{args.scene}_n{args.n}.json")
        if not os.path.exists(p):
            missing.append((label, p))
            continue
        d = load(p)
        if 'reward_breakdown_per_episode' not in d or 'episode_results' not in d:
            stale.append((label, p))
            continue
        d['_per_step'] = per_step_means(d)
        data[label] = d

    for label, p in missing:
        print(f"[缺失] {label}: {p}", file=sys.stderr)
    for label, p in stale:
        print(f"[旧口径] {label}: {p} 无 reward_breakdown_* 字段——该 JSON 是 2026-09-17 之前跑的，需重跑",
              file=sys.stderr)
    if not data:
        sys.exit(1)

    lines = []
    def out(s=""):
        lines.append(s)
        print(s)

    n = next(iter(data.values()))['n_episodes']
    out(f"# 奖励分项均值（{args.scene}，seed 12345，n={n}，max_steps=200）")
    out()
    out("每集 = 一集内累计总量 / 集数；每步 = 一集内累计总量 / 该集步数，再对集数取均值。")
    out("事件项一集只发一次（看每集列），电平项每步都发（看每步列）。")
    out()
    out("| 分项 | 类型 | " + " | ".join(f"{l}<br>每集 / 每步" for l in data) + " |")
    out("|---|---|" + "---|" * len(data))
    for key, name, kind in LABELS:
        row = []
        any_nonzero = False
        for label in data:
            e = data[label]['reward_breakdown_per_episode'].get(key, 0.0)
            s = data[label]['_per_step'].get(key, 0.0)
            if abs(e) > 1e-12:
                any_nonzero = True
            row.append(f"{e:+.3f} / {s:+.5f}")
        if not any_nonzero:
            row = ["— (全部恒 0)"] * len(data)
        out(f"| {name} `<{key}>` | {'事件' if kind == 'event' else '电平'} | " + " | ".join(row) + " |")
    out()
    out("| 合计 | | " + " | ".join(
        f"**{data[l]['avg_episode_reward']:+.1f}**" for l in data) + " |")
    out("| 平均时长 (步) | | " + " | ".join(
        f"{data[l]['avg_episode_length']:.1f}" for l in data) + " |")

    if args.md_out:
        with open(args.md_out, "w", encoding="utf-8") as f:
            f.write("\n".join(lines) + "\n")
        print(f"\nsaved -> {args.md_out}")


if __name__ == "__main__":
    main()
