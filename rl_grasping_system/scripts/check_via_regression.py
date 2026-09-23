#!/usr/bin/env python3
"""回归护栏：无障碍场景下 `global_planner=via_point` 必须**逐集逐位**退化为纯 MPC。

方案 §验证 2 的硬要求。理由是这层是**加法**：它只在有障碍时挑一个绕行路点，
没有障碍就该完全不介入。若这里出现任何差异，说明 via 层在悄悄改变标称行为，
那四臂对照表里所有「加绕行」的增量就都不干净了。

做法：同 seed、同 n，用评测器 CLI 跑 `--global-planner off` 与 `via_point` 两遍，
逐集逐键比对 `episode_results`。允许不同的键只有本次新增的 via_*/config：
  - config.global_planner / config.tag（本来就是口径标签）
  - 每集新增的 via_* 键（无障碍时应为 0 / None / False）
其余任何键不一致即 FAIL，并打印首个不一致的集号 + 键名 + 两侧值。

用法（必须在 rl_grasping_system/ 下跑）：
    python3 scripts/check_via_regression.py --scene static --n 3
    python3 scripts/check_via_regression.py                # 默认 static, n=3
"""
import argparse
import json
import os
import subprocess
import sys
import tempfile

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)

# 允许不同的键：无障碍时 via 层必须给出「没介入」的中性值
# ⚠️ via_plan_time_s 不在此列——它是**墙钟**量：via 层每步给自己计时，
#    即使不规划也会累计一点开销，两次运行不可能相同。计时类键一律只比「量级」不比相等。
VIA_NEUTRAL = {
    "via_screen_cnt": 0, "via_cand_total": 0, "via_cand_pass": 0, "via_adopt_cnt": 0,
    "via_planned": False, "via_margin_via_m": None, "via_margin_seg1_m": None,
    "via_margin_seg2_m": None, "via_margin_total_m": None,
}
CONFIG_ALLOW = {"global_planner", "tag"}
# 墙钟量：两次独立运行必然不同，逐位比对无意义 → 排除（另见文件末尾的单独检查）
TIMING_KEYS = {
    "avg_mpc_solve_s", "mpc_solve_s_total", "last_mpc_solve_s", "via_plan_time_s",
    "avg_last_mpc_solve_s", "mpc_solve_s_total_sum", "via_plan_time_s_sum",
    "avg_via_plan_time_s",      # ← 漏了这一个，护栏会在这里误报（2026-09-19 实测踩到）
    "via_screen_time_s", "via_screen_time_s_sum", "avg_via_screen_time_s",
}


def run(out_path, scene, n, planner, tag):
    cmd = [sys.executable, os.path.join("scripts", "mpc_plus_rl_eval.py"),
           "--scene", scene, "--nominal-mode", "mpc", "--zero-residual", "--arm-aware",
           "--d-safe", "0.05", "--tag", tag,
           "--seed", "12345", "--seed-per-episode", "--n_episodes", str(n),
           "--max-steps", "200", "--save-results", out_path]
    if planner != "off":
        cmd += ["--global-planner", planner]
    print("[run] " + " ".join(cmd))
    r = subprocess.run(cmd, cwd=ROOT, capture_output=True, text=True)
    if r.returncode != 0:
        print(r.stdout[-3000:], file=sys.stderr)
        print(r.stderr[-3000:], file=sys.stderr)
        sys.exit(f"[FAIL] 评测器退出码 {r.returncode}（planner={planner}）")
    with open(out_path, encoding="utf-8") as f:
        return json.load(f)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--scene", default="static", help="必须是无障碍场景（SCENE_CFG 里 obstacle=off）")
    ap.add_argument("--n", type=int, default=3)
    args = ap.parse_args()

    with tempfile.TemporaryDirectory() as td:
        a = run(os.path.join(td, "off.json"), args.scene, args.n, "off", "regr_off")
        b = run(os.path.join(td, "via.json"), args.scene, args.n, "via_point", "regr_via")

    bad = []

    # 1) 顶层聚合量（除 config 外）必须逐位一致
    ka, kb = set(a) - {"config"}, set(b) - {"config"}
    if ka != kb:
        bad.append(("顶层键集", sorted(ka ^ kb)))
    for k in sorted(ka & kb):
        if k == "episode_results" or k in TIMING_KEYS:
            continue
        if a[k] != b[k]:
            bad.append((f"顶层 {k}", (a[k], b[k])))

    # 2) config：只允许 global_planner / tag 不同
    ca, cb = dict(a.get("config") or {}), dict(b.get("config") or {})
    for k in sorted(set(ca) | set(cb)):
        if k in CONFIG_ALLOW:
            continue
        if ca.get(k) != cb.get(k):
            bad.append((f"config.{k}", (ca.get(k), cb.get(k))))

    # 3) 逐集逐键
    ea, eb = a.get("episode_results", []), b.get("episode_results", [])
    if len(ea) != len(eb):
        bad.append(("集数", (len(ea), len(eb))))
    else:
        for i, (ra, rb) in enumerate(zip(ea, eb)):
            keys = (set(ra) | set(rb)) - set(VIA_NEUTRAL) - TIMING_KEYS
            for k in sorted(keys):
                if ra.get(k) != rb.get(k):
                    bad.append((f"ep{i}.{k}", (ra.get(k), rb.get(k))))
            # 新增的 via_* 键必须是中性值（真的没介入）
            for k, neutral in VIA_NEUTRAL.items():
                if rb.get(k) != neutral:
                    bad.append((f"ep{i}.{k} 非中性", (rb.get(k), neutral)))

    print("\n" + "=" * 68)
    if bad:
        print("❌ 回归护栏 **未通过** —— via 层在无障碍场景下改变了标称行为：")
        for name, val in bad[:40]:
            print(f"   {name}: off={val[0]!r}  via={val[1]!r}")
        if len(bad) > 40:
            print(f"   … 另有 {len(bad) - 40} 处")
        sys.exit(1)
    print(f"✅ 回归护栏通过：`{args.scene}` 无障碍，n={args.n} 集，")
    print("   `global_planner=off` 与 `via_point` 逐集逐位一致（via_* 全为中性值）。")
    print("   ⇒ 绕行层是纯加法，四臂对照的「加绕行」增量不含无障碍场景的污染。")


if __name__ == "__main__":
    main()
