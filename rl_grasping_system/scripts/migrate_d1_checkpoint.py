#!/usr/bin/env python3
"""D1 观测空间 55→67 迁移工具（2026-08-24，一周冲刺方案 P0）

背景：
    D1 把观测空间从 55 维扩到最终形态 67 维（末尾 +6 v_nominal[55:61] +3 obstacle_rel[61:64]
    +3 obstacle_vel[64:67]，见 docs/CHANGELOG.md D1 条目 / scripts/check_obs_layout.py）。
    旧课程模型（final_model_stage1/stage2 等，输入 55 维）在当前 67 维环境下无法直接加载：
    PPO.load(model, env=...) 按 env.observation_space 重建网络 → 输入层维度不匹配。

本工具做两处"输入层手术"，产出与原模型行为完全一致的 67 维版本：
    1. PPO 网络输入层 55→67：mlp_extractor.policy_net[0]/value_net[0] 的 Linear 权重零扩展
       （前 55 列拷贝旧权重，新 12 列全 0 → 网络对前 55 维的计算逐 bit 不变 → 行为不变）。
    2. VecNormalize obs_rms 55→67：obs_mean 追加 0（新槽位先验均值）、obs_var 追加 1
       （新槽位先验方差=1 → normalize 后仍≈原始值）、count 沿用旧值（新槽位无样本）。

    新 12 个观测槽位在 delta 模式恒为 0 / 障碍隐藏时 obstacle 槽位为 0 / 无标称时 v_nominal 为 0，
    权重 0 + 先验统计 (0,1) 保证迁移后策略输出 = 原策略输出，成功率不变（--verify 实测 0 差）。

用法：
    python3 scripts/migrate_d1_checkpoint.py --model models/final_model_stage2.zip --verify
    默认产出 models/final_model_stage2_d1.zip(+_vecnormalize.pkl)
    --dry-run：只打印迁移计划不写盘；--verify：迁移后与旧模型随机输入输出一致性校验
"""
import os
import sys
import argparse
import pickle

import numpy as np
import torch
import torch.nn as nn
import cloudpickle
from gymnasium.spaces import Box
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import VecNormalize
from stable_baselines3.common.torch_layers import FlattenExtractor

NEW_DIM = 67          # D1 观测空间最终形态
OLD_DIM = 55          # D1 前课程模型输入维度
ADD_DIM = NEW_DIM - OLD_DIM


def extend_linear(lin: nn.Linear, new_in: int) -> nn.Linear:
    """零扩展 nn.Linear 输入维度：前 old_in 列拷贝旧权重，新增列权重=0，bias 保留。"""
    old_w = lin.weight.detach()
    old_b = lin.bias.detach()
    out_f, old_in = old_w.shape
    new = nn.Linear(new_in, out_f, bias=True)
    with torch.no_grad():
        new.weight.zero_()
        new.weight[:, :old_in] = old_w
        new.bias.copy_(old_b)
    return new


def migrate_model(old_zip: str, new_zip: str, old_dim: int) -> None:
    """输入层手术：55→67，另更新 policy 观测空间 / features_dim。"""
    ppo = PPO.load(old_zip)
    pol = ppo.policy
    pol.mlp_extractor.policy_net[0] = extend_linear(pol.mlp_extractor.policy_net[0], NEW_DIM)
    pol.mlp_extractor.value_net[0] = extend_linear(pol.mlp_extractor.value_net[0], NEW_DIM)
    # 同步观测空间（PPO 重建 policy 与 predict 前的空间一致性校验都依赖它）
    new_space = Box(low=-np.inf, high=np.inf, shape=(NEW_DIM,), dtype=np.float32)
    ppo.observation_space = new_space
    pol.observation_space = new_space
    # features_extractor 换成 67 维 FlattenExtractor（其 forward 按 features_dim reshape，
    # features_dim 是只读 property，不能原地改；policy.features_dim 是普通属性可直接更新）
    pol.features_extractor = FlattenExtractor(new_space)
    pol.features_dim = NEW_DIM
    print(f"  policy_net[0]: {old_dim}→{NEW_DIM}（新 {ADD_DIM} 列权重=0）")
    print(f"  value_net[0] : {old_dim}→{NEW_DIM}（新 {ADD_DIM} 列权重=0）")
    # 重建 optimizer：SB3 2.9.0（policies.py:634）用 optimizer_class(parameters, lr=lr_schedule(1), **kwargs)
    # 创建单 Adam。不重建的话，PPO.save 会把旧 55 维 optimizer state（exp_avg/exp_avg_sq 形状 (512,55)）
    # 一并写入新 zip；运行时 PPO.load(model, env=67 维) 重建 67 维参数却加载 55 维 state，
    # 首次 optimizer.step() 报 "tensor a (55) vs tensor b (67)"（2026-08-24 smoke 实测）。
    # 新 12 个输入槽位本无历史动量，重建为空 state（冷启动）即可，行为等价。
    pol.optimizer = pol.optimizer_class(pol.parameters(), lr=1.0, **pol.optimizer_kwargs)
    print("  optimizer  : 重建（丢弃旧 55 维 Adam state，新 67 维参数冷启动）")
    ppo.save(new_zip)
    print(f"  → 已保存 {new_zip}")


def migrate_vecnormalize(old_pkl: str, new_pkl: str) -> None:
    """obs_rms 55→67：mean 补 0、var 补 1、count 沿用旧值（新槽位无样本先验）。

    注：本机 SB3 的 VecNormalize.load() 强制要求 venv 参数，无法在无环境场景恢复 pkl，
    故直接 cloudpickle 读写（VecNormalize.save 底层即为 cloudpickle，agent.py 的
    VecNormalize.load 兼容此格式）。pkl 内 venv 属性在 save 时已剔除，无需处理。
    """
    with open(old_pkl, "rb") as f:
        vn = cloudpickle.load(f)
    mean, var = vn.obs_rms.mean, vn.obs_rms.var
    cur = int(mean.shape[0])
    if cur != NEW_DIM:
        vn.obs_rms.mean = np.concatenate([mean, np.zeros(NEW_DIM - cur)])
        vn.obs_rms.var = np.concatenate([var, np.ones(NEW_DIM - cur)])
        print(f"  obs_rms.mean/var: {cur}→{NEW_DIM}（新槽位 mean=0, var=1, count={int(vn.obs_rms.count)}）")
    # observation_space 也必须同步为 67 维——VecNormalize.load 会做
    # check_shape_equal(self.observation_space, venv.observation_space)，旧 55 维会断言失败
    vn.observation_space = Box(low=-np.inf, high=np.inf, shape=(NEW_DIM,), dtype=np.float32)
    print("  observation_space: 55→67")
    # SB3 2.9.0 的 VecNormalize.__getstate__ 会 del venv/class_attributes/returns，
    # 而云 pickle 恢复后的实例 __dict__ 缺这 3 个键（原 save 时已被删）→ 补上再 dump，
    # 用标准 pickle.dump（与 VecNormalize.save 同机制；agent.py 的 VecNormalize.load 兼容）。
    vn.venv = None
    vn.class_attributes = []
    vn.returns = []
    with open(new_pkl, "wb") as f:
        pickle.dump(vn, f)
    print(f"  → 已保存 {new_pkl}")


def verify_equivalence(old_zip: str, new_zip: str, n: int = 200, seed: int = 0) -> float:
    """随机输入下对比新旧模型 deterministic 输出，返回最大绝对差（预期 0）。"""
    old = PPO.load(old_zip)
    new = PPO.load(new_zip)
    rng = np.random.default_rng(seed)
    max_diff = 0.0
    for _ in range(n):
        x55 = rng.standard_normal(OLD_DIM).astype(np.float32)
        x67 = np.concatenate([x55, np.zeros(ADD_DIM)]).astype(np.float32)
        with torch.no_grad():
            a_old, _ = old.predict(x55, deterministic=True)
            a_new, _ = new.predict(x67, deterministic=True)
        max_diff = max(max_diff, float(np.max(np.abs(a_old - a_new))))
    return max_diff


def main():
    ap = argparse.ArgumentParser(description="D1 观测空间 55→67 迁移（输入层手术 + obs_rms 扩展）")
    ap.add_argument("--model", default="models/final_model_stage2.zip")
    ap.add_argument("--suffix", default="_d1", help="输出文件名后缀（默认 _d1）")
    ap.add_argument("--dry-run", action="store_true", help="只打印迁移计划，不写盘")
    ap.add_argument("--verify", action="store_true", help="迁移后与旧模型输出一致性校验")
    args = ap.parse_args()

    old_zip = args.model
    base = os.path.splitext(old_zip)[0]
    old_pkl = base + "_vecnormalize.pkl"
    new_zip = base + args.suffix + ".zip"
    new_pkl = base + args.suffix + "_vecnormalize.pkl"

    if not os.path.exists(old_zip):
        raise SystemExit(f"模型不存在: {old_zip}")
    print(f"迁移源: {old_zip}（{os.path.basename(old_pkl)}）→ 目标: {new_zip}")

    # 源维度探测
    src = PPO.load(old_zip)
    old_dim = int(src.observation_space.shape[0])
    if old_dim == NEW_DIM:
        print(f"源模型已是 {NEW_DIM} 维，无需迁移")
        return
    if old_dim != OLD_DIM:
        raise SystemExit(f"不支持源维度 {old_dim}（预期 {OLD_DIM}）")
    print(f"源观测空间: {old_dim} 维 → 目标: {NEW_DIM} 维（+{ADD_DIM}：v_nominal[55:61] + obstacle_rel[61:64] + obstacle_vel[64:67]）")

    if args.dry_run:
        print("[dry-run] 计划：policy_net[0]/value_net[0] 输入层零扩展 55→67；"
              f"obs_rms.mean 补 {ADD_DIM} 个 0、var 补 {ADD_DIM} 个 1；count 沿用。未写盘。")
        return

    # 迁移 + 保存
    migrate_model(old_zip, new_zip, old_dim)
    if not os.path.exists(old_pkl):
        print(f"⚠️ 未找到 {old_pkl}，跳过 obs_rms 迁移（若后续训练需恢复观测统计会失配）")
    else:
        migrate_vecnormalize(old_pkl, new_pkl)

    # 一致性校验（迁移无损的硬证据）
    if args.verify:
        d = verify_equivalence(old_zip, new_zip)
        status = "✅ 一致" if d < 1e-6 else "⚠️ 有差异"
        print(f"[verify] 新旧模型 deterministic 输出最大绝对差 = {d:.3e} → {status}")
        if d >= 1e-6:
            raise SystemExit("迁移后输出与源不一致，请勿使用该产物！")


if __name__ == "__main__":
    main()
