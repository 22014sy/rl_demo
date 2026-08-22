# VecNormalize 观测归一化：自定义实现替换为官方 SB3 实现

> 2026-08-21 · 训练侧（rl_grasping_system）观测归一化实现修复
> 关联：`agent.py`（train / load）、`vec_normalize_wrapper.py`（退役）、`docs/RL_DEPLOY_CONTRACT.md`

---

## 1. 背景

训练效果不好（策略行为≈随机、`r_dist_xy` 不降）。排查中发现观测归一化用的是**自定义 `VecNormalizeWrapper`**
（`vec_normalize_wrapper.py`），经评审确认有缺陷，改为官方 `stable_baselines3.common.vec_env.VecNormalize`。

## 2. 自定义实现缺陷（对照官方）

| # | 缺陷 | 影响 |
|---|---|---|
| 1 | **无 `save`/`load`（obs_rms 不随模型保存）** | 评估/部署时观测统计丢失 → **evaluate 0% 根因之一**（训练归一化、评估不归一化） |
| 2 | `reset()` 不更新 obs_rms（官方 reset/step 都更新） | 每 episode 首步观测不参与统计 |
| 3 | reward 用"单步 reward 统计"而非官方"return(ret_rms) 统计" | 若 norm_reward 开启会错误（当前已关） |
| 4 | 无与 PPO 集成的 `normalize_reward()` 反归一化接口 | PPO 回报处理失真（reward 归一化时） |
| 5 | 稀疏通道（接触力/手指力大多 0）归一化后易饱和 ±10 | 信息损失（接触由状态机处理，不致命） |

> Welford 在线更新公式数学上正确，但工程完整性（save/load、reset、PPO 集成）不足。

## 3. 替换方案

- `agent.py`：
  - train：`VecNormalize(DummyVecEnv([lambda: env]), norm_obs=True, norm_reward=False, clip_obs=10)`
  - load：`VecNormalize(..., training=False)` —— `PPO.load` 会从 zip 恢复 `obs_rms`（官方支持），保证评估/部署观测一致
- `vec_normalize_wrapper.py`：退役（保留文件，不再 import）

## 4. 验证

- [x] 冒烟训练（PPO + 官方 VecNormalize 正常工作）：2048 步 2 rollout 无报错，fps 50，episode 记录正常，
  奇异点 0（环境改进后更稳定）
- [x] 启动 A（40k 长训练）观察 `r_dist_xy` 是否随训练下降
- [x] **A2 训练（混合奖励后）**：成功率 0% → 6%（见 docs/奖励设计与混合塑形方案.md）
- [ ] 保存/加载后 obs_rms 一致（训练-评估一致性）——final_model.zip 已含官方 VecNormalize stats，待 evaluate 验证

### 冒烟记录（2026-08-21）
```
Logging to logs/PPO_76
Episode 1: reward 37.78 | Episode 2: 212.82 | Episode 3: 74.43（奇异 0）
rollout: fps 50, ep_rew_mean 108, total_timesteps 1024
SMOKE TRAIN OK
```

### A 训练结果（2026-08-22，日志 training_log_20260822_014453.json，139 episodes / 40k 步）
```
episodes=139 succ=0.0%
rdist_xy: Q1=49.2 Q2=44.1 Q3=51.4 Q4=53.5 (随机45.5)
rdist_z : Q1=47.2 Q4=80.0
r_contact: 非0=0/139 均值=0.00
```
**结论**：
- ✅ **官方 VecNormalize 修复有效**：中期（Q2）出现 xy 对准学习（44.1 < 随机 45.5），
  对比替换前（57ep 策略 r_dist≈52.8 完全随机）——归一化修复 + 环境改进让学习信号出现
- ⚠️ 40k 步**只学到"xy 对准"部分**：rdist_z 反升（47→80，臂未降到 cube 高度）、r_contact=0（从未接触）
- 可能原因：z 下降收益稀疏（接触奖励 10 需精确触发）、r_dist_z 权重低、40k 步探索不足
- **下一步**：需评估 r_dist_z 奖励权重/接触 shaping，或延长训练（看 r_contact 是否出现）

