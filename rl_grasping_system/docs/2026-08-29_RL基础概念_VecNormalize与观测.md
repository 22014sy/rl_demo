# 2026-08-29 · RL 基础概念笔记：VecNormalize 与观测在算法中的角色

> 面试基础知识点整理（结合本项目 PPO 实现），可作为 `docs/简历项目知识整理.md` 的补充。

---

## 1. VecNormalize（观测归一化）

### 1.1 一句话定义
**Stable-Baselines3 里的"观测归一化"包装器**：把每个观测维度减去在线统计均值、除以标准差，
让所有特征回到相近尺度再喂给网络。本项目 `agent.py` 的归一化靠它。

### 1.2 解决什么问题（为什么需要）
67 维观测里各维度尺度天差地别：
| 特征 | 量级 |
|---|---|
| 关节位置/速度 | ~rad / ~rad/s |
| 末端位置 | ~0.1m 级 |
| 关节力矩 | ~100 N·m 级 |
| 夹爪开度 | ~0.01m 级 |

**不归一化**：梯度被大力矩/大位置特征主导，小尺度特征学不到；方差差异使损失病态 → 震荡、收敛慢。
**归一化后**：所有特征 ≈ 均值 0、方差 1，梯度均衡，训练稳定。

### 1.3 具体机制
```python
obs_norm = (obs - mean) / sqrt(var + eps)     # ① 逐维度标准化
obs_norm = np.clip(obs_norm, -clip_obs, clip_obs)   # ② 裁剪防极端值（本项目 clip=10）
VecNormalize(..., norm_reward=False)          # ③ 奖励归一化：本项目关闭
VecNormalize(..., training=True/False)        # ④ 训练更新统计 / 评估冻结统计
```
- **mean/var 在线更新**（running statistics，Welford 算法）——统计跟随策略的观测分布变化；
- **norm_reward=False 的原因**：奖励已是势能塑形 + 有界（P1.3），不需再归一；避免改变相对奖励尺度；
- **评估必须 training=False**：否则评估会继续更新统计 → 统计漂移 → 结果不可复现、行为改变。

### 1.4 本项目用法 + 关键坑（面试可讲）
```python
vec_env = VecNormalize(DummyVecEnv([...]), training=False)          # 评估模式
vec_env = VecNormalize.load(model_path + '_vecnormalize.pkl', vec_env)  # 恢复训练统计
model = PPO.load(model_path, env=vec_env)
```
**坑**：SB3 的 `PPO.save()` 不把 obs_rms 存进 zip（`_excluded_save_params` 排除 `_vec_normalize_env`）
→ 必须单独存/恢复 `_vecnormalize.pkl`；缺失则评估/部署观测失配、**策略乱走**。
→ 因此模型 `.zip` 与 `_vecnormalize.pkl` 必须**成对保留**（清理文件时已贯彻）。

### 1.5 与其他归一化的区别
| 归一化 | 位置 | 机制 | 本项目 |
|---|---|---|---|
| VecNormalize | 环境层（VecEnv 包装） | 观测在线标准化 | ✅ |
| BatchNorm | 网络层内 | 按 batch 归一化激活 | ❌（RL 时序相关不常用） |
| Reward normalization | 奖励 | norm_reward | ❌（奖励已有界） |
| 早期自定义 `vec_normalize_wrapper.py`（RunningMeanStd） | 环境层 | 手写版 | 已被官方 SB3 替代 |

### 1.6 面试话术
> 观测归一化用 SB3 官方 VecNormalize：在线 mean/var 标准化 + clip=10 解决 67 维观测尺度失衡。
> 三个细节：① norm_reward 关闭（奖励已势能塑形有界）；② 训练在线更新、评估冻结（防统计漂移不可复现）；
> ③ 模型 zip 不含 obs_rms，必须单独存/恢复 pkl——SB3 经典坑，我踩过并修复。

---

## 2. 观测在强化学习算法中的环节

### 2.1 宏观位置：RL 循环的"感知输入主线"
```
环境 → 状态 s →(测量)→ 观测 obs → [VecNormalize] → Actor π(a|obs) → 动作 → 环境
                                              ↘  Critic V(obs) → GAE 优势
                                                      ↘ 更新时：(obs, a, r, A) 作条件
```
观测 = 智能体从环境拿到的信息，是算法输入的唯一来源。

### 2.2 观测在 PPO 各环节的角色
| 环节 | 观测的作用 | 项目体现 |
|---|---|---|
| 采样（rollout） | `obs → π(a\|obs) → a`，收集轨迹 | environment/evaluate 循环 |
| 策略网络 Actor | 观测是输入层，决定动作分布 | 67 维 → [512,512,256] |
| 价值网络 Critic | 输出 V(obs)（状态价值） | 同一观测喂 Actor+Critic |
| GAE 优势估计 | `A_t = δ_t + γλδ_{t+1}+...` 用 V(obs_t)、V(obs_{t+1}) | PPO 内置 GAE（λ=0.95） |
| PPO 更新 | `ratio = π_new(a\|obs)/π_old(a\|obs)` + 裁剪；V(obs) 拟合 | agent.py 内部 |
| 归一化预处理 | obs 先 VecNormalize 再进网络 | agent.py |

### 2.3 本质：观测 = 状态空间 S（或 POMDP 部分观测）
- MDP 五元组 (S, A, P, R, γ) 中的 **S**：决定策略定义域、网络输入维度、"能看到什么"；
- **观测 ≠ 状态**：观测是智能体实际拿到的（本项目 oracle+噪声；真部署=相机测量）；若观测≠完整状态 → **POMDP**；
- **观测设计直接影响可学性**（本项目三个实例）：
  - 显式给**相对接近姿态四元数 q_rel**（P2-2）→ 策略直读对齐误差，不自己合成转角；
  - 观测含**标称参考速度 v_nominal** → 残差策略能只学偏差的前提；
  - 观测含**夹爪相位 / 障碍相对位姿+速度** → 策略感知阶段与障碍才能学绕障。

### 2.4 为什么观测设计是 RL 工程核心技能（面试点）
1. **可学性**（信息够不够：缺障碍速度就学不会避障）；
2. **收敛难度**（显式给误差 vs 让网络合成）；
3. **泛化**（是否含环境无关量——oracle vs 视觉特征）；
4. **sim2real**（部署时观测分布一致性——训练端 oracle+噪声、部署端真相机的核心问题）。

### 2.5 面试话术
> 观测在 PPO 里贯穿采样、价值估计、更新三环节，本质是 MDP 的状态空间、策略和价值的输入。
> 我的 67 维观测设计原则是"本体感知为主 + 显式给对齐误差 + 残差槽位第一天就位"——例如显式 q_rel
> 让策略直读对齐误差，标称参考速度槽位让残差策略只学偏差。观测设计直接决定可学性与 sim2real，
> 所以我做了感知-控制双通道隔离（oracle+噪声量化感知容忍度，D6 实证 σ≤3cm 不降）。
