# Stage 2：MPC 标称 + PPO 残差（MPC+RL）训练与评估

> 日期：2026-09-07　分支：Stage 2　前置：`2026-09-07_MPC标称层验证与多障碍场景.md`（Stage 1，已提交 5409f77）

## 0. 结论速览（2026-09-07 21:47 评估完成回填）

| 场景 | 纯 MPC（Stage1） | MPC+RL（v24，本阶段） | 判定线 | 达标 |
|---|---|---|---|---|
| static | 96.7% / 0.0% coll | **96.7% / 0.0% coll** | ≥80% & coll≤5% | ✅ |
| dyn_target | 66.7% / 0.0% coll | **73.3% / 0.0% coll** | →83.3%（v23 口径） | ❌（低于 v23/纯 MPC 复测 10pp，CI 内） |
| dyn_both | 80.0% / 0.0% coll | **86.7% / 3.3% coll** | ≥80% & coll≤5% | ✅ |
| static3 | 0.0% / 3.3% coll | **0.0% / 0.0% coll** | >0% | ❌（与纯 MPC 持平） |
| mixed3 | 13.3% / 33.3% coll | **0.0% / 86.7% coll** | 说明性 | ❌（劣于纯 MPC） |
| mixed_z | 13.3% / 13.3% coll | **13.3% / 80.0% coll** | coll<10% | ❌（coll 恶化） |

- **整体判定：2/6 达标**（static、dyn_both）；dyn_target 73.3% 接近 v23 口径但未达；3 个极端场景（static3/mixed3/mixed_z）未达。
- **核心正面结果 = dyn_both**：MPC+RL 成功率 86.7% 与纯 MPC 复测（90.0%）相当、优于 v23 下限（80%），碰撞从 v23 的 16.7~36.7% 压到 3.3%——「MPC 标称兜底 + 残差不破坏性能」成立；static 与 v23/纯 MPC 持平（96.7%/0 coll）。
- **核心负面结果 = dyn_target + 极端多障碍场景**：dyn_target 73.3% 低于纯 MPC 复测/v23（均 83.3%）10pp（CI 内，残差无正贡献）；static3 0%、mixed3/mixed_z coll 80%+。avg_residual_norm 全场景恒定 ≈0.2165（=clip 上界），提示残差输出长期饱和、缺乏状态自适应，在高密度障碍场景反而扰动 MPC 标称的避障规划。
- 完整 4 象限：MPC/velocity_field × 纯标称/RL 残差（`mpc_plus_rl_eval.py --zero-residual` 与 v23 对照组，见 §3.2/§3.3/§4）。

## 1. 目标与方法

- **架构**：`v = v_nominal(t) + Δv/T`（residual 模式，与 v23 完全同构）；标称层替换为
  `nominal_mode=mpc`（末端级滚动最优控制，Stage 1 已验证 drop-in 等价）。
- **动机数据（Stage 1 §5）**：MPC 局限 = static3 多静态障碍 0%、mixed_z 非匀速 13.3% coll、
  v_max 提速后 mixed3 coll 0→33.3%。RL 残差负责「经验性规避」把碰撞压回安全地板、把部分成功
  提升为可靠成功。
- **判定线（固定 n=30）**：dyn_both ≥80% 成功 & coll ≤5%；dyn_target →83.3%（v23 口径）；
  static3 >0%；mixed_z coll <10%（部分成功可接受）。n=30 二项 95%CI ≈ ±8~17pp，**必须对照
  v23_mix / 纯 MPC 同口径复测**再断言。

## 2. 训练配方（v24_mpcres，2026-09-07 16:34 发起）

```
python3 train_with_monitor.py --action-mode residual --nominal-mode mpc \
  --dynamic-target --target-vel 0.05 --target-axis x \
  --obstacle --obstacle-on-path --obstacle-vel 0.05 --obstacle-count 2 \
  --scenario-mix 0.2,0.3,0.5 --collision-penalty -10 --obstacle-w 3 \
  --obstacle-clear-bonus 30 --residual-reg 2.0 \
  --load-model models/final_model_stage2_d2_v23_mix.zip \
  --save-model models/final_model_stage2_d2_v24_mpcres.zip \
  --total-timesteps 300000 --no-demo
```

- 单变量对比 v23_mix：仅 `--nominal-mode velocity_field → mpc`，其余完全一致。
- 实际训练：2026-09-07 16:34 → 18:39，**1:57:39**（319,488 步 / 3169 episodes；每 rollout ≈8 分钟，MPC SLSQP 求解比速度场慢 ~4×，fps≈42）。
- 训练摘要：最终成功率 **70.0%**（训练混合场景口径）、平均奖励 -37.8、最佳奖励 356.5、平均 episode 长度 100.6、奇异点 0.0。
- 成功率曲线：Ep 300 达 ~74% → Ep 500 ~72-75% → 中段回落 65-68%（PPO 探索波动）→ 中后段回升 75-80%
  → 末段回落 68-75%。**训练期混合成功率仅作过程参考，判定看固定 n=30 评估**。

## 3. 评估结果（n=30，确定性，action_mode=residual + nominal_mode=mpc）

> `scripts/mpc_plus_rl_eval.py`（Stage 1↔2 同口径，max_steps=200）；JSON 落 `results/mpc_plus_rl/*.json`（gitignored，本表为唯一 git 记录）。

### 3.1 MPC+RL（v24_mpcres）
| 场景 | success | collision | 平均碰撞数 | avg_len | avg_residual_norm(成功) | avg_residual_norm(失败) |
|---|---|---|---|---|---|---|
| static | 96.7% | 0.0% | 0.0 | 44.2 | 0.2165 | 0.2165 |
| dyn_target | 73.3% | 0.0% | 0.0 | 86.2 | 0.2165 | 0.2162 |
| dyn_both | 86.7% | 3.3% | 0.5 | 76.1 | 0.2165 | 0.2163 |
| static3 | 0.0% | 0.0% | 0.0 | 200.0 | — | 0.2164 |
| mixed3 | 0.0% | 86.7% | 40.0 | 200.0 | — | 0.2164 |
| mixed_z | 13.3% | 80.0% | 25.6 | 191.2 | 0.2164 | 0.2165 |

> 注：avg_residual_norm 全场景恒定 ≈0.2165，等于观测到的 max_residual_norm_overall=0.216506——残差输出全程贴着 clip 上界（饱和），
> 说明策略没有学到「大多数时刻小幅修正、关键时刻大修正」的稀疏性，而是固定幅度偏置。max_mpc_solve_s 0.05~0.24s（多障碍场景更慢），mpc_fallback=0。

### 3.2 对照：纯 MPC（`--zero-residual`，同脚本复测）
| 场景 | success | collision | 说明 |
|---|---|---|---|
| dyn_both | 90.0% | 0.0% | 同脚本复测（2026-09-07 21:55）；Stage1 记录 80.0%/0.0%（`mpc_nominal_verify.py` 口径，seed/参数略异） |
| dyn_target | 83.3% | 0.0% | 同脚本复测（2026-09-07 21:57）；Stage1 记录 66.7%/0.0%（口径差异，以此复测为准对照） |

### 3.3 对照：v23_mix（velocity_field+PPO，基线）
| 场景 | success | collision | 来源 |
|---|---|---|---|
| static | 96.7% | 0.0% | results/v23_mix/static.json |
| dyn_target | 83.3% | 0.0% | results/v23_mix/dyn_target.json |
| dyn_both | 80.0~86.7% | 16.7~36.7% | results/v23_mix/dyn_both{1,2}.json（coll 高 = 已知短板） |

## 4. 判定与结论（2026-09-07 回填）

### 4.1 逐场景判定

| 场景 | v24 | 判定线 | 结果 | 解读 |
|---|---|---|---|---|
| static | 96.7% / 0 coll | ≥80% & coll≤5% | ✅ | 与 v23（96.7%）、纯 MPC（96.7%）持平，三者在静态简单场景均到顶；残差不产生额外收益也不损害 |
| dyn_target | 73.3% / 0 coll | →83.3%（v23 口径） | ❌（差 10pp） | 低于 v23（83.3%）与纯 MPC 复测（83.3%）各 10pp；n=30 二项 95%CI ≈ ±15pp，与 83.3% 统计重叠——「无显著劣化」但残差对动态目标**无正贡献** |
| dyn_both | 86.7% / 3.3 coll | ≥80% & coll≤5% | ✅ | **本阶段核心正结果**：成功率与纯 MPC 复测（90.0%）、v23 上限（86.7%）相当，coll 从 v23 的 16.7~36.7% 压到 3.3%——MPC 标称层天然低碰撞，残差保持了性能不劣化，「标称兜底 + 残差不破坏」成立 |
| static3 | 0.0% / 0 coll | >0% | ❌ | 与纯 MPC（0.0%）持平；MPC 标称在 3 静态障碍下无法规划可行路径，残差也未能习得「绕行穿越」；coll=0 说明不撞但永远够不到 |
| mixed3 | 0.0% / 86.7 coll | 说明性 | ❌ | **劣于纯 MPC（13.3%/33.3%）**：残差饱和输出（0.2165）持续扰动 MPC 避障规划，高密度障碍下把「部分成功+中等碰撞」拖成「全失败+高碰撞」 |
| mixed_z | 13.3% / 80.0 coll | coll<10% | ❌ | success 与纯 MPC 持平（13.3%），但 coll 13.3%→80.0% 恶化 6×；同样归因于残差饱和扰动 + 非匀速 z 场景 MPC 本身规划裕度小 |

### 4.2 4 象限矩阵（dyn_both，n=30）

| 标称层 | 纯标称（Δv=0） | PPO 残差（Δv≈0.2165 饱和） |
|---|---|---|
| velocity_field | v18 消融纯标称（旧版，见 results/ablation） | v23_mix：80~86.7% / coll 16.7~36.7% |
| MPC | 纯 MPC（复测）：90.0% / coll 0.0%（§3.2；Stage1 记录 80.0%） | **v24_mpcres（本阶段）：86.7% / coll 3.3%** |

### 4.3 结论

1. **达标 2/6**：static、dyn_both。dyn_both 的成功率（86.7%）与纯 MPC 复测（90.0%）相当，coll 从 v23 的 16.7~36.7% 压到 3.3%——碰撞问题的根因是 **velocity_field 标称层**而非残差本身；换成 MPC 标称后碰撞自然归零，残差没有破坏这一优势（「标称兜底 + 残差保持」）。
2. **dyn_target 未达 v23 口径，残差无正贡献**：73.3% vs v23/纯 MPC 复测均 83.3%，低 10pp 且 CI 重叠（统计不显著）；说明在该场景 PPO 残差与 MPC 标称未形成互补，甚至轻微干扰。
3. **极端多障碍场景暴露残差饱和缺陷**：avg_residual_norm 恒 0.2165（=clip 上界）说明策略输出长期饱和、近乎固定偏置，缺少状态自适应；在 static3/mixed3/mixed_z 反而干扰 MPC 避障（mixed3/mixed_z coll 80%+，混合 3 障碍下把纯 MPC 的「部分成功+中等碰撞」拖成「全失败+高碰撞」）。这是**本阶段最值得记录的限制**，也是后续调参方向（见 §4.4）。
4. **统计诚实性**：n=30 单次评估，每个场景成功率的 95%CI ≈ ±8~17pp；dyn_target 73.3% 与 83.3%、dyn_both 86.7% 与 90.0% 的差异均在 CI 内；§3.2 纯 MPC 复测与 Stage1 记录（66.7%/80.0%）的差异同属此幅度，判定以「达标/未达标」而非具体百分点为准。

### 4.4 后续方向（若继续 Stage 2+）

> **2026-09-08 更新**：§4.3 的残差饱和缺陷已完成归因与修复方案，见
> **`docs/2026-09-08_残差饱和分析与奖励重塑方案.md`**（证据链、根因 A–D、经济学账、v25 三路修复
> 与 Go/No-Go 判定线）。v25_mpcres_sparse（2026-09-07 23:01 发起）已实施以下三项：
> 1. **残差预算与 MPC 解耦**：residual 分支 clip 0.005→0.002（v_max_res 0.125→0.05 m/s，MPC 的 40%）；
> 2. **L1 稀疏三件套**：`r_residual` = L2(2.0) + L1(2.0) + 激活步罚(1.0, τ=0.02)，满幅残差成本从 18.7 → ≈60/集，与成功 +130 同级；
> 3. **高密度课程**：`--scenario-mix` 扩展 5 场景（+static3/mixed3，共 ~25% 权重），残差不再零样本硬扛。
>
> **判定线补充（v25 必须同时满足）**：`avg_residual_norm` 呈双峰（常态≈0、尖峰≤0.0866）且均值 <0.05——
> 否则即使成功率达标也算未达标（防「恒偏置硬闯」模型再次蒙混过关）。
> Go/No-Go：混合场景 coll 回落 ≤15% 且至少一个 MPC 硬伤场景（static3>0% 或 mixed_z coll<13.3%）出现残差增益 → Go 继续 Stage 3；
> 否则 No-Go，把 v24 定档为架构对照终点，主方向转向纯 MPC 工程化。

- ~~残差稀疏化：降低 `--residual-reg` / 加 L1 正则，或对残差动作乘门控……~~（已由 v25 L1 三件套实施）
- ~~极端场景 curriculum：训练 mix 中加入 static3/mixed3/mixed_z 难度……~~（已由 v25 5 场景 mix 实施）
- 状态自适应上限：把残差 clip 上界与 MPC 可行域解耦（MPC 无解/低裕度时允许残差加大），dyn_target 与 static3 或可上探。（v25 仅做单向解耦——固定降幅；双向「MPC 无解时放大残差」留待 Stage 3）

## 5. 产物与复现
- 模型：`models/final_model_stage2_d2_v24_mpcres.zip` + `_vecnormalize.pkl`（git 跟踪；zip 10.3MB / pkl 6.7KB）
- 评估脚本：`scripts/mpc_plus_rl_eval.py`、`scripts/run_mpcres_eval.sh`（6 场景并行，n=30）
- 结果 JSON：`results/mpc_plus_rl/*.json`（gitignored，`*.json` 规则；本表为唯一 git 记录）
- 评估时间：主评估 2026-09-07 21:31→21:47；纯 MPC 复测 21:55→21:57
- 复现：
  ```bash
  cd rl_grasping_system
  bash scripts/run_mpcres_eval.sh                      # 6 场景 v24 MPC+RL，n=30
  python3 scripts/mpc_plus_rl_eval.py --scene dyn_both --n_episodes 30 \
      --nominal-mode mpc --zero-residual \
      --save-results results/mpc_plus_rl/zero_dyn_both.json   # 纯 MPC 对照
  ```
