# 机械臂抓取强化学习系统（混合残差架构）

本仓库是一个面向**强化学习运动控制 / 具身智能方向实习**的作品集项目：让 UR5e + Robotiq 2F-85
在 MuJoCo 仿真中完成**动态目标下的避障抓取**，采用**混合残差架构**——一个自实现的标称层给出
解析参考轨迹，PPO 只学习对它的偏差修正（`v = v_nominal(t) + Δv`）。

> ⚠️ 标称层是**本仓库自实现的仿真速度场 / 运动学 MPC**，**不是 MoveIt**。
> MoveIt 只出现在部署端设计中（见 `docs/混合控制架构设计.md`、`docs/简历项目知识整理.md`）。

## 设计动机：为什么不是端到端

"到达并抓取"是长时序、多目标（接近 + 对齐 + 闭合 + 抬升 + 避障）的任务。端到端 PPO 让网络
同时学"任务分解"和"控制偏差"，实测 8 版端到端训练均失败（归因见
`rl_grasping_system/docs/2026-08-27_RL训练失败分析_v5到v12b.md`）。

残差架构把任务拆开：**标称层负责"任务是什么"**（几何上正确的接近轨迹），
**PPO 负责"差多少"**（局部偏差修正与避障绕行）。这样 RL 的学习目标更小、更局部，
且标称层在域变化时仍然成立——这正是 sim2real 需要的性质。

## 目录结构

```
robotic_arm_control/
├── rl_grasping_system/          # 主系统：环境 / PPO 智能体 / 标称层 / 评估脚本
│   ├── environment.py           # UR5e + Robotiq 2F-85 抓取环境（67 维观测 / 6 维残差动作）
│   ├── reward.py                # 势能塑形奖励
│   ├── mpc_nominal.py           # 第二代标称层：运动学 MPC（SLSQP 滚动）
│   ├── agent.py                 # PPO 智能体封装 + 早停回调
│   ├── scripts/                 # 消融 / MPC+RL / 感知噪声 / 延迟扰动等评估脚本
│   ├── docs/                    # 开发日志、实验记录、学习笔记
│   └── README.md                # 系统技术文档（观测/动作/奖励/配置详解）
├── docs/                        # 简历与面试材料、数字口径总闸、架构设计
└── models/universal_robots_ur5e/ # MuJoCo 场景（含 Robotiq 2F-85 与障碍物生成脚本）
```

## 核心结果：三路消融

为验证"残差"这一设计选择本身是否有效，做了三路对比（端到端 PPO / 残差 PPO / 纯标称零残差），
每场景 n=30、确定性评估，三路模型均在扩大后的桌面上**未重训**下评估：

| 场景 | 端到端 PPO | 残差 PPO | 纯标称（零残差） |
|---|---|---|---|
| 静态无障 | 26.7% | **100%** | 83.3% |
| 动态目标（0.05 m/s） | 63.3% | **90.0%** | 66.7% |
| 动态目标 + 动态障碍 | 43.3% | **86.7%** | 70.0% |

证据：`rl_grasping_system/results/ablation/summary_table.txt` + `results/ablation/*.json`，
复现：`bash rl_grasping_system/scripts/run_ablation.sh`。

两点值得注意：静态任务纯标称已达 83.3%，说明**标称层解决了任务本身**，RL 补的是偏差；
而端到端对域变化最敏感（静态跌到 26.7%）、残差最稳定，说明**学偏差比学任务更抗域偏移**。

## 第二代：MPC 标称层 + 残差（部分成功，主动标注）

标称层由速度场替身升级为**运动学 MPC**（[mpc_nominal.py](rl_grasping_system/mpc_nominal.py)）：
对末端**单积分**模型 `p_{k+1} = p_k + u_k·dt` 做滚动最优控制，决策变量是未来 N 步末端速度，
用 `scipy` SLSQP warm-start 求解（失败降级 P 控制）。代价里既有目标跟踪，也有**障碍软约束**
——所以障碍代价是建进规划里的，不是靠 RL 事后补救：

```
代价 = Σ w_p‖p_k − hover_k‖² + w_o·relu(d_safe − d_k)² + w_s‖u_k − u_{k−1}‖² + w_term·终端
约束 = |u| ≤ v_max = 0.125（契约速度）
```

与速度场标称接口完全一致（输出 6 维 twist），因此**换标称层不动环境、观测槽位与 RL 代码**。

### MPC 标称层对照（n=30，确定性，零残差 = 纯 MPC）

| 场景 | 纯 MPC（Δv=0） | MPC + v25b2 残差 |
|---|---|---|
| 动态目标 | 83.3% / 0 碰撞 | 80.0% / 0 碰撞 |
| 动态目标 + 动态障碍 | **90.0%** / 0 碰撞 | 76.7% / 0 碰撞 |
| static3（3 面障碍墙） | — 无 n=30 基线 | 0.0% |
| mixed3（高密度障碍） | — 无 n=30 基线 | 6.7% / 平均碰撞 1.17 |
| mixed_z（高密度障碍） | — 无 n=30 基线 | 13.3% / 平均碰撞 0.73 |

证据：`results/mpc_plus_rl/zero_*.json`（纯 MPC）、`results/mpc_plus_rl_v25b2/*.json`（残差）。
纯 MPC 只有 `dyn_target`/`dyn_both` 两个场景有 n=30 基线，其余场景仅 n=5 smoke，**留空不可引用**。

### 最强证据：n=200 逐集配对（`--seed 12345 --seed-per-episode`）

上面那张表是 n=30、未固定种子，有 ±8~17pp 波动带，只能看趋势。固定种子后两臂走同一条
场景序列，可做严格 McNemar 检验：

| 标称层（场景 `dyn_both_train`，姿态伺服 off） | 纯标称 | + 残差 | Δ |
|---|---|---|---|
| velocity_field | 63.0% / 158.5 步 | **69.5%** / 144.0 步 | +6.5pp（p=0.047）/ −14.5 步 |
| MPC | 53.5% / 155.4 步 | 57.0% / 142.5 步 | +3.5pp（**不显著**）/ −12.9 步 |

证据：`results/unified7_trainmatch_n200/*.json`。**步数是压倒性的**——四组残差全部显著省步
（配对 t 检验 p ≤ 1.1e−4，Wilcoxon p ≤ 2.7e−12），省 11.1–14.6 步。但成功率只在 velocity_field
标称下站得住（+6.5pp，成功翻转 25↑/12↓，p=0.047）；**换到 MPC 标称就只有 +3.5pp、不显著**。

### 安全增益：残差预算 cap 解耦（v24 → v25b2）

v24 的残差恒满幅（`avg_residual_norm ≡ 0.2165 = √3×0.125`，即 clip 上界），高密度障碍下
几乎必然穿障；v25b2 把残差预算从与 MPC `v_max` 同级降为 40%，并加结构门控：

| 场景 | v24 平均碰撞 | v25b2 平均碰撞 | 降幅 |
|---|---|---|---|
| mixed3 | 40.03 | **1.17** | **34×↓** |
| mixed_z | 25.57 | **0.73** | **35×↓** |

同口径（v24/v25b2 同一 `d_safe`）的对照，证据见 `results/mpc_plus_rl_v24/*.json`
与 `results/mpc_plus_rl_v25b2/*.json`。

> ⚠️ 这两格是在 `D_SAFE=0.05` 口径下测的，而 `D_SAFE` 默认值此后已变（现为 0.20）——
> 按当前默认口径的重算**尚未做**，引用时须带此保留（见 `docs/数字真值表_20260910.md` §2.4）。

### arm-aware 标称 MPC：给规划器补臂身碰撞模型

`mpc_nominal.py` 原本只跟踪末端**一个点**，对臂身会扫到什么没有任何表示。给它在代价里加一项
「臂身碰撞球」软代价（把每个可碰撞 geom 用 MuJoCo 编译出的包围球 `geom_rbound` 近似，共 51 球），
默认关闭（`mpc_nominal_arm_aware=False`）。在 3 面障碍墙场景上：

| static3（`D_SAFE=0.05`，纯 MPC，n=30） | 成功率 | 平均碰撞 | 平均步长 |
|---|---|---|---|
| 基线（仅末端点模型） | 13.3%（4/30） | 169.9 | 184.5 |
| arm-aware `w=120` | **56.7%（17/30）** | **76.9** | **127.7** |

4/30 → 17/30 两比例检验 z≈3.95（p≈8e−5），非噪声；代价是单次求解从 ~0.05 s 涨到 ~0.28 s。
6 场景消融中 4 个含障场景方向一致（成功率升、碰撞降 1.5×–3.8×），但**只有 static3 达统计显著**
（其余 p≈0.11–0.24，属噪声）。证据：`results/arm_aware/*_dsafe005_*_n30.json`。

### 诚实边界（完整口径见 `docs/数字真值表_20260910.md`）

- **成功率的诚实结论取决于标称层**：在 velocity_field 标称下残差确实打赢纯标称（+6.5pp，
  p=0.047）；**换到 MPC 标称就不显著**（+3.5pp，z≈0.70）。所以准确表述是
  「**残差带来的是安全增益与执行效率增益；成功率增益与否取决于标称层**」，
  不是笼统的「打赢/没打赢纯 MPC」。
- **static3 六场景里唯一全版本 0%**：根因是**运动学**——只建模末端点的 MPC 面对 3 个压在
  必经路径上的障碍时陷入局部极小、规划不出绕行（已证伪「`D_SAFE` 调参」旧归因）。arm-aware
  把纯标称抬到 56.7%，但残差通道在该场景贡献≈0（v25b2 叠上去 60.0% vs 56.7%，z≈0.26，无差别）。
- **arm-aware 的收益只在放开安全距离时出现**：默认 `D_SAFE=0.20` 下基线纯 MPC 本身就 0 碰撞
  （它停滞不动），臂身项此时是**空操作**（ON/OFF 逐位一致）。

## 奖励设计

单步奖励 = **势能塑形 + 事件奖励 + 惩罚项**，实现在 [reward.py](rl_grasping_system/reward.py)：

```
势能  Φ(s) = w_xy·min(d_xy, xy_cap)/xy_scale + w_height·min(d_z, height_cap)/height_scale + 近距锚定
塑形  r_shape = Φ(s_prev) − γ·Φ(s)，γ = 0.99
```

| 分量 | 默认值 | 作用 |
|---|---|---|
| `w_xy` / `xy_scale` | 2.0 / 0.3 | 平面距离势能（目标 = 物体正上方 pre-grasp 点） |
| `w_height` / `height_scale` | 8.0 / 0.3 | 高度势能，权重更高——先把末端抬到悬停高度 |
| `anchor_w_xy` / `anchor_w_z` | 1.0 / 2.0（d_target=0.06 m） | 近距锚定，加陡末段斜率，打破「够近但不精调」的收益权衡 |
| `w_contact`（阈值 0.1 N） | 5.0 | 手指接触物体（电平式） |
| `close_trigger_reward` | 5.0 | 进入闭合相位的中间里程碑 |
| `grasp_reward` | 100.0 | 抓取成功，**上升沿一次性**发放 |
| `step_penalty` | 0.05 | 每步时间惩罚 |
| `obstacle_w`（range 0.15 m） | 1.0 | 障碍接近惩罚（线性衰减） |
| `residual_reg_w` | 0.5 | 残差幅度 L2 正则 |

几处踩过坑才定下来的设计（细节见 `rl_grasping_system/docs/`）：

- **诊断键不许进奖励**（P1.3）：早期用 `sum(values())` 求和，把 `orient_align`、`z_axis_z`
  这些诊断量一并算进奖励——而 `z_axis_z` 是「手指朝上 +1 / 朝下 −1」，等于把方向信号反着
  注入奖励。现改为只对 `REWARD_KEYS` 白名单求和，诊断量走 `parts['diag']`。
- **方向项的死区**（P2-1）：旧式 `max(0, align)` 在夹爪与目标夹角 θ>90° 时把梯度截成 0，
  策略于是只学「到位」不学「转向」，可视化里夹爪总是横夹。改为手指朝下投影
  `gain = w_orient·(−z_axis_z)`，全程连续有梯度（当前 `w_orient=0`，3D 动作下该项关闭）。
- **势能塑形替代累计惩罚**：此前距离惩罚按步累加，300 步合计约 −800，把成功奖励 +16 彻底
  淹没；改成势能差 `Φ(s_prev) − γ·Φ(s)` 后单步有界（典型 ±2）、回报不随 episode 长度膨胀。
- **抓取成功的定义**（P3）：不是「夹紧到某开度」，而是**手指被物体挡住合不上**——闭合后开度
  仍大于 `grasp_min_width=0.02` 且双指接触。夹爪由环境内「近距自动闭合状态机」驱动，
  RL 不再输出肌腱命令，动作空间收敛为 6 维末端位姿增量。

## 快速开始

```bash
cd rl_grasping_system
pip install -r requirements.txt

# 训练（本地带实时监控）
python train_with_monitor.py

# 评估某个模型
python evaluate.py --model_path models/final_model_stage2_d2_v18_p3f.zip

# 纯标称对照 / MPC 标称 / arm-aware 标称 / 逐集固定种子
python evaluate.py --model_path models/final_model_stage2_d2_v18_p3f.zip --zero-residual
python scripts/mpc_plus_rl_eval.py --nominal-mode mpc --seed 12345 --seed-per-episode
python scripts/mpc_plus_rl_eval.py --nominal-mode mpc --arm-aware --zero-residual --seed 12345
```

云端无头训练：`MUJOCO_GL=egl python train_cloud.py`。

## 文档导航

| 文档 | 用途 |
|---|---|
| [docs/简历项目知识整理.md](docs/简历项目知识整理.md) | **面试主手册**：知识点 → 对应文件 → 失败回顾 → 刁难问题 |
| [docs/数字真值表_20260910.md](docs/数字真值表_20260910.md) | **数字口径总闸**：每个数字的证据文件与口径；已撤回数字清单 |
| [docs/面试口述脚本_1页.md](docs/面试口述脚本_1页.md) | 一页口述脚本（开场三句 + 高频追问） |
| [docs/混合控制架构设计.md](docs/混合控制架构设计.md) | 三层架构技术契约（标称 / 残差 / 实时安全层） |
| [docs/ROADMAP.md](docs/ROADMAP.md) | 阶段路线图（L1 静态闭环 → L2 动态目标 → L3 动态演示） |
| [rl_grasping_system/README.md](rl_grasping_system/README.md) | 系统技术文档：观测/动作空间、奖励设计、P2/P3/P4 演进 |
| [rl_grasping_system/docs/](rl_grasping_system/docs/) | 开发日志、实验记录、调参失败分析、学习笔记 |

## 技术栈

MuJoCo · Gymnasium · Stable-Baselines3 (PPO) · SciPy (SLSQP) · NumPy · Matplotlib

## 许可证

MIT License
