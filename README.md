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

标称层由速度场升级为**运动学 MPC**（SLSQP 滚动最优控制）后，遇到一个反直觉现象：残差在
成功率上**没有**打赢纯 MPC（动态+障碍 76.7% vs 90.0%），但在高密度障碍下大幅改善了安全性
——碰撞 cap 解耦使 mixed3 平均碰撞从 40.03 降到 1.17（34×↓）。

诚实边界（完整口径见 `docs/数字真值表_20260910.md`、`docs/简历项目知识整理.md`）：

- 残差**未在成功率上超越纯 MPC**——安全增益不等于成功率增益；
- **static3 场景全版本 0%**：根因是只建模末端点的运动学 MPC 规划不出绕行，未解；
- 早期评估脚本**无 seed 控制**，存在 ±8~17pp 采样噪声，故"高几个百分点"级结论不可信。

## 快速开始

```bash
cd rl_grasping_system
pip install -r requirements.txt

# 训练（本地带实时监控）
python train_with_monitor.py

# 评估某个模型
python evaluate.py --model_path models/final_model_stage2_d2_v18_p3f.zip

# 纯标称对照 / MPC 标称 / 执行延迟扰动
python evaluate.py --model_path models/final_model_stage2_d2_v18_p3f.zip --zero-residual
python scripts/mpc_plus_rl_eval.py --nominal-mode mpc --seed 0
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
