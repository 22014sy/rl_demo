# 强化学习抓取系统

这是一个独立的、基于强化学习的机械臂抓取系统，专门用于训练Panda机械臂执行"到达并抓取"任务。

## 系统特点

### 🎯 核心功能
- **独立模块**: 完全独立的抓取系统，不影响现有VLM项目
- **PPO算法**: 基于Stable-Baselines3的PPO实现
- **归一化技术**: 观察归一化、奖励归一化、优势函数归一化
- **云端支持**: 无头渲染，适用于云端训练
- **完整监控**: 训练进度、成功率、奇异点检测等

### 🔧 技术架构
- **环境**: MuJoCo + Panda机械臂
- **任务**: 到达并抓取固定位置物体
- **算法**: PPO with 完整归一化
- **监控**: 实时训练监控和可视化
- **部署**: 云端无头训练支持

## 快速开始

### 1. 安装依赖
```bash
cd rl_grasping_system
pip install -r requirements.txt
```

### 2. 云端训练
```bash
python train_cloud.py
```

### 3. 本地训练（带监控）
```bash
python train_with_monitor.py
```

### 4. 评估模型
```bash
python evaluate.py --model_path models/final_model.zip
```

## 系统架构

### 核心模块

#### 1. 环境模块 (`environment.py`)
- **PandaGraspingEnv**: 基于MuJoCo的抓取环境
- **状态空间**: 关节位置/速度/力矩、末端执行器位置/方向/速度、夹爪状态、目标信息
- **动作空间**: 关节命令 + 夹爪控制
- **奖励函数**: 距离奖励、抓取奖励、完成奖励、惩罚机制

#### 2. 智能体模块 (`agent.py`)
- **GraspingAgent**: PPO智能体封装
- **GraspingCallback**: 抓取任务专用回调函数
- **早停机制**: 基于成功率的智能早停
- **模型管理**: 保存/加载/评估功能

#### 3. 配置管理 (`config.py`)
- **GraspingConfig**: 抓取任务配置
- **NetworkConfig**: 网络架构配置
- **TrainingConfig**: 训练参数配置
- **RewardConfig**: 奖励函数配置
- **SystemConfig**: 系统配置

#### 4. 归一化系统 (`vec_normalize_wrapper.py`)
- **VecNormalizeWrapper**: 观察和奖励归一化
- **RunningMeanStd**: 在线统计计算
- **裁剪机制**: 防止归一化值过大

#### 5. 训练监控 (`training_monitor.py`)
- **TrainingMonitor**: 完整的训练监控系统
- **实时图表**: 奖励、成功率、奇异点统计
- **日志记录**: 详细的训练日志
- **早停支持**: 基于性能的智能早停

#### 6. 安全机制
- **SingularityHandler**: 奇异点检测和处理
- **SafeActionWrapper**: 动作安全约束
- **碰撞检测**: 防止机械臂碰撞

## 配置说明

### 训练配置
```python
# 基本参数
total_timesteps: int = 50000  # 总训练步数
learning_rate: float = 3e-4     # 学习率
batch_size: int = 1024          # 批次大小
n_steps: int = 1024             # 每轮步数

# 归一化参数
normalize_observations: bool = True  # 观察归一化
normalize_rewards: bool = False      # P1: 关闭奖励归一化（奖励已由 P1.3 势能塑形有界化）
norm_obs_clip: float = 10.0         # 观察裁剪
norm_reward_clip: float = 10.0      # 奖励裁剪
```

### 奖励配置（P1.3 势能塑形，见 reward.py）
```python
# 位置目标 = 物体中心正上方（pre-grasp 位姿，hand 基座物理可达）
pre_grasp_offset_z: float = 0.10   # hand 基座悬停高度

# 距离/方向势能（低=好）；单步奖励 = Φ(s_prev) − γ·Φ(s)，γ=shaping_gamma
w_xy: float = 2.0                  # 平面距离权重
w_height: float = 2.0              # 高度权重
xy_scale: float = 0.3              # 平面距离归一化尺度(m)
height_scale: float = 0.15         # 高度归一化尺度(m)
w_orient: float = 2.0              # 手指朝下(目标抓取姿态)对齐权重
shaping_gamma: float = 0.99        # 势能塑形折扣

# 事件奖励（一次性/每步，量级与典型单步奖励相当）
w_contact: float = 1.0             # 手指接触物体
grasp_reward: float = 5.0          # is_grasped（到位+夹紧+接触）
completion_reward: float = 10.0    # grasp_success
step_penalty: float = 0.01         # 时间惩罚
```

## 归一化技术

### 1. 观察归一化
- **目的**: 稳定训练，避免不同特征尺度差异
- **实现**: 在线计算均值和方差
- **裁剪**: 防止归一化值过大

### 2. 奖励归一化
- **目的**: 稳定梯度，避免奖励爆炸
- **实现**: 在线计算奖励统计量
- **裁剪**: 限制奖励范围

### 3. 优势函数归一化
- **目的**: 提高策略更新稳定性
- **实现**: PPO算法内部自动处理
- **方法**: GAE (Generalized Advantage Estimation)

## 训练监控

### 实时指标
- **Episode奖励**: 每个episode的总奖励
- **成功率**: 抓取成功的episode比例
- **平均步数**: 完成任务的步数统计
- **奇异点检测**: 机械臂奇异点统计

### 可视化图表
- **训练曲线**: 奖励、成功率、步数趋势
- **分布图**: 奖励分布、步数分布
- **实时更新**: 每2000个episodes自动保存

## 云端部署

### 环境设置
```bash
# 设置无头渲染
export MUJOCO_GL=egl
export DISPLAY=:0

# 运行云端训练
python train_cloud.py
```

### 特点
- **无头渲染**: 完全避免图形界面依赖
- **CPU优化**: 强制使用CPU，避免GPU警告
- **错误处理**: 完善的异常处理和日志记录
- **自动保存**: 定期保存模型和图表

## 性能优化

### 网络架构
```python
# 策略网络
policy_hidden_sizes: [512, 512, 256]

# 价值网络
value_hidden_sizes: [512, 512, 256]
```

### 训练参数
- **学习率**: 3e-4 (适中)
- **熵系数**: 0.01 (促进探索)
- **GAE lambda**: 0.95 (优势估计)
- **裁剪范围**: 0.2 (PPO裁剪)

## 故障排除

### 常见问题

#### 1. 观察类型错误
```
Exception: Unrecognized type of observation <class 'tuple'>
```
**解决方案**: 确保VecNormalizeWrapper返回正确的numpy数组格式

#### 2. 图形渲染失败
```
Cannot initialize a EGL device display
```
**解决方案**: 使用云端训练脚本，设置无头渲染

#### 3. 训练不收敛
**解决方案**: 
- 检查奖励函数设计
- 调整网络架构
- 增加训练步数
- 启用归一化

## 项目状态

### ✅ 已完成
- [x] 独立RL抓取系统架构
- [x] PPO智能体实现
- [x] 完整归一化技术
- [x] 云端训练支持
- [x] 训练监控系统
- [x] 安全机制实现

### 🔧 技术改进
- **归一化**: 观察/奖励归一化 + PPO内置优势函数归一化
- **云端优化**: 无头渲染、CPU训练、错误处理
- **监控增强**: 实时图表、早停机制、详细日志


### 笔记

奖励曲线
含义：每个 episode 的总奖励是多少
作用：最核心的“学习效果”指标
你应该看：
如果它一直上升，说明策略在变好
如果它震荡很大，说明训练不稳定
如果它长期不升，说明奖励设计或环境可能有问题

成功率曲线
含义：最近 10 个 episode 中，有多少比例成功完成抓取
作用：比奖励更直接，反映“任务是否真的学会了”

奇异点统计
含义：机械臂发生奇异点的次数
作用：反映机械臂是否“走到了不健康的姿态”
你应该看：
奇异点越少越好
如果它突然变多，说明策略开始走偏，可能把机械臂推到危险姿态

Episode 长度
含义：每个 episode 结束前用了多少步
作用：反映任务是否越来越高效
如果长度越来越短，说明机器人越来越快完成任务
如果长度越来越长，说明它可能在原地打转，或者没有学会高效策略

如果奖励在涨，但成功率不涨
说明：你的奖励函数可能“骗过”智能体，它学会了拿到高 reward，但没有真正学会抓取
你可以考虑：
调整奖励函数，让成功抓取的奖励更明显
减少“只靠靠近物体就能拿高分”的奖励设计

如果成功率低，且奇异点多
说明：策略在尝试危险动作，机械臂可能学到了一些不稳定的控制方式
你可以考虑：
增加安全约束
降低动作幅度
调整奖励，惩罚危险姿态

如果奖励和成功率都很波动
说明：训练不稳定，可能是学习率、批次大小、熵系数或归一化设置不合适
你可以考虑：
降低学习率
调整 PPO 的 ent_coef
增加训练稳定性

如果 episode 长度越来越长
说明：智能体可能没有学会快速完成任务，或者环境终止条件不够利于它学习
你可以考虑：
缩短最大步数
调整任务目标，让它更容易学到“快速完成”

最实用的判断标准
你可以把它们看成三个层次：
奖励曲线：看“有没有学到东西”
成功率：看“有没有真正完成任务”
奇异点/episode 长度：看“是不是学得很不稳或很低效”


成功率有没有从低往上走
奇异点有没有明显增多
奖励曲线是否持续上升而不是只偶尔冲高

## 文件结构

```
rl_grasping_system/
├── environment.py              # 抓取环境
├── agent.py                   # PPO智能体
├── config.py                  # 配置管理
├── training_monitor.py        # 训练监控
├── vec_normalize_wrapper.py   # 归一化包装器
├── train_cloud.py            # 云端训练脚本
├── train_with_monitor.py     # 本地训练脚本
├── evaluate.py               # 模型评估
├── singularity_handler.py    # 奇异点处理
├── action_wrapper.py         # 动作安全包装
├── state.py                  # 状态处理
├── reward.py                 # 奖励函数
├── requirements.txt          # 依赖列表
└── README.md                # 本文件
```

## 许可证

MIT License
