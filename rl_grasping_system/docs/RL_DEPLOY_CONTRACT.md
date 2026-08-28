# RL 抓取系统 训练 ↔ 部署 接口契约

> 版本：**v3（2026-08-28）**——对齐训练侧最新形态：**67 维观测 + 3D 位置增量动作 + 残差（residual）语义**。
> 迁移说明：v2（2026-08-21）为 55 维观测 + 6 维位姿增量；训练侧已升级（D1 残差架构、P3 动作 3D），
> 本契约同步。任何改动必须先改本契约，再同步两侧实现。
> 目的：训练侧（`robotic_arm_control/rl_grasping_system`，MuJoCo，无 ROS）与部署侧
> （`ros2_moveit2_ur5e_grasp`，MoveIt2 + UR5e + Robotiq 2F-85 + Gazebo，独立仓库）之间的**唯一契约**。
> 两侧独立实现，但本契约定义的语义（动作/观测/状态机/坐标系/产物）必须严格一致。

---

## 1. 机器人

| 项 | 值 |
|---|---|
| 机械臂 | UR5e（6-DOF 旋转关节） |
| 关节名 | `shoulder_pan_joint`, `shoulder_lift_joint`, `elbow_joint`, `wrist_1_joint`, `wrist_2_joint`, `wrist_3_joint` |
| 关节限位 | ±6.28 rad（shoulder_pan/lift, wrist_*）/ ±3.14 rad（elbow），effort 150/150/150/28/28/28 N·m |
| 夹爪 | **Robotiq 2F-85**（平行二指，开度 0–85 mm）；训练侧 = `mujoco_menagerie` `robotiq_2f85/2f85.xml`（真实四连杆/质量/碰撞网格）集成于 `ur5e_robotiq.xml`，关节 `rq_*_driver/coupler/spring_link/follower` |
| 几何基准 | 链接长度/基座高度/EE 偏移以 `ur5e_gripper.urdf.xacro` 为准；训练侧 MJCF 几何数值与其对齐 |

## 2. 动作空间

- 向量：`[dx, dy, dz]` —— **末端位置增量**（任务空间，臂无关；`action_space_dim=3`）
- 语义（**残差模式**，训练侧默认）：**`v_servo = v_nominal(t) + Δv/T`**
  - `v_nominal(t)`：标称轨迹参考速度（MoveIt 静态层生成，见 §4.1）
  - `Δv/T`：RL 残差 = 动作增量 / 决策周期（决策 25Hz 时 T=0.04s）
  - 姿态**固定朝下顶抓**（3D 动作不输出姿态——训练实证 6D 姿态维度无学习信号）
- 限幅：位置增量 `max_ee_delta = 0.005` m/决策步（v_max = 0.125 m/s，实机对照安全限速）
- 坐标系：`base_link`（世界系，z 向上）
- 兼容说明：旧 6D 位姿增量（`[dx,dy,dz,dax,day,daz]`）仅作对照/历史模型，新策略一律 3D。

## 3. 观测空间

**67 维精确布局**（对齐训练侧 `environment.py::_get_observation`，顺序即拼接顺序）：

| # | 段 | 维度 | 部署侧来源 | 部署侧真值/占位 |
|---|---|---|---|---|
| 1 | 关节位置 | 6 | `/joint_states` | 真值 |
| 2 | 关节速度 | 6 | `/joint_states` | 真值 |
| 3 | 关节力矩 | 6 | `/joint_states`(effort) 或估计 | 尽力真值，缺则 0 |
| 4 | 夹爪开度 | 1 | gripper 反馈 / rq_*_joint | 真值（2F-85 开度） |
| 5 | 夹爪速度 | 1 | 开度差分 | 真值/0 |
| 6 | 握持张力 | 1 | 开度导出（训练侧非线性模型） | 占位 0 或估计 |
| 7 | 末端位置 | 3 | TF `base_link→tool0` | 真值 |
| 8 | 末端姿态 | 4 | TF（四元数 w,x,y,z） | 真值 |
| 9 | 末端速度 | 6 | 位姿差分 / 控制器 | 尽力真值，缺则 0 |
| 10 | 夹爪状态 | 1 | gripper 反馈 | 真值 |
| 11 | 目标位置 | 3 | 视觉（YOLO+深度反投影+TF，见 §5） | 真值（含噪声） |
| 12 | 目标姿态 | 4 | 视觉 / 先验（顶抓固定） | 真值/先验 |
| 13 | 可操作度 | 1 | 关节位置启发式 / 雅可比 | 占位 0 或估计 |
| 14 | 接触力 | 1 | 夹爪电流 / 力传感器 | 占位 0 |
| 15 | 左右手指力 | 3+3 | 同上 | 占位 0 |
| 16 | 相对接近姿态四元数 | 4 | q_target ⊗ q_ee⁻¹（w≥0） | 计算真值 |
| 17 | 夹爪相位 | 1 | 状态机（open=0/closing=1/closed=2） | 真值 |
| 18 | **标称参考速度 v_nominal** | 6 | MoveIt 标称轨迹 | 计算真值（residual 必要） |
| 19 | **障碍相对末端位姿** | 3 | 聚类点云最近障碍 | 真值（静态层感知） |
| 20 | **障碍速度** | 3 | 帧间差分 + 平滑 | 尽力真值，缺则 0 |

> **总计 67 维**。v2 的 55 维缺残差三件套：标称速度(6) + 障碍位姿(3) + 障碍速度(3)，并多出 3 个姿态/夹爪维度——已对齐训练侧。
> ⚠️ **占位通道（力矩/张力/可操作度/接触力/手指力）策略影响有限**：观测经 VecNormalize 归一化，
> 训练分布中这些通道多为诊断量；部署侧缺失时置 0 即可，但**必须保持 67 维形状一致**。
> 物体位姿由视觉节点发布（`/detection` + TF），残差桥按本表拼装。

## 4. 决策 / 控制周期与模式

| 项 | 值 |
|---|---|
| RL 决策频率 | 25 Hz（`action_repeat=2` × 50Hz 控制周期）；动态场景规划提频 50Hz |
| 动作模式 | **`residual`**（训练侧默认）：`v_servo = v_nominal(t) + Δv/T`；`delta`（纯增量）仅历史模型 |
| 控制/规划频率 | 部署侧 MoveIt Servo / `forward_velocity_controller`（高频）；训练侧 MuJoCo（timestep 0.002s） |
| 主控制模式 | **速度控制**：arm = motor(力矩) + 环境内**速度 PID + 重力补偿前馈** `τ=kp·(dq−qvel)+qfrc_bias`（复刻 UR speedJ） |
| 速度级 IK | 雅可比阻尼伪逆 `dq=Jᵀ(JJᵀ+λ²I)⁻¹v`（**λ=0.08**，奇异降速），末端速度 v=Δp/T |
| 速度 PID 增益 | 肩/肘 **200**、腕 **100/60**（N·m·s/rad，按关节标定；v2 为 150/100，随训练侧同步） |
| 命令实现率 | 脉冲 ~96%（决策周期内位移 vs 命令；v1 位置伺服仅 8-41%） |
| 兜底模式 | 位置控制 + 力监控（接触/抬升阶段） |
| 已排除 | 全关节力矩控制（UR5e 无此接口；effort 仅为力矩前馈） |

### 4.1 标称轨迹（residual 的必要输入）

- 训练侧实现：`nominal_trajectory.py`——**两段速度场**（段1 末端先到 cube 正上方高处水平对齐，
  段2 垂直下降到 hover，`nominal_approach_clearance=0.15`、hover 偏移 `0.134`）。
- 部署侧对应：MoveIt2 OMPL 无碰撞标称轨迹（含静态避障，用聚类碰撞体），执行层沿轨迹给
  Servo 下发 twist 参考速度。
- 契约参数：`nominal_approach_speed=0.12`、`nominal_gain=3.0`、`nominal_horizontal_gain=6.0`、
  目标速度前馈 `k_ff=1.0`（动态目标消 P 控制滞后），整体限幅 `v_max=0.125 m/s`。
- **残差语义**：静态世界残差 Δv→0（标称主导）；动态场景残差出手（MoveIt 规划跟不上实时动态）。

## 5. 状态机（两侧语义一致）

| 阶段 | 控制模式 | 触发/说明 |
|---|---|---|
| 接近 (approach) | 速度 | 初始 → 距目标 > 阈值；末端朝 hover（目标上方 `pre_grasp_offset`）接近 |
| 对准 (align) | 速度 | 进入末端对准窗口（pad 中心与物体 XY 对齐 < `grasp_align_xy_tol`、Z < `grasp_align_z_tol`）|
| 接触/闭合 (closing) | 速度微调 + 力控 | 对齐触发夹爪闭合；closing 期间 pad 向物体中心 XY 微调（`closing_align_speed`）；接触力 > 阈值停止微调防侧推 |
| 闭合成功 (closed) | 力控 | 手指被物体挡住合不上（宽度 > `grasp_min_width`）+ 双指接触确认 |
| 抬升 (lift) | 位置（恒定速度） | `grasp_success` 后自动状态机，RL 不参与（`lift_speed=0.003`） |
| 放置 (place) | 位置 | 到达放置位后释放（2F-85 全开） |

> **训练侧实现（`environment.py`）：RL 只控制末端位置；夹爪由"近距自动闭合状态机"驱动。**
> 参数沿用训练侧：`grasp_align_xy_tol=0.030`、`grasp_align_z_tol=0.032`、`closing_align_speed=0.08`、
> `closing_align_force_tol=1.0`、`grasp_min_width=0.02`、`grasp_force_limit=20`、`lift_speed=0.003`、
> `lift_height=0.2`——部署侧按 2F-85 复核。

## 6. 抓取成功判定

**核心定义（P3 延续）：手指被物体挡住合不上 = 抓取成功**。Task3 在此基础上补齐触发条件与验证（防误判/防弹飞）。

### 6.1 触发闭合（open → closing），需同时满足
1. **XY 对齐**：物体中心 XY 与夹爪中心（左右 pad 中点）之差 < `grasp_align_xy_tol = 0.030` m（训练侧 2026-08-22 收紧：0.045 时 pad 在 cube 表面外 2.5cm 夹不到，收紧保证 pad 几乎正对 cube 中心）
2. **Z 对齐**：物体中心 Z 与夹爪中心 Z 之差 < `grasp_align_z_tol = 0.032` m
   > 语义：物体真正落在 pad 之间才夹（比"距离触发"严格，防侧方误触发；同时修复 home 位形
   > pad 悬于物体上方 0.034m 时立即闭合的假成功问题——Z 阈值须 < 0.034；XY 已放宽至逼近夹爪半径）

### 6.2 闭合成功（closing → closed，`grasp_success=True`），需同时满足
| # | 条件 | 参数 |
|---|---|---|
| 1 | 开度收敛：连续 `close_confirm_steps` 步内宽度变化 ≤ `close_width_tol` | 3 步 / 0.004 m |
| 2 | 被物体挡住：宽度 `width > grasp_min_width`（手指合不上） | 0.02 m |
| 3 | 双指接触：左右两 pad 均接触物体（单指不算） | — |

### 6.3 失败 / 重试路径
| 情况 | 结果 |
|---|---|
| 开度收敛但 ≤0.02 或 **无接触**（空闭合） | → open，`grasp_retry_hold=True`（离开物体 >`gripper_open_distance=0.22` 才可再闭合） |
| 闭合超 `max_close_steps`（60） | → open 重试 |
| closed 后滑脱（≤0.02）或失接触 | → open 重试；**抬升中则 `lift_aborted`（失败）** |

### 6.4 抬升验证（P4）
`grasp_success` 后自动抬升（`lift_height=0.2` m、`lift_speed=0.003` m/步）：
- 抬升到位（误差 < `lift_tol=0.015`）→ `lift_done` = **最终成功**
- 抬升中失夹 → `lift_aborted` = **失败**

### 6.5 Task3 附加约束
- **限力闭合 + 闭合速度斜坡（阻抗更轻柔）**：closing 阶段手指每决策步 `ctrl += gripper_close_speed=50`（速度斜坡，pad 慢速接触，
  对应实机 2F-85 `set_gripper_speed`）；pad-cube 接触力 > `grasp_force_limit=20` N 后不再加压（保持中等压力，
  对应实机 2F-85 `set_gripper_force`）
- 接近限速：pad 距物体 < `approach_speed_dist=0.06` m 时末端线速度 ≤ `approach_speed_limit=0.12` m/s
- 速度环：kp=(200,200,200,100,100,60) + 粘性阻尼 kd=(5,5,5,4,4,6) + 加速度前馈（静止微颤归零）
- episode 终止：`terminated`（lift_done/lift_aborted/is_grasped）或 `truncated`（步数 ≥300 或连续奇异 ≥50）

### 6.6 一句话
**抓取成功 = pad 已下探到与物体高度对齐 + 手指闭合到"被物体挡住合不上"（宽度 >0.02m）+ 双指都接触物体**；成功后再完成抬升才算最终任务成功（抬升失夹则失败）。

## 7. 模型产物（唯一跨仓流通物）

- 训练产出：SB3 PPO 模型 `.zip` + `_vecnormalize.pkl`（**必须一起**——SB3 zip 不含 obs_rms，
  缺失则部署观测失配乱走）
- 当前产品模型：`models/final_model_stage2_d2_v18_p3f.zip`（67 维观测 + 3D 残差动作；
  residual 模式，`--action-mode residual` 评估）
- 部署消费：桥接节点加载 `.zip` + `_vecnormalize.pkl`（必要时转 ONNX）在部署侧推理
- 约定：训练侧发布到 `rl_grasping_system/models/` 并记录 SHA/日期；部署侧 `ros2_moveit2_ur5e_grasp`
  读取路径由 `ur5e_rl_bridge` 的 `model_path` 参数指定
- 策略只输出任务空间增量（臂无关），不依赖训练侧 IK/动力学

## 8. 安全（实机部署硬约束）

- 末端速度限幅：`max_ee_delta / 决策周期` 必须 ≤ 实机安全限速
- 关节限位：UR5e ±6.28 rad（训练侧模型与部署 URDF 一致）
- 奇异规避：速度雅可比病态检测；部署侧由 MoveIt2（Trac-IK/KDL + 关节限位 + 碰撞检测）兜底
- 碰撞检测：部署侧 MoveIt planning scene **点云聚类碰撞体**（`cluster_to_collision`，替代 OctoMap，
  决策见 `docs/混合控制架构设计.md` §3）；训练期即模拟限幅
- 安全层（Phase 4-5，规划）：CBF 式约束（最近障碍距离 + 相对速度求解最小修正量）、
  ISO/TS 15066 SSM 分离距离监控（过近先降速后停机）、独立实时节点
