# RL 抓取系统 训练 ↔ 部署 接口契约

> 版本：v2（2026-08-21，Task 3 训练侧迁移落地）
> 目的：训练侧（`robotic_arm_control/rl_grasping_system`，MuJoCo，无 ROS）与部署侧
> （`ros2_moveit2_ur5e_grasp`，MoveIt2 + UR5e + Robotiq 2F-85 + Gazebo，独立仓库）之间的**唯一契约**。
> 两侧独立实现，但本契约定义的语义（动作/观测/状态机/坐标系/产物）必须严格一致。
> 任何改动必须先改本契约，再同步两侧实现。

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

- 向量：`[dx, dy, dz, dax, day, daz]` —— 末端位姿增量（**任务空间，臂无关**）
- 语义：增量 / 决策周期 = **末端速度**（速度控制模式，见 §4）
- 限幅：位置增量 `max_ee_delta = 0.02` m/步；姿态增量 `max_orient_delta = 0.05` rad/步
  （实机部署必须折算为末端速度并对照安全限速）
- 坐标系：`base_link`（世界系，z 向上）

## 3. 观测空间

| 通道 | 维度 | 说明 |
|---|---|---|
| 末端位置 | 3 | base_link 系 |
| 末端姿态 | 4 | 四元数 (w,x,y,z) |
| 关节位置 | 6 | UR5e 六关节 |
| 关节速度 | 6 | UR5e 六关节 |
| 关节力矩 | 6 | qfrc_actuator（训练侧诊断用） |
| 夹爪开度 | 1 | 0–1 归一化（0=全开，1=全闭） |
| 肌腱/握持张力 | 3 | tendon_position/velocity/tension（2F-85 由开度导出） |
| 目标物体位置 | 3 | 桌面目标（部署侧由视觉/TF 提供，见 §5） |
| 目标物体姿态 | 4 | 四元数 |
| 手指接触力 | 6 | 左右手指力（简化模型） |
| 相对接近姿态 | 4 | q_target ⊗ q_ee⁻¹（w≥0） |
| 夹爪相位 | 1 | open=0 / closing=1 / closed=2 |
| 其他 | 4 | 末端速度(6)、可操作度(1)、接触力(1) 等 |

> **总计 55 维**（Task3：机械臂 6 关节；v1 时 Panda 7 关节为 58 维）。
> 注：训练观测以 `environment.py` 实际实现为准；部署桥接必须能提供同样通道（物体位姿由视觉节点发布）。

## 4. 决策 / 控制周期与模式

| 项 | 值 |
|---|---|
| RL 决策频率 | 25 Hz（`action_repeat=2` × 50Hz 控制周期） |
| 控制/规划频率 | 部署侧 MoveIt Servo / `forward_velocity_controller`（高频）；训练侧 MuJoCo（timestep 0.002s） |
| 主控制模式 | **速度控制**：arm = motor(力矩) + 环境内**速度 PID + 重力补偿前馈** `τ=kp·(dq−qvel)+qfrc_bias`（复刻 UR speedJ） |
| 速度级 IK | 雅可比阻尼伪逆 `dq=Jᵀ(JJᵀ+λ²I)⁻¹v`（λ=0.05），末端速度 v=Δp/T |
| 速度 PID 增益 | 肩/肘 150、腕 100（N·m·s/rad，按关节标定） |
| 命令实现率 | 脉冲 ~96%（决策周期内位移 vs 命令；v1 位置伺服仅 8-41%） |
| 兜底模式 | 位置控制 + 力监控（接触/抬升阶段） |
| 已排除 | 全关节力矩控制（UR5e 无此接口；effort 仅为力矩前馈） |

## 5. 状态机（两侧语义一致）

| 阶段 | 控制模式 | 触发/说明 |
|---|---|---|
| 接近 (approach) | 速度 | 初始 → 距目标 > 阈值 |
| 对准 (align) | 速度 | 进入末端对准窗口 |
| 接触/闭合 (contact/close) | 位置 + 力监控 | `gripper_close_distance` 触发夹爪闭合（2F-85） |
| 抬升 (lift) | 位置（恒定速度抬升） | `grasp_success` 后自动状态机，RL 不参与 |
| 放置 (place) | 位置 | 到达放置位后释放（2F-85 全开） |

> 参数沿用 `rl_grasping_system` 的 lift/grasp 状态机（`lift_speed`、`gripper_close_distance`、
> `gripper_open_distance`），部署侧按 2F-85 复核。

## 6. 抓取成功判定

**核心定义（P3 延续）：手指被物体挡住合不上 = 抓取成功**。Task3 在此基础上补齐触发条件与验证（防误判/防弹飞）。

### 6.1 触发闭合（open → closing），需同时满足
1. **XY 对齐**：物体中心 XY 与夹爪中心（左右 pad 中点）之差 < `grasp_align_xy_tol = 0.045` m（逼近夹爪半径 0.0468）
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

- 训练产出：SB3 PPO 模型 `.zip`（含归一化参数）
- 部署消费：桥接节点加载 `.zip`（必要时转 ONNX）在部署侧推理
- 约定：训练侧发布到 `rl_grasping_system/models/` 并记录 SHA/日期；部署侧 `ros2_moveit2_ur5e_grasp` 读取路径在 `D1 桥接` 阶段确定
- 策略只输出任务空间增量（臂无关），不依赖训练侧 IK/动力学

## 8. 安全（实机部署硬约束）

- 末端速度限幅：`max_ee_delta / 决策周期` 必须 ≤ 实机安全限速
- 关节限位：UR5e ±6.28 rad（训练侧模型与部署 URDF 一致）
- 奇异规避：速度雅可比病态检测；部署侧由 MoveIt2（Trac-IK/KDL + 关节限位 + 碰撞检测）兜底
- 碰撞检测：部署侧 MoveIt2 octomap（`ur5e_octomap_moveit`）
