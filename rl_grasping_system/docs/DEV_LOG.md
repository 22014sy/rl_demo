# RL 抓取系统 开发与研究日志 (DEV_LOG)

> 目的：记录「RL 抓取训练 0% 成功率」的修复过程与关键技术决策（servo 滞后、action_repeat、
> MoveIt2 部署、速度控制、Robotiq 2F-85、仓库组织），以及每一轮问答的结论与依据。
> 起始于 2026-08-20，持续追加。工程变更条目见 `CHANGELOG.md`；训练↔部署接口见 `RL_DEPLOY_CONTRACT.md`。

---

## 2026-08-22 · 轮：混合奖励方案——成功率 0% → 6%

【问题】A 训练（官方 VecNormalize 后）策略"大幅靠近但未精确对准"：`r_contact=0`、`r_grasp=0`、成功率 0%。

【根因】（见 `docs/奖励设计与混合塑形方案.md`）
1. `w_contact=0`、`w_orient=0`（当日调试时被禁用）→ 接触/方向无学习信号
2. **驻留陷阱**：`shaping_gamma=0.9 < PPO γ=0.99` → 停驻每步白拿 `(0.99−0.9)·Φ≈0.09·Φ` 正收益
   （d≈0.05 处 0.105/步，300 步 ≈31 ≫ 步惩罚−15）→ 策略停在"够近但不精调"处是最优驻留
3. 唯一事件奖励 grasp_reward=100 需走完"对齐→闭合→width 收敛"全链路，40k 步探索概率≈0

【修复】
- 恢复 `w_contact=5.0`、`w_orient=3.0`（git 原值）
- `shaping_gamma` 0.9 → **0.99**（= PPO γ，消除驻留收益，满足 Ng 势能塑形等价性）
- **近距离锚定势能**：`Φ_anchor = anchor_w·min(d/anchor_dist, 1)` 叠加到主势能
  （anchor_w_xy=1.0/0.06m、anchor_w_z=1.0/0.04m；近距离斜率 xy 6.7→23.3、z 20→45，
  每靠近一步塑形收益 0.115 > step_penalty 0.05，精调最后一步有利可图）
- ⚠️ 曾误用 `max(0,1−d/d_target)` 使近距离势能非单调（靠近反被惩罚），已修正为单调形式

【A2 训练结果】（40k 步 / 136 episodes，日志 `training_log_20260822_113525.json`）
```
成功率: 6.0% (8次成功，A 训练 0%) | 平均奖励: -2.46 | 最佳: 121.8
r_contact 非0 = 23 次 | r_grasp 非0 = 8 次
r_dist_xy/z 累计 ≈ 7（vs A 的 53/80：策略停在目标而非大幅移动）
r_orient 均值 -8.3（手指未朝下，但斜抓仍能触发闭合成功）
```
**结论**：混合方案打破 0%——策略学会"接近→接触→闭合→抓取"全链路（接触→闭合成功率 ~35%）。
方向未学对（斜抓成功），is_grasped 判定与方向无关。

【下一步】① evaluate 验证加载一致性（final_model.zip 含官方 VecNormalize stats）；
② 若需顶抓 → 强化方向（action_space_dim=3 固定朝下 / 提高 w_orient）；③ 更长训练看成功率上升。

---


## 2026-08-22 · 轮：观测归一化替换为官方 VecNormalize + A 训练验证

【变更】自定义 `VecNormalizeWrapper`（缺陷：无 save/load → 评估观测统计丢失 / reset 不更新 / 无 PPO 集成）
替换为官方 `stable_baselines3.common.vec_env.VecNormalize`（`agent.py` train 与 load 两处）。记录见 `docs/VecNormalize观测归一化替换.md`。

【A 训练结果】（40k 步 / 139 episodes，日志 `training_log_20260822_014453.json`）
- `rdist_xy`: Q1=49.2 Q2=44.1 Q3=51.4 Q4=53.5（随机基线 45.5）→ **中期出现 xy 对准学习信号**
- `rdist_z`: 47→80（臂未降到 cube 高度）、`r_contact`=0（从未接触）
- **结论**：归一化修复有效（替换前 57ep 策略 r_dist≈52.8 完全随机），但 40k 步只学到"xy 对准"，
  接触/抓取未出现——z 下降收益稀疏、接触奖励需精确触发是候选原因。

【下一步】评估 `r_dist_z` 奖励权重与接触 shaping，或延长训练观察 `r_contact` 是否出现。

---


## 2026-08-20 · 轮 1：RL 抓取 0% 成功率根因分析（三个"陷阱"裁决）

【问题】RL 抓取训练成功率恒为 0%。

【候选陷阱】
1. Kp=45000 高增益 → PD 震荡
2. lift_speed 削足适履（削 max_steps 掩盖问题）
3. 50Hz RL vs 1000Hz PD 频率错配 → 噪声

【实测裁决】

| 陷阱 | 裁决 | 依据 |
|---|---|---|
| 高 Kp→震荡 | ❌ 不成立 | ζ≈10~15 极度过阻尼 + forcerange ±87N·m 饱和 → 单调趋近、无振荡 |
| lift_speed 削足适履 | ❌ 架构不成立 | lift 是 grasp_success 后自动状态机，`step()` 覆盖 RL 动作，RL 不参与抬升 |
| 频率错配→噪声 | ⚠️ 结论修正 | 实际 50Hz RL vs 500Hz 物理（substeps=10）；非噪声，是**每 RL 步伺服收敛不足** |

【关键实测】命令-实际位移实现率：

| 命令 | substeps=10 | substeps=20 |
|---|---|---|
| Δx=0.02（最大步进） | ~28% | ~28% |
| Δz=0.003（RL 早期小动作） | **8.1%** | 41.3% |

【结论】小增量实现率仅 ~8% → RL 早期随机策略几乎推不动末端 → 训练 0% 的物理根因。
【决定】增加每决策步内的伺服收敛时间（即后来的 `action_repeat`）。

---

## 2026-08-20 · 轮 2：P5 action_repeat 实现与验证

【问题】如何让单决策步内伺服收敛更多，又不破坏物理与抬升。

【改动】
- 修正 `substeps = (1/control_freq)/timestep`；control_freq 不变，改用 `action_repeat=2`
- `config.py`: `action_repeat=2`；`environment.py`: 物理循环 `range(substeps*ctrl_cycles)`；
  **lift 阶段 ctrl_cycles=1**

【实测】
- 小增量实现率 **8% → 23%**（↑3×），保持 50Hz 控制 / 25Hz RL 决策
- lift 若用 action_repeat：每步"脉冲式"抬升、物体滑移 41% → lift_aborted 失夹 → 故 lift 必须绕过

【验证】`smoke_lift` PASS（EE dz=0.185 / obj dz=0.1857，100% 刚性跟随，抬升期奖励恒 0）；
`verify_grasp_success_criterion` PASS；`verify_ee_pose_control` PASS；truncation PASS
（`max_steps=10` 不截断抬升，`trunc=False`/`term=True`）；PPO smoke train PASS（~6.31s/episode，仍 0%）。

【根因闭环】环境可端到端完成任务（smoke_lift 直接 IK 流程能完整抓取+抬升）；剩余 0% 是 **RL 学习层**问题，非环境物理。

【遗留·层 2】贪心策略（朝物体方向最大增量）从 safe init 可接近 ~0.10~0.12m，但未完成精确对准抓取；
修正 z 目标（obj.z+FINGER_REACH）后**奇异点=0**——此前日志的高奇异计数是策略乱动的**症状**而非独立障碍。

---

## 2026-08-21 · 轮 3：概念澄清——「刷新频率追不上 RL 目标？」

【问题】刷新频率类似执行器约束导致"点击追不上 RL 输出的目标"？

【结论】方向对但不精确：不是刷新频率，而是**「每决策步收敛时间 vs 伺服时间常数」不匹配**
（0.02s vs τ≈0.1s → 只实现 8%）。精确类比：每 40ms 发一条指令，但每条要 100ms 执行完 →
指令积压、实际位置恒定滞后于指令流。命令等效速度 0.15 m/s vs 实际末端 0.012 m/s（8%）。

【实机部署建议】解耦：RL 低频输出目标、规划层高频平滑、伺服小步跟踪；用阶跃响应/实现率标定
决策周期与动作尺度。

---

## 2026-08-21 · 轮 4：MoveIt2 方案确定

【用户要求】1) 记录对话到 md；2) 执行方案一（MoveIt2 规划层）；3) 未来的建议和改动从实机部署角度考虑。

【调研】MoveIt2 2.5.9 已装（含 `moveit_servo`）；UR5e 全套（`ur_moveit_config`/`ur_robot_driver`/`ur_bringup`）；
RL 动作空间 = **6D 末端增量 `[dx,dy,dz,dax,day,daz]`，臂无关**。

【方案一】`RL(6D 末端增量) → MoveIt2 规划层 → ros2_control → 仿真/实机后端`。
【决策】实机是 **UR5e**；RL 训练也迁移 UR5e（训练=部署同臂，消域差）。

---

## 2026-08-21 · 轮 5：Robotiq 2F-85 夹爪确认

【问题】夹爪能换成 robotiq 的吗？

【重大发现】`/home/zrq/githubprojects/ros2_moveit2_ur5e_grasp/` 是**已建成**的完整部署栈：
UR5e + **Robotiq 2F-85** + MoveIt2 + Gazebo + YOLO 视觉 + OctoMap。

| 事实 | 依据 |
|---|---|
| 末端挂 2F-85 | `ur5e_gripper.urdf.xacro` include `robotiq_2f_85_macro`，挂 `tool0` |
| 2F-85 关节模型 | `robotiq_85_left_knuckle_joint` 主动 + 其余 mimic（四连杆并联） |
| 夹爪控制 | `/gripper_controller/gripper_cmd` action，`grasp(0.36)` 夹 / `grasp(0)` 放 |
| MoveIt 规划组 | `ur_manipulator` |
| 当前抓取逻辑 | `demo.cpp` 对 `target_pose_list.yaml` 硬编码位姿 plan+execute（**非 RL**） |

【结论】部署**已是 2F-85，无需换**；要换的是训练模型夹爪（我原计划"简化 2 指"是错的）→
训练模型也用 2F-85 对齐几何/开度。2F-85 vs Panda 手爪：开度 85 vs 80mm（`grasp_min_width` 参数可复用）；
平行指尖 vs 曲线手指（接触判定需对应）；同为位置控制（部署映射 Robotiq Position 命令）。

---

## 2026-08-21 · 轮 6：力矩控制 vs 速度控制

【问题】力矩控制还是速度控制好？

【裁决】

| 模式 | UR5e 实机 | 裁决 |
|---|---|---|
| 力矩 | ❌ 无真正全关节力矩（`forward_effort_controller` 的 effort 只是**力矩前馈叠加在位置/速度回路**上） | **排除** |
| 速度 | ✅ 原生 `speedJ/speedL` + `forward_velocity_controller`（interface=velocity） | **推荐主模式** |
| 位置 | ✅ MoveIt2 标准（joint_trajectory_controller position） | **接触/抬升兜底** |

【机制】位置：位置误差收敛 τ≈0.1s → 决策步 0.02s 只实现 8%；速度：`v×dt` 精确实现 → 实现率≈100%。
`action_repeat` 是位置控制下的补丁，**速度控制是根治**。
【代价与缓解】速度积分漂移→位置闭环校正；加速度限制→决策周期内平滑（MoveIt Servo/UR acc）；
MuJoCo 换 velocity actuator→标定速度 PID；奇异→速度雅可比病态检测（UR5e 关节限位 ±6.28 rad 较宽裕）。
【关键事实】`ur5e_gripper_controllers.yaml` 已同时配 position 与 velocity 控制器 → 切换成本低。
【决策】**速度控制为主**：训练 velocity actuator（`action_repeat` 退役）；部署接近用速度
（MoveIt Servo/`forward_velocity_controller`），接触/抬升用现有位置控制 + 力监控兜底。

---

## 2026-08-21 · 轮 7：仓库组织

【问题】训练侧与部署侧会合在当前项目文件里吗？

【结论】**不物理合并**：两仓分离 + 接口契约。会合点是 ① 接口契约 ② 策略产物（SB3 `.zip`→部署桥接）
③ 状态机语义。生命周期（训练频繁迭代 vs 部署稳定）、依赖（训练无 ROS vs 部署 ROS2）、稳定性差异决定分离。

【隐患】`robotic_arm_control/ros2_ws` 是**半成品**：`robotic_arm_description/launch/` 空、
`controllers.yaml` 是 panda 注释残留草稿（mujoco_ros2_control 接 UR5e 实验未完成），与部署仓功能重叠。
【决策】**两仓分离**：训练侧留 `robotic_arm_control/rl_grasping_system`（迁 UR5e+2F-85），
部署侧留 `ros2_moveit2_ur5e_grasp`；新建接口契约；归档清理 `robotic_arm_control/ros2_ws` 半成品。
## 2026-08-21 · 轮 8：menagerie Robotiq 2F-85 下载与集成（真实模型替代手写简化版）

【问题】训练侧夹爪原为手写简化平行二指，几何/质量与部署 2F-85 偏差大。连网后能否拿到官方 menagerie 模型？

【网络事实】PyPI **无** `mujoco-menagerie`/`mujoco_menagerie`（pypi.org 均 404，pip 永远装不上）；
GitHub `google-deepmind/mujoco_menagerie` 可达，但 raw/git clone 极慢（带宽受限）；**jsDelivr CDN**
（`cdn.jsdelivr.net/gh/...`）加速下载成功 → 8 个碰撞 STL + `2f85.xml` 全部拉取。

【menagerie 2F-85 结构】真实四连杆并联：
- 主动关节 `right_driver_joint`（range [0,0.8] rad）+ equality 同步 `left_driver_joint`
- `split` tendon（右/左 driver 各 0.5）→ 单 actuator `fingers_actuator`（ctrl 0–255，kp=100 位置伺服）
- spring_link 弹簧预紧（springref=2.62）保自锁；碰撞网格 + 真实质量/惯性

【velocity servo 关键修正】`biastype` 枚举**无** `velocity`/`integrated`/`pid`
（mjtBias 只有 NONE/AFFINE/DCMOTOR/MUSCLE/USER，查 mjmodel.h + python 枚举确认）。
正确写法（版本无关，实测收敛）：
```xml
<general joint="..." gaintype="fixed" biastype="affine" gainprm="30" biasprm="0 0 -30" .../>
```
即 f = kp·ctrl − kp·qvel = kp·(ctrl − qvel)，biasprm=[0,0,−kp]（const, pos, vel 系数）。

【集成要点】`rq_` 前缀防 2F-85 与 ur5e 命名冲突（body/mesh/class/material/tendon/contact 全改名）；
`base_mount` 挂 `attachment_site`（pos=0,0.1,0 / quat=绕x −90° `1 -1 0 0`）→ **指尖朝下、开合沿世界 x 水平**；
左右 pad 各加 site（`rq_pad_*_site`）量抓取宽度。

【验证】nq14（6 臂 + 8 个 2F-85 hinge）/ nu7 / nbody22；pinch 在 site 下方 0.149m（指尖朝下）；
闭合 ctrl255→width 0.009m（夹紧），全开→0.093m；臂速度控制 ctrl=0.5 → qvel 0.498 ✓。

【新耦合点（Task 3 T2 必须处理）】2F-85 的 **8 个关节全是 hinge** → `_find_components` 的
HINGE 臂关节过滤会把它们混入。T2 需按关节名前缀（`rq_`）或 qposadr < 6 排除夹爪关节，
并实现 2F-85 宽度状态机（`grasp_min_width=0.02` 沿用，开度 85mm）。

【结论】手写简化 MJCF fallback **退役**；训练模型现为 menagerie 2F-85（几何/质量/碰撞网格与
部署 `robotiq_2f_85_macro` 对齐，几何基准仍以 `ur5e_gripper.urdf.xacro` 为准）。

---

## 2026-08-21 · 轮 9：Task 3 训练侧迁移 UR5e + 2F-85 + 速度控制（完整闭环）

【任务】按 RL_DEPLOY_CONTRACT v1 把训练侧从 Panda 迁到 UR5e + Robotiq 2F-85 + 速度控制；
归档 `robotic_arm_control/ros2_ws` 半成品。

### A. cube 场景 + 环境改造
- `models/universal_robots_ur5e/ur5e_robotiq_cube.xml`：ur5e_robotiq.xml + 固定桌面 + `target_cube`(0.04m, freejoint)。
  桌面半边长 **0.45 → 0.24**：0.45 会伸到上臂活动区（y<0.14）→ `upper_arm_link<->table` 接触，
  qfrc_constraint 高达 **356 N·m** 直接把速度环压崩（排查 8 轮才定位，见下）。
- `config.py`：xml_path→cube；新增 `control_mode/velocity_kp/arm_joint_names/end_effector_body/
  gripper_actuator_idx/pad_site_*/gripper_max_width(0.093)/velocity_ik_lam/table_top_z/object_fixed_pos`；
  `object_rest_z` 0.02→0.32（Panda 桌面在 z=0，UR5e 桌面顶 0.30+半边长）；`pre_grasp_offset_z` 0.10→0.134
  （2F-85 pad 中心距 rq_base_mount=0.134m）；workspace_bounds 收窄到 IK 可靠区；action_repeat 保持 2（25Hz 决策）。
- `environment.py`：`PandaGraspingEnv→GraspingEnv`（留别名）；`_find_components` 排除 `rq_` 铰链（8 个 2F-85
  hinge 全是 hinge，原过滤会混入）；gripper 宽度改 **pad site 距离**；手指 geoms 改 `rq_*_pad`；夹爪 actuator
  index=6；step 双模式（velocity=增量→速度→IK→PID；position=旧位置伺服对照）；观测 58→55 维（6 关节）。
- `state.py`：arm_joint_ids 索引 + tendon 语义改 2F-85 开度；`get_end_effector_position` 用 env.end_effector_id
  （原 fallback xpos[-1] 在 cube 场景会取到 target_cube！）。
- `ik.py`：新增 `velocity_ik`（mj_jacBody + 阻尼伪逆 `dq=Jᵀ(JJᵀ+λ²I)⁻¹v`）+ `finger_reach` 动态指尖测量。
- `singularity_handler.py`：UR5e 6 关节版（去掉 shoulder_lift±π/2 误报——UR5e 肩奇异需肘伸直+wrist_2=0 组合）。

### B. 速度控制标定（关键）
1. **纯 velocity-servo（biasprm=[0,0,-kp]）在重力静载荷下拉塌**：kp=30 重力不平衡（shoulder_lift 需 ~21 N·m），
   kp=1000 又遇 **wrist forcerange ±28 N·m 饱和**。正解 = **arm 改 motor(力矩) + 环境内"速度 PID + 重力补偿前馈"**
   `τ=kp·(dq−qvel)+qfrc_bias`（复刻真实 UR speedJ 内置 PID+重力补偿）。kp 按关节标定：肩/肘 150、腕 100。
2. **速度环能力上限 ~0.5 m/s**：v=0.5 时 dq 大→kp·dq 接近饱和→相位滞后振荡（shoulder_pan qvel 反转到 +1.17）。
   v=0.125 干净跟随（qvel 精确匹配 dq）。决策周期 T=0.04s、max_ee_delta=0.02（v_max=0.5）**脉冲实现率 96%**
   （vs Panda 位置伺服 8-41%）。
3. **桌面碰撞是"训练时 arm 乱漂"的真正根因**：home 位形 pad 距 cube 顶 0.014m 看似安全，实际
   `upper_arm_link` 与桌面边缘接触（qfrc_constraint 105-356 N·m）→ 速度环被约束力压过。桌面缩小后消除。

### C. 2F-85 手指实测
- `fingers_actuator`（menagerie）：**ctrl=0 → 全开(width 0.0934)；ctrl=255 → 闭合(0.0458)**——与 Panda
  `255=张开` 相反，`_update_gripper_control` 方向反了导致"closing 阶段手指反而张开"、抓取永不成功。
- 空闭合（无物）width≈0.046 > grasp_min_width=0.02 → 成功判定必须靠 **contact**（pad 被 cube 挡住），宽度阈值
  不能单独用（verify_grasp_success_criterion 场景 B 相应放宽）。

### D. 端到端验证
- e2e（速度模式 + 固定目标 P 控制）：**step10 phase=closed success=True**，cube 0.32→0.509（夹着抬升），lift_done。
- verify 迁移全部 PASS：`verify_ee_pose_control`、`verify_grasp_feasibility`、`verify_grasp_success_criterion`、
  `verify_position_randomization`（可达 100%）。`verify_grasp_feasibility` 抓取朝向修正：2F-85 开合沿局部 x，
  `frame_to_quat([0,0,-1],[0,-1,0])`（错误 `[1,0,0]` 会 roll 偏 90°，远端位置 pad 夹偏）。
- `smoke_train_p1.py` 迁移 → **SMOKE PASS**（4096 步，episode reward 33.5/length 300）。
- `train_with_monitor.py` 无头化（render_gui=False）→ **B3 重训练（seed 42, 40k 步产物）后期 rollout
  成功率 0.4–0.5、平均奖励 ~100+**——速度模式根治了位置伺服滞后，训练首次稳定出现成功。
- B4 产物 `models/final_model.zip`（2026-08-21 16:27，SHA `27c8ad1af7e0f3978d1f448eec8bd043`）。
  独立评估 20 episodes 成功率 0%（avg_reward 160.5、final_dist 0.65）——训练已出现成功（rollout 0.4-0.5），
  但策略未充分收敛（40k 短训练），且 `evaluate.py` 未加载 VecNormalize 观测统计（训练侧已启用归一化），
  评估观测分布失配；评估脚本迁移 + 更长训练收敛留待后续（属 Task 4 部署桥接/调参范畴）。

### E. ros2_ws 归档
- `ros2_ws` 确认被 `.gitignore` 忽略、0 个 git-track 文件 → 改名 `ros2_ws_archived_20260821`。
- 部署侧独立仓库 `ros2_moveit2_ur5e_grasp` 为唯一部署载体。

【结论】Task 3 训练侧迁移完成：UR5e+2F-85+速度控制（motor+PID+重力前馈）全链路验证通过，训练出现成功；
契约升级 v2（见 RL_DEPLOY_CONTRACT.md）。Task 4 只余部署侧桥接（D1，另仓）。

---

