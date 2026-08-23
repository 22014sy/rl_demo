# 阶段路线图：深度相机 + MoveIt2 静态抓取/避障 + RL 动态避障

> 状态：规划稿（2026-08-23）
> 范围：仅文档，不涉及代码改动
> 关联：`docs/混合控制架构设计.md`（技术契约）；`rl_grasping_system/`（RL 侧）；`../ros2_moveit2_ur5e_grasp/`（ROS2 侧）

---

## 1. 项目目标

在现有基础上分阶段落地一套"双控制器"抓取系统：

| 能力 | 技术方案 | 责任 |
|---|---|---|
| 感知 | 深度相机（Gazebo 仿真 → 真机 D435i） | 目标 3D 定位 + 环境障碍点云 |
| 静态抓取 + 静态避障 | MoveIt2（OMPL + planning scene 几何碰撞体） | 确定性、可证明无碰撞 |
| 动态避障 / 动态抓取 | 强化学习（PPO 残差策略） | 对快速移动的障碍/目标做出反应 |
| 安全兜底 | 独立实时安全层（CBF 式约束 + 限幅） | 任何模式下都保证安全 |

**架构决策：不使用 OctoMap**，理由与替代方案详见《混合控制架构设计.md》§3。

---

## 2. 现状盘点

### 2.1 已有的两套系统

| | RL 抓取系统（本仓库 `rl_grasping_system/`） | MoveIt2 系统（兄弟仓库 `ros2_moveit2_ur5e_grasp/`） |
|---|---|---|
| 仿真引擎 | MuJoCo 3.10 | Gazebo + ROS2 Humble |
| 机械臂 | UR5e + Robotiq 2F-85（MJCF） | UR5e + Robotiq 2F-85（URDF） |
| 感知 | ❌ 无相机；58 维 oracle 观测（目标位置直给） | ✅ RealSense 仿真插件 + YOLO + 深度反投影 + TF |
| 规划/控制 | PPO 端到端 → 末端速度增量 → 速度级 IK | MoveIt2 OMPL 规划 → 轨迹执行 |
| 避障 | ❌ 无 | ⚠️ OctoMap 链路已写但 launch 中被注释关闭 |
| 部署文档 | ✅ sim2real 契约（`docs/RL_DEPLOY_CONTRACT.md`） | ✅ Docker/启动脚本 |

### 2.2 关键缺口（按重要度）

1. **RL 是"上帝视角"**：目标位置由 oracle 直接给出，没有视觉输入 → 需要把 RL 改造成 POMDP 感知（最难，独立成阶段）。
2. **静态避障链路未启用**：`simulation.launch.py` 里 `octo_launch` 被注释；`ur5e_octomap_moveit` 的 `octomap_to_planning_scene_node` 也未启用。且本规划决定**弃用 OctoMap**，改用点云聚类碰撞体。
3. **两个仿真世界未打通**：MuJoCo 训练出的策略无法直接跑在 MoveIt2/Gazebo 世界（关节名、动作语义、动力学不一致）。
4. **无真实相机硬件**：`lsusb` 未检测到 RealSense → 前期全部用 Gazebo 仿真相机。

### 2.3 基于现有 RL 代码的改造可行性（结论：可改，无需重写）

现有 `rl_grasping_system/` 是"末端速度增量控制 + 环境内夹爪状态机"，**恰好是残差策略的直接前身**。改造面集中在 4 个文件 + 3 个小模块。

#### 直接复用（零改动）

| 模块 | 复用理由 |
|---|---|
| `ik.py` → `velocity_ik` | 末端速度→关节速度阻尼伪逆，残差策略的落点 |
| `environment.py` step 速度执行链路（484~517 行） | 速度 PID + 重力/惯性前馈，`v_max=0.125` 已与 MoveIt Servo 对齐 |
| 夹爪自动闭合状态机（`_update_gripper_state`） | RL 不管夹爪、状态机驱动，与真机 Robotiq 一致 |
| `reward.py` 势能塑形框架（`reward_breakdown` + `REWARD_KEYS`） | 结构清晰，Phase 4 只需新增项并扩展 `REWARD_KEYS`（勿混入诊断键） |
| `singularity_handler.py`、`agent.py`、SB3 训练管线 | 完全不动 |

#### 必须改造（函数级定位）

| 改造 | 位置 | 阶段 |
|---|---|---|
| 🔴 干掉 oracle 观测：`target_position/orientation` 从物理真值改为感知估计（`use_oracle` 开关保留对照） | `environment.py::_get_observation`（834 行）、`state.py::get_proprioceptive_state`、`_set_object_position` 双通道分离（物理位置仍走 freejoint，观测只吃感知通道） | Phase 2 |
| 🔴 动作叠加：增量 → 残差 `v = v_nominal(t) + Δv`，观测加"参考速度"段 | `environment.py::step` 速度解析段（388~420 行）；`action_mode: 'delta'\|'residual'` 开关 | Phase 3 |
| 🟡 动态障碍实体 + 机械臂-障碍碰撞检测 | MJCF 加移动障碍 body；碰撞查询仿 `_check_grasp_contact`（796 行）用 `data.contact` | Phase 4 |
| 🟡 决策频率提升 25→50Hz | `config.py`：`action_repeat=2→1` | Phase 4 |

#### 新增模块（3 个小模块）

```
rl_grasping_system/
├── perception.py          # Phase 2：目标感知估计器（噪声/漏检/深度特征占位）
├── nominal_trajectory.py  # Phase 3：标称轨迹生成器（MuJoCo 侧假 MoveIt）
└── (MJCF 内) 障碍物实体     # Phase 4：动态障碍 body
```

#### 改造注意事项

1. **物理位置与感知估计必须双通道分离**：物理真实位置仍由 `freejoint` 承载，观测只吃感知通道，否则仍是 oracle。
2. **残差策略需要"标称轨迹参考速度"进观测**：Phase 2 就预留该槽位（架构文档 §5.1）。
3. **奖励扩展**：只在 `reward.py` 加项 + 扩 `REWARD_KEYS`，`reward_breakdown` 的 dict 结构直接兼容。
4. **姿态固定朝下**（`action_space_dim=3`）：对"绕障后顶抓"够用；需侧向抓取时再开 6D 模式（代码已支持）。

---

## 3. 分阶段目标

### Phase 0：基线打通（约 1 周）

**目标**：确认 MoveIt2 静态抓取链路可用，RL 有可用基线策略，两套系统的模型/语义对齐表建立。

任务清单：
- [ ] 跑通现有 MoveIt2 抓取 demo（`ros2 launch ur_bringup simulation.launch.py` + `start_grasp.launch.py`），记录成功率基线
- [ ] 跑通 RL 侧 `rl_grasping_system` 已有策略评估，记录成功率基线
- [ ] 建立 **URDF ↔ MJCF 对齐表**：关节名、link 名、末端坐标系（`rq_base_mount` / pad sites）、动作语义、速度上限
- [ ] 确认深度相机数据流（仿真）：`/color/image_raw`、`/depth_registered/image_rect`、`/depth/points` 在 RViz 可见
- [ ] 记录环境版本（MuJoCo 3.10 / SB3 2.9 / ROS2 Humble / MoveIt2 Humble）

验收标准：
- [ ] 两套系统各自有可复现的运行基线与成功率记录
- [ ] 对齐表文档评审通过（后续所有阶段依赖它）

### Phase 1：MoveIt2 静态抓取 + 静态避障闭环（2~4 周）

**目标**：让 MoveIt2 真正"看见并绕开静态障碍物"，不依赖 OctoMap。

任务清单：
- [ ] 视觉包鲁棒化：`vision/obj_detect.py` 多目标、遮挡；深度反投影误差标定
- [ ] 新增点云→碰撞体节点 `cluster_to_collision`（替代 octomap_server）：
      `depth/points` → 平面移除 → 欧式聚类 → AABB/OBB/凸包 → planning scene `add_collision_object`
- [ ] 抓取 demo 升级为状态机：预抓取 → 抓取 → 抬升 → 放置
- [ ] Gazebo 静态障碍物场景测试（箱子/立柱），OMPL 自动绕障后抓取
- [ ] （可选对照）保留 `use_octomap: bool` 开关，OctoMap 仅作论文对照组

验收标准：
- [ ] 固定场景成功率 ≥ 95%
- [ ] 有静态障碍物场景成功率 ≥ 90%，全程零碰撞
- [ ] 随机 50 场景成功率 ≥ 85%

### Phase 2.0：随机抓取——改造前置验证（约 1 天）

**目标**：以最低成本验证"策略基于观测泛化"这一核心假设，为 Phase 2 感知升级扫清风险。

**背景**：
- 固定位置抓取已收敛（最新成功率 0.82~0.87，2026-08-23 日志）——策略可能部分过拟合 `(0.1, 0.42)` 单点。
- P0-2 已验证"环境级随机化 + IK 可达性"（`docs/P0-2_物体位置随机化.md`），但**从未验证"随机位置下 RL 训练收敛"**。
- 随机抓取与 Phase 2 感知升级共享同一核心难点：策略必须利用观测里的 `target_position` 做条件化。改动量天差地别（一行配置 vs 4 个文件+新模块），故先做低风险验证。

任务清单：
- [ ] `config.py`：`use_fixed_position=False`；按实测 IK 可达域**收紧 `workspace_bounds`**（P0-2 警示：边界处可能超出 IK 可达域，先跑 `scripts/verify_position_randomization.py` 打点）
- [ ] 沿用现有 `train_cloud.py` / `train_with_monitor.py` 重训，**不动任何架构代码**
- [ ] 完整过一遍 评估 → 可视化 → 调参 工作流（为后续改造建立操作手感与对比基线）
- [ ] 记录随机抓取成功率 = **泛化基线**，留档进本仓库

验收标准：
- [ ] 随机抓取成功率 ≥ 70%（比固定 85% 低一档为合理），成功率曲线单调上升
- [ ] 达标即关停，不无限调参

分支决策：
- **收敛** → 泛化链路通，直接进入 Phase 2 感知升级（观测从真值换成噪声/深度数据源即可）
- **不收敛** → 先诊断泛化问题（观测利用不足 / 奖励 / 网络容量），这是 Phase 2 的前置调试，比"改造后调试"便宜一个量级

### Phase 2：RL 感知升级——从上帝视角到"看得见"（2~3 周）

**目标**：给 MuJoCo RL 环境加相机/感知，观测从 oracle 改为部分可观测。

任务清单：
- [ ] 降级方案先行：目标位置加高斯噪声 + 随机漏检（模拟检测误差），验证 PPO 在感知不确定性下仍收敛
- [ ] 用 `mujoco.Renderer` 渲染深度图/点云作为观测（现有 renderer 已存在，改造低成本）
- [ ] 观测空间从 58 维 oracle 升级为"状态 + 目标感知特征 + 障碍距离特征"
- [ ] 为残差策略预留"标称轨迹参考速度"槽位（用脚本轨迹充当假 MoveIt 标称轨迹）
- [ ] 障碍物版观测预留位：相对位姿 + 速度（Phase 4 用）

验收标准：
- [ ] 带感知噪声/深度输入下，成功率相对 oracle 版下降 < 10%

### Phase 3：双世界桥接（2~4 周）⭐ 工程核心

**目标**：把 MuJoCo 训练出的策略能在 MoveIt2/Gazebo 世界运行。

任务清单：
- [ ] 按 Phase 0 对齐表统一两边动作语义：RL 末端速度增量 ↔ MoveIt **Servo** twist 接口
- [ ] 用 sim2real 思路标定 Gazebo 与 MuJoCo 动力学差距（复用 `docs/sim_to_real动力学匹配方案.md` 方法）
- [ ] 轻量 gym→ROS2 桥：订阅 Gazebo 状态/相机，发布 Servo 命令，把 RL 策略包装为 ROS2 节点
- [ ] 在 MuJoCo 侧实现"标称轨迹 + RL 残差"叠加机制，与 Gazebo 侧行为对比

验收标准：
- [ ] 同一 PPO 策略在 MuJoCo 与 Gazebo 中抓取轨迹相似度 > 90%
- [ ] 两端成功率差距 < 15%

### Phase 4：动态避障 RL（3~6 周）

**目标**：训练"动态避障"残差策略（不是重新学整个任务）。

任务清单：
- [ ] MuJoCo 与 Gazebo 引入同构动态障碍物（摆锤 / 移动球 / 随机轨迹物体），保证域一致
- [ ] `reward.py` 扩展：障碍物接近惩罚、碰撞终止、安全距离保持项
- [ ] 课程学习：静态 → 慢速单障碍 → 中速 → 多障碍快速，每级达标再升级
- [ ] 安全层加入训练（训练期即模拟限幅，部署一致性）

验收标准：
- [ ] 课程最高级下动态场景成功率 ≥ 70%
- [ ] 碰撞率 < 5%

### Phase 5：混合控制闭环（MoveIt2 + RL 融合）（3~5 周）

**目标**：静态用 MoveIt2、动态用 RL 残差，安全层兜底。

任务清单：
- [ ] MoveIt2 标称轨迹（含静态避障）→ RL 残差修正 → Servo 执行 → 安全层监控
- [ ] 模式状态机：静态模式（MoveIt2）/ 动态模式（RL 残差），无扰过渡
- [ ] 建立对比基准：纯 MoveIt2 高频 replan vs 混合方案，对比成功率/碰撞率/时延

验收标准：
- [ ] 动态场景成功率 ≥ 80%
- [ ] 静态场景不劣于纯 MoveIt2 的 90%

### Phase 6：真机部署（4~8 周，依赖硬件）

**目标**：真实 UR5e + RealSense D435i。

任务清单：
- [ ] 手眼标定（eye-in-hand / eye-to-hand 外参）
- [ ] 真实点云 → 聚类碰撞体 → MoveIt2 静态避障
- [ ] RL 域随机化与实机校准（复用已有 sim2real 文档）
- [ ] 安全层实机化：急停、速度/力矩限幅、分离距离监控（ISO/TS 15066 风格）

验收标准：
- [ ] 真机静态抓取成功率 ≥ 80%
- [ ] 动态避障完成现场演示

---

## 4. 风险与关键决策

| 风险 | 等级 | 缓解 |
|---|---|---|
| Phase 2 RL 从 oracle 转视觉感知，收敛难度骤增 | 高 | 先做"加噪声降级方案"兜底；动作/奖励不变只换观测 |
| Phase 3 两世界不一致导致 Phase 4 成果作废 | 高 | 先对齐再动动态障碍；以轨迹相似度 90% 为硬门槛 |
| 动态避障对观测频率要求高（现状 25Hz 偏慢） | 中 | 动态场景决策频率提升到 50~100Hz + Servo 实时模式 |
| 真机硬件未到位 | 中 | Phase 0~5 全部可用 Gazebo 仿真完成，真机只影响 Phase 6 时间 |

关键决策（已定）：
1. **不用 OctoMap**：静态用点云聚类几何碰撞体，动态用距离反应式（RL 残差），安全用 CBF 式约束。（业界依据见《混合控制架构设计.md》§3）
2. **混合架构 = 残差策略**：MoveIt2 标称轨迹 + RL 残差修正 + 独立实时安全层。
3. **训练侧继续用 MuJoCo**（快、可并行），通过"域一致性 + ROS2 桥"部署到 Gazebo/真机。

---

## 5. 时间线总览（单人估算）

```
Phase 0 ██████                              1 周
Phase 1 ██████████████████                  2~4 周
Phase 2.0 ███                               1 天（改造前置验证）
Phase 2 ██████████████                      2~3 周
Phase 3 ██████████████████                  2~4 周
Phase 4 ██████████████████████████          3~6 周
Phase 5 ██████████████████████              3~5 周
Phase 6 ██████████████████████████████████  4~8 周（依赖硬件）
--------------------------------------------------
合计                                        4~6 个月
```

## 6. 配套文档

- `docs/混合控制架构设计.md` —— 三层架构、避障方案对比、接口契约、与现有代码衔接点
