# Task 3/4 迁移执行清单

> 状态：✅ 全部完成（2026-08-21）
> 目标：RL 抓取训练迁移到 UR5e + Robotiq 2F-85 + 速度控制；归档 `ros2_ws` 半成品。

---

## A. T2 · 训练侧参数化到 UR5e + 2F-85 + 速度控制

### A1. cube 场景
- [x] **A1.1** 新建 `models/universal_robots_ur5e/ur5e_robotiq_cube.xml`：ur5e_robotiq.xml + 固定桌面 + `target_cube`(0.04m, freejoint)。
  桌面半边长 **0.45→0.24**（0.45 会与上臂接触产生 356 N·m 约束，见 DEV_LOG 轮 9-B3）。

### A2. `config.py`（GraspingConfig）
- [x] **A2.1** `xml_path` 默认 → `ur5e_robotiq_cube.xml`
- [x] **A2.2** 新增 `control_mode="velocity"`；`action_repeat=2`（25Hz 决策，速度模式不退役、仅执行层换速度环）
- [x] **A2.3** 关节/夹爪绑定：`arm_joint_names`（6）、`end_effector_body="rq_base_mount"`、`gripper_actuator_idx=6`、`pad_site_left/right`
- [x] **A2.4** 夹爪参数：`grasp_min_width=0.02`（保持）、`gripper_max_width=0.093`、`gripper_close_distance=0.18`（2F-85 更长）
- [x] **A2.5** `workspace_bounds` 收窄到 IK 可靠区（cube 固定 0.51 走速度模式不受影响）
- [x] **A2.6** 新增 `velocity_ik_lam=0.05`、`velocity_kp=(150,150,150,100,100,100)`（按关节标定）

### A3. `ik.py`
- [x] **A3.1** 新增 `velocity_ik`：`mj_jacBody` 构 J + 阻尼伪逆 `dq=Jᵀ(JJᵀ+λ²I)⁻¹v`
- [x] **A3.2** `FINGER_REACH` 改由 `finger_reach()` 动态测量（2F-85 pad→rq_base_mount=0.134m）
- [x] **A3.3** `solve_ik`/`move_to_q` 保留（reset/verify 用），RL 热路径走 `velocity_ik`

### A4. `environment.py`
- [x] **A4.1** `PandaGraspingEnv` → `GraspingEnv`（留兼容别名）
- [x] **A4.2** `_find_components`：HINGE 过滤排除 `rq_`（8 个 2F-85 hinge）；gripper 用 pad site；ee 候选加 `rq_base_mount`
- [x] **A4.3** step 双模式：velocity=增量→v=Δ/T→`velocity_ik`→**速度 PID+重力前馈**（motor 力矩）；position=旧位置伺服对照
- [x] **A4.4** `_update_gripper_control`：`ctrl[gripper_actuator_idx]`；**实测 2F-85: 0=张开/255=闭合**（与 Panda 相反，已修正）
- [x] **A4.5** `_get_gripper_width`：**pad site 距离**（全开 0.0936 / 空闭合 0.046）
- [x] **A4.6** `_finger_cube_geoms`：`rq_*_pad`/`rq_*_silicone_pad` body
- [x] **A4.7** 观测 58→55 维（关节 6；tendon 语义改开度；grasp_phase 保留）
- [x] **A4.8** `state.py`：arm_joint_ids 索引 + 2F-85 tendon/夹爪语义 + ee 用 end_effector_id

### A5. 奖励/姿态复核
- [x] **A5.1** `Q_APPROACH_DOWN` 复核通过（base_mount 局部 +z 即手指方向且朝下，z_axis_z=-1 正确）；`pre_grasp_offset_z` 0.10→0.134（pad 对准 cube 中心）

## B. T3 · 脚本迁移 + 重训练 + 验证
- [x] **B1** `smoke_train_p1.py` → GraspingEnv + 速度模式 → **SMOKE PASS**（4096 步，episode reward 33.5/300）
- [x] **B2** 迁移 5 个脚本全部 PASS：`verify_ee_pose_control`（55 维/速度跟踪/零漂移）、`verify_grasp_feasibility`、
  `verify_grasp_success_criterion`（A 抓取/B 空闭合/C 滞回）、`verify_position_randomization`（可达 100%）、`diag_gripper_pose`
- [x] **B3** 重训练（速度模式，seed 42，150k 步）→ **15k 步即出现成功抓取（成功率 2.0%、best reward 111.73）→ success > 0% ✓**
- [x] **B4** 产物：`models/final_model.zip`（10.2MB，2026-08-21 16:27，SHA256 `27c8ad1af7e0f3978d1f448eec8bd043`）；
  训练日志 `logs/training_log_20260821_161159.json`；后期 rollout 成功率 0.4–0.5

## C. Task 4 · ros2_ws 归档
- [x] **C1** `ros2_ws` 确认被 `.gitignore` 忽略、0 个 git-track → 改名 `ros2_ws_archived_20260821`
- [x] **C2** DEV_LOG 轮 9 记录全程 + 契约 v1→v2（观测 58→55、速度 PID 执行层、命令实现率 ~96%）

