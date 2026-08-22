# 变更记录 / CHANGELOG

> 约定：**从 P3.1 起**，每次代码改动 + 相关讨论/决策总结都追加到本文件，条目按日期倒序。
更早的阶段性设计与讨论见 docs/P0-2、P1、P2-2、P3 等设计文档。

---

## 2026-08-22 · 方案A+B：打通 closing→closed 最后一环（首次出现成功）

### 背景
PPO_90（惩罚调整后，200k 步）奖励转正（+2.78）、接触率 50%，但仍 0 成功。
模型评估 73% 能进入 closing（pad 到达触发区），**closing→closed 0/60**。
实验实录 closing 期间行为：
- pad 在 cube 上方（XY 差 4.1cm）但接触力 0——pad 在 cube 表面外 2cm，手指夹不到
- 策略持续输出动作（pad 一直动）+ 手指 8 步快速合拢（宽度 0.09→0.013）
- "连续 3 步宽度收敛"（close_width_tol 0.004）永不满足 → 空闭合回退

### 根因
1. `grasp_align_xy_tol=0.045` 太宽松：pad 距 cube 中心 4.5cm 时 pad 在表面外 2.5cm（cube 半宽 2cm），夹不到。
2. closing 阶段策略不停手 + 手指合拢快 → 宽度不收敛。

### 改动
| 方案 | 位置 | 改动 |
|---|---|---|
| A | `config.py` | `grasp_align_xy_tol` 0.045→**0.030**（pad 必须几乎正对 cube 中心才触发 closing） |
| B | `environment.py` step | closing 阶段覆盖 `_vel_target=0`（类似 lift 阶段），pad 停住让手指自然合拢 |

### 验证（PPO_90 模型 + 新环境，60 episodes）
- **首次出现成功：1/60（1.7%）**（ep6 phase=closed 成功=True）——B 机制打通闭合链路
- 进入 closing：37/60（62%；阈值收紧后旧模型部分不达标，预期重训后适配）
- `verify_grasp_success_criterion` PASS（状态机无回归）

### 下一步
**重训**（PPO_90 学的是 XY 4cm 对齐，新阈值 3cm 需策略学更精确；XY 锚定势能 0.06 支撑）。
预期重训后成功率显著提升。

---

### 背景
方案A 后完整训练（PPO_89，200k 步）仍 0 成功。breakdown 显示：
- `r_step=-20.00`：200 步 × step_penalty 0.1，吞掉策略全部靠近收益（XY+Z ≈ +11.7）
- `r_orient=-5.99`：**3D 固定姿态下方向项是纯损耗**（恒 -0.03/步，姿态不可变无学习信号）
→ 平均奖励 -13.8，事件奖励（grasp=100）占比被稀释。
（详见 `docs/调参失败分析.md` PPO_89）

### 改动（`config.py` RewardConfig）
| 项 | 前 | 后 | 理由 |
|---|---|---|---|
| `w_orient` | 3.0 | **0.0** | 3D 姿态固定，方向项纯损耗；6D 时恢复 3.0 |
| `step_penalty` | 0.1 | **0.05** | 200 步成本 -20→-10；方案A 下压收益 0.17-0.33 仍 >> 0.05 |

### 验证
- `o_gain=0.0000`（方向项纯损耗消除）
- Z 差 1-6cm 下压收益 +0.30~+0.33 > 0.05 全部 ✅
- 语法 OK

### 预期
平均奖励 -13.8 → 约 -3~+2；重训后看接触率/成功率是否提升。

---

### 背景
用户观察：训练演示时机械臂 pad 已到 cube 上方"夹取位置"，但夹爪张开、停住不抓。
实验定位（`docs/训练效果分析与VecNormalize保存修复.md`）：
- 闭合触发需 pad 中心与 cube 中心 **Z 差 < 3.2cm**（= pad 距 cube 顶面 < 1.2cm），非常严格。
- 实测：pad 悬停 cube 顶上方 2cm + XY 偏 2cm 时 Z 差 3.96cm → **差 0.76cm 不触发**。
- 根本：Z 主势能 slope = w_height/height_scale = 2.0/0.3 = 6.67/m，下压一步收益
  ~0.033 **< step_penalty 0.1**（净亏 0.067/步）→ 策略停在悬停区无下压动力。

### 改动（`config.py` RewardConfig）
| 项 | 前 | 后 | 效果 |
|---|---|---|---|
| `w_height` | 2.0 | **8.0** | 主势能 slope 6.67→26.7/m |
| `anchor_w_z` | 1.0 | **2.0** | 锚定 slope 25→33.3/m |
| `anchor_dist_z` | 0.04 | **0.06** | 锚定覆盖 pad 悬停区（Z差4-6cm） |

### 量化验证（修复后）
Z 差 15cm→1cm 全程下压一步收益 +0.17~+0.33（> step_penalty 0.1）：
```
Z差6cm(锚定边界): +0.333 ✅   Z差2cm(pad悬停): +0.309 ✅   Z差1cm: +0.303 ✅
```
修复前悬停区（Z差4-6cm）下压亏本 → 停驻；修复后全程有强下压动力。

### 回归验证
`verify_grasp_success_criterion` PASS（触发/重试逻辑无回归）。
`w_height` 调大不影响 Ng 势能塑形等价性（γ_shape=γ_mdp=0.99，不改变最优策略）。

### 下一步
重训验证：预期 pad 能从悬停区降到接触 → 触发闭合 → 出现首次真实成功。
若仍卡（XY 对齐不足）：闭合触发 Z 阈值受 home 位形约束（0.034m）不能直接放宽，
可考虑"进入 closing 后 pad 自动微降"（方案 C）。

---

### 背景
并行改造后首次完整训练（200k 步 / 9 分钟 / 1104 episodes）成功率 0%。
深度分析（见 `docs/训练效果分析与VecNormalize保存修复.md`）发现：
策略真实水平是「接触率升到 29%、XY 对齐在学」，但评估/演示全部显示「乱走远离」——
**观测归一化统计根本没被保存**。

### 根因（实证）
1. SB3 `PPO.save` 的 `_excluded_save_params` 排除 `_vec_normalize_env` → **zip 里没有
   `vec_normalize.pkl`**（实测 zip 内容只有 data/policy 等）。
2. 加载后 `obs_rms.mean 非零元素数 = 0/55`（初始值）→ normalize 失效。
3. 训练时观测被归一化（自洽学习，obs_rms 45/55 非零），评估/演示 stats=0/1 → **观测分布
   失配 → 策略输出乱动作**。对照实验：随机策略 d 0.286→0.280 稳定，模型确定性策略
   d 0.286→0.460（远离 = 失配铁证）。
4. `GraspingCallback._play_demo` 演示回调也喂原始 obs（未归一化），演示同样失真。

### 改动（`agent.py`）
- `save`：`self.agent.save(path)` 后追加 `get_vec_normalize_env().save(path→_vecnormalize.pkl)`。
- `load`：创建 VecNormalize 后，若 `<path>_vecnormalize.pkl` 存在则
  `VecNormalize.load(pkl, vec_env)` 恢复统计，再 `PPO.load`。
- `predict`：统一先 `vn.normalize_obs(obs)` 再预测（SB3 PPO.predict 不自动归一化）。
- `_play_demo`：演示前归一化 obs，临时关 `vn.training` 避免污染统计。
- `train`：内部保存改走 `self.save`（连带存 stats）。

### 验证
save/load 往返测试 PASS：
- 训练后 `obs_rms.mean` 非零 45/55（归一化确实生效）
- save 生成 `_vecnormalize.pkl` ✓
- load 后统计恢复 45/55 ✓
- predict 归一化链路 OK ✓

### 遗留
旧 `final_model.zip`（无 pkl）无法恢复统计，评估仍会失配——**需重训**后用新 save 逻辑
保存，评估才反映真实水平。若重训后仍 0 成功（Z 方向弱是主要嫌疑）：提高 Z 权重/锚定、
闭合触发加学习信号、或课程学习。

---

### 背景
训练"成功率不高且很慢"：40k 步 87 分钟（~7.7 步/s，全程 GLFW 渲染），A2 训练成功率 5.1%
（`training_log_20260822_113525.json`），后续 run 退化到 1.4%（策略不稳定）。

### 根因（诊断 + 脚本实证）
1. **慢**：单环境串行采样（`num_envs=1`，16 核只用 1 核）；每决策步 20 物理子步
   （`action_repeat=2 × substeps=10`）重复执行 `velocity_ik` + `mj_fullM`；全程渲染。
2. **成功率低**：40k 步≈136 episodes 样本量远不够；6D 动作空间姿态维度无学习信号
   （`w_orient=0.0`，`diag_orientation_reward` 实证朝下/朝上/水平 r_orient 全为 0）；
   成功奖励链稀疏（对齐→闭合→3 步宽度收敛+接触才拿 grasp_reward）。
3. 验证脚本实证环境物理/动作链路全部正常（`verify_grasp_feasibility` /
   `verify_grasp_success_criterion` / `verify_ee_pose_control` 全 PASS），确认根因在训练层。

### 改动
- `config.py`：`num_envs` 1→12、`n_steps` 1024→2048、`n_epochs` 5→10、
  `total_timesteps` 40000→200000、`max_steps` 300→200、`action_space_dim` 6→3（仅位置、姿态固定朝下）、
  `w_orient` 0.0→3.0（恢复方向梯度）、`render_gui` True→False（训练 worker 无头）。
- `agent.py`：`SubprocVecEnv` 多进程并行（工厂函数在子进程内新建环境、显式 `start_method="fork"`
  避免 forkserver 的 `__main__` 依赖、worker 内包 `Monitor`）；`GraspingCallback` 重写 episode 统计
  （改用 `info['episode']` 每 episode 一次，多 env 安全）+ **主进程统一写 TrainingMonitor**
  （避免多进程写同一 JSON 冲突）+ **演示播放**（`demo_env` 定期用当前策略弹窗，不拖慢训练）。
- `environment.py`：速度模式 IK 去重（`mj_fullM` 每决策步 1 次、`velocity_ik` 每控制周期 1 次，
  目标速度决策步内不变，语义不变）；`render_gui=False` 时不创建 Renderer
  （避免 SubprocVecEnv fork 后子进程继承 GLFW/X 连接触发 XIO fatal error，同时每 worker 省内存）；
  `_get_info` 补 `singularity_count`/`episode_breakdown`（主进程汇总 episode 记录用）。
- `singularity_handler.py`：初始位形 `spread` 0.3→0.15（课程式：先学近端精确微调）。
- `train_with_monitor.py`：并行 + 演示入口（**先 fork workers 再创建 demo_env**，避免子进程继承 GLFW 连接）。
- `scripts/verify_ee_pose_control.py`：适配 `action_space_dim=3`（3D 下姿态增量测试跳过，语义固定朝下）。

### 验证
- `verify_ee_pose_control` PASS：dx 30 步 0.137m 与 IK 去重前**完全一致**（实现率不受影响）、
  零漂移 5.8e-9；`verify_grasp_success_criterion` PASS。
- smoke 并行训练（12 环境，24576 步）：fork 正常、无 XIO 错误；**吞吐 70.9 步/s（原 7.7，↑9.2 倍）**；
  TrainingMonitor 主进程记录 120 episodes；demo_env 渲染链路正常（viewer 弹出，30 步渲染 OK）。

### 遗留/注意事项
- `demo_env` 退出时可能有 `GLXBadDrawable` X Error 告警（GLFW 窗口销毁时机），不影响功能。
- 演示播放期间训练主进程暂停数秒（每 5 个 rollout 一次，约 1% 开销）。
- 3D 动作空间打通链路后，后续可再开 `action_space_dim=6`（w_orient 已恢复 3.0，姿态维度有梯度了）。

---

### 背景
用户提出三个候选陷阱：(一) Kp=45000 高增益→PD 震荡、(二) lift_speed 削足适履、
(三) 50Hz RL vs 1000Hz PD 频率错配→噪声。逐一实测裁决：

| 陷阱 | 裁决 | 依据 |
|---|---|---|
| 高 Kp→震荡 | ❌ 不成立 | ζ=Kv/(2√(Kp·I_eff))≈10~15 极度过阻尼 + forcerange=±87N·m 力矩饱和 → 单调趋近、**无振荡** |
| lift_speed 削足适履 | ❌ 架构不成立 | lift 是 grasp_success 后自动状态机，`step()` 直接覆盖 RL 动作；RL 不参与抬升 |
| 频率错配→噪声 | ⚠️ 结论修正 | 实际 50Hz RL vs **500Hz** 物理（timestep=0.002、substeps=10）；不是噪声，是**每 RL 步伺服收敛不足** |

### 实测（根因：命令-实际位移脱节）
50 步命令累计位移的**实际 EE 实现率**（reset 默认位形）：

| 命令 | 1×50Hz | 2×(action_repeat=2) |
|---|---|---|
| Δz=0.003（RL 早期小动作） | **8.1%** | **23.2%**（↑3×） |
| Δx=0.02（最大步进） | 28.2% | 30.6% |

小增量实现率仅 8% → RL 早期随机策略几乎推不动末端 → 训练 0% 成功率的物理根因。
lift 阶段每步物理时间 0.02s（1 控制周期），抬升曲线与 P4 一致（obj 100% 刚性跟随）。

### 决策/改动
- `config.py`：新增 `action_repeat: int = 2`——每 RL 决策步执行 2 个控制周期再观测/给奖励，
  不改 control_freq（物理仍 50Hz），RL 决策/观测频率 50→25Hz；
- `environment.py`：物理循环 `range(substeps * ctrl_cycles)`；**lift 阶段 ctrl_cycles=1**
  （实测 action_repeat 作用于 lift 会让每步"脉冲式"抬升、物体滑移 41% → lift_aborted 失夹）。

### 验证
`verify_grasp_success_criterion` PASS；`verify_ee_pose_control` PASS；
`smoke_lift` PASS（EE dz=0.185 / obj dz=0.1857，100% 刚性跟随，抬升期奖励恒 0，无失夹）；
truncation：`max_steps=10` 抬升完整跑完（499 步、trunc=False）；`smoke_train_p1 512` SMOKE PASS。

### 遗留发现（层2，训练层面，未在本次改动）
贪心策略（朝物体方向最大增量）在 reset 默认位形下可接近到 ~0.10~0.12m，但未完成精确
对准抓取；修正 z 目标（`obj.z+FINGER_REACH`）后奇异点=0——此前训练日志的奇异点多为
策略乱动的**症状**而非独立障碍。接近/对准阶段 RL 增量动作效率是训练层面的下一步问题。

---

## 2026-08-20 · P4 抓取成功后的抬升（lift）物理修复 + 速度确认

### 背景
`grasp_success` 后环境自动抬升 0.2m（`lift_height`），但物体跟随只有 ~0.16~0.17m
（相对 EE 滑移 ~2cm）。此前归因于"64g 立方体侧向夹持蠕动倾覆"。

### 排查结论（关键）
- 根因**不是**摩擦（法向力 84N、余量 130 倍），也不是臂身型号（换 UR5 无益）；
- **速度扫描（`lift_speed`）证明 0.02m 滑移是伺服/动态滞后，不是几何极限**：

| lift_speed | EE 升幅 | 物体升幅 | 跟随率 | 步数 |
|---|---|---|---|---|
| 0.005 | 0.1857 | 0.1740 | 93.7% | 265 |
| **0.003** | 0.1850 | 0.1851 | **100.0%** | 494（3 seed 稳定） |
| 0.002 | 0.1851 | 0.1897 | 102.5% | 861 |

### 决策/改动
- `config.py`：`lift_speed` 0.005 → **0.003**、`lift_max_steps` 350 → **700**；
- 保留此前 `panda.xml` 衬垫加硬（`solref=0.001/0.999`）+ 主衬垫加大（接触斑）；
- 物体被抬至 **0.205m**（>0.2m 目标）、100% 刚性跟随、宽度恒 0.0345；
- 尝试过但**放弃**：货叉/托片（物体贴桌无底部空间、托片穿地）、μ=2.0（更差）、
  Newton 求解器 + 400 迭代（边际、已还原）；
- `hand.xml` 历史残留（μ=2.0 等无效尝试）已还原——该文件不被任何场景引用。

### 验证
`verify_grasp_success_criterion.py` PASS（A 0.0345 / B 0.0074）；
`smoke_lift` PASS（+5/+5/+10 边沿触发一次、抬升期奖励恒 0、lift_done、无截断）；
truncation：`max_steps=10` 下抬升仍完整跑完（episode 499 步、无截断）。

---

## 2026-08-20 · P3.1 仿真卡顿修复 —— 只做方案 A（IK 最坏情况）

### 背景（讨论摘要）
用户反馈“现在仿真有点卡”。经逐环节 benchmark（本机，MUJOCO_GL=egl）定位：

| 环节 | 耗时 |
|---|---|
| 纯物理 1 控制周期（10 子步） | 0.29 ms |
| IK 正常收敛（小增量） | 2 ~ 5 ms |
| **IK 目标不可达（5000 次迭代跑满）** | **≈ 838 ms/步（卡顿主因）** |
| 每步新建 MjData scratch | 0.43 ms |
| env.step 平均（随机动作） | ≈ 3.0 ms |

根因（按影响排序）：
1. **IK 病态最坏情况（主因，P3 引入）**：`ik.solve_ik(iters=5000)` 只在位置误差
   <3e-4 且旋转误差<1e-3 时提前 break；目标不可达（超工作空间/接近奇异/姿态冲突）时
   会把 5000 次迭代全部跑完（每次 mj_forward+mj_jacBody+6×6 解方程），单步 ≈0.84s。
   未训练策略/随机动作频繁给出不可达目标 → 一阵一阵卡。
   且 action_wrapper 是在 IK 解完之后才判 `err>ik_error_hold → 保持原位`，白算。
2. GUI 渲染（render_gui=True 默认、训练脚本未关、每 5 步 sync 窗口）—— 未处理。
3. 每步新建 MjData scratch（0.43ms）—— 未处理。

### 决策
只做方案 A（修 IK 最坏情况）；B（关/降渲染）、C（复用 scratch）、D（零动作短路）、
E（训练并行化）暂不做，留待后续讨论。

### 改动内容
- `ik.py` `solve_ik`：
  - `iters` 默认 5000 → **500**（正常收敛实测只需几十次迭代）；
  - 新增**停滞提前退出**（`stall_patience=40`）：连续 40 次总误差不再下降即 break；
  - 跟踪并返回**历史最优解 best_q**（及其位置误差），避免返回末次更差的迭代解；
  - 效果：不可达单步 **838 ms → ~9 ms**（≈94×），收敛路径 2~5ms 不变。
- `scripts/verify_grasp_feasibility.py`：删除本地重复的 `solve_ik` 实现，改为
  `from ik import solve_ik`（单一来源；`verify_position_randomization.py` 从本文件
  导入 solve_ik，自动继承修复）。

### 验证结果（全绿）
- [x] py_compile 全量
- [x] benchmark：不可达 838ms → 8.9ms；env.step avg ≈2.9ms
- [x] verify_ee_pose_control.py PASS
- [x] verify_grasp_success_criterion.py PASS
- [x] verify_grasp_feasibility.py PASS
- [x] verify_position_randomization.py --n 12 PASS（12/12 可达、抽查抓取成功）
- [x] smoke_train_p1.py 2048 SMOKE PASS（PPO 管线端到端正常）
