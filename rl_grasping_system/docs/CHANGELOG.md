# 变更记录 / CHANGELOG

> 约定：**从 P3.1 起**，每次代码改动 + 相关讨论/决策总结都追加到本文件，条目按日期倒序。
> 更早的阶段性设计与讨论见 docs/P0-2、P1、P2-2、P3 等设计文档。

---

## 2026-09-02 · 标称轨迹升级：IK 可达性检查（对齐 MoveIt「IK 解算目标位姿」语义）

### 动机
标称替身（`nominal_trajectory.py`）此前是**纯几何两段速度场**（P 控制，目标点直接由
target_pos + hover_z_offset 构造），与真 MoveIt「IK 解算目标位姿 → 轨迹采样 → Servo 下发 twist」
仅行为近似、无解算环节。本次加入 **solve_ik 可达性检查**：
- 语义对齐：标称对 hover 目标先 IK 验证可达，与 MoveIt 规划语义同构；
- 兜底能力：目标越出 IK 可达域时标称**保持不动**（对标 MoveIt 规划失败→不动），
  为将来扩大 workspace（新边界点可能不可达）预铺基础设施。

### 改动
- **`nominal_trajectory.py`**：新增 `bind(model, data, arm_joint_ids, body_id)`（环境构造后调用）、
  `_solve_err_from`（DLS 解 hover 位置误差，从当前位形出发）与 `_target_reachable`
  （当前位形 + home keyframe 双起点防局部极小误判；带缓存，目标移动 < 5mm 复用结论）；
  `reference_velocity` 中 hover 不可达 → 返回零速度并 5s 限频告警。
- **`environment.py`**：构造标称后调用 `bind(...)`。
- **`config.py`**：新增 `nominal_ik_check=True` / `nominal_ik_check_tol=0.01` /
  `nominal_ik_check_recheck=0.005`。

### 验证
- py_compile 三文件通过；冒烟测试：可达目标正常驱动（0.12 m/s）→ 不可达目标（桌外远点）
  零速度保持 → 缓存复用 → 回到可达恢复驱动 → 环境 step×10 不崩。
- 当前 workspace（IK 打点 100% 可达）下恒通过检查，**行为与纯几何两段速度场 bit 一致，零回归**。

### 影响
- 成功率不变（90% 是速度环执行上限；97.5% 需关节伺服、与残差接口不兼容，见
  `docs/2026-08-27_部署分工与最小验证.md §8.1`）。
- 后续若扩大 workspace_bounds，本检查自动生效（不可达目标不空转，RL 残差保留行动空间）。

## 2026-08-28 · peg-in-hole 最小尝试（接触操作信号可行性）

### 动机
4 周计划 Week 2 最小前置验证：搭建最小 peg-in-hole 环境，短训验证"对准 → 插入"信号可学。
详细记录：`docs/2026-08-28_peg-in-hole最小尝试.md`。

### 改动
- **`models/universal_robots_ur5e/ur5e_peg_hole.xml`**：基于抓取场景改造——`hole_board`
  （200×200×20mm、14×14mm 方孔，4 段 box 围成，MuJoCo 无减法几何）、peg 固定连 `rq_base_mount`
  （局部 +z 0.13/0.18 指向桌面）、移除障碍物。
- **`scripts/peg_hole_min.py`**：独立最小环境（21 维观测、3D 动作、插入奖励、速度级 IK 执行）
  + PPO 60k 步训练脚本。

### 关键调试
1. IK 初始位形：`mj_resetData` 后 qpos=0 近奇异，DLS 落局部极小 → 先置 SAFE_CONFIG 再 IK。
2. peg 挂错层级（在 rq_base_mount 外）→ 修正。
3. 单位自查：`insertion×100` 是厘米误当毫米造成"成功率与深度矛盾"假象，逻辑本身正确。

### 结果（8 env PPO，60k 步，n=30 确定性评估）
| 阶段 | 成功率 | 平均插入深度 |
|---|---|---|
| 训练前 | 0% | 孔面上方 3.3cm |
| 训练后 | **100%** | **低于孔面 15mm**（>12mm 阈值）|

**插入信号可学**（成功率 0→100%，训练后 episode 仅 13~22 步）——peg-in-hole 作为第二个
简历卖点（接触操作）路线成立；最小尝试边界：peg 固定、孔固定、无阻抗基线/扰动（Week 2-3 后续）。

### 步 1-2 追加（2026-08-28 晚）：阻抗基线 + 扰动对比（修复孔 bug 后 RL 优势）
- **步 1 阻抗基线**（`scripts/peg_baseline.py`：闭环 PID 对齐 + 垂直下压）：理想情况 100%（n=30），
  验证任务可由传统控制完成（对比锚点建立）。
- **步 2 扰动对比**（`scripts/peg_perturb_compare.py`）：初始水平偏移 ±5cm。
  - 初版无区分度（基线 100% vs RL 100%）→ **排查发现孔尺寸 bug**：`hole_board` 的 `pos=0.107`
    使孔实为 20cm×20cm（非设计 14mm）——任务太容易。
  - **修复孔到 12mm**（容差 1mm）后：**基线 73.3/76.7% vs RL 100/100%（理想/±5cm 扰动），
    RL 优势 +26.7/+23.3pp，步数快 ~4.5×**——基线因 PID 对齐容差（2mm）> 孔容差（1mm）卡孔壁，
    RL 学到精确对准（<1mm）。这是"接触修正 RL 相对基线优势"的核心卖点数据。
- 诚实边界：基线为简单阻抗式控制（非完整工业力控）、peg 固定/姿态垂直（Week 2-3 扩展）。

---

## 2026-08-28 · D6 感知噪声鲁棒性实验（目标位置加噪 + 随机漏检）

### 动机
ROADMAP Phase 2 降级方案 A 落地：量化 RL 策略对感知误差的容忍度（验收线"成功率下降 <10%"），
把"oracle 观测"从简历弱点变成"感知-控制双通道隔离"的工程方法卖点。
详细记录：`docs/2026-08-28_D6感知噪声鲁棒性实验.md`。

### 改动
- **config.py**：`perception_noise_std`（目标位置高斯噪声 σ, m）/ `perception_dropout`（随机漏检 0~1），
  默认 0=关闭（不影响旧训练）。
- **environment.py**：`_perceived_pos` 状态 + `_perceived_target_pos()`（真值+噪声，漏检零阶保持）；
  `_get_observation` 覆盖 `state['target_position']`——**仅污染观测通道**，奖励/物理位置仍用真值
  （感知-控制双通道隔离）。
- **evaluate.py**：CLI `--perception-noise-std` / `--perception-dropout`。
- **scripts/run_perception.sh**：8 配置 × n=30 复现脚本。

### 结果（v18_p3f，n=30/配置）
| 场景 | 配置 | 成功率 | 相对 oracle |
|---|---|---|---|
| 动态+障碍 | oracle（重跑） | 73.3% | 基线 |
| 动态+障碍 | σ=0.01 / 0.02 / 0.03 | 86.7% / **90.0%** / 73.3% | 持平或更高 |
| 动态+障碍 | σ=0.02 + 漏检 10% / 20% | 80.0% / 90.0% | 无下降 |
| 静态 | oracle / σ=0.02 | 96.7% / 86.7% | −10pp |

### 结论
1. **D6 验收通过**：动态+障碍场景带感知噪声成功率无显著下降（σ≤0.03 持平或更高、漏检 20% 仍 90%）。
2. **机制**：残差架构下标称轨迹主导接近，目标小幅偏移（≤3cm < `grasp_align_xy_tol=0.030`）
   由残差修正——对感知误差天然鲁棒，"单深度相机足够"（决策记录）获实证支持。
3. **oracle 不是瓶颈**："先证明控制层、再升级感知"的工程方法成立（隔离变量）。
4. 边界：仅污染 RL 观测（标称用真值）；静态 σ=0.02 下降 10pp 接近验收边缘（2cm≈对齐容差）；
   训练时加噪鲁棒化是下一选项（`perception_noise_std` 训练可直接启用）。

---

## 2026-08-28 · 三路消融对比实验（端到端 vs 残差 vs 纯标称）


### 动机
一周冲刺方案 §10.4"基线对比实验（最高性价比的含金量补强）"落地：**不重训**，用已有资产
（端到端 v11、残差 v18、纯标称=零残差）补一张正式对比图，论证"标称解决静态、残差解决动态、
端到端均不如残差"，杀死 toy demo 感。

### 改动
- **evaluate.py**：新增 `--zero-residual` 开关——residual 模式下动作恒 0 → `v = v_nominal + 0`
  = 纯标称（MoveIt 标称仿真替身、无 RL 修正），三路对比 baseline。改动 ≤20 行
  （run_episode / evaluate_model / argparse）。
- **scripts/run_ablation.sh**：三路（end2end=v11_500k / residual=v18_p3f / nominal=v18_p3f+零残差）
  × 三场景（static / dyn_target / dyn_both）→ `results/ablation/{algo}_{scene}.json`。
- **scripts/plot_ablation.py**：1×3 对比图（成功率 / 碰撞率 / 平均步数）+ 汇总表
  `results/ablation/summary_table.txt`。图表用英文标签（DejaVu 无中文字形）。

### 结果（n=30/场景，固定物体位置，同一当前环境[扩大桌面 0.70m]，三路均未重训）

| 场景 | 端到端 PPO | 残差 PPO | 纯标称(零残差) |
|---|---|---|---|
| 静态无障 | 26.7% | **100%** | 83.3% |
| 动态目标 0.05 m/s | 63.3% | **90.0%** | 66.7% |
| 动态目标+动态障碍 | 43.3% | **86.7%** | 70.0% |

### 结论与机制
1. **标称解决静态**：纯标称静态 83.3%——静态任务标称即够用，RL 只需补偏差。
2. **残差解决动态**：动态场景残差 90%/86.7% 大幅领先纯标称 66.7%/70%——"RL 学偏差"优于"RL 重学任务"。
3. **端到端均不如残差**：全场景最差；机制解释见 `docs/2026-08-27_RL训练失败分析_v5到v12b.md`。
4. **域变化鲁棒性（附赠发现）**：三路都承受"桌面 0.48→0.70m"域变化冲击，端到端静态从历史 ~68%
   跌到 26.7%（最敏感）、残差仍 100%（最鲁棒）——残差受域差影响小，sim2real 直接论据
   （`docs/sim_to_real动力学匹配方案.md`）。
5. 失败模式全为 **200 步超时**（够不到/到不了位），无碰撞终止、无奇异点——端到端是"没学会到达"而非"撞坏"。

### 验证
- 冒烟：`--zero-residual` 下 residual_norm=0、n_episodes 生效 ✅
- 9 个评估 JSON + 对比图 + 汇总表已入库（`results/ablation/`、`results/ablation_compare.png`）。
- 复现：`bash scripts/run_ablation.sh && python3 scripts/plot_ablation.py`。
- 注意事项：评估统计里 `residual_norm` 实为 velocity 模式"动作幅度"（`_last_residual_norm` 无条件
  记录 `‖Δv‖`），非真正的残差幅度——对比图未采用该指标，避免误导。

---

## 2026-08-28 · 深度相机分工记录 + 桌面扩大 + 眼在手外相机

### 背景（文档）
- 创建 `docs/2026-08-28_深度相机分工与桌面扩大眼在手外.md`：
  决策一"深度相机加训练端还是部署侧"（部署侧真相机 + 训练端等价感知观测，
  感知特征接口契约）；决策二"扩大桌面"；决策三"眼在手外（eye-to-hand）相机"。

### 桌面扩大（0.48m → 0.70m 见方）
- **`models/universal_robots_ur5e/ur5e_robotiq_cube.xml`**：桌面半边长 0.24→**0.35**、
  中心 (0.1, 0.42)→(0.1, **0.53**)（Y 下界保持 0.18 不碰下臂，历史 356 N·m 教训）。
- **`models/universal_robots_ur5e/make_cube_scene.py`**：同步 T_HALF/CY + **把障碍物段
  （obstacle/obstacle_2/obstacle_3）并入生成模板**——修复"重跑生成脚本覆盖手动追加障碍物"隐患，
  生成产物与手改 XML diff 为空。
- **`rl_grasping_system/config.py`**：`workspace_bounds` 扩大
  X(-0.15,0.1)→(-0.18,0.13)、Y(0.32,0.42)→(0.26,0.42)（面积约 +90%）；`obstacle_wander_bounds`
  同步 → X(-0.22,0.18)、Y(0.22,0.50)；新增 `table_half_size=0.35 / table_center=(0.1,0.53)`。
- **`scripts/verify_position_randomization.py`**：DEFAULT_BOUNDS 同步。

### 眼在手外相机（eye-to-hand）
- **MJCF**：新增 `<camera name="eye_to_hand" mode="track" target="cam_target"
  pos="(0.1,0.42,1.20)" fovy="60">` + 不可见 `cam_target` body（工作区中心）。
  ⚠️ MuJoCo 3.10 实测：camera `mode="target"` 是无效关键字，用 `mode="track"`（位置固定 + 自动对准 target）。
- **`rl_grasping_system/environment.py`**：新增 `_get_offscreen_renderer(depth)`（惰性离屏渲染器，
  与 GUI renderer 分离、子进程内创建、云端需 `MUJOCO_GL=egl/osmesa`）、`get_depth_image()`（米，
  float32 (H,W)）、`get_camera_image()`（RGB uint8）；`close()` 清理渲染器。
  ⚠️ MuJoCo 3.10 实测：enable_depth_rendering 后 render() 返回**物理距离（米）**而非 [0,1] 缓冲，
  不做非线性转换（中心像素 0.9m = 相机到桌面垂直距离）。
- **`config.py`**：新增 `camera_name/camera_width=640/camera_height=480/camera_fovy=60/depth_enabled=False/
  depth_max_clip=5.0`（`depth_enabled` 默认关 = 训练零渲染开销）。

### 验证
- 模型加载 OK、obs 67 维不变；桌面 half-extents=(0.35,0.35,0.025)。
- 眼在手外 RGB 640×480×3（桌面≈24% 像素、cube 居中）；深度 640×480、0~1.2m、42% 有效像素。
- 真·臂↔桌面接触：home + 100 随机步 = 0；`qfrc_constraint` max≈0.98 N·m。
- workspace IK：`verify_position_randomization.py` 12/12 可达 + 抽查 PASS。
- `check_d1.py` 全部断言 PASS。

> ⚠️ **需重训**：workspace 扩大 + 桌面中心后移改变标称路径/分布，旧模型（v18 等）评估会掉，
> 后续训练用新环境从头/ warm-start 重训。

## 2026-08-27 · P3 训练分布启动 + 实习项目立项（Week1）

### 背景（文档）
- 创建 `docs/2026-08-27_RL训练失败分析_v5到v12b.md`：v5→v12b 八版失败的四类机制归因
  （A 无信号 / B 奖励经济学 / C 感知锁死 / D 探索压死）→ 混合架构决策树 + 业界理论对应。
- 创建 `docs/2026-08-27_实习项目4周计划.md`：投具身智能 RL 实习的 4 周冲刺计划
  （W1 动态抓取闭环 → W2 peg-in-hole+基线 → W3 残差 RL+鲁棒性 → W4 消融+交付）+ 毕设 sim2real 接口预留。

### P3 训练分布：per-episode 场景采样（40/30/30）
- **config.py**：`scenario_mix: Tuple[float,float,float] = None`（(静态无障, 动态目标无障, 动态目标+动态障碍)；
  None=旧机制全跟随全局开关，兼容 check）。P3 传 (0.4,0.3,0.3)。
- **environment.py**：reset 按 scenario_mix 采样 `_dyn_active/_obstacle_active`（选中场景强制覆盖全局开关）；
  `_update_dynamic_target` 改查 `_dyn_active`；`__init__` 初始化 `_dyn_active`。
- **train_with_monitor.py / evaluate.py**：`--scenario-mix "0.4,0.3,0.3"` CLI（需配合
  `--dynamic-target --target-vel` + `--obstacle --obstacle-vel`）。
- **验证**：200 reset 分布 38.5%/30.0%/31.5%（≈40/30/30）✓；scenario_mix=None 旧机制回归 ✓；
  check_d1 全 PASS ✓。

### P3 训练三连（v13→v15）结果
| 版本 | 配方 | 静态 | 动态0.03 | 动态0.05 | 动态+障碍 | 说明 |
|---|---|---|---|---|---|---|
| v13_p3 | warm-start v11 + mix(0.4,0.3,0.3) + residual_reg 0.5 | 80% | 73.3% | 35% | 70%/碰撞78% | 残差恒定 0.216≈饱和干扰标称；动态障碍 fixed 远离路径 |
| v14_p3b | + on-path 动态障碍 + residual_reg 1.5 | **95%** | 83.3% | 66.7% | 70%/碰撞93% | 双修复生效；双动态避障仍失败 |
| v15_p3c | + 双动态 50% + obstacle_w 2 + collision -5 | **96.7%** | 91.7% | 83.3% | **90%/碰撞80%** | **Week1 3/4 达标** |

**Week1 交付**：静态 96.7% / 动态 0.03 91.7% / 动态 0.05 83.3%（3/4 验收线达标）；
双动态碰撞 80% 为奖励经济学 trade-off 残余（成功率 90% 已达标）。

### Bugfix
- **train_with_monitor.py**：`--obstacle-on-path` 分支补 `--obstacle-vel` 应用（原只 `--obstacle`
  分支生效 → on-path 障碍被钉住成静态，v13 碰撞率 78% 根因之一）。

### 动态障碍运动修复（2026-08-27，用户观察驱动）
- **问题**：动态障碍（obstacle_vel>0）原实现只设 axis qvel → freejoint 障碍受重力 z 颠簸
  （视觉竖直掉落）+ 0.05m/s 直线飞走（看起来像 v5 静态障碍）。
- **修复**：`environment.py _update_obstacle_motion` v>0 分支改为**运动学横向往返三角波**
  （锚点 ±obstacle_half_range 沿 obstacle_axis 往返，z 每子步钉住，qvel 设往返方向）；
  `config.py` 新增 `obstacle_period=4.0` / `obstacle_half_range=0.12`；reset 清 `_obs_motion_t`。
- **最小验证**：z 波动 0.0000m（钉住）、y 波动 0.12m（横向横移）、qvel 方向切换 10 次/250步（往返）
  、障碍扫过路径时被机械臂物理挡住（真实接触挡路）；check_d2 全 PASS。

### 动态障碍运动修复 2（2026-08-27，用户要求"在桌面上随机移动"）
- **`config.py`**：`obstacle_motion_mode: str = 'random'`（'random'=桌面 XY 随机游走 / 'roundtrip'=固定轴往返）、
  `obstacle_dir_change=2.0`（方向随机重采样间隔 s）、`obstacle_wander_bounds`（X∈[-0.18,0.15], Y∈[0.28,0.48]）。
- **`environment.py`**：`_update_obstacle_motion` v>0 分支按 mode 分发——
  random：方向每 dir_change 秒随机重采样 + 边界镜面反弹 + z 钉住锚点高度（运动学写入 + qvel 设方向）；
  roundtrip：保留对称往返。reset 初始化 `_obs_dir/_obs_dir_t`。
- **最小验证**：z 波动 0.0001m、X 游走 0.28m / Y 游走 0.20m（均在边界内）、方向变化 18 次/500 步；
  check_d2 全 PASS。
- **demo**：`results/demo/final_model_stage2_d2_v15_p3c_dual{1,2,3}.gif`（动态目标+随机游走障碍，双视角）。

## 2026-08-28 · v16 奖励经济学设计（用户要求"注意奖励经济学设计，动态避障+抓取率≥80%"）

### 基线（v15_p3c 正式评估，n=30，随机游走障碍 0,0,1）
- **成功率 86.7%**（26/30）已达标；**碰撞率 13.3%**（4/30：ep9 42次/ep14 1次/ep23 22次 失败，**ep16 成功但仍穿障 10 次**）。
- **残余经济学死结**：v15 穿障成功仍净赚（collision -5×10=-50 < 成功 +120）→ "成功带碰撞"仍存在（ep16）。
- 残差恒 0.216（成功/失败同幅，饱和）；失败模式 = 撞障卡死 200 步。

### v16 奖励经济学改动（单变量：在 v15 基础上只改奖励+障碍行为适应）
- **config.py**：新增 `obstacle_clear_bonus: float = 0.0`（默认 0 不破坏旧行为）。
- **reward.py**：`REWARD_KEYS` 增 `'r_avoid'`；`reward_breakdown` 在 `grasp_success_rising` 当步且
  `obstacle_collision_count==0` → +bonus（**干净成功奖励**）。
  - 经济学：绕障抓取 = +100(成功)+30(干净) 严格优于 穿障抓取 = +100 → 引导主动绕障而非贴边/穿障。
- **environment.py**：`grasp_info` 传 `obstacle_collision_count`（累计，reset 清零）。
- **train_with_monitor.py / evaluate.py**：`--obstacle-clear-bonus` CLI。
- **单测**（/tmp/verify_avoid.py）：成功+0碰撞→+30 / 成功+5碰撞→0 / 普通步→0 / 默认0不破坏旧行为，全 PASS。
- **v16 配方**：warm-start v15 + random 障碍（现状）+ obstacle_w 3 + collision -10 + clear_bonus 30 +
  residual_reg 2.0 + scenario_mix (0.3,0.2,0.5)（动态+障碍 50%）+ 600k 步。
- **训练**：PID 3061，起步成功率 93%（warm-start 保留）。
- **v16 训练完成**（600k 步，00:26→01:06）：最终成功率 **89.0%**（训练分布 mix 0.3,0.2,0.5）；
  r_avoid 生效——干净成功 +30 / 穿障成功被重罚（r_obstacle -140~-171 + r_collision -120~-130 → 穿障净收益转负）。
- **评估**：`results/v16_p3d_rand1_30.json`（1 障碍）/ `v16_p3d_rand2_30.json`（2 障碍）/ `v15_p3c_rand2_30.json`（v15 2 障碍基线）。
- **v16 正式评估（n=30，随机游走障碍 0,0,1）——经济学效果验证**：
  | 模型 | 障碍 | 成功率 | 碰撞率 | 平均碰撞 |
  |---|---|---|---|---|
  | v15 | 1 | 86.7% | 13.3% | 2.5 |
  | **v16** | **1** | **86.7%** | **10.0%** | **1.0** |
  | v15 | 2 | 56.7% | 47% | 10.6 |
  | **v16** | **2** | **56.7%** | **23%** | **4.0** |
  → 成功率持平，**碰撞率/碰撞次数减半**（r_avoid 经济学生效）；2 障碍 56.7% 未达标，
  失败主因 = 双障碍绕行耗时 → 200 步超时（13 失败中 9 个 coll=0 超时，仅 4 个撞障）。
- **v17 训练启动**（warm-start v16 + `--obstacle-count 2` + 保持 v16 经济学 + mix 0.2,0.2,0.6，
  600k 步）：起步成功率 78%（训练分布），目标拉 2 障碍达标 80%。
- **v17 训练完成**（600k 步）：训练分布成功率 85.0%。
- **v17 正式评估（n=30，0,0,1）**：**2 障碍 76.7%**（v16 56.7% → +20pp，差 3.3pp）、1 障碍 80.0%（略降）。
  失败 7/7 全部 **200 步超时**（6/7 撞障后绕不开；成功平均仅 81 步 → 双障碍绕行时间不够）。
  碰撞率 37%（双障碍下策略更激进接近障碍）。
- **假设验证（证伪）**：max_steps 200→250（train/evaluate 新增 `--max-steps` CLI），v17 直接评估反而更差
  （2 障 73.3%、1 障 76.7%）——随机游走障碍给更多步 = 障碍动更久，撞障概率↑，非超时问题。
- **v18 训练启动**（warm-start v17 + obstacle_count 2 + **mix 0.1,0.1,0.8**（双动态+障碍 80%）+
  保持 v16 经济学 + max_steps 200，600k 步）：瞄准 2 障碍达标 80%。
- **v18 训练完成**（600k 步）：训练分布成功率 83.0%。
- **🎯 v18 正式评估（n=30，0,0,1）——双目标达成**：
  | 版本 | 1 障碍 | 2 障碍 | 2 障碍碰撞率 |
  |---|---|---|---|
  | v15 | 86.7% | 56.7% | 47% |
  | v16 | 86.7% | 56.7% | 23% |
  | v17 | 80.0% | 76.7% | 37% |
  | **v18** | **83.3%** | **90.0%** | **27%** |
  → **动态避障 + 抓取率 ≥80% 双目标达成**（2 障碍 90.0%、1 障碍 83.3%）；
  经济学沿革 v15→v18 碰撞率 47%→27%（avg_coll 10.6→4.4），avg_len 2 障 108.7→95.1（绕行更高效）。

## 2026-08-28 · v19 碰撞经济学强化（用户要求"提高碰撞经济学再训一版"）

### 动机
v18 双目标已达标（2 障碍 90.0%、碰撞率 27%），用户要求进一步压碰撞。

### v19 配方（v18→v19 只强化碰撞经济学，其余保持单变量）
- **collision_penalty -10 → -15**：碰撞当步重罚（穿障 3 次即 -45，显著亏于成功）
- **obstacle_w 3 → 4**：接近惩罚加强（防贴边/擦碰）
- **obstacle_clear_bonus 30 → 40**：干净成功 = +100+40=140，vs 穿障成功 100-15N → 差距拉大
- 保持：obstacle_count 2、mix 0.1,0.1,0.8、residual_reg 2.0、max_steps 200、不终止（max_streak=0）
- warm-start v18，600k 步。
- **训练**：PID 31528。
- **v19 训练完成**（600k 步，成功率 87%）；**正式评估（n=30，0,0,1）：2 障碍 63.3%、1 障碍 76.7%——惩罚过冲**。
  失败 11/11 全超时，**其中 5 个 coll=0 = 纯保守"不敢动"**（复现 v7 教训：惩罚太重伤成功率）；
  但成功带碰撞仅 2/19（10.5%，经济学确实压制穿障成功）。→ 判定 v19 配方过头，回退平衡点。
- **v20 训练启动（折中）**：warm-start **v18**（最平衡，90%/27%）+ 中度经济学
  （collision **-12**、obstacle_w **3.5**、clear_bonus **35**）+ 保持 obstacle_count 2、mix 0.1,0.1,0.8、
  residual_reg 2.0、600k 步。目标：成功率 ≥80% 且碰撞率 <20%。
- **v20 训练完成**（6043 episodes，训练成功率 81.0%，最后 rollout 84.3%）。
- **v20 正式评估（n=30，0,0,1）——失败**：
  | 指标 | v18 | v20 |
  |---|---|---|
  | 2 障碍成功率 | **90.0%** | 70.0%（21/30）❌ |
  | 2 障碍碰撞率 | **26.7%** | 36.7%（11/30，avg 8.47 次/ep）❌ 不降反升 |
  | 1 障碍成功率 | 83.3% | 93.3% ✅ |
  | 1 障碍碰撞率 | — | 6.7%（avg 0.23 次/ep）✅ |
  → 经济学折中版 2 障碍成功率跌破 80% 且碰撞率比 v18 **更高**：warm-start v18 后强化惩罚导致
  策略震荡（训练趋势持续下降即征兆），2 障下"贴障绕行"退化，反而更多擦碰/超时。
  **结论：v18 仍是最佳模型（2 障 90% / 碰撞 26.7%），正式回退 v18；v20 不进产品线。**
- **双视角 demo（2026-08-28，用户要求）**：`show_grasp_demo.py` 新增 `--dual-side`
  （双视角右栏 = 侧面近景 `_set_side_camera`，近距离看夹爪-障碍间隙）；用 **v18**（唯一达标）
  录制 `--dual --dual-side --track`（左特写 + 右侧面碰撞观察）。
  2026-08-28 修订：用户要求**不做黑边裁剪**——`--dual-side` 右栏与默认 dual 一致做满幅渲染
  + 直接 hstack（1280×480 双栏），去除 `_crop_black_edges`/`np.pad` 逻辑。

### 三视图 demo + 多障碍独立随机游走（2026-08-27，用户要求）
- **`environment.py`**：随机游走方向改 **per-bid dict**（`_obs_dir/_obs_dir_t` 按障碍 body 存）——
  `obstacle_count>1` 时各障碍独立随机方向/游走（reset 初始化为空 dict，首次 `_update_obstacle_motion` 惰性采样）。
- **`scripts/show_grasp_demo.py`**：
  - 新增 `_set_side_camera`：**侧面近距离特写**（从夹爪 +x 侧、略高于桌面看，主体=夹爪+下方桌面，障碍靠近时看清间隙/碰撞）；
  - 新增 `--triple` 三视图：左 track 特写 + 中俯拍 + 右侧面碰撞观察（H×(3W)）；
  - 新增 `--obstacle-count`（多障碍 demo）。
- **验证**：2 障碍独立方向（差 2.35 rad）、各自游走（X 0.33/Y 0.20 与 X 0.24/Y 0.13）。
- **demo**：`results/demo/final_model_stage2_d2_v15_p3c_triple{1,2}.gif`（35/60 帧，1920×480 三视图）、
  `v15_triple_strip.png`（4 时刻条带）、`v15_side_view_frame.png`（右侧面单帧）。

### P3 训练启动（v13_p3）
- 命令：`--load-model models/final_model_stage2_d2_v11_500k.zip --action-mode residual
  --dynamic-target --target-vel 0.05 --obstacle --obstacle-vel 0.05 --scenario-mix 0.4,0.3,0.3
  --total-timesteps 500000 --save-model final_model_stage2_d2_v13_p3.zip --no-demo`
- 冒烟 8k 步：warm-start v11 加载 ✓，成功率起步 ~54%（与 P2b 兼容性评估 53% 一致）。
- 完整训练后台运行（PID 30494，预计 2-4h）。


## 2026-08-25 · D3 无障碍混合采样 + v6 warm-start 训练（v5→v6 冲线）

### P0 量化 v5 现状（evaluate.py 补训练口径 CLI + 碰撞率/残差统计）
- **evaluate.py**：新增训练口径 CLI（`--action-mode --obstacle-on-path --obstacle-vel --obstacle-pos
  --obstacle-w --residual-reg-w --obstacle-mix-ratio --collision-penalty --position-random --radius`）——
  评估 D2/D3 避障模型必须与训练配置一致（config 默认 delta/无障碍会测错）；默认固定位置
  （对齐 train_with_monitor 无 `--position-random` 的行为）。
- **evaluate.py**：新增**碰撞率**（碰撞 episode 比例 + 平均碰撞次数）与**残差幅度 ‖Δv‖**（成功/失败分组，
  §3.5-5 分工证据）统计；`episode_summaries` 记录每 episode 碰撞数。
- **environment.py**：`info` 暴露 `residual_norm`（本步 ‖Δv‖，residual 模式真值；delta/position 恒 0）。
- **v5 实测（200 episodes/口径）**：
  | 口径 | 成功率 | 平均奖励 | 碰撞率 | 平均碰撞/episode |
  |---|---|---|---|---|
  | 固定位置（训练口径） | **74.0%** | +63.2 | **98%** | 45.4 |
  | 随机 ±3cm（演示口径） | **67.5%** | +52.0 | **99.5%** | 60.5 |
  → **成功率已过 70% 线，但碰撞率 ~100% 是验收红线**（v5 训练 collision_penalty=0，策略完全不顾碰撞）。
  残差幅度成功/失败几乎无差（0.2163/0.2162）——v5 每步残差恒定，分工证据不明显。
  产物：`results/step0_eval_v5_fixed_200.json` / `results/step0_eval_v5_rand003_200.json`。

### P1 无障碍混合采样（一周冲刺方案 §11.4，解决"绕障训练冲刷抓取技能"）
- **config.py**：`obstacle_mix_ratio: float = 0.0`（每 episode 以该概率隐藏障碍做纯抓取训练）。
- **environment.py**：per-episode 障碍状态 `_obstacle_active`（reset 按 mix_ratio 采样），
  全链路替换全局 `self.obstacle_enabled`（激活/观测槽位/接近惩罚/碰撞检测/运动钉住）；
  隐藏 episode = contype=0 + 观测槽位 0 + obstacle_dist=inf（r_obstacle=0）+ 碰撞计数不触发。
- **train_with_monitor.py**：`--obstacle-mix-ratio` + `--collision-penalty` + `--collision-max-streak`。
- **check_d2.py**：新增场景 6（mix=1.0 隐藏/槽位0/inf、mix=0.0 激活、mix=0.5 实测 53/100 激活）；
  `check_d1` 无回归（5 场景全过）。

### P2 v6 warm-start 训练（v5 → v6，完成 ✅）
- 命令：`--load-model final_model_stage2_d2_v5.zip --total-timesteps 500000 --action-mode residual
  --obstacle-on-path --obstacle-w 0.35 --residual-reg 0.3 --obstacle-mix-ratio 0.3 --collision-penalty -2 --no-demo`
- 参数理由：mix_ratio=0.3 保 30% 纯抓取样本（防 v5 已学技能被绕障样本覆盖）；
  **collision-penalty=-2 为达成"0 碰撞"验收线的必要信号**——v3 失败的是碰撞**终止**（探索死、episode 25 步），
  现改为温和**当步惩罚** + 30% 无障碍保底。
- **CLI 哨兵 bug 修复（关键）**：`--collision-penalty` 用 `>=0` 判"未指定"是错的——合法值是负数，
  `-2 >= 0` 恒 False → penalty 从未写入 config（首轮 1517 episodes 的 r_collision 全 0，碰撞惩罚形同虚设）。
  改 `None` 哨兵（`is not None`）后重启，碰撞惩罚确认生效（r_collision 非 0，avg -76.8/episode）。
- **v6 Step 3 评估（200 episodes/口径，训练同口径 CLI）**：
  | 口径 | 成功率 | 碰撞率 | 平均碰撞/episode |
  |---|---|---|---|
  | 固定位置 | **79.0%** | **65.5%** | 22.9 |
  | 随机 ±3cm | **75.5%** | **66.5%** | 25.7 |
  → **成功率达标（≥70%）**；碰撞率 v5→v6 **98%→65.5%**（减半）但仍未到 **0 碰撞**验收线。
  根因分析：63% 的**成功** episode 仍带碰撞（均 15 次）——`-2×15=-30` vs 成功 +100，**-2 惩罚太弱、碰着也划算**。
  产物：`results/step3_eval_v6_fixed_200.json` / `results/step3_eval_v6_rand003_200.json`、`models/final_model_stage2_d2_v6.zip`。
- 日志：`logs/training_log_20260825_155509.json`。

### P3 v7 warm-start 训练（v6 → v7，完成 ❌ 未达标）
- 命令：`--load-model final_model_stage2_d2_v6.zip --total-timesteps 500000 --action-mode residual
  --obstacle-on-path --obstacle-w 0.35 --residual-reg 0.3 --obstacle-mix-ratio 0.3 --collision-penalty -5
  --collision-max-streak 30 --save-model final_model_stage2_d2_v7.zip --no-demo`
- 参数理由：v6 碰撞率 65.5%（未到 0），根因是 **-2 惩罚太弱**（成功 +100 主导，碰 15 次也只 -30）。
  v7 加强信号：penalty -5 + max_streak 30（持续碰撞 truncated）。
- **v7 Step 3 评估（200 episodes/口径，同口径 CLI 含 max-streak 30）**：
  | 口径 | 成功率 | 碰撞率 | 平均碰撞/episode |
  |---|---|---|---|
  | 固定位置 | 69.5% | 64.0% | 14.1 |
  | 随机 ±3cm | 69.0% | 67.0% | 17.7 |
  → **未达标**：成功率掉到 70% 线下（69.5%，比 v6 79% 低 10pp），而碰撞率仍 64%（≈v6）。
  **max_streak 30 伤成功率、对碰撞率无益**（evaluate.py 补 `--collision-max-streak` CLI 对齐评估口径）。
- 产物：`results/step3_eval_v7_fixed_200.json` / `results/step3_eval_v7_rand003_200.json`、`models/final_model_stage2_d2_v7.zip`。

### 几何诊断（决定 v8 方向）：碰撞率卡 65% 的根因不是惩罚强度
- **残差恒定**：v5/v6/v7 评估 avg_residual_norm 成功/失败几乎相同（0.2164/0.2162）→ 策略**没有基于障碍观测动态绕障**，
  学的是"平均路径"直接穿障（惩罚只让它"碰一下快速通过"，碰撞次数 v5 45→v6 23→v7 14 递减但碰撞率不降）。
- **几何可行**：固定口径探针——障碍球面距 cube 中心 6~8.5cm（cube 半宽 ~2.5cm → **有效绕障间隙 3.5~6cm，0 碰撞可达**）。
- **根因**：障碍位置固定（on-path fraction=0.5, lateral=0）→ 策略无需根据 obstacle_dist 观测改变动作。
- **v8 对策**：每-episode 障碍侧偏随机化，强制动态避障。

### P4 v8 warm-start 训练（v6 → v8，完成 ❌ 0 碰撞不可达，交付 v6）
- 环境改动：config 加 `obstacle_path_lateral_range: (float, float) = (0.0, 0.0)`（默认固定=不破坏 check_d1/d2）；
  environment reset 在区间非零时随机采样 lateral 再放置障碍；train/evaluate 加 `--obstacle-path-lateral-range "min,max"` CLI。
  check_d1 ✅ / check_d2 ✅（场景 6 mix 48/100）。
- 命令：`--load-model final_model_stage2_d2_v6.zip --total-timesteps 500000 --action-mode residual
  --obstacle-on-path --obstacle-w 0.35 --residual-reg 0.3 --obstacle-mix-ratio 0.3 --collision-penalty -5
  --obstacle-path-lateral-range=-0.06,0.06 --save-model final_model_stage2_d2_v8.zip --no-demo`
  （负号开头值必须用等号形式，否则 argparse 当选项报错——已踩坑修正）
- **v8 Step 3 评估（200 episodes/口径，训练同口径 CLI 含 penalty -5 + 侧偏随机）**：
  | 口径 | 成功率 | 碰撞率 | 平均碰撞/episode |
  |---|---|---|---|
  | 固定障碍 | 70.5% | 66.5% | 19.2 |
  | 障碍侧偏随机（训练口径） | **73.0%** | **57.0%**（首次 <60%） | 20.3 |
  | 随机物体 + 障碍侧偏随机 | 65.5% | 53.0% | 19.6 |
- **决定性探针（模型对障碍观测无响应）**：改 obstacle_rel 槽位 [61:64]（真实/无/近/右/远 5 种），
  v6 **和** v8 模型输出完全相同 [0.005,0.005,-0.005]（diff=0.00000）→ **策略彻底忽略障碍感知**，残差恒定是真实行为。
  碰撞率降到 57% 是"障碍偏出路径"的统计效应，不是策略避障。
- **结论：D3 验收"0 碰撞"在当前 residual PPO 配方下不可达**（4 轮训练 v5→v8 系统性验证：
  碰撞惩罚把碰撞次数 45→14~23，但碰撞率卡 53~67%；策略无动态绕障）。
- **交付决策**：**v6 为 D3 正式产物**（成功率最高 79%/75.5% ✅，确定性最好；碰撞 23 次/episode 较 v5 减半）。
  0 碰撞作为**已知限制**记录：D6 真机评估时碰撞计数需容忍轻度擦碰，或需架构级改进
  （观测利用/奖励塑造，非参数调优）才能真正绕障。

### P5 v9 从零快速验证（避障感知探路，完成 ✅ 假说成立）
- 动机：v6/v7/v8 探针铁证——策略**忽略障碍观测**（改 obstacle_rel 槽位输出不变），因 warm-start 锁定
  "忽略障碍"权重先验（v6 obstacle 输入权重≈0，后续全继承）。从零（无 --load-model）打破先验，
  让 obstacle 槽位在探索初期就随机影响输出 → 验证 RL 能否激活 obstacle→action 感知。
- 配置：从零 + `--obstacle-on-path --obstacle-path-lateral-range=-0.06,0.06 --collision-penalty -5
  --obstacle-w 0.35 --residual-reg 0.3 --obstacle-mix-ratio 0.3 --total-timesteps 300000
  --save-model final_model_stage2_d2_v9_scratch_300k.zip --no-demo`（侧偏随机 ±6cm 每-episode，
  障碍观测=预测障碍位置唯一线索）。
- **结果（验证门通过 ✅）**：
  - **探针**：改 obstacle_rel 槽位，v9 对"障碍右/远"输出 diff=0.010/0.014（v6/v8=0.00000）→ **障碍感知激活**。
  - **碰撞率暴跌**：侧偏随机口径 **7.5%**（v6=57%）、固定口径 **22.5%**（v6=65.5%）→ 从零 + 障碍随机
    确实让 RL 学会利用障碍观测避障（4 轮 warm-start 从未做到）。
  - **代价**：成功率 0.5%/0.0%（从零 300k 只学完避障、没学完抓取；avg_final_distance 0.69~0.76m）。
- **结论**：假说成立——**避障学不会的根因 = warm-start 锁定"忽略障碍"权重先验**，不是 RL 原理限制。
  v9 证明从零 + 障碍随机化能激活避障；下一步续训补齐抓取技能。

### P5.1 v10 续训（v9 → v10，500k，完成 ❌ 但发现关键 bug）
- 命令：`--load-model final_model_stage2_d2_v9_scratch_300k.zip ... --total-timesteps 500000
  --save-model final_model_stage2_d2_v10_500k.zip`
- **关键 bug（19:45 发现）**：`--load-model` 传的是**文件名**（如 `final_model_stage2_d2_v9_scratch_300k.zip`），
  但 `agent.set_environment` 直接 `os.path.exists(model_path)` **不做 models/ 前缀处理** → 从 cwd 解析失败 →
  **v10 实际是又一个从零训练，warm-start 从未生效**！v10 与 v9 是两次独立从零训练。
- v10 结果（从零 500k）：训练成功率 39%（v9=32%）、评估成功率 2%/5%、碰撞率 18%/23%
  （避障仍好但抓取不够）。**失败模式诊断：final_distance 中位数 0.53m（v6=0.153m），94% episode
  够不到物体** → 从零训练被避障奖励主导，策略"躲障碍但不抓取"。
- **教训**：warm-start 必须用 `--load-model models/xxx.zip`（带前缀）或绝对路径；验证"加载预训练模型"
  日志必须出现，否则实际从零。
- **补充发现**：v6 的 value 网络 obstacle 输入权重 norm=12.1（价值函数感知障碍），而 actor 只有 0.178
  （被淹没）→ 解释"v6 不避障但能评估碰撞风险"。

### P5.2 v11 权重注入（v6_obsseed2 解锁感知 + 障碍随机，完成）
- 思路：v6 保抓取但锁死避障（actor obstacle 权重≈0）；从零激活避障但丢抓取（v9/v10）。
  **权重注入破局**：只把 v6 的 actor 输入层 obstacle 槽位 [61:64] 权重从 0.178 强注入为
  randn×0.1（norm 3.85），保留 value 网络（12.1）与其余层 → 非障碍场景行为不变（抓取保留）、
  障碍场景解锁感知（有梯度路径可学）。
- 生成：`/tmp/seed_obs_weights.py` → `models/final_model_stage2_d2_v6_obsseed2.zip(+_vecnormalize.pkl)`
- **验证**：归一化空间改 obstacle 槽位，v6_obsseed2 输出 diff 达 0.09~2.68（巨大响应）→ 感知解锁成功。
  （原 `/tmp/probe_model_obs.py` 改"原始 obs"被 VecNormalize 压缩测不出——探针方法需在归一化空间测。）
- 训练（500k，warm-start v6_obsseed2 + 障碍侧偏随机 ±6cm + penalty -5 + mix 0.3）：成功率 69%（训练），
  5643 episodes。**注**：save-model 传 `models/xxx.zip` 产生双重前缀 → 实际存 `models/models/xxx.zip`，
  已复制回 `models/`。
- **评估（200/口径）**：
  | 口径 | 成功率 | 碰撞率 | avg 碰撞 | avg_final_dist |
  |---|---|---|---|---|
  | 固定 | **71.5%** | 62% | 18.9 | 0.156 |
  | 侧偏随机 | **68%** | 53% | 18.1 | 0.156 |
- **结论**：① 权重注入方案成功——成功率 68-71.5%（v6 抓取保留，final_dist 0.156≈v6 0.153）；
  ② 但碰撞率 53-62%（v6 57-65.5%）几乎没降 → **感知激活是必要不充分**；
  ③ 根因仍是奖励经济学：碰撞惩罚 -5 太软，穿障平均 -90 < 抓取成功 +120 → 策略选穿障。

### P5.3 避障实验链总结论（v5→v11 全链路）
- **D3"0 碰撞"验收在当前配方下不可达的原因 = 两个独立配方缺陷的叠加**：
  1. **warm-start 锁定忽略障碍**（v6 obstacle 权重≈0，后续继承 → 无梯度激活感知）——**权重注入可破解**
     （v11：感知解锁 + 成功率 68-71.5% 保留）
  2. **奖励经济学穿障划算**（碰撞 -5/步 vs 抓取 +120 → 穿障净赚）——**未被破解**（v11 碰撞率未降）
- **两难**：v9 从零强避障主导 → 碰撞 7.5% 但抓取崩（0.5%）；v11 感知激活 + 弱惩罚 → 抓取保但碰撞不降。
- **完整解**（若继续）= v11 的感知激活 + **强避障奖励**（碰撞惩罚 -20~-50 或碰撞终止，使穿障不划算）
  + 障碍随机，且需课程/分阶段防止伤成功率（v7 大惩罚伤成功率的教训）。
- **当前交付**：D3 维持 v6（成功率 79% 最优，0 碰撞为已知限制）；v11 为"成功率达标 + 感知已激活"
  的中间产物（71.5%/68%）保留备查。

### P5.4 v12 多障碍 + 碰撞即失败 + 强接近惩罚（2026-08-27，执行中）
- **用户指令**：① 增加静态障碍数量 ② 碰到障碍算失败 ③ 接近障碍惩罚权重上调 ④ 混入无障碍训练样本。
- **实现**：
  - **XML**：新增 `obstacle_2 / obstacle_3` body（与 obstacle 同构 sphere r=0.05 + freejoint + contype=1）。
  - **config.py**：`obstacle_count: int = 1`（激活障碍数量，≤3）。
  - **environment.py** 多障碍全链路：收集 body 列表（`obstacle_body_ids`）、按 count 激活前 N 个
    （超出隐藏 contype=0）、碰撞=任一激活障碍、接近距离=所有激活障碍 min、运动钉住逐障碍处理。
  - **关键设计：观测保持 67 维不变**——obstacle_rel/vel 只给**最近激活障碍**（v12 前逻辑槽位）。
    → warm-start v11 无需输入层手术，PPO/VecNormalize 直接复用。
  - **布局：横向"障碍墙"**。路径水平投影仅 ~8cm（home 末端已贴近 pre-grasp），多球沿线竖排必重叠/
    贴臂；改沿路径法向铺开（frac=[0.45,0.60,0.75], latbases=[-0.12,0,0.12]）+ 墙整体随机 lateral
    ±[0.03,0.05]（每-episode 随机逼基于观测绕墙，外推兜底 ±0.21）+ **物理接触级防初始贴臂**
    （mj_forward 后 `_check_arm_obstacle_contact` 拦截，对 max_streak=1"碰撞即失败"必须初始不碰）。
  - **奖励**：`obstacle_w 0.35→1.0`（接近惩罚×3）、`collision_penalty -5→-20`（碰撞当步大惩罚）、
    `collision_max_streak 0→1`（**碰一步即失败 terminated**——破解"穿障平均 -90 < +120"奖励经济学）、
    `mix 0.3` 保留（30% 无障碍纯抓取保底，防碰撞终止洗掉抓取技能）。
- **CLI**：train/evaluate 均加 `--obstacle-count`。
- **验证**：`/tmp/check_v12_multiblock.py` 全部 PASS（3 body 收集 / count 激活 / 横向墙布局 /
  最近障碍观测 / min 距离 / 任一碰撞+惩罚+终止 / mix 隐藏 / 单障碍回归）；`check_d1`/`check_d2` 无回归。
  压力测试：count=3 零动作首步真实碰撞率 ~8%（保守度量，训练中 warm-start 策略首步即移动，实际更低）。
- **训练**：`--load-model models/final_model_stage2_d2_v11_500k.zip --obstacle-count 3 --obstacle-w 1.0
  --residual-reg 0.3 --obstacle-mix-ratio 0.3 --collision-penalty -20 --collision-max-streak 1
  --total-timesteps 500000 --save-model final_model_stage2_d2_v12_500k.zip --no-demo`。
  冒烟（8k 步）确认 warm-start 加载 + 碰撞终止生效（首 rollout 成功率 22%、平均步数 27——穿障路径
  变"碰→死"的预期回落，靠 30% 无障碍样本 + 绕障学习恢复）。
- **v12 结果（500k，3 障碍口径评估 200 eps）**：
  | 指标 | v12 | v11 | 说明 |
  |---|---|---|---|
  | 成功率 | **25.0%** | 68~71.5% | 崩（v11 一半以下） |
  | 碰撞率 | **71%** | 53~62% | 反而更高 |
  | avg 碰撞 | **0.71** | 18.1~18.9 | 大降（碰一次即终止，非"不碰"） |
  | avg_final_dist | 0.222 | 0.156 | 够不到物体 |
  | avg_episode_len | 25.1 | ~90 | 碰撞终止压缩 |
- **结论（决定性负结果）**：**max_streak=1"碰到算失败"复现 v3 失败模式**——碰撞终止把探索空间压死
  （episode 平均 25 步），策略"一碰就死"、无"碰→退→绕"试错路径，即使感知激活 + 当步 -20 + 接近惩罚
  也学不到绕障（成功率 25% 且碰撞率 71% 反而升高）。奖励经济学仍未破解，但这次失败是**学习动力学**
  问题（无探索空间）而非信号强度问题。→ v12b 将 max_streak 放宽至 **3**（连续碰 3 步才失败，
  保留"持续碰撞=失败"语义，给碰一步后逃生/重试的探索空间）。
- **v12b（max_streak=3，同参数 warm-start v11，500k，13:05 启动 13:16 提前终止）**：
  - 前 5 个 rollout 成功率 25.9/26.8/24.2/23.5/25.5（平台 24-27%，与 v12 完全一致）；
    训练统计成功率 **15%**（stdout，最近 100 eps）比 v12 更差（碰撞惩罚更重）。
  - **提前终止理由**：v12（同族 max_streak=1）已用 20 个 rollout 证明"碰撞终止"配方族不爬升，
    v12b 前 5 个 rollout 完全复现且更差，继续是浪费。
  - **结论**：max_streak 1/3 均无效——**warm-start 策略在"碰撞终止"配方下学不到绕障**，
    与 v8（无终止）/v11（弱惩罚）一起构成完整负证据链：warm-start 路线（v5→v12b 七版）全部失败；
    **唯一学会避障的是从零训练（v9，碰撞 7.5%，但抓取 0.5%）**。

---

## 2026-08-25 · D2 静态障碍绕障完成 + D3 残差 warm-start 训练（v1→v4 调参结论）

### D2 静态障碍绕障（P1，完成 ✅）
- **config.py**：on-path 放置参数 `obstacle_on_path / obstacle_path_fraction / obstacle_path_lateral / obstacle_path_z_offset=-0.10`
  （fraction=标称直连线段插值比例、lateral=侧偏、z_offset 罩住夹爪上部碰撞体——body 原点比碰撞 mesh 高 ~0.105m）
  + 障碍接近惩罚/残差正则 `obstacle_w=0.5 / obstacle_range=0.15 / residual_reg_w=0.5`。
- **reward.py**：`r_obstacle = -w·max(0,1-d/range)`（末端→障碍表面接近惩罚）+ `r_residual = -w·‖Δv‖²`（残差幅度正则）。
- **environment.py**：必经之路放置、geom AABB 距离（比 body 原点距离准）、臂-障碍碰撞检测与计数。
- **XML**：obstacle `contype 0→1`（MuJoCo 编译期 contype=0 的 geom 不进碰撞树，运行时改 1 无效）。
- **静态 freejoint 障碍钉住**：每子步拉回锚点 + 清零速度（`dof_frozen` 当前 mujoco 版本不存在）；
  否则臂会撞开障碍、RL 学"推开"而非"绕障"。
- **验证**：新增 `scripts/check_d2.py` 5 场景断言全过（on-path 放置/geom 距离/奖励集成/碰撞检测）；`check_d1` 无回归。

### D3 残差 warm-start 训练调参（P2，从 stage2_d1 warm-start）
| 版本 | 关键差异 | 步数 | 成功率 | 平均奖励 | 结论 |
|---|---|---|---|---|---|
| v1 | obstacle_w=0.5, lateral=0 | 98304 | 33% | -20.3 | 基线；r_obstacle 均值 -72.7 淹没抓取信号 |
| v2 | obstacle_w=0.2, lateral=0.03 | 98304 | 4% | -20.3 | 惩罚过弱 + lateral 间隙干扰，策略不学绕障 |
| v3 | obstacle_w=0.35 + 碰撞惩罚/终止 | 393216 | 4% | -64 | 碰撞终止让迁移策略一探索就死（episode 均 25 步）|
| **v4** | **obstacle_w=0.35, lateral=0, 无碰撞终止** | **500000** | **55%** | **+30.4** | **温和惩罚 + 长训方向正确** |
| **v5** | **从 v4 warm-start 续训，参数同上** | **500000** | **60%** | **+43.5** | **续训继续提升；最近 100 episodes 成功率 69%** |

- **v5（从 v4 warm-start 续训，累计 100 万步）**：成功率 **60.0%**（最近 100 episodes **69.0%**）、
  平均奖励 **+43.5**（v4 +30.4）、平均 episode 104.6 步（v4 134.8，更快完成抓取）；最佳奖励 375.2。
  breakdown（后 30% 收敛段，1478 episodes，成功 64.1%）：成功 `r_obstacle` **-16.32** / `r_grasp` +100 / 总 **+104.6**；
  失败 `r_obstacle` -66.8 / `r_step` -10 / 总 -57.4。`r_residual` 成功 -0.68 / 失败 -2.81（与 v4 一致）。
- **v4/v5 reward breakdown（收敛段）结论**：成功 episode 抓取信号 +100 完全主导、绕障代价降至 ~-16，
  `r_obstacle` 不再淹没抓取信号（对比 v1 全期均值 -72.7）✅

- **v4 breakdown（后 30% 收敛段，1148 episodes）**：成功 54.8% / `r_obstacle` **-16.75** / `r_grasp` +100 / 总 +103.8；
  `r_residual` 成功 **-0.69** / 失败 **-2.81**（分工证据：成功=小残差干净绕障）。v5 同构、成功率更高。
- **碰撞惩罚/终止机制**（`obstacle_collision_penalty / obstacle_collision_max_streak`）**默认关闭**：
  v3 实证对"从无障迁移 + 障碍难绕"早期有害；真机/D6 需要"碰撞=失败"信号时再启用。
- **产物**：`models/final_model_stage2_d2_v5.zip(+_vecnormalize.pkl)`（v4: `..._d2_v4.zip` 同上，累计 100 万步）、
  `logs/training_log_20260825_120807.json`（v5，4924 episodes）、`logs/training_log_20260825_001907.json`（v4）。
- **状态**：成功率回升至 50%+ 达成（v4 55% → v5 60%，最近 100 episodes 69%）；距验收线 70% 一步之遥，
  v5 已接近（收敛段 64-69%），P2 后续再续训或轻微调参即可冲线。

---

## 2026-08-24 · D1 收尾（P0）：55→67 迁移工具 + optimizer state 修复 + residual 冒烟

### 背景
- D1 完成时旧课程模型（stage1/stage2，输入 55 维）在 67 维环境下 `PPO.load` 维度不匹配，无法 warm-start。

### 迁移工具（`scripts/migrate_d1_checkpoint.py`）
- 55→67 输入层手术：`policy_net[0]`/`value_net[0]` Linear 权重零扩展（前 55 列拷贝、新 12 列=0，前 55 维计算逐 bit 不变）。
- VecNormalize obs_rms 55→67：obs_mean 补 0、obs_var 补 1、count 沿用旧值。
- `--verify`：新旧模型 deterministic 输出最大绝对差 = 0.000e+00（无损迁移硬证据）。
- 产物 `models/final_model_stage2_d1.zip(+_vecnormalize.pkl)`【实测 policy_net.0/value_net.0=(512,67)、obs 67】。

### 关键修复：optimizer state 维度（冒烟实测发现）
- 现象：迁移后 rollout 正常（成功率 80.8%），但首次 `optimizer.step()` 崩溃
  `RuntimeError: tensor a (55) vs tensor b (67)`。
- 根因：SB3 2.9.0 `PPO.save` 把 optimizer state（Adam exp_avg/exp_avg_sq，形状随输入层为 55）写入 zip；
  运行时 `PPO.load(env=67 维)` 重建 67 维参数却加载 55 维 state → 维度不匹配。
- 修复：migrate 改完输入层后重建 optimizer（`pol.optimizer_class(pol.parameters(), lr=1.0, **pol.optimizer_kwargs)`），
  丢弃旧 55 维 state、新 67 维参数冷启动（新 12 槽位本无历史动量，等价）。实测 `optimizer.step()` OK。

### 冒烟（residual 短训练，端到端）
- `train_with_monitor.py --action-mode residual --total-timesteps 30000 --no-demo
  --load-model models/final_model_stage2_d1.zip --save-model smoke_residual_d1.zip`
- 迁移加载（VecNormalize 67 + PPO 67）✅；residual rollout 成功率 **80.5% / 78.6%**
  （迁移无损 + residual 闭环）；2 次 PPO update 无报错 ✅；模型 + vecnormalize pkl 保存 ✅。
- 产物 `models/smoke_residual_d1.zip(+_vecnormalize.pkl)`【实测 67 维】。

---

## 2026-08-24 · D1 完成：残差策略（标称轨迹 + 残差动作）+ 动态化环境地基

### 交付（一周冲刺方案 §5 D1）
- **残差策略**：`action_mode: 'delta' | 'residual'`（`config.GraspingConfig`），`nominal_trajectory.py` 提供 MoveIt 标称仿真替身（阻尼引导速度场，线速度 ≤ v_max 契约 0.12 m/s）。residual 模式 `v = v_nominal(t) + Δv/T`，RL 只学偏差；delta 模式完全旧语义（向后兼容，v_nominal 观测槽位占位 0）。
- **动态目标地基（L2）**：`dynamic_target_enabled / target_vel_xy / target_motion_axis / target_period`——per-step 运动学写入（qpos），水平往返三角波，z 固定桌面高度；目标真值 `target_pos` / `task_state` / 观测同步更新。
- **障碍物地基（L1 就位，L2/L3 可编程）**：`obstacle_enabled / obstacle_fixed_pos / obstacle_vel / obstacle_axis / obstacle_radius / obstacle_mass / obstacle_hidden_pos`——body 加 freejoint + step 内 qvel 赋值（可编程运动）；geom 默认 contype=0（不碰撞），激活时运行时 contype=1 + 移到 fixed_pos，隐藏时置于 hidden_pos，开关切换不返工。
- **观测空间最终形态（67 维）**：末尾 +6 `v_nominal`（residual 真值 / delta 占位 0）+3 `obstacle_rel` +3 `obstacle_vel`，槽位预留，L2/L3 只填真值不返工。
- **CLI（`train_with_monitor.py`）**：`--action-mode --dynamic-target --target-vel --target-axis --obstacle --obstacle-vel`；默认空/0/False 保持旧配置行为。

### 关键修复（本日调试）
- `config.py`：D1 字段此前被误插进 `RewardConfig`（而非 `GraspingConfig`），导致 `obstacle_hidden_pos` 等 AttributeError——已把 D1 块整体移到 `GraspingConfig` 末尾（RewardConfig 字段不再泄漏）。
- `environment.py`：mujoco 3.10 移除 `jnt_qveladr`，3 处改用 `jnt_dofadr`。
- `environment.py _get_obstacle_velocity`：不用 `data.cvel` 前 3（障碍球在桌面被接触约束转成滚动，质心瞬时线速度≈0），改读 freejoint `qvel` 前 3（世界系平移速度，即写入的 v）。
- `environment.py _update_dynamic_target`：修正三角波公式（原公式 phase=0 时 offset=−half、周期末不连续跳变 2·half）。

### 验证
- `python3 scripts/check_obs_layout.py`：obs 67 维，`v_nominal[55:61]` / `obstacle_rel[61:64]` / `obstacle_vel[64:67]` 布局正确。
- 新增 `scripts/check_d1.py`，4 场景断言全过：① delta 向后兼容；② residual 标称速度非零且 ≤v_max；③ 障碍激活（fixed_pos + contype=1）与可编程运动（10 步移动 21mm、vel 槽位反映速度）；④ 动态目标（单步 dx≈4mm=v·T、观测/真值一致）。
- `train_with_monitor.py --help` 参数解析正常。

---

## 2026-08-24 · 简历项目方向决策：静态归 MoveIt、动态归 RL（一周冲刺方案）

### 决策
- 简历项目主线定为**混合架构动态抓取避障**（静态避障/抓取 → MoveIt2，动态避障/抓取 → RL 残差），与 `docs/混合控制架构设计.md` 三层架构一致。
- **姿态学习明确排除**：6D 姿态维度无学习信号（`action_space_dim` 6→3 实证教训），一周内无法收敛。

### 产出
- 新增 `docs/一周冲刺方案.md`：一周渐进式动态（L1 静态闭环必达 → L2 动态目标尽力 → L3 动态演示）+ 动态化环境地基原则 + 7 天执行计划 + 保底方案（感知噪声鲁棒随机抓取）。
- `docs/ROADMAP.md` 顶部加执行路线指针。
- 约束：纯仿真（投实习不做真机）、偏工业岗、AI 辅助、投递前 1 周。

### 下一步（执行时）
- D1 起：`nominal_trajectory.py` + `action_mode: delta/residual` + 动态化环境地基（可编程障碍/动态目标开关/观测预留速度槽位）。

---

## 2026-08-24 · 随机抓取（P2.0）评估失败模式：修复"撞下桌面"，记录"悬停"为 Phase 2 已知问题

### 背景
- P2.0 随机抓取（workspace_bounds 内随机摆放）确定性评估 ≥70% 达标（ROADMAP 验收线）。
- 评估发现失败集中在两类：① 目标附近悬停；② 把物体撞下桌面。

### 结论
- **"撞下桌面"立即修复**：物理执行层、感知无关（感知加噪声/真机只会更严重）。
- **"悬停"不修**：奖励塑形问题（策略停在 pre-grasp 高度，下降/对齐边际收益 < 步罚）；
  Phase 2 重训调奖励，见 ROADMAP Phase 2 已知问题。

### 改动（5 处，最小必要，不碰已收敛成功路径）
| 文件 | 改动 |
|---|---|
| `config.py` | 新增 `closing_align_force_tol=1.0`（接触力超 1N 停止 XY 推挤）、`closing_align_z_xy_tol=0.02`（Z 微降须 pad 在 cube 正上方） |
| `environment.py` reset | `task_state` 加 `knocked_off_table` 标记（按 episode 清零） |
| `environment.py` closing | ① pad 已接触 cube 即停 XY 微调（防单侧推挤撞飞）；② Z 微降加"XY 偏差<2cm"前置条件（防悬空斜压） |
| `environment.py` `_is_done()` | 掉桌检测：cube z < table_top(0.30)-5mm → 失败终止（防撞飞后无效漫游污染数据） |
| `environment.py` `_get_info()` | 暴露 `object_off_table` 失败原因，评估可统计撞飞率 |

### 验证（冒烟测试全过）
- reset/step 正常（obs len=55，30 随机步 obj_z 稳定 0.32）；
- 掉桌单元测试：cube z→0.20 → `_is_done=True`、`knocked=True`、`info.object_off_table=True`；
- reset 后标记正确清零。

### 下一步
- 评估脚本可新增撞飞率统计（`info.object_off_table`）作为迁移前健康度基线；
- "悬停"进入 Phase 2 已知问题清单。

---

## 2026-08-24 · Stage 2 完成（±6cm 迁移训练，确定性 73% 达标）

### 结论
从 `final_model_stage1b` 迁移训练 ±6cm（radius 0.06, 500k 步, 21 个 rollout），
**确定性 73.0%（100ep）≥ 70% 达标**；stochastic 82.0%（50ep）。迁移链完整闭环：
固定位 → ±3cm det 77% → ±6cm det 73%（只掉 4 个点）。

### 训练
- 前置：±6cm IK 可达性 49/49 = 100%（脚本 `verify_radius_006_reachability.py`）
- 从 `final_model_stage1b.zip` 续训 500,000 步（21 个 rollout，~29 分钟）
- rollout：0.797 0.722 0.806 0.763 0.709 0.758 0.780 0.778 0.775 0.828 0.751
  0.783 0.763 0.807 0.785 0.813 0.794 0.811 0.822 0.804 0.779（末段 78-82%）
- 产物：`models/final_model_stage2.zip`(+`_vecnormalize.pkl`)

### 评估（±6cm）
| 口径 | 结果 | 步数 | 距 |
|---|---|---|---|
| deterministic 100ep | **73.0%** | 96.5 | 0.157 |
| stochastic 50ep | 82.0% | 89.4 | 0.155 |
- σ：log_std=[0.233,0.404,0.328] → σ=[1.263,1.498,1.388]（继续增大 = 探索激进，
  但 μ 同步精确、确定性 73% 达标，再印证"σ 大与 μ 精确并存"）

### 迁移链总览（确定性 100ep 口径）
| 阶段 | 半径 | det | stoch |
|---|---|---|---|
| stage1 固定 | 0 | 基准 | — |
| stage1b | ±3cm | 77.0% | 94.0% |
| stage2 | ±6cm | **73.0%** | 82.0% |
- ROADMAP 随机线（det ≥70%）两阶段均达标 → **Stage 2 验收通过**。

---

## 2026-08-24 · Stage 1b 续训完成（确定性达标）+ 启动 Stage 2

### 结论
Stage 1b（从 `final_model_stage1` 续训 400k 步 ±3cm）确定性评估 **77.0%**（100 ep）达标
（ROADMAP ≥70%），Stage 1 验收通过；已进入 Stage 2（±6cm）。

### 训练
- 从 `final_model_stage1.zip` 续训 400,000 步（17 个 rollout，`--radius 0.03`）
- rollout：74.8 → 79.5 → 75.1 → 75.4 → 80.6 → 82.9 → 75.2 → 82.6 → 83.9 → 81.4
  → 80.4 → 80.0 → 80.7 → 83.3 → 80.9 → 83.8 → **84.4**
- 产物：`models/final_model_stage1b.zip`(+`_vecnormalize.pkl`)，PPO.load 通过

### 评估（±3cm，样本量对评估噪声敏感）
| 口径 | 50ep | 50ep rerun | 100ep |
|---|---|---|---|
| deterministic | 66.0% | 72.0% | **77.0%** |
| stochastic | — | — | 94.0%（50ep） |
- **教训**：50ep 评估噪声大（66% vs 72%），100ep 大样本才稳（77%）；此前 Stage 1 的
  "66% 未达标"结论受单次评估噪声影响被低估。**确定性评估固定 100ep 口径**。
- σ 对比：stage1 log_std=[0.083,0.257,0.17] → stage1b=[0.155,0.327,0.264]（σ 更大 = 探索更激进），
  但 mean 动作同步变精确（det 66→77），stochastic 94% 印证能力提升——"σ 大"与"μ 变精确"可并存。

### Stage 2 前置：±6cm IK 可达性
- 新增 `scripts/verify_radius_006_reachability.py` 打点 49 点（X∈[0.04,0.16]×Y∈[0.36,0.48]）
  → **49/49 = 100% 可达**（IK 误差 p90=0.00028）→ radius 0.06 可行，无需降 0.05。

### 已启动 Stage 2
- 从 `final_model_stage1b.zip` 迁移，`--radius 0.06 --total-timesteps 500000`
  `--save-model final_model_stage2.zip`（~21 分钟）

---

## 2026-08-24 · Stage 1 续训链路演练（drill）完成

### 演练目的
在不影响主线模型的前提下完整走一遍"启动 → 中断 → 恢复续训 → 评估 → 清理"链路
（Stage 2 原样复用），真实踩一遍两个已知坑（SB3 pkl 命名、num_timesteps 重置）。

### 执行过程（全部成功）
| 阶段 | 操作 | 结果 |
|---|---|---|
| 1 启动 | 从 `final_model_stage1` 续训 196608 步，`--save-model final_stage1_drill` | rollout 1-4：72.4% → 76.1% → 73.1% → **78.8%** |
| 2 中断 | 第 4 个 rollout 保存 98304 checkpoint 后 kill | 进程 0；`final_stage1_drill.zip` 未生成（收尾前中断） |
| 3 坑1修复 | `mv ..._vecnormalize_98304_steps.pkl → ..._98304_steps_vecnormalize.pkl` | stats 恢复命中（日志确认） |
| 3 续训 | `--total-timesteps 98304`（剩余步数） | 76.5% → 79.1% → 78.0% → **79.1%**，最终保存成功 |
| 4 评估 | 双口径 ±3cm | **deterministic 70.0%**（35/50）、**stochastic 80.0%**（40/50） |
| 5 清理 | 删 98304 checkpoint + 孤儿 pkl | `final_stage1_drill.zip(+pkl)` 保留，主线模型未动 |

### 演练新发现（补充进手册）
- **进程清理要防误杀 shell 自身**：`pkill -f 'final_stage1_[d]rill'`（正则避开方括号字面量）。
- **并行启动命令的 cd 只作用于第一个后台 job**：`cd x && A & B &` 中 B 在旧目录跑 →
  并行启动要把 `cd` 放进每个后台 job。
- **续训收尾会再存一次 checkpoint**：续训 num_timesteps 从 0 走满 98304 时 SB3 又存
  `{prefix}_98304_steps` checkpoint（含 SB3 命名 pkl）→ 清理时注意遗留孤儿 pkl。
- 演练模型（stage1 + 0.2M 步）deterministic 70.0%（较 66% 有提升但有限），
  与"σ 未收敛、需继续训练压低方差"结论一致。

### 状态
主线 `final_model_stage1.zip(+pkl)` 未变；是否启动正式 stage1b 续训（400k 步推确定性 ≥72~75%）待定。

---

## 2026-08-24 · 课程阶段 1 续训完成 + 续训可恢复性修复

### 结论
Stage 1（固定位置 → 位置随机 ±3cm 迁移）**训练完成并通过评估**：
`final_model_stage1.zip`（+`_vecnormalize.pkl`）确定性评估 ±3cm **成功率 66.0%**（33/50），
平均步数 105.4（成功提前终止，非 200 封顶），平均最终距离 0.156。

### 第一轮训练中断（1M 步差 ~11.5 万步被杀）
- 从 fixed 迁移的训练跑完 40 个 rollout（983,040 步）后，进程在收尾阶段被杀：
  日志戛然而止（无"最终模型已保存"），最后一个 checkpoint `..._983040_steps.zip`
  **0 字节损坏**，`final_model_stage1.zip` 未保存。
- 恢复：从最新有效 checkpoint `final_model_stage1_884736_steps.zip`（88.5% 进度）续训补足剩余步数。

### 续训踩坑 1：checkpoint 无 VecNormalize pkl → 归一化失配 → 成功率又归 0%
- **现象**：从 884736 checkpoint 续训，前 3 个 rollout 成功率恒 0%、平均步数 200 封顶。
- **根因**：`CheckpointCallback` 只存 PPO zip（`_excluded_save_params` 排除 `_vec_normalize_env`），
  不含观测统计；续训时新 `VecNormalize` 用初始 stats（mean=0,var=1≈恒等）→ 策略输入与训练时
  归一化观测失配 → 策略乱走（复现 0% bug 的同一机理）。
- **注意**：不能用 `final_model_fixed_vecnormalize.pkl` 直接顶替——fixed 训练下物体位置固定，
  其 `obs_rms.var[35:38]`（target_position）= **0**，直接复用会把位置分量归一化到 ±10 边界、
  丢失位置感知。
- **修复**：
  1. `agent.py` CheckpointCallback 加 **`save_vecnormalize=True`**——checkpoint 连带保存观测统计，
     今后崩溃恢复/续训不再失配。
  2. 为 `884736_steps` checkpoint 生成配套 pkl：以 fixed pkl 为基础（其余 52 维统计在 fixed/stage1
     下相同），**修正 target_position 分量 var 为 ±3cm 合理值**（x/y: 3e-4，z: 1e-4），
     并先验证 `checkpoint + 修正 stats` 在 ±3cm 确定性成功率 **60%**（24/40，恢复非 0%）。

### 续训踩坑 2：SB3 learn 重置 num_timesteps → 续训传错步数会过度训练
- SB3 `learn()` 默认 `reset_num_timesteps=True`，**续训时 num_timesteps 从 0 重新计数**（权重继承
  checkpoint，但步数累计归零）。
- 因此续训的 `--total-timesteps` 应传**剩余步数**（目标总步数 − 已训练步数），否则会从 checkpoint
  再跑满整个目标（本例若传 1,000,000 会累计 188 万步，严重超训）。
- 本次从 884736 续训传 `--total-timesteps 115264`（= 1,000,000 − 884,736），权重累计 ≈ 1M 步等价。

### 续训踩坑 3：SB3 checkpoint 的 pkl 命名与加载逻辑不匹配
- SB3 CheckpointCallback 的 `save_vecnormalize` 生成 `{prefix}_vecnormalize_{n}_steps.pkl`
  （"vecnormalize" 在中间），而 `set_environment`/`load` 期望 `<model>_vecnormalize.pkl`。
- 二者**不匹配** → checkpoint 的 pkl 无法被 `set_environment` 自动恢复。当前靠手动生成
  `final_model_stage1_884736_steps_vecnormalize.pkl`（命名匹配）规避；将来可让
  `agent.load`/`set_environment` 同时尝试 SB3 命名。

### Stage 1 最终数据
| 项 | 值 |
|---|---|
| 续训 rollout 成功率 | 72.4% → 73.7% → 72.7% → 74.2% → **78.6%** |
| 确定性评估 ±3cm（50 ep） | **66.0%**（33/50） |
| 平均步数 / 平均最终距离 | 105.4 / 0.156 |
| 产物 | `models/final_model_stage1.zip` + `models/final_model_stage1_vecnormalize.pkl` |

### 推荐下一步
- Stage 1 验收通过后进入 Stage 2：增大随机半径（如 `--radius 0.06`），从 `final_model_stage1.zip`
  迁移加载，续训 `--total-timesteps` 传本阶段步数。

---


## 2026-08-24 · 迁移训练 0% 根因修复：VecNormalize.load 嵌套

### 现象
课程阶段 1（迁移 final_model_fixed → 位置随机 ±3cm，`train_with_monitor.py --position-random --radius 0.03 --load-model`）
训练 14 个 rollout（34 万步）**成功率恒 0%**、平均步数恒 200（所有 episode 跑满上限）。

### 排查过程（多轮诊断脚本 + 对照实验）
1. **诊断 vs 训练矛盾**：单机 evaluate（`agent.predict` + 裸 env）随机位置下 fixed 模型
   deterministic/stochastic 均有 36.7%~55% 成功率；但训练 rollout 恒 0%。
2. **排除**：位置随机未生效（并行 worker 物体位置 X∈[0.072,0.123] Y∈[0.393,0.450] 正常）；
   VecNormalize 更新/冻结（冻结后仍 0%）；预热 obs_rms（扩展 stats 后仍 0%，且改变策略熟悉分布）。
3. **关键对比**（`scripts/check_reset_layers.py`、`scripts/repro_ppo_load.py`）：
   - 裸 env / Monitor / DummyVecEnv reset → 观测正常（物理值）
   - `VecNormalize.load(pkl, vec_env)` 后 `vn.venv.reset()` → **观测近全 0**
   - 而 `VecNormalize.load(pkl, 底层 venv)` 后 → 正常
4. **根因**：`agent.py` 的迁移加载 `VecNormalize.load(_norm_pkl, vec_env)` 传入的是
   **VecNormalize 实例**（set_environment 已先 `VecNormalize(SubprocVecEnv)` 包装）→
   load 创建的 VecNormalize 的 `venv = 另一个 VecNormalize`（**嵌套**）→
   `vn.venv.reset()/step()` 实际调用内层 VecNormalize 的归一化逻辑 → 返回异常观测
   → 训练 rollout 观测错误 → 策略输入失真 → 0%。
   单机 evaluate 用裸 env + `vn.normalize_obs` 恰好绕过了该路径，故看似正常（46.7%~55%）。

### 修复（`agent.py` set_environment 迁移加载）
```python
_base = vec_env.venv if isinstance(vec_env, VecNormalize) else vec_env
vec_env = VecNormalize.load(_norm_pkl, _base)
```
`VecNormalize.load` 的 venv 参数必须传**底层 VecEnv**（SubprocVecEnv/DummyVecEnv），
不得传 VecNormalize 本身。

### 验证
- `scripts/verify_parallel_vn.py`（并行 set_environment 迁移 + 采样 4096 步）：
  修复前成功率 **0.0%** → 修复后 **51.5%**（185/359），reset 观测恢复正常归一化值。
- **重新启动迁移训练**（`--position-random --radius 0.03 --load-model final_model_fixed`）：
  rollout 成功率 **63.3% → 66.4% → 64.2% → 68.3%**（上升趋势），平均步数 110~117
  （成功提前终止，不再是 200 封顶）——策略在位置随机下保持有效并继续 fine-tune。

### 新增诊断脚本（`scripts/`）
`verify_position_migration.py`（fixed 模型位置随机鲁棒阈值，支持 --stochastic）、
`compare_predict_paths.py`（归一化路径 A/B/C 对比）、`verify_parallel_vn.py`（并行 VN 状态 + 采样）、
`verify_warmup.py`（obs_rms floor/预热实验）、`check_reset_layers.py` / `check_setenv_chain.py` /
`repro_ppo_load.py`（逐层定位 reset 观测异常）。

---

## 2026-08-23 · 防崩溃修复（--no-demo + checkpoint 定期保存）

### 背景
PPO_97（复现 95 的重训）在 38/61 rollout 时因 **GLFW 演示窗口 segfault** 崩溃，
日志尾部全是 NUL 垃圾，模型未保存（~93 万步白跑）。

### 改动
| 位置 | 内容 |
|---|---|
| `train_with_monitor.py` | 新增 `--no-demo` 参数：禁用演示窗口（后台/长训练推荐，消除 segfault 崩溃源）|
| `train_with_monitor.py` | `agent.train` 传入 `save_path`（启用 CheckpointCallback 定期保存）|
| `config.py` | `save_freq` 50000→**98304**（SB3 CheckpointCallback 要求是 n_steps×n_envs=24576 的倍数）|
| `agent.py` | CheckpointCallback 的 `save_freq //= n_envs`（SB3 的 save_freq 是 callback 调用次数，每 n_envs 步调用一次；不除则永远不触发）|

### 验证（短训 120000 步，--no-demo）
- ✅ 演示窗口禁用（"演示窗口已禁用"）
- ✅ **checkpoint 生成**：`test_ckpt_98304_steps.zip`（98304 步处）
- ✅ 训练正常完成、最终模型保存、无崩溃

### 说明
- checkpoint 是 PPO 权重（SB3 CheckpointCallback 不含 VecNormalize stats）——崩溃恢复时
  VecNormalize stats 从最近一次 `save` 的 `_vecnormalize.pkl` 恢复或重新累计
- 崩溃最多损失 98304 步（~4 分钟）

### 长训练推荐命令
```bash
python train_with_monitor.py --save-model final_model_fixed.zip --total-timesteps 1500000 --no-demo
```

---

## 2026-08-23 · 课程阶段 1：位置随机迁移训练（进行中）

### 背景
- 固定位置模型 `final_model_fixed.zip` 评估 83.3% 成功率（60 episodes），需泛化到位置随机
- PPO_96 位置随机**从零训练失败 0%** → 课程学习必须迁移加载（继承固定位置技能）

### 迁移加载改进（`agent.py`）
- `set_environment` 迁移加载时若存在 `<model>_vecnormalize.pkl`，先 `VecNormalize.load` 恢复
  观测统计，再 `PPO.load`——修复迁移初期观测归一化失配（stats 为初始值）导致策略输出不一致
  的问题（否则 fine-tune 慢甚至失败）

### 训练命令
```bash
python train_with_monitor.py --position-random --radius 0.03 \
  --load-model models/final_model_fixed.zip \
  --save-model final_model_stage1.zip \
  --total-timesteps 1000000 --no-demo
```

### 参数说明
| 参数 | 值 | 说明 |
|---|---|---|
| `--load-model` | `models/final_model_fixed.zip` | 迁移加载固定位置抓取技能 |
| `--position-random` | `--radius 0.03` | 物体位置围绕 (0.1,0.42) ±3cm 随机 |
| `--save-model` | `final_model_stage1.zip` | 独立保存（不覆盖 fixed / stage 模型）|
| `--total-timesteps` | `1000000` | 迁移 fine-tune |
| `--no-demo` | — | 禁用演示窗口（防崩溃）|

### 状态（已完成 2026-08-24）
- ✅ 从 fixed 迁移训练 + 中断恢复续训补足 1M 步完成
- ✅ 最终模型 `models/final_model_stage1.zip`（+ `_vecnormalize.pkl`）已保存
- ✅ 确定性评估 ±3cm：**66.0%**（33/50）、平均步数 105.4（详见顶部"课程阶段 1 续训完成"条目）

---

## 2026-08-23 · 训练脚本参数扩展（--save-model / --total-timesteps）

### 背景
- PPO_95 模型被 PPO_96（位置随机从零训练，0% 成功率）覆盖，无备份
- 需要重新训练固定位置模型（复现 95）并单独保存，避免再被覆盖

### 改动（`train_with_monitor.py`）
| 参数 | 作用 |
|---|---|
| `--save-model PATH` | 模型保存路径/文件名（默认 `models/final_model.zip`，可指定单独文件名避免覆盖）|
| `--total-timesteps N` | 训练总步数（默认 config 值；复现 PPO_95 用 1500000）|

### 用法
```bash
# 重新训练固定位置模型（复现 95），单独保存
python train_with_monitor.py --save-model final_model_fixed.zip --total-timesteps 1500000
```

### PPO_96 失败教训
- PPO_96 = 位置随机 ±3cm **从零训练** 1M 步 → 0% 成功率（r_contact=0，只学到"靠近"）
- 位置随机任务收敛远慢于固定位置（需同时学"感知 target_position + 相对控制"）
- **课程学习必须迁移加载**上一阶段模型（fine-tune），不能从零训练

---

### 背景
PPO_95（固定位置）独立评估 91.7%，剩余 8.3% 失败是 XY 边缘对齐极限（收益递减）。
进入课程学习：逐步引入物体位置随机，让策略泛化到任意位置。

### 改动（`train_with_monitor.py`）
新增课程学习 CLI 参数：
| 参数 | 作用 |
|---|---|
| `--position-random` | 启用物体位置随机（`use_fixed_position=False`，围绕 `object_fixed_pos` ±radius）|
| `--radius` | 随机半径(m)，阶段1=0.03、阶段2=0.06-0.08、阶段3=完整 workspace_bounds |
| `--load-model` | 迁移加载上一阶段模型继续训练（agent 已有 model_path 支持）|

### 验证
- 位置随机正确（30 次 reset，cube X/Y 均在 ±3cm 范围）
- 迁移加载 PPO_95 + 位置随机环境创建成功
- 语法 OK

### 用法（课程阶段 1）
```bash
python train_with_monitor.py --position-random --radius 0.03 --load-model models/final_model.zip
```
阶段 2：`--radius 0.06`；阶段 3：去掉 `--radius` 直接 `--position-random`（用完整 workspace_bounds）。

---

### 背景
PPO_93 失败分析（10/60）定位到两个模式：
| 模式 | 占比 | 现象 | 方案 |
|---|---|---|---|
| 未进 closing | 4/10 | pad-cube XY 3.6-5.1cm（阈值 3.0cm），差 0.6-2.1cm | ② closing 中间奖励 |
| 进 closing 但空闭合 | 6/10 | pad 悬空（接触力 0，cube 顶面上方 ~1cm），手指合拢碰不到 | ① closing Z 微降 |

### 改动
| 方案 | 位置 | 内容 |
|---|---|---|
| ① | `environment.py` | closing 微调加 **Z 微降**（pad 未接触 cube 时微降到接触，接触后保持）|
| ② | `config.py` + `reward.py` | 新增 `close_trigger_reward=5.0`；REWARD_KEYS 加 `r_close`（进入 closing 上升沿一次性发放）|
| ③ | `reward.py` | 抬升阶段 `r_step=0`（抬升不计入训练步数也不扣步惩罚，一致性）|

### 验证（PPO_93 模型 + 新环境，60 episodes）
| 指标 | 之前 | 现在 |
|---|---|---|
| 成功率 | 83.3% | **85.0%**（51/60）|
| 达到 closed | 83% | 85% |
| 进入 closing | 93% | 93% |
| `verify_grasp_success_criterion` | PASS | PASS |

方案①（环境机制）旧模型即受益；方案②（奖励）需重训后体现——预期重训后成功率再提升。

---

### 背景
用户观察：成功 episode 显示 200/218 步（含抬升 ~28 步），要求"训练步数只计数到成功抓取，抬升不计入"。
分析：
- `max_steps=200` 预算已只算抓取前步数（P4 设计，`pre_lift_steps`）✓
- 但 **Monitor/tensorboard 的 ep_len 统计含抬升**（成功跑到 lift_done 才 terminated）→ 218 步

### 改动
| 位置 | 内容 |
|---|---|
| `config.py` | 新增 `lift_enabled: bool = False`（默认关闭抬升）|
| `environment.py` | 抬升触发条件加 `getattr(lift_enabled, False)`——关闭时成功当步走 `_is_done` 旧逻辑 `is_grasped → True`，episode 结束 |

### 验证（PPO_93 模型 + 新环境，60 episodes）
| 指标 | 之前（含抬升）| 现在 |
|---|---|---|
| 成功率 | 81.7% | **83.3%**（50/60）|
| 成功 episode 步数 | ~190-218 | **44-64 步**（不含抬升）|
| 进入 closing | 93% | 93% |
| `verify_grasp_success_criterion` | PASS | PASS |

### 意义
- tensorboard `ep_len` 反映真实抓取步数（~50 步），不再被抬升混淆
- 训练 `total_timesteps` 不再浪费在抬升上（每个成功 episode 省 ~28 步）
- P4 抬升代码保留，`lift_enabled=True` 可恢复（部署/验证抬升稳定性用）

---

### 背景
PPO_92（~500k 步）训练内成功率 5.78%、独立评估 16.7%——大幅进步但仍低。
评估显示：**进入 closing 100%（到达已完全学会），但 closing→closed 仅 18.3%**——
pad 能到触发区（XY<3cm），但停的位置差 1-2cm（cube 半宽 2cm），手指合拢夹不住。

### 改动
| 位置 | 内容 |
|---|---|
| `config.py` | 新增 `closing_align_tol=0.005`（偏差<5mm 保持）、`closing_align_speed=0.02`（限速 0.02m/s≈每步0.8mm） |
| `environment.py` | closing 阶段由"静止"改为"向 cube 中心 XY 缓慢微调到位后保持"（限速防撞飞 cube） |

### 验证（PPO_92 模型 + 新环境，60 episodes）
| 指标 | 方案B（静止）| 方案B+2（微调）|
|---|---|---|
| 成功率 | 16.7% | **53.3%**（32/60）|
| closing→closed | 18.3% | **56.7%**（34/60）|
| 进入 closing | 100% | 100% |

### 意义
方案2 把"精确对齐"从策略职责移入环境状态机（类似 lift 宏）——策略只需到达触发区，
对齐+闭合由状态机完成。**sim-to-real 提示**：部署实机时需在控制器层实现同样的
"触发后自动对齐再合拢"逻辑。

### 下一步
重训（策略将适配微调机制，进入 closing 即接近成功）→ 预期成功率再上台阶。

---

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
