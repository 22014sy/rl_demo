"""
强化学习抓取系统配置文件
"""

import os
from dataclasses import dataclass
from typing import List, Tuple

@dataclass
class GraspingConfig:
    """抓取任务配置"""
    # 环境配置
    xml_path: str = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "models/universal_robots_ur5e/ur5e_robotiq_cube.xml")
    max_steps: int = 200  # 并行改造(2026-08-22): 300→200。初始位形本就悬于 cube 上方附近，原 300 步大量浪费在失败漫游；初始 spread 同步收窄到 0.15 后 200 步足够走完行程
    control_freq: int = 50  # 控制频率 (Hz)，P1: 现在真正生效——每个环境动作执行 1/control_freq 秒的物理仿真（子步进）
    # P5 (2026-08-20): action_repeat——每个 RL 决策步连续执行 N 个控制周期再观测/给奖励。
    # 实测：单步命令小增量(Δ~0.003m)在 1×50Hz 下实现率仅 ~8%（伺服过阻尼+力矩饱和追不完），
    # RL 早期随机小动作几乎推不动末端（训练成功率 0% 的根因之一）；
    # 2× 后每决策步物理时间 0.04s，小增量实现率 ~41%（↑5×），大步进不受影响。
    # control_freq 语义不变（物理控制周期仍 50Hz），仅 RL 决策/观测频率降为 25Hz。
    # Task3 (UR5e+2F-85 速度控制)：执行层改为"速度 PID + 重力补偿"（motor+PID），
    #   决策周期 T=action_repeat×0.02s；实测 Δp=0.02/T=0.04 脉冲实现率 96%（远超市面位置伺服的 8-41%）。
    action_repeat: int = 2
    # Task3: 控制模式。'velocity' = 末端速度增量经速度级 IK(雅可比阻尼伪逆) 写关节速度，
    #   环境内速度 PID+重力前馈执行（根治位置伺服滞后，契约 §4）；'position' = 旧位置伺服模式（对照用）。
    control_mode: str = "velocity"
    # Task3: 速度环 PID 增益（N·m·s/rad，按关节标定：肩/肘 150，腕 100；实测 v=0.75m/s 实现率~90%）
    velocity_kp: List[float] = (200.0, 200.0, 200.0, 100.0, 100.0, 60.0)
    # Task3 抖动抑制 A：速度环粘性阻尼（D 项等效，τ −= kd·qvel），抑制静止/扰动微颤。
    # 注意：加速度差分阻尼（kd·(qvel−prev)/dt）在 dt=0.002 下放大数值噪声反而更抖，弃用。
    velocity_kd: List[float] = (5.0, 5.0, 5.0, 4.0, 4.0, 6.0)
    # Task3 抖动抑制 B：RL 动作一阶低通平滑（0=不平滑、1=全跟随新动作；0.3 消除决策步跳变冲击，对应 MoveIt Servo smoothing）
    action_smoothing_alpha: float = 0.7
    render_gui: bool = True  # 并行改造(2026-08-22): 训练 worker 一律无头（多进程下每进程开窗会拖垮训练）。想看训练画面：训练脚本主进程单独建演示环境 render_gui=True，定期用当前策略播放
    render_interval: int = 1  # 每隔多少步显示一次画面
    render_pause_sec: float = 0.001  # 刷新窗口时的短暂暂停
    
    # --- P3 动作空间：末端位姿增量控制（RL 只控制末端，不再直接控制关节）---
    # P3 重构：动作空间由"7 关节增量 + 1 肌腱"（P1）改为"末端位姿增量"。
    #   action_space_dim = 6 -> [dx, dy, dz, dax, day, daz]
    #       位置增量 (m) + 姿态增量 (rad，旋转向量，叠加到当前朝向)
    #   action_space_dim = 3 -> [dx, dy, dz]
    #       仅位置，姿态固定为"朝下接近"抓取姿态（退化/对照方案）
    # 增量经 DLS IK（ik.solve_ik）反解为目标关节位形，再写 ctrl = 位置伺服目标。
    action_space_dim: int = 3  # 并行改造(2026-08-22): 6→3（仅位置，姿态固定朝下）。实证(2026-08-22 diag_orientation_reward) w_orient=0.0 时 6D 的姿态维度无学习信号，纯熵探索浪费样本；先降维打通"到达→对齐→闭合"链路
    max_ee_delta: float = 0.005   # 每决策步位置增量上限 (m)；v_max=Δp/T=0.125m/s（速度环可靠区，训练/部署两端一致；见 docs/sim_to_real动力学匹配方案.md）
    # v25 残差预算与 MPC 可行域解耦（2026-09-08，见 docs/2026-09-08_残差饱和分析与奖励重塑方案.md）：
    #   residual 模式下残差分支每决策步位置增量上限 m——v_max_res=cap/T=0.05m/s = MPC v_max 的 40%。
    #   残差只能是"微调"（0.5s 内 ~2.5cm 修正量），不能"覆盖"MPC 刚算出的安全速度方向；
    #   delta 模式不受影响（仍用 max_ee_delta）。v24 恒 0.2165 的饱和根因即 clip 上界与此处同级。
    residual_delta_cap: float = 0.002
    max_orient_delta: float = 0.05  # 每步姿态增量上限 (rad，旋转向量模长上限；角速度 1.25rad/s)
    ik_error_hold: float = 0.02     # IK 位置误差超过该值(m)时保持当前位置（不朝不可达目标乱动）
    # Task3: 速度级 IK 阻尼系数（雅可比阻尼伪逆 dq=(JᵀJ+λ²I)⁻¹Jᵀv；越大越稳但末端速度实现率越低）
    # B+D 方案（见 docs/奇异点处理方案讨论记录.md）：0.05→0.08，奇异邻域数值更稳（类实机奇异降速）
    velocity_ik_lam: float = 0.08
    # Task3 B 方案：奇异 = 失败信号（而非原地卡到超时）
    singularity_penalty: float = -0.5     # 每处于奇异状态一步的负奖励（shaping 惩罚）
    singularity_max_streak: int = 50      # 连续奇异步数超阈值 -> truncated（对应实机"持续不可达 -> 任务失败"）

    # --- Task3 (UR5e + Robotiq 2F-85) 模型绑定 ---
    # 臂关节：UR5e 六关节（_find_components 须排除 2F-85 的 8 个 rq_*_joint 铰链）
    arm_joint_names: List[str] = ("shoulder_pan_joint", "shoulder_lift_joint", "elbow_joint",
                                  "wrist_1_joint", "wrist_2_joint", "wrist_3_joint")
    end_effector_body: str = "rq_base_mount"   # 末端执行器 body（挂 attachment_site 的根）
    gripper_actuator_idx: int = 6              # fingers_actuator 在 ctrl 数组的索引（0-5 为臂速度）
    pad_site_left: str = "rq_pad_left_site"    # 左右衬垫测量 site（抓取宽度 = 两 site 距离）
    pad_site_right: str = "rq_pad_right_site"
    gripper_max_width: float = 0.093           # 2F-85 全开时 pad site 间距 (m)（实测 0.0936）

    # --- P3 抓取：夹爪由"近距自动闭合状态机"驱动（RL 不再输出肌腱命令）---
    # 状态机：open -> closing -> closed（失败/空闭合/滑脱 -> 重开重试，带滞回，见 environment）。
    # 抓取成功 = "手指被物体挡住合不上"：闭合后手指开度仍 > grasp_min_width（物体卡在
    # 两指间），并辅以左右双指真实接触确认。空闭合时开度收敛到 ≈0 -> 失败重开。
    gripper_close_distance: float = 0.18  # [退役] 不再用于闭合触发（闭合时机改对齐判定，见 grasp_align_*），保留兼容
    #   (2F-85 base_mount 在 pinch 上方 ~0.145m，比 Panda hand 长；Panda 0.12 -> 0.18)
    gripper_open_distance: float = 0.22   # 滞回：失败重开后须离开物体到该距离(m)外才允许再次闭合（防抖振）
    # Task3 闭合时机（对齐触发）：物体中心须与夹爪中心(左右 pad 中点)对齐才闭合——
    # 物体真正落在 pad 之间才夹（比"距离触发"更严格、物理合理，防侧方误触发）
    grasp_align_xy_tol: float = 0.030     # XY 对齐阈值。方案A(2026-08-22): 0.045→0.030——0.045 时 pad 中心距 cube 中心
                                           # 4.5cm 即 pad 在 cube 表面外 2.5cm（cube 半宽 2cm），手指合拢夹不到；
                                           # 收紧到 3cm 保证 pad 几乎正对 cube 中心（手指能包住物体）。
    grasp_align_z_tol: float = 0.032      # Z 对齐阈值（受 home 假成功约束：pad 距 cube 0.034，须 <0.034）
    # 方案2(2026-08-22): closing 阶段 pad 向 cube 中心 XY 缓慢微调（解决 PPO_92 卡点：
    # 进入 closing 100% 但 closing→closed 仅 18.3%——pad 停的位置差 1-2cm 手指夹不住）
    closing_align_tol: float = 0.005      # 偏差 < 该值(m) 即保持不动（5mm 内手指已能夹住）
    closing_align_speed: float = 0.08     # closing 微调限速 (m/s)。2026-08-27 选项①：0.02→0.08（每决策步 ~3.2mm）。
                                          # 纯标称（residual+零残差）静态抓取 42%→70%——closing 阶段 pad 快速对正到 cube
                                          # 中心再合拢（手指合拢 ~10 步内 pad 移动 32mm，3cm 宽触发下也能夹住 4cm cube），
                                          # 见 docs/2026-08-27_部署分工与最小验证.md §7.3。v11 模型无障碍评估 83.3%（不破坏 RL）。
                                          # 原 0.02：手指合拢太快、pad 微调来不及对正 → 空闭合重开（PPO_92 卡点同因）。
    # Task3 冲击抑制（见 docs/抓取冲击抑制方案.md）：
    approach_speed_limit: float = 0.12    # pad 距 cube < approach_speed_dist 时末端线速度限幅 (m/s)（对应实机 UR 慢速 Servo 接近）
    approach_speed_dist: float = 0.06     # 接近限速触发距离（pad 中心→cube；仅抓取最后一段限速，避免 home 全程触发）
    grasp_force_limit: float = 20.0       # closing/closed 阶段 pad-cube 接触力上限 (N)；超限不再加压（对应 2F-85 set_gripper_force 力控）

    # 2026-08-24 随机抓取评估防侧推：closing 阶段 pad 已接触 cube（接触力 > 该值(N)）即停止 XY 微调，
    #   避免单侧压着 cube 侧向推挤把物体撞下桌面；未接触（悬空对正）时不受影响。
    closing_align_force_tol: float = 1.0
    # 2026-08-24 方案① Z 微降的安全条件：pad 中心与 cube 中心 XY 偏差 < 该值(m) 才允许悬空 Z 微降，
    #   悬空斜压 cube 边缘是"撞下桌面"另一根源；pad 偏斜时先由 XY 微调对正再下降。
    closing_align_z_xy_tol: float = 0.02
    gripper_close_speed: float = 50.0     # Task3 阻抗更轻柔：closing 每决策步 ctrl 增量（速度斜坡，pad 慢速接触；
                                          # 对应实机 2F-85 set_gripper_speed）
    grasp_min_width: float = 0.02        # 成功阈值：闭合后手指开度(m) 仍 > 该值 = 被物体挡住合不上
    close_confirm_steps: int = 3          # 闭合阶段连续 N 步"开度稳定且 > 阈值 + 接触"才判定成功
    close_width_tol: float = 0.004        # 开度收敛判定容差 (m)
    max_close_steps: int = 60             # 闭合阶段最长步数；超时视为空闭合重开

    # --- P4 成功后的抬升（lift）：判定抓取成功后，夹爪夹着物体向上抬升 lift_height ---
    # 抬升由环境内状态机自动执行（RL 不参与）；到位或失夹/超时即结束 episode。
    # P4 实测（2026-08-20）：lift_speed=0.01 太快会在 ~10 步内把物体甩脱；
    # 0.005 配合指尖衬垫高刚接触（panda.xml solref=0.001/0.999）只能跟随 ~93.7%
    # （物体相对滑移 ~0.02m，属伺服/动态滞后而非蠕动倾覆）。
    # 降到 0.003 后（实测 3 个 seed 稳定）：立方体 100.0% 刚性跟随、宽度恒 0.0345 不滑脱，
    # ~494 步 EE 到位 0.2m（obj 同步抬 ~0.185m = 0.2-lift_tol）。
    # max_steps 只计抬升前步数（见 environment），抬升步数不消耗 RL 预算。
    lift_enabled: bool = False     # 2026-08-23: 抬升默认关闭——成功抓取当步 episode 即结束，
                                   # 训练步数只计数到成功抓取，抬升不计入（tensorboard ep_len 干净，
                                   # 不再显示 218 步含抬升）。部署/验证需要抬升时设 True（P4 逻辑保留）
    lift_height: float = 0.2       # 抬升总高度 (m)
    lift_speed: float = 0.003      # 每步抬升增量上限 (m/step)：0.005 有 ~6% 滞后，0.003 物体 100% 跟随
    lift_tol: float = 0.015        # 到达目标高度的容差 (m)
    lift_max_steps: int = 700      # 抬升阶段最长步数（防死循环；0.003 到 0.2m 实测需 ~494 步）

    # 工作空间配置 - 围绕XML中的物体位置设计（Task3: UR5e + cube@(0.1, 0.42, 0.32)）
    # 2026-08-28 扩大：X (-0.15,0.1)→(-0.18,0.13)、Y (0.32,0.42)→(0.26,0.42)（面积约 +90%）。
    #   从 home 位形 DLS-IK 打点 100% 可达（决策记录 docs/2026-08-28_深度相机分工与桌面扩大眼在手外.md）。
    workspace_bounds: Tuple[Tuple[float, float], Tuple[float, float], Tuple[float, float]] = (
        (-0.18, 0.13),  # X轴范围（原 (-0.15, 0.1)；IK 直接设位模式在距 base>~0.45 处 pad 夹偏，速度模式增量逼近不受限）
        (0.26, 0.42),   # Y轴范围（原 (0.32, 0.42)；下界外扩至 0.26，仍在桌面内）
        (0.25, 0.6)     # Z轴范围 (围绕rq_base_mount home z≈0.48)
    )

    # 任务配置
    target_object: str = "target_cube"  # 目标物体名称
    use_fixed_position: bool = False  # True=固定桌面位置(object_fixed_pos)；False=在workspace_bounds的X/Y内随机（P0-2）
    object_fixed_pos: Tuple[float, float] = (0.1, 0.42)  # Task3: cube 桌面水平位置（换位后；原 -0.134,0.492 对齐 home pinch）
    object_rest_z: float = 0.32  # Task3: 物体落定高度 = 桌面顶(0.30) + 半边长(0.02)；Panda 桌面在 z=0 时原为 0.02
    table_top_z: float = 0.30    # Task3: 桌面顶面高度（初始位形防碰撞校验用）
    # ============================================================
    # 2026-08-28 桌面扩大 + 深度相机（眼在手外 eye-to-hand）
    # ============================================================
    # 桌面（MJCF ur5e_robotiq_cube.xml）：半边长 0.24→0.35（0.70m 见方），中心 (0.1, 0.42)→(0.1, 0.53)
    #   （保持 Y 下界 0.18 不碰下臂，历史教训见 docs/DEV_LOG.md）；顶面 z=0.30 不变。
    #   工作区中心 = object_fixed_pos (0.1, 0.42) = 眼在手外相机的对准点。
    table_half_size: float = 0.35
    table_center: Tuple[float, float] = (0.1, 0.53)
    # 深度相机（眼在手外）：固定在工作区正上方、垂直向下看，不随机械臂运动（区别于 eye-in-hand）。
    #   对应部署侧 RealSense D435i 支架布局；训练端 mujoco.Renderer 渲染等价深度图
    #   （environment.get_depth_image），两端感知特征同构（决策记录 docs/2026-08-28_...）。
    camera_name: str = "eye_to_hand"
    camera_width: int = 640
    camera_height: int = 480
    camera_fovy: float = 60.0    # 垂直视场角（°），与 XML <camera fovy> 一致
    depth_enabled: bool = False  # 感知改造 Phase 2 启用深度渲染（默认关 = 无渲染开销）
    depth_max_clip: float = 5.0  # 深度裁剪上限 (m)，> 该值置 0（远背景清零）
    # ============================================================
    # D6 感知噪声（2026-08-28）：模拟部署侧相机测量误差（ROADMAP Phase 2 降级方案 A）
    # ============================================================
    # 仅污染"观测通道"的目标位置（感知通道）：物理位置/奖励仍用真值（感知-控制双通道隔离，
    #   隔离变量：先只换观测，量化 RL 策略对感知误差的容忍度；验收线"成功率下降 <10%"）。
    # 对应部署侧 RealSense 深度反投影误差（~0.01-0.02m）+ YOLO 漏检（tracker 丢帧零阶保持）。
    perception_noise_std: float = 0.0  # 目标位置高斯噪声标准差 (m)；0=关闭
    perception_dropout: float = 0.0    # 随机漏检概率 (0~1)；0=关闭；漏检时目标估计保持上次值

    # ============================================================
    # D1（2026-08-24，一周冲刺方案 §5）残差策略 + 动态化环境地基
    # 目标：L1 静态闭环（MoveIt 标称 + RL 残差绕障抓取）的地基第一天就位，
    #       动态目标（L2）/ 动态障碍（L3）只改开关、不返工。
    # ============================================================

    # --- D1 残差策略：action_mode ---
    # 'delta'    = 现有行为：RL 输出末端增量，v_raw = Δ/T（无标称轨迹，旧模型语义）
    # 'residual' = 混合残差：v = v_nominal(t) + Δv/T（MoveIt 标称主导 + RL 只学偏差，
    #              对应 docs/混合控制架构设计.md §5.2 Servo 叠加执行语义）
    action_mode: str = "delta"

    # --- D1 标称轨迹（MoveIt 标称仿真替身，见 nominal_trajectory.py）---
    # residual 模式自动启用；delta 模式下标称参考速度槽位为 0（观测预留，见观测空间最终形态）
    nominal_approach_speed: float = 0.12    # 标称接近速度上限 (m/s)，≤ v_max 契约 0.125（docs/sim_to_real动力学匹配方案.md）
    nominal_hover_z_offset: float = 0.134   # 标称悬停高度 = 物体上方 pre_grasp_offset_z（pad 对准 cube 中心）
    nominal_gain: float = 3.0               # 标称速度场增益（阻尼引导：远端匀速 v_max，近端线性减速收敛）
    nominal_horizontal_gain: float = 6.0    # 标称水平(X/Y)对准增益（2026-08-27 选项①：与垂直分离，提高近端水平
                                            # 收敛精度——纯标称需在触发闭合前把 pad 对正到 <2cm（cube 半宽），
                                            # 否则手指从 cube 边缘滑过空闭合，见 docs/2026-08-27_部署分工与最小验证.md §7.3）
    nominal_ik_two_stage: bool = True       # 标称两段轨迹模式（2026-08-27 P2a：90%）。复刻 MoveIt 无碰撞轨迹：
                                            # 段1 末端先到 cube 正上方高处（水平对齐、不扫过物体），段2 垂直下降到 hover。
                                            # 单段速度场直插会扫过 cube 撞飞物体（60%→90%），见 §8.3。速度环执行 + RL 残差兼容。
    # P2b（2026-09-02）：标称 IK 可达性检查（对齐真 MoveIt「IK 解算目标位姿」语义）。
    #   solve_ik 从当前位形验证 hover 位姿可达：可达→正常速度场（行为零回归）；不可达→标称保持
    #   （对标 MoveIt「规划失败→不动」，为扩大 workspace/越出 IK 可达域提供兜底），见 nominal_trajectory.py docstring。
    nominal_ik_check: bool = True           # 是否启用标称目标 IK 可达性检查（默认开；当前 workspace 100% 可达恒通过）
    nominal_ik_check_tol: float = 0.01      # 可达判定容差 (m)：solve_ik 位置误差 < 该值视为可达
    nominal_ik_check_recheck: float = 0.005 # 目标位置变化超过该值(m)才重新解 IK（动态目标节流；0.05m/s×0.04s≈2mm/步）
    nominal_approach_clearance: float = 0.15   # 段1 悬停高度 = hover + 该值（m），保证不碰 cube（cube 高 4cm）
    nominal_stage_switch_tol: float = 0.02     # 段1→段2 切换阈值（末端距段1目标 < 该值 m）
    nominal_feedforward_gain: float = 1.0   # 动态目标速度前馈增益（2026-08-27）：v_nominal += k·v_target。
                                            # 消纯 P 控制对移动目标的稳态跟踪滞后（err≈v_target/gain，
                                            # 见 docs/2026-08-27_部署分工与最小验证.md §4.1）
    # --- P2c（2026-09-07）MPC 标称层：nominal_mode ---
    # 'velocity_field' = 现有速度场标称（假 MoveIt，无状态 O(1)，训练快）
    # 'mpc'            = 末端级滚动最优控制（scipy SLSQP，见 mpc_nominal.py）。
    #                    v = v_mpc(t) + Δv（MPC 标称 + RL 残差）：MPC 处理确定性可预测障碍
    #                    （含 z 方向运动），RL 残差只补 MPC 模型误差/随机游走预测失效。
    #                    与 velocity_field 共用同一 reference_velocity 接口/观测槽位（零改动）。
    nominal_mode: str = "velocity_field"
    # MPC 标称超参（nominal_mode='mpc' 时生效；与 scripts/mpc_ur5_grasp.py 基线同口径）
    mpc_nominal_horizon: int = 10           # 预测步数（0.4s 时域 @ dt=0.04）
    mpc_nominal_w_p: float = 40.0           # 到达 hover 权重
    mpc_nominal_w_o: float = 1500.0         # 障碍分离软约束权重（MPC 自带避障，D_SAFE 外无惩罚）
    mpc_nominal_d_safe: float = 0.20        # 避障安全距离 (m)（球 r=0.05 + margin）
    mpc_nominal_w_s: float = 0.5            # 控制平滑权重（对齐 mpc_ur5_grasp.py 基线 W_S=0.5）
    mpc_nominal_w_term: float = 800.0       # 终端硬权重（强制末步到位，破 warm-start 粘滞）
    mpc_nominal_xy_align_tol: float = 0.035 # XY 对准阈值：对准后才允许降 z（防 pad 侧撞推走 cube）
    mpc_nominal_approach_z: float = 0.06    # 未对准时目标悬高 (m)：先水平对准再下降（对标称两阶段语义）

    # --- D1 动态目标（L2 激活；L1 静止占位）---
    # 物体位置可 per-step 更新（运动学 qpos 写入，z 固定桌面高度 object_rest_z）
    dynamic_target_enabled: bool = False    # 是否启用动态目标
    target_vel_xy: float = 0.0              # 目标水平移动速度 (m/s)；0=静止（L1）
    target_motion_axis: str = "x"           # 运动轴：'x' 或 'y'
    target_period: float = 6.0              # 往返运动周期 (s)（target_vel_xy>0 时生效，三角波往返）

    # --- D1 障碍物地基（D1 起就位；L1 隐藏/固定，L2/L3 激活可编程运动）---
    # 障碍物 body 加 freejoint + step 内 qvel 赋值（可编程运动）；geom 默认 contype=0
    # （不参与碰撞），环境激活时运行时改 contype=1 并移动到 obstacle_fixed_pos——
    # 开关切换不返工（一周冲刺方案 §4.1）。
    obstacle_enabled: bool = False          # 是否启用障碍物
    obstacle_count: int = 1                 # v12: 激活的静态障碍数量（≤ XML 提供的 body 数 obstacle/obstacle_2/obstacle_3）
                                            #     多个障碍沿标称路径不同 fraction/lateral 排布；观测仍只给"最近激活障碍"槽位
                                            #     （67 维不变，兼容旧模型 warm-start）
    obstacle_fixed_pos: Tuple[float, float, float] = (0.06, 0.40, 0.35)  # 固定障碍位置（桌面上的球，球底=桌面顶0.30）
    obstacle_vel: float = 0.0               # 障碍移动速度 (m/s)；0=静态障碍（L1）
    obstacle_axis: str = "y"                # 移动方向轴：'x' 或 'y'
    obstacle_radius: float = 0.05           # 障碍半径 (m)（与 XML geom size 一致）
    obstacle_mass: float = 0.5              # 障碍质量 (kg)（与 XML geom mass 一致）
    obstacle_hidden_pos: Tuple[float, float, float] = (1.0, 1.0, 1.0)  # 未启用时放置处（远离场景，不参与碰撞）
    # P3 修复（2026-08-27，用户观察"动态障碍竖直掉落/像静态"）：动态障碍（obstacle_vel>0）改为
    # 横向往返三角波运动（沿 obstacle_axis，锚点 ±obstacle_half_range），z 每子步钉住——消除
    # freejoint 重力颠簸（视觉掉落）+ 直线飞走 + 0.05m/s 看似静止的缺陷。
    obstacle_period: float = 4.0            # 横向往返周期 (s)
    obstacle_half_range: float = 0.12       # 横向往返半幅 (m)（覆盖路径 ±侧偏，障碍反复扫过）
    # P3 修复 2（2026-08-27，用户要求"在桌面上随机移动"）：随机游走模式。
    # 'random'=桌面 XY 随机游走（方向每 obstacle_dir_change 秒随机变，边界反弹，z 钉住）；
    # 'roundtrip'=固定轴往返（上面 obstacle_period/half_range）。
    obstacle_motion_mode: str = 'random'
    obstacle_dir_change: float = 2.0        # random 模式：随机改方向间隔 (s)
    obstacle_wander_bounds: Tuple[Tuple[float, float], Tuple[float, float]] = (
        (-0.22, 0.18),   # X 游走范围（2026-08-28 随 workspace 扩大，桌面内 X∈[-0.25,0.45]）
        (0.22, 0.50))    # Y 游走范围（2026-08-28 随 workspace 扩大，桌面内 Y∈[0.18,0.88]）

    # --- D2 静态障碍绕障（一周冲刺方案 §5；架构文档 §5.3）---
    # "放标称必经之路"：障碍按当前目标位置自动放在 home→pre-grasp 线段上（reset 时计算），
    # 物体位置随机化（use_fixed_position=False）时障碍随目标跟随，保证每个 episode 都逼 RL 绕障。
    # 与 obstacle_fixed_pos 互斥：obstacle_on_nominal_path=True 时忽略 fixed_pos（优先级更高）。
    obstacle_on_nominal_path: bool = False   # 是否放"标称必经之路"（home→pre-grasp 线段上自动放置）
    # D3 §11.4 无障碍混合采样（2026-08-25）：每 episode 以该概率隐藏障碍做"纯抓取训练"，
    # 防止残差策略在绕障训练中把已学抓取技能覆盖（v5 失败 episode r_obstacle=-66.8 的实证——
    # 只有 70% 绕障样本时旧技能被冲刷）。0=全激活（保持旧行为/check 脚本）；D3 训练传 0.3。
    obstacle_mix_ratio: float = 0.0
    # P3 训练分布（2026-08-27，实习项目 Week1）：per-episode 场景类型采样（None=旧机制
    # 全跟随全局开关，兼容 check_d1/check_d2）。三元组 = (静态无障, 动态目标无障, 动态目标+动态障碍)
    # 概率（归一化）。P3 训练传 (0.4, 0.3, 0.3)：40% 静态（残差→0 防漂移）+ 30% 动态目标（学追踪修正）
    # + 30% 动态目标+动态障碍（学综合修正）。选中的场景**强制覆盖**全局开关（dynamic_target_enabled/
    # obstacle_enabled 仍需配合 CLI 开启以提供 target_vel_xy/obstacle_vel 等参数）。
    scenario_mix: Tuple[float, float, float] = None
    obstacle_path_fraction: float = 0.5      # 沿线段的放置比例（0=home 端，1=pre-grasp 端，0.5=中点）
    obstacle_path_lateral: float = 0.0       # 侧偏(m，相对路径 XY 法向；>0 向 +y 侧，<0 向 -y 侧，0=正落路径）
    # D3 v8（2026-08-25）：每-episode 障碍侧偏随机化区间（m）。默认 (0.0, 0.0)=固定。
    # 障碍位置固定时策略学会"平均路径"直接穿障（残差恒定、无动态绕障，v5/v6/v7 碰撞率卡 65% 根因）；
    # 设置非零区间（如 (-0.06, 0.06)）→ reset 随机 lateral，强制策略基于障碍观测（obstacle_dist 槽位）
    # 学"看障碍在左绕左、在右绕右"的泛化避障。评估时同步随机以测泛化。
    obstacle_path_lateral_range: Tuple[float, float] = (0.0, 0.0)
    # z 偏移默认 -0.10：障碍相对"标称路径高度"下移，罩住夹爪上部碰撞体（rq_base_mount mesh
    # 在 body 原点下方 ~0.105m）——否则球放在 body 路径高度会从夹爪上方掠过、挡不住。绝对值随
    # 末端结构可调（正=抬高，负=降低）；自动约束 z ≥ 桌面顶 + 半径 + 0.02。
    obstacle_path_z_offset: float = -0.10
    # D3 调参（架构文档 §5.3）：碰撞 = 失败信号。实测（v3）对"从无障迁移 + 障碍难绕"的早期，
    # 碰撞终止会让策略一探索就死（episode 平均 25 步），学不到绕障——故**默认关闭**：
    #   obstacle_collision_penalty=0  （碰撞当步无额外惩罚；接近惩罚已由 r_obstacle 提供梯度）
    #   obstacle_collision_max_streak=0（0=禁用提前终止；>0 时连续碰撞超阈值 truncated）
    # 真机/评估（D6）需要"碰撞=失败"时再启用。
    obstacle_collision_penalty: float = 0.0   # 撞上当步额外惩罚（<0 生效，叠加在接近惩罚之上）
    obstacle_collision_max_streak: int = 0    # 连续碰撞步数阈值 -> truncated（0=禁用；如 20 ≈0.8s 持续撞障）

    # --- P2c（2026-09-07）per-obstacle 规格：更多障碍（静态+动态混合、z 方向运动）---
    # None = 旧统一逻辑（obstacle_count / obstacle_vel / obstacle_motion_mode 应用于前 N 个障碍）。
    # 设置后：每个激活障碍独立配置（XML 提供 obstacle/obstacle_2/...，按下标 i 匹配）：
    #   {'type': 'static',  'pos': (x,y,z)}                       静态钉住锚点
    #   {'type': 'dynamic', 'vel': 0.05, 'mode': 'random'|'roundtrip',
    #    'axis': 'x'|'y', 'pos': (x,y,z),                         运动锚点/初始位置
    #    'z_motion': True, 'z_amp': 0.10, 'z_period': 3.0}        z 方向往返运动（半幅 m / 周期 s）
    # 动态障碍在 XY（random/roundtrip）基础上可选叠加 z 方向三角波往返（默认关闭）。
    # 观测仍只给\"最近激活障碍\" rel/vel 槽位（67 维不变）；MPC 标称（nominal_mode='mpc'）用全部
    # 激活障碍做避障软约束预测。obstacle_on_nominal_path=True 时位置由\"必经之路\"布局覆盖，
    # 运动行为（static/dynamic/z_motion）仍按 specs 生效。
    obstacle_specs: List[dict] = None
    # 旧统一逻辑下动态障碍的 z 方向往返运动开关与参数（obstacle_specs=None 时生效）：
    obstacle_z_motion: bool = False          # 动态障碍是否叠加 z 方向三角波往返
    obstacle_z_amp: float = 0.10             # z 往返半幅 (m)（锚点 ±z_amp；自动约束 ≥ 桌面顶+半径+0.02）
    obstacle_z_period: float = 3.0           # z 往返周期 (s)



@dataclass
class NetworkConfig:
    """网络配置（P0-3：移除从未消费的 policy_activation/value_activation，
    PPO 实际使用默认 relu；如需自定义激活请走 policy_kwargs 显式传入）"""
    # 策略网络
    policy_hidden_sizes: List[int] = None
    # 价值网络
    value_hidden_sizes: List[int] = None
    
    def __post_init__(self):
        if self.policy_hidden_sizes is None:
            self.policy_hidden_sizes = [512, 512, 256]  # 增加网络深度和宽度
        if self.value_hidden_sizes is None:
            self.value_hidden_sizes = [512, 512, 256]  # 增加网络深度和宽度

@dataclass
class TrainingConfig:
    """训练配置"""
    # 基本参数
    total_timesteps: int = 1000000  # 并行改造(2026-08-22): 40000→200000。A2(40k 步/136ep) 成功率仅 5.1%，样本量是成功率低的根因之一；配合 12 环境并行，墙钟时间反而更短
    learning_rate: float = 2e-4     # 适中的学习率
    batch_size: int = 1024          # 与n_steps匹配，避免警告
    n_steps: int = 2048          # 并行改造(2026-08-22): 1024→2048（12 环境并行，每 worker 每 rollout 跑 ~171 步；batch_size 1024 不变）
    n_epochs: int = 10             # 并行改造(2026-08-22): 5→10（数据量↑，多学几轮，充分利用并行采集的样本）
    
    # PPO参数 - 增加探索和归一化
    gamma: float = 0.99
    gae_lambda: float = 0.95
    clip_range: float = 0.2
    ent_coef: float = 0.01          # 增加熵系数，促进探索
    vf_coef: float = 0.25
    max_grad_norm: float = 0.5
    
    # 归一化参数
    normalize_advantage: bool = True  # 优势函数归一化
    normalize_observations: bool = True  # 观察归一化
    normalize_rewards: bool = False  # P1: 关闭奖励归一化。早期奖励接近常量、方差≈0，除以小方差会把奖励放大到±10以上导致不稳定；奖励本身在[-3,10]有界，交给PPO优势归一化处理
    norm_obs_clip: float = 10.0  # 观察归一化裁剪
    norm_reward_clip: float = 10.0  # 奖励归一化裁剪
    
    # 训练控制
    save_freq: int = 98304          # 检查点保存频率（2026-08-23: 50000→98304。SB3 CheckpointCallback 要求
                                    # save_freq 是 n_steps×n_envs(2048×12=24576) 的倍数，否则永不触发；
                                    # 98304 = 4 个 rollout ≈ 每 ~4 分钟存一个 checkpoint，崩溃最多损失 4 分钟）
    # 早停条件 - 放宽条件
    success_threshold: float = 0.6   # 降低成功率阈值 (原0.8)
    patience: int = 50              # 增加耐心值 (原30)

@dataclass
class RewardConfig:
    """奖励配置（P1 重新设计）

    位置目标 = 物体中心正上方 pre_grasp_offset_z，而不是物体中心：因为
    ee_position 是 hand 基座位置，物理上到不了物体中心，原设计会让奖励饱和、
    对“最后对准→抓取”这一段失去梯度。
    """
    # --- 位置：以 pre-grasp 位姿为目标 ---
    pre_grasp_offset_z: float = 0.134  # 末端基座悬停高度（物体中心上方；Task3 UR5e: pad 中心距 rq_base_mount 0.134m，此值使 pad 对准 cube 中心）
    w_xy: float = 2.0          
    w_height: float = 8.0             # 高度(Z)到位权重。方案A(2026-08-22): 2.0→8.0。主势能 slope 6.67→26.7/m，
                                       # 每下降一步收益 ~0.13 > step_penalty 0.1——修复"pad 悬停 cube 上方 2-4cm
                                       # 不触发闭合(需 Z差<3.2cm)且无下压动力"的停驻陷阱（闭合触发边界实测见
                                       # docs/训练效果分析与VecNormalize保存修复.md）
    xy_scale: float = 0.3             # 平面距离归一化尺度(m)，0.3m 处惩罚=1，把xy和height的尺度拉开，避免高度到位时xy仍然远离目标
    height_scale: float = 0.3        # 高度误差归一化尺度(m)

    # --- 距离势场封顶（P1.1 去饱和）：线性梯度覆盖可达空间，仅极端远处封顶防量级失控 ---
    xy_cap: float = 0.60             # d_xy 封顶(m)，> 最大可达距离，正常状态下不触发
    height_cap: float = 0.30         # dz 封顶(m)，> 最大可达误差，正常状态下不触发

    # --- 方向：手指朝下 ---
    # P2-1: 由“四元数对齐 cos(θ) 再 max(0,·)”（θ>90° 时无梯度/死区）改为
    # “手指朝下投影 −z_axis_z”（朝下=+1，水平=0，朝上=−1），全程连续有梯度。
    w_orient: float = 0.0             # 方向奖励权重：gain = w_orient * (-z_axis_z)。
                                      # 2026-08-22 调整: 3.0→0.0。当前 action_space_dim=3（姿态固定朝下），
                                      # 方向项是纯损耗（恒 -0.03/步，每 episode 白扣 ~6 分）无学习信号；
                                      # 切回 6D 动作空间时恢复 3.0（此时姿态维度才需要方向梯度）。

    # --- 接触（真实物理接触）---
    w_contact: float = 5.0            # 手指接触物体奖励
    # 方案②(2026-08-23): 进入 closing 相位（pad 到达闭合触发区）的中间奖励——激励策略
    # 把 pad 从"触发区边缘(4cm)"精确推到"正中心(3cm内)"（PPO_93 失败分析：4/10 未进 closing，
    # pad-cube XY 差 0.6-2.1cm；closing 是抓取的中间里程碑，此前无中间信号，grasp_reward 太稀疏）
    close_trigger_reward: float = 5.0
    contact_force_threshold: float = 0.1  # 接触判定阈值(N)

    # --- 抓取事件 ---
    grasp_reward: float = 100.0         # 到位+夹紧+接触（is_grasped）
    completion_reward: float = 0.0   # grasp_success（P1 中与 is_grasped 相同，见 MD）

    # --- 时间惩罚 ---
    step_penalty: float = 0.05       # 轻时间惩罚，防止原地磨蹭。2026-08-22 调整: 0.1→0.05（200 步成本 -20→-10；
                                      # 原 0.1 吞掉策略全部靠近收益，见 docs/调参失败分析.md PPO_89）

    # --- 势能塑形（P1.3）---
    shaping_gamma: float = 0.99      # 势能塑形折扣 γ：r_shape = Φ(s_prev) − γ·Φ(s)
                                      # = PPO γ(0.99)，满足 Ng 势能塑形等价性（不改变最优策略）。
                                      # ⚠️ 曾用 0.9 < γ_mdp(0.99)：策略停驻时每步白拿
                                      # (γ_mdp−γ_shape)·Φ ≈ 0.09·Φ 的正收益（d≈0.05 处 ≈0.105/步，
                                      # 300 步 ≈31 ≫ 步惩罚−15）→ "停在够近不精调"驻留陷阱。
                                      # γ_shape=γ_mdp 后停驻收益=0，塑形只提供梯度方向，行为由事件奖励驱动。
                                      # （2026-08-22 混合方案修复，见 docs/奖励设计与混合塑形方案.md）

    # --- 近距离锚定势能（2026-08-22 混合方案，见 docs/奖励设计与混合塑形方案.md）---
    # A 训练策略"大幅靠近但未精确对准"（闭合触发需 XY<3cm/Z<2cm）：主势能在近距离
    # 每靠近一步的塑形收益 γ·slope·Δd（≈0.03） < step_penalty(0.05) → 策略停在"够近
    # 但不精调"。叠加单调锚定势能使近距离斜率变陡：
    #   Φ_anchor = anchor_w · min(d/anchor_dist, 1)   （单调：d=0 时 0，之外封顶）
    # 近距离斜率 = 主势能 + anchor_w/anchor_dist（xy 6.7→23.3、z 20→45），
    # 每靠近一步收益 > 步惩罚 → 有动力精调到最后一步；仍为势能差形式（有界、不变策略）。
    anchor_w_xy: float = 1.0         # xy 锚定势能权重（d_xy=anchor_dist 时 Φ_anchor=1.0）
    anchor_dist_xy: float = 0.06     # xy 锚定激活距离(m)
    anchor_w_z: float = 2.0          # z 锚定势能权重（方案A 2026-08-22: 1.0→2.0，配合 anchor_dist_z 0.04→0.06，近距离下压梯度更强）
    anchor_dist_z: float = 0.06      # z 锚定激活距离(m)（方案A 2026-08-22: 0.04→0.06，覆盖 pad 悬停区 Z差4-6cm，
                                      # 让"降到接触"的最后一段持续有塑形梯度）

    # --- D2 障碍接近惩罚 + 残差幅度正则（一周冲刺方案 §5；架构文档 §5.3）---
    # 障碍接近惩罚（连续负奖励 shaping，d=末端 base_mount → 障碍表面的距离）：
    #   r_obstacle = -obstacle_w * max(0, 1 - d / obstacle_range)
    #   仅障碍启用时非零；obstacle_enabled=False 时 env 传 obstacle_dist=inf → 恒 0，不干扰旧训练。
    # 残差幅度正则（仅 residual 模式；delta 模式 env 传 residual_norm=0 → 恒 0）：
    #   r_residual = -residual_reg_w * ||Δv||²     （L2 鼓励小残差：静态场景 Δv≈0，动态才出手——
    #   对应"残差幅度统计：静态小/动态大"架构分工，见 nominal_trajectory.py / 混合控制架构设计.md）
    obstacle_w: float = 1        # 障碍接近惩罚权重（d=0 时惩罚 -0.5/步）
    obstacle_range: float = 0.15   # 接近惩罚触发距离(m)（末端→障碍表面 > 该距离 → r_obstacle=0）
    # v16 奖励经济学（2026-08-28，用户要求"注意奖励经济学设计"）：
    #   obstacle_clear_bonus——"干净成功"一次性奖励：抓取成功当步若全程 0 碰撞 → +bonus。
    #   经济学：v15 穿障成功仍净赚（collision -5×10=-50 < 成功 +120）；该 bonus 让"绕开障碍
    #   再抓取"成为独立正收益（+100+30）vs 穿障成功（+100 无 bonus）→ 引导策略主动绕障而非贴边。
    #   默认 0 保持旧行为（旧模型/旧训练不引入额外奖励，避免观测口径漂移）。
    obstacle_clear_bonus: float = 0.0
    residual_reg_w: float = 0.5    # 残差幅度正则权重（||Δv||=0.125 上限时约 -0.0078/步，温和不淹没抓取信号）
    # v25 残差稀疏化（2026-09-08，见 docs/2026-09-08_残差饱和分析与奖励重塑方案.md）——经济学修复：
    #   根因：v24 用 L2(reg_w=2.0) 对满偏置残差惩罚仅 ~0.094/步（200 步 ≈18.7）≪ 成功 +130，
    #   满幅残差"几乎免费"→ PPO 停在 clip 上界（avg_residual_norm 恒 0.2165）。三件套治本：
    #   r_residual = -residual_reg_w·||Δv||² - residual_l1_w·||Δv||₁ - residual_step_w·𝟙[||Δv||>τ]
    residual_l1_w: float = 0.0    # L1 稀疏权重（满幅残差 ‖Δv‖₁=0.15@cap=0.002 → -0.15·w/步；w=2 时 200 步 ≈-60，与成功 +130 同级）
    residual_step_w: float = 0.0  # 残差激活硬门控步罚：||Δv|| > τ 时当步再罚 -residual_step_w（进一步惩罚"持续激活"）
    residual_step_threshold: float = 0.02  # 残差"激活"判定阈值 (m/s)：‖Δv‖ > 该值视为出手（≈30% 残差预算）
    # v25 §4.5 残差引起的碰撞双倍惩罚：collision 且 ‖Δv‖ > τ 时，在 obstacle_collision_penalty 之外再叠加该罚，
    # 让策略学到"不确定时别碰 MPC 标称"（直接对应 dyn_target 残差轻微干扰问题）。
    residual_collision_threshold: float = 0.02     # 残差碰撞双倍罚判定阈值 (m/s)
    residual_collision_penalty_extra: float = 0.0  # 残差激活下碰撞的额外惩罚（默认 0 关闭；训练 CLI --residual-collision-penalty-extra 覆盖）

@dataclass
class SystemConfig:
    """系统配置"""
    # 路径配置 - 确保所有路径都在rl_grasping_system目录下
    base_dir: str = os.path.dirname(os.path.abspath(__file__))
    models_dir: str = os.path.join(os.path.dirname(os.path.abspath(__file__)), "models")
    logs_dir: str = os.path.join(os.path.dirname(os.path.abspath(__file__)), "logs")
    results_dir: str = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results")
    
    # 设备配置
    device: str = "auto"  # "auto", "cpu", "cuda"
    num_envs: int = 12             # 并行改造(2026-08-22): 1→12。16 核 CPU / 11GB 内存，12 个 SubprocVecEnv worker（每 worker ~400-600MB，留内存余量；吃紧可降 8）
    
    # 随机种子
    seed: int = 42

    def __post_init__(self):
        # 创建必要的目录
        for dir_name in [self.models_dir, self.logs_dir, self.results_dir]:
            os.makedirs(dir_name, exist_ok=True)

@dataclass
class Config:
    """总配置"""
    grasping: GraspingConfig = None
    network: NetworkConfig = None
    training: TrainingConfig = None
    reward: RewardConfig = None
    system: SystemConfig = None
    
    def __post_init__(self):
        if self.grasping is None:
            self.grasping = GraspingConfig()
        if self.network is None:
            self.network = NetworkConfig()
        if self.training is None:
            self.training = TrainingConfig()
        if self.reward is None:
            self.reward = RewardConfig()
        if self.system is None:
            self.system = SystemConfig()

# 默认配置实例
default_config = Config()

def get_config() -> Config:
    """获取配置"""
    return default_config

def update_config(config: Config, **kwargs):
    """更新配置"""
    for key, value in kwargs.items():
        if hasattr(config, key):
            setattr(config, key, value)
        else:
            # 尝试更新子配置
            for sub_config_name in ['grasping', 'network', 'training', 'reward', 'system']:
                sub_config = getattr(config, sub_config_name)
                if hasattr(sub_config, key):
                    setattr(sub_config, key, value)
                    break
    return config
