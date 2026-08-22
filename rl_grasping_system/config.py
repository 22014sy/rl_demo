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
    # Task3 冲击抑制（见 docs/抓取冲击抑制方案.md）：
    approach_speed_limit: float = 0.12    # pad 距 cube < approach_speed_dist 时末端线速度限幅 (m/s)（对应实机 UR 慢速 Servo 接近）
    approach_speed_dist: float = 0.06     # 接近限速触发距离（pad 中心→cube；仅抓取最后一段限速，避免 home 全程触发）
    grasp_force_limit: float = 20.0       # closing/closed 阶段 pad-cube 接触力上限 (N)；超限不再加压（对应 2F-85 set_gripper_force 力控）
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
    lift_height: float = 0.2       # 抬升总高度 (m)
    lift_speed: float = 0.003      # 每步抬升增量上限 (m/step)：0.005 有 ~6% 滞后，0.003 物体 100% 跟随
    lift_tol: float = 0.015        # 到达目标高度的容差 (m)
    lift_max_steps: int = 700      # 抬升阶段最长步数（防死循环；0.003 到 0.2m 实测需 ~494 步）

    # 工作空间配置 - 围绕XML中的物体位置设计（Task3: UR5e + cube@(-0.134, 0.492, 0.32)）
    workspace_bounds: Tuple[Tuple[float, float], Tuple[float, float], Tuple[float, float]] = (
        (-0.15, 0.1),   # X轴范围（IK 直接设位模式在距 base>~0.45 处 pad 夹偏；cube 固定 0.51 走速度模式）
        (0.32, 0.42),   # Y轴范围（max dist≈0.45）
        (0.25, 0.6)     # Z轴范围 (围绕rq_base_mount home z≈0.48)
    )

    # 任务配置
    target_object: str = "target_cube"  # 目标物体名称
    use_fixed_position: bool = True  # True=固定桌面位置(object_fixed_pos)；False=在workspace_bounds的X/Y内随机（P0-2）
    object_fixed_pos: Tuple[float, float] = (0.1, 0.42)  # Task3: cube 桌面水平位置（换位后；原 -0.134,0.492 对齐 home pinch）
    object_rest_z: float = 0.32  # Task3: 物体落定高度 = 桌面顶(0.30) + 半边长(0.02)；Panda 桌面在 z=0 时原为 0.02
    table_top_z: float = 0.30    # Task3: 桌面顶面高度（初始位形防碰撞校验用）

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
    total_timesteps: int = 500000  # 并行改造(2026-08-22): 40000→200000。A2(40k 步/136ep) 成功率仅 5.1%，样本量是成功率低的根因之一；配合 12 环境并行，墙钟时间反而更短
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
    save_freq: int = 50000          # 检查点保存频率（仅当 train(save_path=...) 时生效）
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
