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
    xml_path: str = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "models/franka_emika_panda/single_cube_scene.xml")
    max_steps: int = 300  # P1: 增量式位置控制下给足行程（原 100 太短，手臂走不完）
    control_freq: int = 50  # 控制频率 (Hz)，P1: 现在真正生效——每个环境动作执行 1/control_freq 秒的物理仿真（子步进）
    render_gui: bool = True  # P0-2: 默认关闭 GUI（无头服务器友好）；本地要看画面时显式置 True
    render_interval: int = 5  # 每隔多少步显示一次画面
    render_pause_sec: float = 0.001  # 刷新窗口时的短暂暂停
    
    # 抓取配置 - 大幅放宽条件
    grasp_distance_threshold: float = 0.1   # 抓取距离阈值 (大幅放宽)
    grasp_success_threshold: float = 0.08   # 抓取成功阈值 (大幅放宽)
    gripper_closed_threshold: float = 0.05  # 夹爪闭合阈值 (大幅放宽)
    max_joint_delta: float = 0.05  # P1: 每步关节增量上限 (rad)，增量式位置控制（动作空间前7维边界）
    
    # 工作空间配置 - 围绕XML中的物体位置设计
    workspace_bounds: Tuple[Tuple[float, float], Tuple[float, float], Tuple[float, float]] = (
        (0.3, 0.7),   # X轴范围 (围绕0.5)
        (-0.2, 0.2),  # Y轴范围 (围绕0.0)
        (0.05, 0.3)   # Z轴范围 (围绕0.1)
    )
    
    # 任务配置
    target_object: str = "target_cube"  # 目标物体名称
    use_fixed_position: bool = True  # True=固定桌面位置(0.5,0)；False=在workspace_bounds的X/Y内随机（P0-2）
    object_rest_z: float = 0.02  # 物体落定高度(=0.04m立方体半边长)；X/Y随机或固定时Z都用此值，物体直接放桌面

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
    total_timesteps: int = 50000  # 总训练步数
    learning_rate: float = 3e-4     # 适中的学习率
    batch_size: int = 1024          # 与n_steps匹配，避免警告
    n_steps: int = 1024             # 适中的n_steps
    n_epochs: int = 5              # 增加epochs，充分利用数据
    
    # PPO参数 - 增加探索和归一化
    gamma: float = 0.99
    gae_lambda: float = 0.95
    clip_range: float = 0.2
    ent_coef: float = 0.005          # 增加熵系数，促进探索
    vf_coef: float = 0.5
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
    pre_grasp_offset_z: float = 0.10  # hand 基座悬停高度（物体中心上方 0.10 m，≈手指包住物体）
    w_xy: float = 2.0          
    w_height: float = 2.0             # 高度(Z)到位权重（下降是最后关键一步，权重更高）
    xy_scale: float = 0.3             # 平面距离归一化尺度(m)，0.3m 处惩罚=1，把xy和height的尺度拉开，避免高度到位时xy仍然远离目标
    height_scale: float = 0.15        # 高度误差归一化尺度(m)

    # --- 距离势场封顶（P1.1 去饱和）：线性梯度覆盖可达空间，仅极端远处封顶防量级失控 ---
    xy_cap: float = 0.60             # d_xy 封顶(m)，> 最大可达距离，正常状态下不触发
    height_cap: float = 0.30         # dz 封顶(m)，> 最大可达误差，正常状态下不触发

    # --- 方向：手指朝下 ---
    w_orient: float = 2.0             # 手+Z(手指)与目标抓取姿态的四元数对齐 cos(θ)(max0) 的奖励权重（P1.2）

    # --- 接触（真实物理接触）---
    w_contact: float = 1.0            # 手指接触物体奖励
    contact_force_threshold: float = 0.1  # 接触判定阈值(N)

    # --- 抓取事件 ---
    grasp_reward: float = 5.0         # 到位+夹紧+接触（is_grasped）
    completion_reward: float = 10.0   # grasp_success（P1 中与 is_grasped 相同，见 MD）

    # --- 时间惩罚 ---
    step_penalty: float = 0.01        # 轻时间惩罚，防止原地磨蹭

    # --- 势能塑形（P1.3）---
    shaping_gamma: float = 0.99       # 势能塑形折扣 γ：r_shape = Φ(s_prev) − γ·Φ(s)
                                      # （单步有界、episode 回报不随长度膨胀，见 reward.py）

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
    num_envs: int = 1
    
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
