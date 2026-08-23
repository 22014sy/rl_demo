"""
强化学习抓取环境：基于 MuJoCo 的 UR5e + Robotiq 2F-85 抓取任务环境（gymnasium.Env）。

 - 动作空间：末端位姿增量（action_space_dim=6 -> [dx,dy,dz,dax,day,daz]；=3 -> [dx,dy,dz]，姿态固定朝下），
 - Task3 速度模式：增量经速度级 IK（雅可比阻尼伪逆）→ 关节速度目标 → 环境内"速度 PID + 重力补偿前馈"
   （arm 用 motor 力矩输出；根治位置伺服滞后，脉冲实现率 ~96%）；
 - 观测空间：本体感知状态组成的 58 维向量（见 get_proprioceptive_state / _get_observation）；
 - 任务：RL 只控制末端位姿；夹爪由环境内"近距自动闭合状态机"驱动，抓取成功 = 手指被物体挡住合不上；
 - 控制：每决策步执行 action_repeat 个控制周期（50Hz），速度模式下每物理步刷新速度目标。
"""

import numpy as np
import mujoco
import mujoco.viewer as mjviewer
import gymnasium as gym
from gymnasium import spaces
from typing import Dict, Tuple, Optional, Any
import logging
import os
import time

from config import GraspingConfig, RewardConfig
from singularity_handler import SingularityHandler
from action_wrapper import SafeActionWrapper
from state import get_proprioceptive_state
from reward import calculate_reward, reward_breakdown, REWARD_KEYS, quat_rotate_world_z, Q_APPROACH_DOWN
import ik

class GraspingEnv(gym.Env):
    """
    UR5e + Robotiq 2F-85 抓取环境（Task3 迁移；PandaGraspingEnv 保留为兼容别名）

    任务: 控制机械臂到达并抓取指定位置的物体（速度控制模式）
    观察空间: 本体感知状态 + 肌腱状态
    动作空间: 末端位姿增量（速度控制）
    """
    # 构造函数
    def __init__(self, grasping_config: GraspingConfig, reward_config: RewardConfig):
        super().__init__()
        
        self.grasping_config = grasping_config
        self.reward_config = reward_config
        # Task3: 控制模式（velocity / position）
        self.control_mode = str(getattr(grasping_config, 'control_mode', 'velocity'))
        
        # 设置日志
        self.logger = logging.getLogger(__name__)
        
        # 检查XML文件是否存在
        if not os.path.exists(grasping_config.xml_path):
            raise FileNotFoundError(f"MuJoCo XML文件不存在: {grasping_config.xml_path}")
        
        # 初始化MuJoCo模型
        self._init_mujoco()
        
        # 设置观察空间和动作空间
        self._setup_spaces()
        
        # 初始化奇异点处理器
        self.singularity_handler = SingularityHandler()
        
        # 初始化安全动作包装器（位置模式用；速度模式 step 直接解析增量转速度）
        self.action_wrapper = SafeActionWrapper(self)
        
        # 目标信息
        self.target_pos = np.array([0.5, 0.0, 0.3])
        self.target_quat = np.array([1, 0, 0, 0])
        
        # 任务状态
        self.task_state = {
            'object_position': None,
            'is_grasped': False,
            'grasp_success': False,
            'episode_steps': 0,
            'previous_joint_pos': None,
            'was_singular': False
        }
        
        # 前一步状态（用于奖励计算）
        self.prev_state = None
        
        # 初始化训练监控器
        self.training_monitor = None
        self.episode_start_time = None
        self.episode_singularity_count = 0
        self.render_gui = getattr(grasping_config, "render_gui", False)
        self.render_interval = max(1, int(getattr(grasping_config, "render_interval", 1)))
        self.render_pause_sec = float(getattr(grasping_config, "render_pause_sec", 0.001))
        self.viewer_handle = None
        self.viewer_available = mjviewer is not None
        self._viewer_error_logged = False
        
        # 重置警告控制
        self.singularity_handler.reset_warning_control()
        
        # P4: 成功提示叠加文本去重标志（mujoco>=3.x 用 set_texts）
        self._success_overlay_shown = False

        # 重置环境
        self.reset()
    
    def _init_mujoco(self):
        """初始化MuJoCo环境"""
        try:
            # 渲染后端由入口脚本统一设置：本地 train_with_monitor 默认 GLFW，
            # 云端 train_cloud 设 EGL。此处不写死，避免覆盖云端设置；
            # 未设置任何后端时 mujoco.Renderer 创建失败 → 走下方 except 无头模式。

            # 加载模型
            self.model = mujoco.MjModel.from_xml_path(self.grasping_config.xml_path)
            self.data = mujoco.MjData(self.model)
            
            # 查找关键组件
            self._find_components()
            
            # P1: 控制周期 = 1/control_freq；每个环境动作执行控制周期内的物理子步
            # （action repeat / 子步进）。此前只跑 1 个 mj_step(2ms)，位置伺服几乎不动，
            # 实测 0.02 rad 命令一步仅推进 0.0006 rad，手臂在 max_steps 内走不完。
            self.control_freq = float(getattr(self.grasping_config, 'control_freq', 50))
            self.control_period = 1.0 / self.control_freq
            self.substeps = max(1, int(round(self.control_period / self.model.opt.timestep)))
            # P5: action_repeat——每个 RL 决策步执行 N 个控制周期再观测/给奖励。
            # 提高单决策步内伺服收敛量（小增量实现率 8%→41%），不改控制频率。
            self.action_repeat = max(1, int(getattr(self.grasping_config, 'action_repeat', 1)))
            
            # 尝试创建渲染器（并行改造 2026-08-22）：仅 render_gui=True 时创建。
            # 训练 worker 全部无头——不创建渲染器可避免 SubprocVecEnv fork 后子进程
            # 继承主进程 GLFW/X 连接导致的 XIO fatal error，同时每个 worker 省一块渲染缓冲。
            if bool(getattr(self.grasping_config, 'render_gui', False)):
                try:
                    self.renderer = mujoco.Renderer(self.model)
                    self.headless = False
                    print("✅ 图形渲染模式已启用")
                except Exception as e:
                    print(f"⚠️  图形渲染失败，使用无头模式: {e}")
                    self.renderer = None
                    self.headless = True
            else:
                self.renderer = None
                self.headless = True
                
        except Exception as e:
            print(f"❌ MuJoCo初始化失败: {e}")
            raise
    
    def _find_components(self):
        """查找关键组件:arm gripper body 目标物体target_body 末端执行器end 
            两种方法
        """
        try:
            # 查找机械臂关节：只取铰链(hinge)关节，排除 2F-85 的 rq_* 夹爪铰链和物体 freejoint。
            # Task3：优先用 config.arm_joint_names 白名单（UR5e 六关节），兜底用"非 rq_ 前缀"过滤。
            self.arm_joint_names = []
            _white = list(getattr(self.grasping_config, 'arm_joint_names', ()) or ())
            for i in range(self.model.njnt):
                joint_name = mujoco.mj_id2name(self.model, mujoco.mjtObj.mjOBJ_JOINT, i)
                if not joint_name or self.model.jnt_type[i] != mujoco.mjtJoint.mjJNT_HINGE:
                    continue
                if _white:
                    if joint_name in _white:
                        self.arm_joint_names.append(joint_name)
                elif not joint_name.startswith('rq_'):
                    self.arm_joint_names.append(joint_name)
            # 实际关节 ID（freejoint 存在时 qpos 地址 ≠ 关节 ID，但 arm HINGE 均为单自由度、位于数组前段）
            self.arm_joint_ids = [mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, n)
                                  for n in self.arm_joint_names]
            
            # 查找夹爪关节：2F-85 的 rq_* 铰链（driver/coupler/spring_link/follower，共 8 个）
            self.gripper_joint_names = []
            self.gripper_joint_ids = []  # 关节 ID（reset 时设初始全开 = qpos=0）
            for i in range(self.model.njnt):
                joint_name = mujoco.mj_id2name(self.model, mujoco.mjtObj.mjOBJ_JOINT, i)
                if joint_name and joint_name.startswith('rq_'):
                    self.gripper_joint_names.append(joint_name)
                    self.gripper_joint_ids.append(i)
            
            # Task3: 夹爪 actuator 索引（fingers_actuator，ctrl 0-255 位置伺服）+ 衬垫测量 site
            self.gripper_actuator_idx = int(getattr(self.grasping_config, 'gripper_actuator_idx', 6))
            self.pad_site_left_id = mujoco.mj_name2id(
                self.model, mujoco.mjtObj.mjOBJ_SITE,
                getattr(self.grasping_config, 'pad_site_left', 'rq_pad_left_site'))
            self.pad_site_right_id = mujoco.mj_name2id(
                self.model, mujoco.mjtObj.mjOBJ_SITE,
                getattr(self.grasping_config, 'pad_site_right', 'rq_pad_right_site'))
            
            # 获取所有body名称，用于调试
            all_body_names = []
            for i in range(self.model.nbody):
                body_name = mujoco.mj_id2name(self.model, mujoco.mjtObj.mjOBJ_BODY, i)
                if body_name:
                    all_body_names.append((i, body_name))
            
            self.logger.info(f"模型中的所有body: {[name for _, name in all_body_names]}")
            
            # 查找目标物体 - 更加灵活的方法
            self.target_body_id = -1
            target_candidates = [self.grasping_config.target_object] + ["red_cube", "blue_cube", "green_cube", "target_cube", "cube"]
            
            for target_name in target_candidates:
                self.target_body_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, target_name)
                if self.target_body_id != -1:
                    self.logger.info(f"找到目标物体: {target_name} (ID: {self.target_body_id})")
                    break
            
            if self.target_body_id == -1:
                # 如果还是找不到，尝试查找包含关键词的body
                for i, name in all_body_names:
                    if any(keyword in name.lower() for keyword in ['cube', 'box', 'target', 'object', 'ball']):
                        self.target_body_id = i
                        self.logger.info(f"通过关键词找到目标物体: {name} (ID: {i})")
                        break
                
                if self.target_body_id == -1:
                    self.logger.warning(f"找不到目标物体，使用默认")
                    self.target_body_id = None
            
            # 查找末端执行器 - 更加灵活的方法（Task3：UR5e 用 rq_base_mount 优先）
            self.end_effector_id = -1
            end_effector_candidates = ["rq_base_mount", "panda_hand", "panda_gripper", "end_effector", "gripper", "hand", "gripper_hand", "ee", "end_effector_link"]
            
            for name in end_effector_candidates:
                self.end_effector_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, name)
                if self.end_effector_id != -1:
                    self.logger.info(f"找到末端执行器: {name} (ID: {self.end_effector_id})")
                    break
            
            if self.end_effector_id == -1:
                # 如果还是找不到，尝试查找包含关键词的body
                for i, name in all_body_names:
                    if any(keyword in name.lower() for keyword in ['hand', 'gripper', 'ee', 'end', 'finger']):
                        self.end_effector_id = i
                        self.logger.info(f"通过关键词找到末端执行器: {name} (ID: {i})")
                        break
            
            if self.end_effector_id == -1:
                # 如果还是找不到，使用最后一个body作为末端执行器
                if all_body_names:
                    self.end_effector_id = all_body_names[-1][0]
                    self.logger.warning(f"找不到末端执行器，使用最后一个body: {all_body_names[-1][1]} (ID: {self.end_effector_id})")
                else:
                    raise ValueError("模型中没有找到任何body")
            
            # 查找物体ID（用于设置物体位置）
            self.object_id = self.target_body_id  # 使用相同的ID
            
            self.logger.info(f"找到 {len(self.arm_joint_names)} 个机械臂关节")
            self.logger.info(f"找到 {len(self.gripper_joint_names)} 个夹爪关节")
            self.logger.info(f"目标物体ID: {self.target_body_id}")
            self.logger.info(f"末端执行器ID: {self.end_effector_id}")
            self.logger.info(f"物体ID: {self.object_id}")
            
        except Exception as e:
            self.logger.error(f"组件查找失败: {e}")
            raise
    
    def _setup_spaces(self):
        """设置观察空间和动作空间"""
        # 动作空间: 前7维=每步关节增量(rad)，第8维=肌腱命令(0=闭合,1=张开)（P1: 增量式位置控制）
        # P3: 动作空间 = 末端位姿增量（action_space_dim=6 -> 位置+姿态；3 -> 仅位置，姿态固定朝下）
        dim = int(self.grasping_config.action_space_dim)
        lo = np.array([-self.grasping_config.max_ee_delta] * 3
                      + [-self.grasping_config.max_orient_delta] * 3)[:dim]
        hi = -lo.copy()
        self.action_space = spaces.Box(low=lo.astype(np.float32), high=hi.astype(np.float32), dtype=np.float32)
        
        # 观测空间（Task3：机械臂 6 关节 -> 55 维）：
        # 6关节位置+6关节速度+6关节力矩+1肌腱位置+1肌腱速度+1肌腱张力+
        # 3末端位置+4末端方向+6末端速度+1夹爪状态+3目标位置+4目标方向+1可操作度+1接触力+6手指力
        # +4相对接近姿态四元数(P2-2, q_target_approach ⊗ q_ee⁻¹，w≥0 规范化，见 _get_observation)
        # +1 夹爪状态机相位(P3)  —— 7→6 后由 58 降为 55
        obs_dim = 6 + 6 + 6 + 1 + 1 + 1 + 3 + 4 + 6 + 1 + 3 + 4 + 1 + 1 + 6 + 4 + 1
        self.observation_space = spaces.Box(  # 无界 Box，由 SB3 VecNormalize 负责归一化
            low=-np.inf,
            high=np.inf,
            shape=(obs_dim,),
            dtype=np.float32
        )
    
    def reset(self, seed: Optional[int] = None) -> Tuple[np.ndarray, Dict[str, Any]]:
        """重置环境"""
        super().reset(seed=seed)
        
        # 重置MuJoCo数据
        mujoco.mj_resetData(self.model, self.data)
        
        # 使用安全的初始配置（P0-2：传入环境 RNG，初始位形由 seed 决定、可复现）
        # Task3：UR5e 六关节初始位形（围绕 home 小范围随机；home 时 pad 已悬于 cube 上方）
        joint_positions = self.singularity_handler.generate_safe_initial_config(self.np_random)
        self.data.qpos[self.arm_joint_ids] = joint_positions
        
        # 初始化夹爪（2F-85 全开 = 各 rq_ 关节 qpos=0；driver=0 → 衬垫间距 ~0.0936）
        if hasattr(self, 'gripper_joint_ids') and self.gripper_joint_ids:
            for joint_id in self.gripper_joint_ids:
                addr = self.model.jnt_qposadr[joint_id] if joint_id < self.model.njnt else joint_id
                if addr < len(self.data.qpos):
                    self.data.qpos[addr] = 0.0  # 全开（P3：自动闭合状态机从 open 开始）
        # 夹爪 actuator 初始为张开（ctrl=0；menagerie 2F-85：0=全开, 255=闭合）
        if len(self.data.ctrl) > self.gripper_actuator_idx:
            self.data.ctrl[self.gripper_actuator_idx] = 0.0
        
        # P0-2：采样物体初始位置（use_fixed_position=True 固定 (0.5,0)；False 在
        # workspace_bounds 的 X/Y 内随机，Z 固定为落定高度 object_rest_z）
        object_pos = self._sample_object_position()
        self._set_object_position(object_pos)
        
        # 重置episode监控
        self.episode_start_time = time.time()
        self.episode_singularity_count = 0
        
        # 重置任务状态（object_position 在 mj_forward 之后才读取，
        # 否则 data.xpos 是上一 episode 的陈旧值）
        self.task_state = {
            'is_grasped': False,
            'grasp_success': False,
            'episode_steps': 0,
            'episode_reward': 0.0,  # P1: 累计本 episode 奖励，用于监控输出
            'episode_breakdown': {**{k: 0.0 for k in REWARD_KEYS}, 'r_singularity': 0.0},  # Task3: 奖励分项累计（事后回放用；含奇异惩罚）
            'previous_joint_pos': joint_positions.copy(),
            # P4: 抬升阶段状态
            'lift_active': False,
            'lift_target_z': None,
            'lift_origin_xy': None,
            'lift_steps': 0,
            'lift_done': False,
            'lift_aborted': False
        }

        # P3: 夹爪近距自动闭合状态机初始化为 open（RL 不再直接控制肌腱）
        self.gripper_phase = 'open'
        self.grasp_retry_hold = False
        self.closing_steps = 0
        self.closing_width_prev = None

        # Task3 速度模式：目标末端速度（每决策步由动作增量设定）
        self._vel_target = np.zeros(6)

        # Task3 B 方案（奇异 = 失败信号）：连续奇异步计数 + 单步奇异惩罚
        self._singularity_streak = 0
        self._singularity_penalty = 0.0

        # Task3 抖动抑制：动作平滑状态
        self._v_smooth = np.zeros(6)
        # Task3 sim-to-real 动力学匹配：速度环加速度前馈用上一物理步 dq
        self._dq_prev = np.zeros(6)

        # P4: 清除上一 episode 的"成功"叠加文本，重置去重标志
        self._success_overlay_shown = False
        if self.viewer_handle is not None:
            try:
                self.viewer_handle.clear_texts()
            except Exception:
                pass
        
        # 前向动力学（更新 xpos/xquat 等派生量）
        mujoco.mj_forward(self.model, self.data)

        # Task3 防初始碰撞：随机初始位形可能使 pad 低于桌面顶（手指插进桌面），
        # 接触力会瞬间数值爆炸（qvel 数百 rad/s）。校验 pad 高度，若进入桌面则回退 home。
        if (getattr(self, 'pad_site_left_id', -1) >= 0
                and getattr(self, 'pad_site_right_id', -1) >= 0):
            table_top = float(getattr(self.grasping_config, 'table_top_z', 0.30))
            pad_z = min(self.data.site_xpos[self.pad_site_left_id][2],
                        self.data.site_xpos[self.pad_site_right_id][2])
            if pad_z < table_top + 0.01:
                joint_positions = self.singularity_handler.safe_config.copy()
                self.data.qpos[self.arm_joint_ids] = joint_positions
                mujoco.mj_forward(self.model, self.data)
                self.task_state['previous_joint_pos'] = joint_positions.copy()

        # P1.3 势能塑形：prev_state 置为初始状态（而非 None），保证首步也按
        # Φ(s_prev) − γ·Φ(s) 的势能差计算，episode 回报望远镜式相消、有界
        self.prev_state = get_proprioceptive_state(self.data, self.model, self)

        # P0-2：目标位置 = 物体实际位置，与观测/奖励保持一致
        # （消除此前 target_pos=(0.5,0,0.3) 与真实物体位置不符的偏差）
        self.target_pos = self._get_object_position().copy()
        self.target_quat = np.array([1, 0, 0, 0])
        self.task_state['object_position'] = self.target_pos.copy()
        
        # 获取初始观察
        observation = self._get_observation()
        info = self._get_info()
        
        self.logger.debug(f"环境重置完成，物体位置: {self.task_state['object_position']}")
        
        return observation, info
    #########################################
    def step(self, action: np.ndarray) -> Tuple[np.ndarray, float, bool, bool, Dict[str, Any]]:
        """执行一步动作：速度模式(默认)解析增量->目标速度->速度IK+PID 或 位置模式写伺服 ctrl，然后子步进物理仿真 -> 更新任务/奖励/观测"""
        # 获取当前状态信息
        current_state = get_proprioceptive_state(self.data, self.model, self)
        
        # ---- Task3 动作执行：速度模式（默认）或位置模式（对照） ----
        if self.control_mode == 'velocity':
            # RL 动作 = 末端位姿增量（每决策步）-> 目标末端速度 v = Δ/T（T=决策周期 s）
            a = np.asarray(action, dtype=np.float64).ravel()
            dim = int(getattr(self.grasping_config, 'action_space_dim', 6))
            if a.size < dim:
                a = np.concatenate([a, np.zeros(dim - a.size)])
            pos_delta = np.clip(a[:3], -self.grasping_config.max_ee_delta,
                                self.grasping_config.max_ee_delta)
            ori_delta = (np.clip(a[3:6], -self.grasping_config.max_orient_delta,
                                 self.grasping_config.max_orient_delta)
                         if dim >= 6 else np.zeros(3))
            T = self.substeps * max(1, self.action_repeat) * self.model.opt.timestep
            # Task3 抖动抑制 B：RL 动作一阶低通平滑（消除决策步跳变冲击；对应 MoveIt Servo smoothing_filter）
            _alpha = float(getattr(self.grasping_config, 'action_smoothing_alpha', 0.3))
            v_raw = np.concatenate([pos_delta / T, ori_delta / T])
            self._v_smooth = _alpha * v_raw + (1.0 - _alpha) * self._v_smooth
            self._vel_target = self._v_smooth
            # Task3 冲击抑制 A：接近限速——pad 距 cube < approach_speed_dist 时限制末端线速度，防高速撞击弹飞。
            # 用 pad→cube 距离而非 base_mount→cube（home 位形后者 0.17 会全程触发限速导致实现率暴跌，
            # 见 docs/sim_to_real动力学匹配方案.md）
            _pad_c = (self.data.site_xpos[self.pad_site_left_id] + self.data.site_xpos[self.pad_site_right_id]) / 2.0
            _dist = float(np.linalg.norm(_pad_c - self._get_object_position()))
            _v_lim = float(getattr(self.grasping_config, 'approach_speed_limit', 0.12))
            if _dist < float(getattr(self.grasping_config, 'approach_speed_dist', 0.06)):
                _vn = float(np.linalg.norm(self._vel_target[:3]))
                if _vn > _v_lim:
                    self._vel_target[:3] *= _v_lim / _vn
            # P4: 抬升阶段覆盖 RL 动作为"竖直上抬"速度（夹爪由状态机保持闭合）
            if (self.task_state.get('lift_active')
                    and not self.task_state.get('lift_done')
                    and not self.task_state.get('lift_aborted')):
                v_z = float(getattr(self.grasping_config, 'lift_speed', 0.01)) / T
                self._vel_target = np.array([0.0, 0.0, v_z, 0.0, 0.0, 0.0])
            # 方案B+2(2026-08-22): closing 阶段覆盖 RL 动作——pad 向 cube 中心 XY 缓慢微调到位后保持。
            # 方案B（静止）实证解决"策略不停手→宽度不收敛→空闭合"（PPO_90 卡点）；
            # 方案2（微调）解决"pad 停的位置差 1-2cm 手指夹不到"（PPO_92 卡点：进入 closing 100%
            # 但 closing→closed 仅 18.3%，cube 半宽 2cm）。
            # 微调限速 closing_align_speed（0.02m/s = 每决策步 ~0.8mm），偏差 < closing_align_tol 即保持，
            # 避免挤压撞飞 cube。closing 与 lift 不会同时出现（lift 需 grasp_success=closed 之后）。
            if self.gripper_phase == 'closing':
                _gc = (self.data.site_xpos[self.pad_site_left_id]
                       + self.data.site_xpos[self.pad_site_right_id]) / 2.0
                _obj = self._get_object_position()
                _err = _obj[:2] - _gc[:2]
                _z_err = _obj[2] - _gc[2]   # 方案①(2026-08-23): Z 对齐——pad 悬空时微降到接触高度
                _tol = float(getattr(self.grasping_config, 'closing_align_tol', 0.005))
                _tol_z = float(getattr(self.grasping_config, 'closing_align_z_tol', 0.003))
                _v = float(getattr(self.grasping_config, 'closing_align_speed', 0.02))
                _v_target = np.zeros(6)
                if np.linalg.norm(_err) > _tol:
                    _v_target[:2] = np.clip(_err / T, -_v, _v)
                # 方案①: pad 未接触 cube 时 Z 微降（解决 PPO_93 失败模式2：进 closing 但 pad 悬空
                # 在 cube 顶面上方 ~1cm，手指合拢碰不到 → 空闭合）；已接触则保持
                _cf = self._get_grasp_contact_force()
                if abs(_z_err) > _tol_z and _cf < 0.1:
                    _v_target[2] = np.clip(_z_err / T, -_v, _v)
                self._vel_target = _v_target
        else:
            # 位置模式（旧行为）：DLS IK -> 位置伺服目标（仅对 position-servo 型 XML 有效）
            safe_action = self.action_wrapper.apply(action, current_state)
            if (self.task_state.get('lift_active')
                    and not self.task_state.get('lift_done')
                    and not self.task_state.get('lift_aborted')):
                safe_action = self._lift_action(current_state)
            self.data.ctrl[:7] = safe_action['joint_commands']
        
        # P3: 夹爪由近距自动闭合状态机驱动（RL 不再输出肌腱命令）
        self._update_gripper_control()
        
        # 检查当前关节位置是否奇异（使用警告控制）
        current_joint_pos = self.data.qpos[self.arm_joint_ids].copy()
        current_time = time.time()
        is_singular, singularity_type, singularity_score, should_warn = self.singularity_handler.detect_singularity_with_warning_control(current_joint_pos, current_time)
        
        if is_singular:
            self.episode_singularity_count += 1  # 记录奇异点次数
            # Task3 B: 累计连续奇异 + 单步惩罚（短暂路过不终止；困住过久由 truncated 兜底）
            self._singularity_streak += 1
            self._singularity_penalty = float(getattr(self.grasping_config, 'singularity_penalty', -0.5))
            if should_warn:
                self.logger.warning(f"检测到奇异点 {singularity_type} (程度: {singularity_score:.3f})")
            if self.control_mode != 'velocity':
                safe_joint_action = self.singularity_handler.get_safe_config(current_joint_pos)
                self.data.ctrl[:7] = safe_joint_action
            else:
                self._vel_target = np.zeros(6)  # 速度模式奇异 -> 停（阻尼伪逆本身鲁棒，仅防御）
        else:
            self._singularity_streak = 0
            self._singularity_penalty = 0.0
        
        # 模拟一步（每个 RL 决策步执行 action_repeat 个控制周期，每周期 substeps 个物理子步）
        # Task3 速度模式：每物理步刷新速度目标，执行"速度 PID + 重力补偿前馈"（arm 为 motor）。
        # P5 修正（位置模式）：lift 阶段 ctrl_cycles=1，保持已验证的稳定抬升曲线。
        ctrl_cycles = 1 if self.task_state.get('lift_active') else self.action_repeat
        if self.control_mode == 'velocity':
            kp = np.asarray(getattr(self.grasping_config, 'velocity_kp', (150.0,) * 6), dtype=float)
            kd = np.asarray(getattr(self.grasping_config, 'velocity_kd', (15.0,) * 6), dtype=float)
            lam = float(getattr(self.grasping_config, 'velocity_ik_lam', 0.05))
            lim = np.array([150.0, 150.0, 150.0, 28.0, 28.0, 28.0])  # 臂 forcerange（与 XML 一致）
            arm_dof = self.arm_joint_ids
            # 并行改造(2026-08-22) IK 去重（每决策步物理计算量的主要热点）：
            #   - mj_fullM 全质量矩阵移到决策步外算 1 次（决策步内位形变化极小，惯性矩阵近似不变）
            #   - velocity_ik（雅可比+阻尼伪逆）改为每控制周期 1 次（substeps 次，原 substeps*ctrl_cycles 次）
            # 子步间只做 PID + 重力/惯性前馈 + mj_step。目标末端速度一整个决策步内不变，
            # 该优化不改变指令语义；加速度前馈 dq_ff 在控制周期边界仍产生脉冲（更接近实机速度环语义）。
            _Mfull = np.zeros((self.model.nv, self.model.nv))
            mujoco.mj_fullM(self.model, self.data, _Mfull)   # (model, data, dst)：解压压缩对称阵后取 arm 子块
            M_eff = _Mfull[np.ix_(arm_dof, arm_dof)]
            for _ in range(ctrl_cycles):
                dq = ik.velocity_ik(self.model, self.data, self.end_effector_id,
                                    self._vel_target, arm_dof, lam)
                for _ in range(self.substeps):
                    qvel = self.data.qvel[arm_dof]
                    # Task3 sim-to-real 动力学匹配：加速度前馈（计算力矩前馈）——
                    # M_eff·d(dq)/dt 抵消加速所需力矩，速度环瞬间跟上目标（匹配实机 UR 速度环的前馈）。
                    # 纯 P 环（kp·err）在大负载下 0.04s 决策周期内速度建立慢（实现率仅 ~18%），
                    # 加前馈后目标 ~90%+（见 docs/sim_to_real动力学匹配方案.md）。
                    dq_ff = (dq - self._dq_prev) / self.model.opt.timestep
                    self._dq_prev = dq.copy()
                    tau = (kp * (dq - qvel) - kd * qvel
                           + M_eff @ dq_ff
                           + self.data.qfrc_bias[arm_dof])   # 速度PD + 粘性阻尼 + 惯性前馈 + 重力前馈
                    self.data.ctrl[arm_dof] = np.clip(tau, -lim, lim)
                    mujoco.mj_step(self.model, self.data)
        else:
            for _ in range(self.substeps * ctrl_cycles):
                mujoco.mj_step(self.model, self.data)
        
        # 更新任务状态
        self.task_state['episode_steps'] += 1
        self.task_state['object_position'] = self._get_object_position()
        
        # 按配置间隔显示实时画面
        if self.render_gui and self.task_state['episode_steps'] % self.render_interval == 0:
            self.render(mode='human')
        
        # 检查是否从奇异点恢复（关节目标与当前位形一致且非奇异即视为恢复）
        if self.task_state['previous_joint_pos'] is not None:
            recovered = self.singularity_handler.check_singularity_recovery(
                self.data.qpos[self.arm_joint_ids], self.task_state['previous_joint_pos']
            )
            if recovered:
                # 只在调试模式下输出恢复信息
                if self.logger.isEnabledFor(logging.DEBUG):
                    self.logger.debug("从奇异点恢复")
        
        # 更新前一步关节位置
        self.task_state['previous_joint_pos'] = self.data.qpos[self.arm_joint_ids].copy()
        
        # P3: 夹爪自动闭合状态机 + 抓取成功判定
        was_grasped = self.task_state['grasp_success']
        prev_phase = self.gripper_phase  # 方案②(2026-08-23): 记录上一相位，检测 closing 上升沿
        self._update_gripper_state()

        # P4: 抬升阶段推进——抓取成功即开始抬升；每步计数 + 到位判定
        # 2026-08-23: 受 lift_enabled 控制（默认 False 不抬升——成功当步 episode 结束，
        # 训练步数只计到成功抓取；部署/验证需要抬升验证时设 True）
        if (self.task_state['grasp_success']
                and not self.task_state.get('lift_active')
                and not self.task_state.get('lift_done')
                and not self.task_state.get('lift_aborted')
                and float(getattr(self.grasping_config, 'lift_enabled', False))
                and float(getattr(self.grasping_config, 'lift_height', 0.0)) > 0.0):
            ee = self._get_end_effector_position()
            self.task_state['lift_active'] = True
            self.task_state['lift_target_z'] = float(ee[2]) + float(self.grasping_config.lift_height)
            self.task_state['lift_origin_xy'] = np.array([float(ee[0]), float(ee[1])])
            self.task_state['lift_steps'] = 0
            self._show_success_overlay()

        if self.task_state.get('lift_active'):
            self.task_state['lift_steps'] = self.task_state.get('lift_steps', 0) + 1
            ee_z = float(self._get_end_effector_position()[2])
            if ee_z >= (float(self.task_state.get('lift_target_z', ee_z))
                        - float(getattr(self.grasping_config, 'lift_tol', 0.015))):
                self.task_state['lift_done'] = True
        
        # 获取新的状态
        new_state = get_proprioceptive_state(self.data, self.model, self)
        
        # 计算奖励（P1: 新奖励函数，传入真实物理接触力与抓取事件）
        contact_force = self._get_grasp_contact_force()
        grasp_info = {
            'is_grasped': self.task_state['is_grasped'],
            'grasp_success': self.task_state['grasp_success'],
            # P4: 成功上升沿 + 抬升阶段标志——奖励只在成功当步一次性发放，
            # 抬升期间 grasp_success 持续为 True，若仍按电平式发放会重复累加事件奖励
            'grasp_success_rising': bool(self.task_state['grasp_success'] and not was_grasped),
            # 方案②(2026-08-23): 进入 closing 上升沿——中间里程碑奖励（激励 pad 精确对齐到触发区）
            'closing_rising': bool(self.gripper_phase == 'closing' and prev_phase != 'closing'),
            'lift_active': bool(self.task_state.get('lift_active', False)),
            'contact_force': contact_force,
        }
        reward = calculate_reward(new_state, self.prev_state, grasp_info, self.reward_config)
        # Task3 B: 奇异单步惩罚并入本步奖励（RL 可感知）
        reward += self._singularity_penalty
        self.task_state['episode_reward'] += reward
        # Task3: 奖励分项累计（reward_breakdown 一次计算，calculate_reward 即其 REWARD_KEYS 之和，
        # 避免重复计算；用于训练结束后的"每个奖励项"事后回放）
        parts = reward_breakdown(new_state, self.prev_state, grasp_info, self.reward_config)
        for k in REWARD_KEYS:
            self.task_state['episode_breakdown'][k] += parts[k]
        self.task_state['episode_breakdown']['r_singularity'] += self._singularity_penalty
        
        # 更新前一步状态
        self.prev_state = current_state
        
        # 获取观察和信息
        observation = self._get_observation()
        terminated = self._is_done()
        # P4: max_steps 只计抬升前步数——抬升是抓取成功后的自动阶段（0.003 需 ~494 步），
        # 不应消耗 RL 的 episode 预算；RL 阶段（接近+抓取）仍须在 max_steps 内完成。
        pre_lift_steps = self.task_state['episode_steps'] - self.task_state.get('lift_steps', 0)
        truncated = pre_lift_steps >= self.grasping_config.max_steps
        # Task3 B: 连续困在奇异超过阈值 -> 提前终止（对应实机"持续不可达 -> 任务失败"，不再白耗 300 步）
        if self._singularity_streak >= int(getattr(self.grasping_config, 'singularity_max_streak', 50)):
            truncated = True
        info = self._get_info()
        
        # 如果episode结束且有监控器，记录episode信息（P1: 传累计奖励而非最后一步奖励）
        if (terminated or truncated) and self.training_monitor is not None:
            episode_time = time.time() - self.episode_start_time
            self.training_monitor.log_episode(
                episode=self.task_state.get('episode_number', 0),
                reward=self.task_state['episode_reward'],
                length=self.task_state['episode_steps'],
                success=self.task_state['grasp_success'],
                singularity_count=self.episode_singularity_count,
                episode_time=episode_time,
                breakdown=dict(self.task_state.get('episode_breakdown', {}))
            )
        
        return observation, reward, terminated, truncated, info
    
    def _sample_object_position(self) -> np.ndarray:
        """采样物体初始位置

        P0-2：use_fixed_position=True 时返回固定桌面位置 (0.5, 0, object_rest_z)；
        False 时在 workspace_bounds 的 X/Y 范围内随机（Z 固定为落定高度），
        随机数取自 self.np_random，从而完全由 reset(seed) 决定、可复现。
        返回的位置即物体“落定后”的桌面位置（物体直接放在桌面上，不会漂移）。
        """
        x, y = float(getattr(self.grasping_config, 'object_fixed_pos', (0.5, 0.0))[0]), \
               float(getattr(self.grasping_config, 'object_fixed_pos', (0.5, 0.0))[1])
        if not self.grasping_config.use_fixed_position:
            (x_lo, x_hi), (y_lo, y_hi), _ = self.grasping_config.workspace_bounds
            x = float(self.np_random.uniform(x_lo, x_hi))
            y = float(self.np_random.uniform(y_lo, y_hi))
        z = float(getattr(self.grasping_config, 'object_rest_z', 0.02))
        return np.array([x, y, z], dtype=float)

    def _set_object_position(self, position: Optional[np.ndarray] = None):
        """设置物体位置（固定位置）

        P0-1 兼容性修复：目标物体 target_cube 已加入 <freejoint/>，
        此前该循环因物体无关节永远无法命中（静默无效）；现在会命中 freejoint，
        需要把“位置(3)+四元数(4)”完整写入，否则只写一个 qpos 分量会损坏物体位姿。
        """
        if position is None:
            position = np.array([0.5, 0.0, getattr(self.grasping_config, 'object_rest_z', 0.02)])
        position = np.asarray(position, dtype=float)

        if self.object_id is not None:
            # 找到物体的自由关节并写入完整位姿
            # P0-1 兼容性修复 + Task3：freejoint 存在时 qpos 地址=jnt_qposadr ≠ 关节 ID，
            # 必须用 qposadr 索引（UR5e 14 个铰链后才是 freejoint，此前的 i 索引会写坏夹爪/桌位姿）
            for i in range(self.model.njnt):
                if self.model.jnt_bodyid[i] == self.object_id:
                    jt = self.model.jnt_type[i]
                    if jt == mujoco.mjtJoint.mjJNT_FREE:
                        qadr = self.model.jnt_qposadr[i]
                        self.data.qpos[qadr:qadr + 3] = position          # 位置
                        self.data.qpos[qadr + 3:qadr + 7] = np.array([1, 0, 0, 0])  # 四元数
                    elif jt == mujoco.mjtJoint.mjJNT_BALL:
                        self.data.qpos[self.model.jnt_qposadr[i]:self.model.jnt_qposadr[i] + 4] = np.array([1, 0, 0, 0])
                    else:
                        self.data.qpos[self.model.jnt_qposadr[i]] = position[0]
                    break

        self.logger.debug(f"设置目标物体位置: {position}")
    
    def _get_object_position(self) -> np.ndarray:
        """获取物体位置"""
        if self.object_id is not None:
            return self.data.xpos[self.object_id].copy()
        else:
            # 如果没有找到物体，返回XML中定义的固定位置
            return np.array([0.5, 0.0, 0.1])
    
    def _update_gripper_state(self):
        """P3: 夹爪近距自动闭合状态机 + 抓取成功判定（手指被物体挡住合不上）。

        状态机：
          open    -> closing  ：末端距物体中心 < gripper_close_distance
          closing -> closed   ：开度收敛且 > grasp_min_width 且双指接触（抓取成功）
          closing -> open     ：开度收敛但 ≤ grasp_min_width（空闭合）或超时 -> 重开重试
          closed  -> open     ：开度滑脱 ≤ grasp_min_width 或失接触 -> 重开重试
        重开后的滞回：grasp_retry_hold 期间须离开物体到 > gripper_open_distance 才允许再次闭合，
        避免策略悬停在物体旁导致 open<->closing 抖振。
        """
        cfg = self.grasping_config
        width = self._get_gripper_width()
        dist = float(np.linalg.norm(self._get_end_effector_position()
                                    - self._get_object_position()))
        contact = self._check_grasp_contact()

        phase = self.gripper_phase
        if phase == 'open':
            if self.grasp_retry_hold:
                if dist > cfg.gripper_open_distance:
                    self.grasp_retry_hold = False
            else:
                # Task3 闭合时机（对齐触发）：物体中心 XY/Z 均与夹爪中心(左右 pad 中点)对齐才闭合。
                # 语义：物体真正落在 pad 之间才夹（比"距离<close_distance"更严格，防侧方误触发；
                # 详见 docs/RL_DEPLOY_CONTRACT.md §6.1）
                _gc = (self.data.site_xpos[self.pad_site_left_id]
                       + self.data.site_xpos[self.pad_site_right_id]) / 2.0
                _obj = self._get_object_position()
                _align_xy = float(np.linalg.norm(_obj[:2] - _gc[:2]))
                _align_z = float(abs(float(_obj[2]) - float(_gc[2])))
                if (_align_xy < float(getattr(cfg, 'grasp_align_xy_tol', 0.03))
                        and _align_z < float(getattr(cfg, 'grasp_align_z_tol', 0.02))):
                    phase = 'closing'
                    self.closing_steps = 0
                    self.closing_width_prev = width
        elif phase == 'closing':
            self.closing_steps += 1
            dw = abs(width - self.closing_width_prev) if self.closing_width_prev is not None else 1e9
            self.closing_width_prev = width
            if self.closing_steps >= cfg.close_confirm_steps and dw <= cfg.close_width_tol:
                if width > cfg.grasp_min_width and contact:
                    phase = 'closed'          # 手指被物体挡住合不上 -> 抓取成功
                else:
                    phase = 'open'            # 空闭合 -> 重开重试
                    self.grasp_retry_hold = True
            elif self.closing_steps > cfg.max_close_steps:
                phase = 'open'
                self.grasp_retry_hold = True
        elif phase == 'closed':
            if width <= cfg.grasp_min_width or not contact:
                phase = 'open'                # 滑脱/失夹 -> 重开重试
                self.grasp_retry_hold = True
                if self.task_state.get('lift_active'):
                    self.task_state['lift_aborted'] = True  # P4: 抬升中失夹 -> 判定失败

        self.gripper_phase = phase
        self.task_state['is_grasped'] = (phase == 'closed')
        self.task_state['grasp_success'] = (phase == 'closed')

    def _update_gripper_control(self):
        """P3: 按状态机相位写夹爪控制（fingers_actuator，Task3 索引 gripper_actuator_idx）。
        menagerie 2F-85 实测（/tmp/test_fingers）：ctrl=0 -> 全开(width 0.0934)；ctrl=255 -> 闭合(0.0458)。
        open -> 0(张开)，closing/closed -> 255(闭合)。"""
        idx = getattr(self, 'gripper_actuator_idx', 6)
        if idx < len(self.data.ctrl):
            if self.gripper_phase == 'open':
                self.data.ctrl[idx] = 0.0
            else:
                # Task3 冲击抑制 B + 阻抗更轻柔：闭合速度斜坡——closing 每决策步 ctrl += gripper_close_speed，
                # pad 慢速靠近物体减小接触冲击（接触力超 grasp_force_limit 后保持中等压力）。
                # 对应实机 2F-85 set_gripper_speed + set_gripper_force
                f_lim = float(getattr(self.grasping_config, 'grasp_force_limit', 20.0))
                if self._get_grasp_contact_force() > f_lim:
                    self.data.ctrl[idx] = min(self.data.ctrl[idx], 160.0)
                else:
                    _step = float(getattr(self.grasping_config, 'gripper_close_speed', 50.0))
                    self.data.ctrl[idx] = min(self.data.ctrl[idx] + _step, 255.0)

    def _get_gripper_width(self) -> float:
        """夹爪开口宽度 = 两衬垫测量 site 的欧氏距离（Task3：2F-85 全开 ~0.0936，闭合 ~0.02）"""
        if (getattr(self, 'pad_site_left_id', -1) >= 0
                and getattr(self, 'pad_site_right_id', -1) >= 0):
            return float(np.linalg.norm(self.data.site_xpos[self.pad_site_left_id]
                                        - self.data.site_xpos[self.pad_site_right_id]))
        return 0.0

    def _finger_cube_geoms(self) -> Tuple[set, set, set]:
        """左手指 / 右手指 / 目标物体的 geom id 集合（惰性缓存）

        Task3：2F-85 手指按 body 名含 pad/finger 且含 left/right 区分（rq_left_pad、rq_right_pad、
        rq_*_silicone_pad 等）；成功抓取要求左右两指都接触物体，因此不再合并成一个集合。
        """
        if (getattr(self, '_left_finger_geom_ids', None) is None
                or getattr(self, '_right_finger_geom_ids', None) is None
                or getattr(self, '_cube_geom_ids', None) is None):
            left_ids, right_ids, cube_ids = set(), set(), set()
            for i in range(self.model.ngeom):
                body_id = self.model.geom_bodyid[i]
                body_name = mujoco.mj_id2name(self.model, mujoco.mjtObj.mjOBJ_BODY, body_id) or ''
                lower = body_name.lower()
                if 'pad' in lower or 'finger' in lower:
                    if 'left' in lower:
                        left_ids.add(i)
                    elif 'right' in lower:
                        right_ids.add(i)
                if self.object_id is not None and body_id == self.object_id:
                    cube_ids.add(i)
            self._left_finger_geom_ids = left_ids
            self._right_finger_geom_ids = right_ids
            self._cube_geom_ids = cube_ids
        return self._left_finger_geom_ids, self._right_finger_geom_ids, self._cube_geom_ids

    def _check_grasp_contact(self) -> bool:
        """真实物理接触：左右两个夹爪手指都需与物体发生接触（单指碰到不算成功）"""
        left_ids, right_ids, cube_ids = self._finger_cube_geoms()
        if not left_ids or not right_ids or not cube_ids:
            return False
        left_hit, right_hit = False, False
        for c in self.data.contact:
            if c.geom1 in cube_ids:
                if c.geom2 in left_ids:
                    left_hit = True
                elif c.geom2 in right_ids:
                    right_hit = True
            elif c.geom2 in cube_ids:
                if c.geom1 in left_ids:
                    left_hit = True
                elif c.geom1 in right_ids:
                    right_hit = True
            if left_hit and right_hit:
                return True
        return False

    def _get_grasp_contact_force(self) -> float:
        """左、右手指与物体接触的总法向力 (N)，用于接触奖励

        法向力取自约束力 data.efc_force[c.efc_address]（接触的第一个约束即法向）。
        """
        left_ids, right_ids, cube_ids = self._finger_cube_geoms()
        if not left_ids or not right_ids or not cube_ids:
            return 0.0
        finger_ids = left_ids | right_ids
        total = 0.0
        for c in self.data.contact:
            if (c.geom1 in finger_ids and c.geom2 in cube_ids) or \
               (c.geom2 in finger_ids and c.geom1 in cube_ids):
                if c.efc_address is not None and 0 <= c.efc_address < len(self.data.efc_force):
                    total += abs(float(self.data.efc_force[c.efc_address]))
        return total
    
    def _get_observation(self) -> np.ndarray:
        """组合本体感知状态为 58 维观测向量"""
        # 获取本体感知状态
        state = get_proprioceptive_state(self.data, self.model, self)

        # P2-2: 相对接近姿态四元数 —— 从当前手姿态转到目标抓取姿态的旋转（在手坐标系表达）。
        # 目标抓取姿态 = 物体朝向(target_orientation) ⊗ 朝下接近旋转(Q_APPROACH_DOWN)，
        # 与 reward.py 的 _orient_gain 同款约定；完全对齐时 q_rel=[1,0,0,0]（identity）。
        # 规范化符号（保证 w≥0）消除四元数双覆盖歧义（q 与 −q 表示同一姿态）。
        # 此前观测只有绝对的四元数（ee_orientation / target_orientation），策略需自行合成
        # 相对转角，方向任务难学；显式给出后策略可直接读取对齐误差。
        q_ee = np.asarray(state['ee_orientation'], dtype=float)
        q_obj = np.asarray(state['target_orientation'], dtype=float)
        q_target = ik.quat_mul(q_obj, Q_APPROACH_DOWN) if q_obj.size >= 4 else Q_APPROACH_DOWN
        q_rel = ik.quat_mul(q_target, ik.quat_inv(q_ee))
        if q_rel[0] < 0.0:
            q_rel = -q_rel
        # P3: 夹爪状态机相位（open=0 / closing=1 / closed=2），显式告知策略当前夹爪阶段
        _phase_map = {'open': 0.0, 'closing': 1.0, 'closed': 2.0}
        grasp_phase = _phase_map.get(self.gripper_phase, 0.0)

        # 组合观察向量
        observation = np.concatenate([
            state['joint_positions'],      # 7: 关节位置
            state['joint_velocities'],     # 7: 关节速度
            state['joint_torques'],        # 7: 关节力矩
            [state['tendon_position']],    # 1: 肌腱位置
            [state['tendon_velocity']],    # 1: 肌腱速度
            [state['tendon_tension']],     # 1: 肌腱张力
            state['ee_position'],          # 3: 末端位置
            state['ee_orientation'],       # 4: 末端方向
            state['ee_velocity'],          # 6: 末端速度
            [state['gripper_state']],      # 1: 夹爪状态
            state['target_position'],      # 3: 目标位置
            state['target_orientation'],   # 4: 目标方向
            [state['manipulability']],     # 1: 可操作度
            [state['contact_force']],      # 1: 接触力
            state['left_finger_force'],    # 3: 左手指力
            state['right_finger_force'],   # 3: 右手指力
            q_rel,                          # 4: 相对接近姿态四元数 (P2-2)
            [grasp_phase]                 # 1: 夹爪状态机相位 (P3)
        ])
        
        return observation.astype(np.float32)
    
    def _get_end_effector_position(self) -> np.ndarray:
        """获取末端执行器位置"""
        if self.end_effector_id is not None:
            # 确保前向动力学已经计算
            if len(self.data.xpos) > self.end_effector_id:
                return self.data.xpos[self.end_effector_id].copy()
            else:
                # 如果xpos数组还没有计算，返回机械臂前方的默认位置
                return np.array([0.4, 0.0, 0.2])
        else:
            # 如果没有找到末端执行器，使用最后一个关节的位置
            if len(self.data.xpos) > 0:
                return self.data.xpos[-1].copy()
            else:
                # 如果xpos为空，返回机械臂前方的默认位置
                return np.array([0.4, 0.0, 0.2])

    def _lift_action(self, current_state) -> dict:
        """P4: 抬升动作——目标末端位置 = 当前 xy + z 上抬 lift_speed，IK 反解为关节伺服目标。

        保持当前朝向不变（顶抓姿态已朝下）；IK 误差超 ik_error_hold 时保持当前位形。
        """
        ee = np.asarray(current_state.get('ee_position', np.zeros(3)), dtype=float)
        target_z = float(self.task_state.get('lift_target_z', ee[2] + 0.2))
        z_step = float(getattr(self.grasping_config, 'lift_speed', 0.01))
        new_z = ee[2] + min(z_step, max(0.0, target_z - ee[2]))
        # 锚定抬升起点 XY（起点在物体正上方）：若跟随当前 XY，伺服/重力的每步
        # 横向误差会累积并带偏物体导致滑脱；锚定后抬升是真正垂直的轨迹
        origin = self.task_state.get('lift_origin_xy')
        if origin is None:
            origin = np.array([ee[0], ee[1]])
        target_pos = np.array([float(origin[0]), float(origin[1]), new_z])
        cur_quat = np.asarray(current_state.get('ee_orientation',
                                                np.array([1.0, 0.0, 0.0, 0.0])),
                              dtype=float)
        q_target, err = ik.solve_ik(self.model, self.data, self.end_effector_id,
                                    target_pos, cur_quat, self.arm_joint_ids)
        if err > float(getattr(self.grasping_config, 'ik_error_hold', 0.02)):
            q_target = self.data.qpos[self.arm_joint_ids].copy()
        return {'joint_commands': q_target}
    
    def _is_done(self) -> bool:
        """判断 episode 是否结束（成功抓取+抬升完成 / 抬升失败 / 超过 max_steps 超时）"""
        # P4: 抬升阶段——到位(成功) / 失夹或超时(失败) 才结束
        if self.task_state.get('lift_active'):
            if self.task_state.get('lift_done') or self.task_state.get('lift_aborted'):
                return True
            if self.task_state.get('lift_steps', 0) >= self.grasping_config.lift_max_steps:
                return True
            return False

        # 检查是否成功抓取（未启用抬升时的旧行为）
        if self.task_state['is_grasped']:
            return True
        
        # 检查是否超时
        if self.task_state['episode_steps'] >= self.grasping_config.max_steps:
            return True
        
        return False
    
    def _get_info(self) -> Dict[str, Any]:
        """获取信息"""
        end_effector_pos = self._get_end_effector_position()
        object_pos = self._get_object_position()
        
        info = {
            'episode_steps': self.task_state['episode_steps'],
            'is_grasped': self.task_state['is_grasped'],
            'grasp_success': self.task_state['grasp_success'],
            'end_effector_pos': end_effector_pos.copy(),
            'object_pos': object_pos.copy(),
            'distance_to_object': np.linalg.norm(end_effector_pos - object_pos),
            'gripper_width': self._get_gripper_width(),
            'gripper_phase': self.gripper_phase,
            'lift_active': bool(self.task_state.get('lift_active', False)),
            'lift_done': bool(self.task_state.get('lift_done', False)),
            'lift_aborted': bool(self.task_state.get('lift_aborted', False)),
            # 并行改造(2026-08-22): SubprocVecEnv 下 worker 不挂 training_monitor，
            # episode 统计由主进程 GraspingCallback 汇总；这里补两个字段让主进程能拿到完整记录
            'singularity_count': self.episode_singularity_count,
            'episode_breakdown': dict(self.task_state.get('episode_breakdown', {})),
        }
        
        return info

    def _start_native_viewer(self):
        """启动 MuJoCo 原生窗口"""
        if not self.render_gui or not self.viewer_available:
            return
        if self.viewer_handle is not None:
            return
        try:
            self.viewer_handle = mjviewer.launch_passive(
                self.model,
                self.data,
                show_left_ui=False,
                show_right_ui=False,
            )
            self.logger.info("已启动 MuJoCo 原生窗口")
        except Exception as e:
            self.viewer_handle = None
            if not self._viewer_error_logged:
                self.logger.warning(f"启动原生窗口失败: {e}")
                self._viewer_error_logged = True

    def _sync_native_viewer(self):
        """同步原生窗口显示"""
        if self.viewer_handle is None:
            return
        try:
            self.viewer_handle.sync()
        except Exception as e:
            self.viewer_handle = None
            if not self._viewer_error_logged:
                self.logger.warning(f"更新原生窗口失败: {e}")

    def _show_success_overlay(self):
        """P4: 抓取成功后，在 MuJoCo 原生窗口左上角叠加显示 'succeed'（mujoco>=3.x 用 set_texts）。"""
        if self._success_overlay_shown:
            return
        self._success_overlay_shown = True
        if not (self.render_gui and self.viewer_available and self.viewer_handle is not None):
            return
        try:
            self.viewer_handle.set_texts(
                (mujoco.mjtFontScale.mjFONTSCALE_100,
                 mujoco.mjtGridPos.mjGRID_TOPLEFT,
                 "succeed", None)
            )
            self._sync_native_viewer()
        except Exception as e:
            if not self._viewer_error_logged:
                self.logger.warning(f"显示 succeed 叠加文本失败: {e}")
                self._viewer_error_logged = True

    def render(self, mode='human'):
        """渲染环境，优先使用 MuJoCo 原生窗口"""
        if not self.render_gui:
            return None

        if self.viewer_available:
            self._start_native_viewer()
            if self.viewer_handle is not None:
                self._sync_native_viewer()
                return None

        if self.headless or self.renderer is None:
            return None

        try:
            self.renderer.update_scene(self.data)
            image = self.renderer.render()
            return image
        except Exception as e:
            self.logger.warning(f"渲染失败: {e}")
            return None
    
    def close(self):
        """关闭环境"""
        if self.viewer_handle is not None:
            try:
                self.viewer_handle.close()
            except Exception:
                pass
            self.viewer_handle = None


# Task3 迁移：兼容别名（旧代码/入口脚本引用 PandaGraspingEnv 时仍可用）
PandaGraspingEnv = GraspingEnv
