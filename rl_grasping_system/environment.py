"""
强化学习抓取环境：基于 MuJoCo 的 Panda 机械臂抓取任务环境（gymnasium.Env）。

- 动作空间：前 7 维 = 关节增量式位置伺服命令（±max_joint_delta），第 8 维 = 夹爪开度 [0,1]；
- 观测空间：本体感知状态组成的 53 维向量（见 get_proprioceptive_state / _get_observation）；
- 任务：到达物体正上方的预抓取位姿并对齐抓取姿态（手指朝下），判定 is_grasped/grasp_success；
- 控制：每步执行 1/control_freq 秒物理仿真（子步进），ctrl=位置伺服目标。
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
from reward import calculate_reward

class PandaGraspingEnv(gym.Env):
    """
    Panda机械臂抓取环境
    
    任务: 控制机械臂到达并抓取指定位置的物体
    观察空间: 本体感知状态 + 肌腱状态
    动作空间: 关节角度 + 肌腱控制
    """
    # 构造函数
    def __init__(self, grasping_config: GraspingConfig, reward_config: RewardConfig):
        super().__init__()
        
        self.grasping_config = grasping_config
        self.reward_config = reward_config
        
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
        
        # 初始化安全动作包装器
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
            
            # 尝试创建渲染器，如果失败则使用无头模式
            try:
                self.renderer = mujoco.Renderer(self.model)
                self.headless = False
                print("✅ 图形渲染模式已启用")
            except Exception as e:
                print(f"⚠️  图形渲染失败，使用无头模式: {e}")
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
            # 查找机械臂关节：只取铰链(hinge)关节，排除手指滑动关节和物体 freejoint
            self.arm_joint_names = []
            for i in range(self.model.njnt):
                joint_name = mujoco.mj_id2name(self.model, mujoco.mjtObj.mjOBJ_JOINT, i)
                if joint_name and self.model.jnt_type[i] == mujoco.mjtJoint.mjJNT_HINGE:
                    self.arm_joint_names.append(joint_name)
            
            # 查找夹爪关节
            self.gripper_joint_names = []
            self.gripper_joint_ids = []  # 添加这个属性
            for i in range(self.model.njnt):
                joint_name = mujoco.mj_id2name(self.model, mujoco.mjtObj.mjOBJ_JOINT, i)
                if joint_name and ('finger' in joint_name.lower() or 'gripper' in joint_name.lower()):
                    self.gripper_joint_names.append(joint_name)
                    self.gripper_joint_ids.append(i)  # 记录关节ID
            
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
            
            # 查找末端执行器 - 更加灵活的方法
            self.end_effector_id = -1
            end_effector_candidates = ["panda_hand", "panda_gripper", "end_effector", "gripper", "hand", "gripper_hand", "ee", "end_effector_link"]
            
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
        delta = self.grasping_config.max_joint_delta
        self.action_space = spaces.Box(
            low=np.array([-delta] * 7 + [0.0]),
            high=np.array([delta] * 7 + [1.0]),
            dtype=np.float32
        )
        
        # 观测空间 53 维：7关节位置+7关节速度+7关节力矩+1肌腱位置+1肌腱速度+1肌腱张力+
        # 3末端位置+4末端方向+6末端速度+1夹爪状态+3目标位置+4目标方向+1可操作度+1接触力+6手指力
        obs_dim = 7 + 7 + 7 + 1 + 1 + 1 + 3 + 4 + 6 + 1 + 3 + 4 + 1 + 1 + 6  # 53维
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
        joint_positions = self.singularity_handler.generate_safe_initial_config(self.np_random)
        self.data.qpos[:7] = joint_positions
        
        # 初始化夹爪位置（半开）—— freejoint 存在时 qpos 地址≠关节ID，必须用 jnt_qposadr 索引
        if hasattr(self, 'gripper_joint_ids') and self.gripper_joint_ids:
            for joint_id in self.gripper_joint_ids:
                addr = self.model.jnt_qposadr[joint_id] if joint_id < self.model.njnt else joint_id
                if addr < len(self.data.qpos):
                    self.data.qpos[addr] = 0.02  # 半开状态
        
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
            'previous_joint_pos': joint_positions.copy()
        }
        
        # 前向动力学（更新 xpos/xquat 等派生量）
        mujoco.mj_forward(self.model, self.data)

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
        """执行一步动作：安全动作包装 → 位置伺服写 ctrl → 子步进物理仿真 → 更新任务/奖励/观测"""
        # 获取当前状态信息
        current_state = get_proprioceptive_state(self.data, self.model, self)
        
        # 应用安全动作包装器
        safe_action = self.action_wrapper.apply(action, current_state)
        
        # 设置关节控制（ctrl = 位置伺服目标，由 SafeActionWrapper 给出）
        self.data.ctrl[:7] = safe_action['joint_commands']
        
        # 设置肌腱控制
        if len(self.data.ctrl) > 7:
            # 将肌腱命令映射到控制范围 ?
            tendon_control = int(safe_action['tendon_command'] * 255)
            self.data.ctrl[7] = tendon_control
        
        # 检查当前关节位置是否奇异（使用警告控制）
        current_joint_pos = self.data.qpos[:7].copy()
        current_time = time.time()
        is_singular, singularity_type, singularity_score, should_warn = self.singularity_handler.detect_singularity_with_warning_control(current_joint_pos, current_time)
        
        if is_singular:
            safe_joint_action = self.singularity_handler.get_safe_config(current_joint_pos)
            self.data.ctrl[:7] = safe_joint_action
            self.episode_singularity_count += 1  # 记录奇异点次数
            if should_warn:
                self.logger.warning(f"检测到奇异点 {singularity_type} (程度: {singularity_score:.3f})，使用渐进安全配置")
        
        # 模拟一步（控制周期内执行 substeps 个物理子步，让位置伺服收敛到目标）
        for _ in range(self.substeps):
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
                self.data.qpos[:7], self.task_state['previous_joint_pos']
            )
            if recovered:
                # 只在调试模式下输出恢复信息
                if self.logger.isEnabledFor(logging.DEBUG):
                    self.logger.debug("从奇异点恢复")
        
        # 更新前一步关节位置
        self.task_state['previous_joint_pos'] = self.data.qpos[:7].copy()
        
        # 检查抓取状态
        self._update_grasp_state()
        
        # 获取新的状态
        new_state = get_proprioceptive_state(self.data, self.model, self)
        
        # 计算奖励（P1: 新奖励函数，传入真实物理接触力与抓取事件）
        contact_force = self._get_grasp_contact_force()
        grasp_info = {
            'is_grasped': self.task_state['is_grasped'],
            'grasp_success': self.task_state['grasp_success'],
            'contact_force': contact_force,
        }
        reward = calculate_reward(new_state, self.prev_state, grasp_info, self.reward_config)
        self.task_state['episode_reward'] += reward
        
        # 更新前一步状态
        self.prev_state = current_state
        
        # 获取观察和信息
        observation = self._get_observation()
        terminated = self._is_done()
        truncated = self.task_state['episode_steps'] >= self.grasping_config.max_steps
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
                episode_time=episode_time
            )
        
        return observation, reward, terminated, truncated, info
    
    def _sample_object_position(self) -> np.ndarray:
        """采样物体初始位置

        P0-2：use_fixed_position=True 时返回固定桌面位置 (0.5, 0, object_rest_z)；
        False 时在 workspace_bounds 的 X/Y 范围内随机（Z 固定为落定高度），
        随机数取自 self.np_random，从而完全由 reset(seed) 决定、可复现。
        返回的位置即物体“落定后”的桌面位置（物体直接放在桌面上，不会漂移）。
        """
        x, y = 0.5, 0.0
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
            for i in range(self.model.njnt):
                if self.model.jnt_bodyid[i] == self.object_id:
                    jt = self.model.jnt_type[i]
                    if jt == mujoco.mjtJoint.mjJNT_FREE:
                        self.data.qpos[i:i + 3] = position          # 位置
                        self.data.qpos[i + 3:i + 7] = np.array([1, 0, 0, 0])  # 四元数
                    elif jt == mujoco.mjtJoint.mjJNT_BALL:
                        self.data.qpos[i:i + 4] = np.array([1, 0, 0, 0])
                    else:
                        self.data.qpos[i] = position[0]
                    break

        self.logger.debug(f"设置目标物体位置: {position}")
    
    def _get_object_position(self) -> np.ndarray:
        """获取物体位置"""
        if self.object_id is not None:
            return self.data.xpos[self.object_id].copy()
        else:
            # 如果没有找到物体，返回XML中定义的固定位置
            return np.array([0.5, 0.0, 0.1])
    
    def _update_grasp_state(self):
        """更新抓取状态（P1: 加入真实接触判定，成功 = 到位 + 夹紧 + 接触）"""
        end_effector_pos = self._get_end_effector_position()
        object_pos = self._get_object_position()
        distance_to_object = np.linalg.norm(end_effector_pos - object_pos)
        gripper_width = self._get_gripper_width()
        contact = self._check_grasp_contact()

        is_grasped = (
            distance_to_object < self.grasping_config.grasp_distance_threshold
            and gripper_width < self.grasping_config.gripper_closed_threshold
            and contact
        )
        self.task_state['is_grasped'] = is_grasped
        # P1: 以“到位+夹紧+接触”为成功；物体是否被抬起/搬运留到后续细化
        self.task_state['grasp_success'] = is_grasped

    def _get_gripper_width(self) -> float:
        """夹爪开口宽度 = 两个手指关节位置之和（0=闭合, 0.08=全开）"""
        width = 0.0
        if hasattr(self, 'gripper_joint_ids') and self.gripper_joint_ids:
            for joint_id in self.gripper_joint_ids:
                addr = self.model.jnt_qposadr[joint_id] if joint_id < self.model.njnt else joint_id
                if addr < len(self.data.qpos):
                    width += self.data.qpos[addr]
        return float(width)

    def _finger_cube_geoms(self) -> Tuple[set, set]:
        """手指与目标物体的 geom id 集合（惰性缓存）"""
        if getattr(self, '_finger_geom_ids', None) is None or getattr(self, '_cube_geom_ids', None) is None:
            finger_ids, cube_ids = set(), set()
            for i in range(self.model.ngeom):
                body_id = self.model.geom_bodyid[i]
                body_name = mujoco.mj_id2name(self.model, mujoco.mjtObj.mjOBJ_BODY, body_id) or ''
                if 'finger' in body_name.lower():
                    finger_ids.add(i)
                if self.object_id is not None and body_id == self.object_id:
                    cube_ids.add(i)
            self._finger_geom_ids = finger_ids
            self._cube_geom_ids = cube_ids
        return self._finger_geom_ids, self._cube_geom_ids

    def _check_grasp_contact(self) -> bool:
        """真实物理接触：手指任一 geom 与物体任一 geom 发生接触"""
        finger_ids, cube_ids = self._finger_cube_geoms()
        if not finger_ids or not cube_ids:
            return False
        for c in self.data.contact:
            if (c.geom1 in finger_ids and c.geom2 in cube_ids) or \
               (c.geom2 in finger_ids and c.geom1 in cube_ids):
                return True
        return False

    def _get_grasp_contact_force(self) -> float:
        """手指-物体接触的总法向力 (N)，用于接触奖励

        法向力取自约束力 data.efc_force[c.efc_address]（接触的第一个约束即法向）。
        """
        finger_ids, cube_ids = self._finger_cube_geoms()
        if not finger_ids or not cube_ids:
            return 0.0
        total = 0.0
        for c in self.data.contact:
            if (c.geom1 in finger_ids and c.geom2 in cube_ids) or \
               (c.geom2 in finger_ids and c.geom1 in cube_ids):
                if c.efc_address is not None and 0 <= c.efc_address < len(self.data.efc_force):
                    total += abs(float(self.data.efc_force[c.efc_address]))
        return total
    
    def _get_observation(self) -> np.ndarray:
        """组合本体感知状态为 53 维观测向量"""
        # 获取本体感知状态
        state = get_proprioceptive_state(self.data, self.model, self)
        
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
            state['right_finger_force']    # 3: 右手指力
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
    
    def _is_done(self) -> bool:
        """判断 episode 是否结束（成功抓取或超过 max_steps 超时）"""
        # 检查是否成功抓取
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
