"""
状态空间实现
包含本体感知状态和肌腱控制
"""

import numpy as np
import mujoco
from typing import Dict, Any

def calculate_finger_forces(tendon_tension: float, gripper_state: float, 
                           object_position: np.ndarray, 
                           hand_position: np.ndarray) -> tuple:
    """
    计算Panda夹爪的手指力 - 符合真实结构
    
    Args:
        tendon_tension: 肌腱张力
        gripper_state: 夹爪状态 (0-0.04m)
        object_position: 物体位置
        hand_position: 夹爪基座位置
        
    Returns:
        left_force: 左手指力 [Fx, Fy, Fz]
        right_force: 右手指力 [Fx, Fy, Fz]
    """
    # Panda夹爪参数
    finger_length = 0.0584  # 手指长度 (从XML中的pos)
    max_opening = 0.04      # 最大开口 (从joint range)
    
    # 计算手指尖端位置
    # 手指沿Z轴方向移动，左右对称
    left_finger_tip = hand_position + np.array([0.0, 0.0, finger_length + gripper_state])
    right_finger_tip = hand_position + np.array([0.0, 0.0, finger_length + gripper_state])
    
    # 基础抓取力（基于肌腱张力）
    base_grasp_force = tendon_tension * 0.5
    
    # 检查是否接触物体
    left_distance = np.linalg.norm(left_finger_tip - object_position)
    right_distance = np.linalg.norm(right_finger_tip - object_position)
    
    contact_threshold = 0.03  # 3cm接触阈值
    
    if left_distance < contact_threshold and right_distance < contact_threshold:
        # 接触物体时的力
        # Panda夹爪主要产生Z轴方向的抓取力
        left_contact_force = base_grasp_force * (1.0 - left_distance / contact_threshold)
        right_contact_force = base_grasp_force * (1.0 - right_distance / contact_threshold)
        
        # 计算力的方向（主要沿Z轴，少量X-Y分量）
        left_direction = (object_position - left_finger_tip) / (left_distance + 1e-6)
        right_direction = (object_position - right_finger_tip) / (right_distance + 1e-6)
        
        # 增强Z轴分量（符合Panda夹爪特性）
        left_direction[2] *= 2.0  # 增强Z轴分量
        right_direction[2] *= 2.0
        
        # 归一化
        left_direction = left_direction / np.linalg.norm(left_direction)
        right_direction = right_direction / np.linalg.norm(right_direction)
        
        # 计算三个方向的力
        left_force = left_direction * left_contact_force
        right_force = right_direction * right_contact_force
        
    else:
        # 未接触时的力（主要是Z轴方向的预紧力）
        preload_force = base_grasp_force * 0.1
        left_force = np.array([0.0, 0.0, preload_force])
        right_force = np.array([0.0, 0.0, preload_force])
    
    return left_force, right_force

def get_proprioceptive_state(data: mujoco.MjData, model: mujoco.MjModel, env) -> Dict[str, np.ndarray]:
    """
    获取本体感知状态（无视觉输入）
    
    Args:
        data: MuJoCo数据
        model: MuJoCo模型
        env: 环境实例
        
    Returns:
        state: 本体感知状态字典
    """
    # 本体状态（Task3：只取机械臂 6 关节；freejoint 存在时 qpos 地址≠关节ID，必须用 arm_joint_ids）
    arm_joint_ids = list(getattr(env, 'arm_joint_ids', [])) or list(range(6))
    joint_positions = data.qpos[arm_joint_ids].copy()
    joint_velocities = data.qvel[arm_joint_ids].copy()
    joint_torques = data.qfrc_actuator[arm_joint_ids].copy()

    # 夹爪状态（Task3：2F-85 开口宽度 = 两衬垫 site 距离，由环境测量）。
    # tendon_position/velocity/tension 保留原维度语义：
    #   tendon_position = 开度 width(m)；tendon_velocity 无跨步差分，置 0；
    #   tendon_tension = 由归一化闭合量导出的握持张力（越闭合越大，0~100）
    if hasattr(env, '_get_gripper_width'):
        tendon_position = float(env._get_gripper_width())
    else:
        tendon_position = 0.0
    tendon_velocity = 0.0
    _cfg = getattr(env, 'grasping_config', None)
    _w_min = float(getattr(_cfg, 'grasp_min_width', 0.02))
    _w_max = float(getattr(_cfg, 'gripper_max_width', 0.093))
    _closure = float(np.clip((_w_max - tendon_position) / max(_w_max - _w_min, 1e-6), 0.0, 1.0))
    tendon_tension = _closure * 100.0
    
    # 末端状态 - 使用末端 body（Task3：rq_base_mount；Panda 兼容 'hand'）的真实位姿/速度
    end_effector_pos = get_end_effector_position(data, model, env)
    hand_id = int(getattr(env, 'end_effector_id', -1))
    if hand_id < 0:
        hand_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "hand")
    if hand_id >= 0 and hand_id < len(data.xquat):
        end_effector_orientation = data.xquat[hand_id].copy()
    else:
        end_effector_orientation = np.array([1, 0, 0, 0])
    if hand_id >= 0 and hand_id < len(data.cvel):
        end_effector_velocity = data.cvel[hand_id].copy()  # [旋转(3), 平移(3)]
    else:
        end_effector_velocity = np.zeros(6)

    # 夹爪状态（归一化开度 width/max_width，0=闭合, 1=全开）
    gripper_state = float(tendon_position) / _w_max if _w_max > 0 else 0.0
    
    # 目标信息
    target_position = env.target_pos if hasattr(env, 'target_pos') else np.array([0.5, 0.0, 0.3])
    target_orientation = env.target_quat if hasattr(env, 'target_quat') else np.array([1, 0, 0, 0])
    
    # 可操作度指标
    manipulability = calculate_manipulability(joint_positions)
    
    # 简化的接触力（基于肌腱张力）
    contact_force = tendon_tension if tendon_tension > 5.0 else 0.0
    
    # 使用改进的手指力计算（符合Panda夹爪结构）
    left_finger_force, right_finger_force = calculate_finger_forces(
        tendon_tension, gripper_state, target_position, end_effector_pos
    )
    
    state = {
        # 本体状态
        'joint_positions': joint_positions,
        'joint_velocities': joint_velocities,
        'joint_torques': joint_torques,
        'tendon_position': tendon_position,
        'tendon_velocity': tendon_velocity,
        'tendon_tension': tendon_tension,
        
        # 末端状态
        'ee_position': end_effector_pos,
        'ee_orientation': end_effector_orientation,
        'ee_velocity': end_effector_velocity,
        'gripper_state': gripper_state,
        
        # 目标信息
        'target_position': target_position,
        'target_orientation': target_orientation,
        
        # 可操作度指标
        'manipulability': manipulability,
        
        # 接触信息
        'contact_force': contact_force,
        'left_finger_force': left_finger_force,
        'right_finger_force': right_finger_force
    }
    
    return state

def get_end_effector_position(data: mujoco.MjData, model: mujoco.MjModel, env=None) -> np.ndarray:
    """
    获取末端执行器位置（Task3：优先用 env.end_effector_id=rq_base_mount；Panda 兼容 'hand'）
    """
    hand_id = int(getattr(env, 'end_effector_id', -1)) if env is not None else -1
    if hand_id < 0:
        hand_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "hand")
    if hand_id >= 0 and len(data.xpos) > hand_id:
        return data.xpos[hand_id].copy()
    
    # 如果无法获取hand位置，使用最后一个关节的位置
    if len(data.xpos) > 0:
        return data.xpos[-1].copy()
    else:
        return np.array([0.4, 0.0, 0.2])  # 默认位置

def calculate_tendon_tension(tendon_position: float) -> float:
    """
    基于肌腱位置计算张力 - 改进的非线性模型，增加敏感性
    """
    # 肌腱物理参数
    rest_length = 0.02  # m
    max_length = 0.04   # m
    min_length = 0.01   # m
    
    # 非线性张力模型 - 增加敏感性
    if tendon_position <= rest_length:
        # 压缩阶段 - 指数增长，增加敏感性
        compression = rest_length - tendon_position
        tension = 200.0 * (np.exp(compression * 100) - 1)  # 增加系数
    else:
        # 拉伸阶段 - 二次增长，增加敏感性
        stretch = tendon_position - rest_length
        tension = 400.0 * stretch + 1000.0 * stretch**2  # 增加系数
    
    # 限制最大张力
    tension = min(tension, 2000.0)  # 增加最大张力
    
    return max(0.0, tension)

def calculate_manipulability(joint_positions: np.ndarray) -> float:
    """
    计算可操作度（雅可比矩阵条件数的倒数）
    
    Args:
        joint_positions: 关节位置
        
    Returns:
        manipulability: 可操作度 (0-1, 越大越好)
    """
    # 简化的可操作度计算
    # 检查肘部奇异点 (joint3 接近 ±π/2)
    elbow_angle = joint_positions[2]  # joint3
    elbow_singularity = abs(abs(elbow_angle) - np.pi/2)
    
    # 检查腕部奇异点 (joint5 接近 ±π/2)
    wrist_angle = joint_positions[4]  # joint5
    wrist_singularity = abs(abs(wrist_angle) - np.pi/2)
    
    # 检查肩部奇异点 (joint2 接近 ±π/2)
    shoulder_angle = joint_positions[1]  # joint2
    shoulder_singularity = abs(abs(shoulder_angle) - np.pi/2)
    
    # 计算最小奇异距离
    min_singularity_distance = min(elbow_singularity, wrist_singularity, shoulder_singularity)
    
    # 转换为可操作度 (0-1)
    # 距离奇异点越远，可操作度越高
    manipulability = 1.0 / (1.0 + np.exp(-10 * (min_singularity_distance - 0.3)))
    
    return manipulability
