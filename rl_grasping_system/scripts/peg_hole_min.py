#!/usr/bin/env python3
"""
peg-in-hole 最小尝试（2026-08-28）
==================================================
目标：最小可运行的"接触插入"RL 环境 + 极短训练，验证"对准 → 插入"信号可学（成功率从 0 上升）。

简化（最小尝试范围）：
- peg 固定连在夹爪根部（跳过"抓 peg"环节，聚焦接触插入）
- 孔固定（hole_board：200×200×20mm，14×14mm 方孔，位于桌面 (0.1, 0.42)）
- 姿态固定朝下（垂直顶插），动作 3D 位置增量（复用抓取环境速度级 IK 执行）
- reset 用 IK 把 peg 尖端放到孔正上方附近（课程式先易后难）

成功判定：peg 尖端插入深度 ≥ 12mm（peg_tip_z < 孔面 z − 0.012）。

场景：models/universal_robots_ur5e/ur5e_peg_hole.xml
运行：MUJOCO_GL=egl python3 scripts/peg_hole_min.py
"""
import os
import sys

os.environ.setdefault('MUJOCO_GL', 'egl')
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import gymnasium as gym
import mujoco
import numpy as np
from gymnasium import spaces
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv

import ik

# ---------------- 场景常量 ----------------
XML = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..', 'models',
                   'universal_robots_ur5e', 'ur5e_peg_hole.xml')
ARM_JOINTS = ['shoulder_pan_joint', 'shoulder_lift_joint', 'elbow_joint',
              'wrist_1_joint', 'wrist_2_joint', 'wrist_3_joint']
SAFE_CONFIG = np.array([-1.5708, -1.5708, 1.5708, -1.5708, -1.5708, 0.0])
HOLE_CENTER = np.array([0.1, 0.42, 0.32])       # 孔中心（孔面 z=0.32）
PEG_TIP_OFFSET = 0.18                            # peg 尖端相对 rq_base_mount 局部 +z 偏移(m)
SUCCESS_DEPTH = 0.012                            # 插入深度阈值(m) = peg_tip 低于孔面 ≥ 12mm
MAX_EE_DELTA = 0.005                             # 每决策步位置增量上限(m)
DT, SUBSTEPS, ACTION_REPEAT = 0.002, 10, 2       # 决策周期 T = 0.04s
KP = np.array([200., 200., 200., 100., 100., 60.])
KD = np.array([5., 5., 5., 4., 4., 6.])
LAM, LIM = 0.08, np.array([150., 150., 150., 28., 28., 28.])
Q_APPROACH_DOWN = np.array([0.0, 1.0, 0.0, 0.0])  # 手指/peg 朝下

OBS_DIM = 21  # joint_pos(6)+joint_vel(6)+peg_tip_pos(3)+peg_tip_vel(3)+hole(3)

class PegHoleEnv(gym.Env):
    """最小 peg-in-hole 环境：UR5e 末端带固定 peg，对准并插入桌面孔板。"""

    def __init__(self, render_gui=False):
        super().__init__()
        self.model = mujoco.MjModel.from_xml_path(XML)
        self.data = mujoco.MjData(self.model)
        self.arm_joint_ids = np.array(
            [mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, j) for j in ARM_JOINTS])
        self.ee_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, 'rq_base_mount')
        self.peg_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, 'peg')
        self.tip_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_SITE, 'peg_tip_site')
        self.action_space = spaces.Box(low=-MAX_EE_DELTA, high=MAX_EE_DELTA,
                                       shape=(3,), dtype=np.float32)
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf,
                                            shape=(OBS_DIM,), dtype=np.float32)
        self.steps = 0
        self.max_steps = 200
        self.T = DT * SUBSTEPS * ACTION_REPEAT
        self.lateral_range = 0.02  # 初始水平偏移范围(m)：课程 ±2cm；扰动评估可改大（±5cm）

    def _set_initial_pose(self):
        """IK 把 peg 尖端放到孔正上方附近（+水平/高度小随机，课程式起点）。

        必须先置 SAFE_CONFIG 再 IK：mj_resetData 后 qpos=0（臂下垂接近奇异），
        DLS 从奇异位形出发会落局部极小（实测 tip 水平偏 12cm、高度错 6cm）。
        """
        self.data.qpos[self.arm_joint_ids] = SAFE_CONFIG
        mujoco.mj_forward(self.model, self.data)
        rng = self.np_random
        tip_target = HOLE_CENTER.copy()
        tip_target[:2] += rng.uniform(-self.lateral_range, self.lateral_range, 2)  # 初始水平偏移
        tip_target[2] += rng.uniform(0.06, 0.10)        # 孔上方 6~10cm
        ee_target = tip_target + np.array([0.0, 0.0, PEG_TIP_OFFSET])
        q, err = ik.solve_ik(self.model, self.data, self.ee_id, ee_target,
                             Q_APPROACH_DOWN, self.arm_joint_ids, iters=800)
        if err > 0.01:                                   # IK 失败回退 safe_config
            q = SAFE_CONFIG.copy()
        self.data.qpos[self.arm_joint_ids] = q
        if len(self.data.ctrl) > 6:                      # 夹爪张开，peg 固定在根部
            self.data.ctrl[6] = 0.0

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        mujoco.mj_resetData(self.model, self.data)
        self._set_initial_pose()
        mujoco.mj_forward(self.model, self.data)
        self.steps = 0
        return self._get_obs(), {}

    def _get_obs(self):
        tip = self.data.site_xpos[self.tip_id].copy()
        tip_vel = self.data.cvel[self.peg_id][:3].copy()
        return np.concatenate([
            self.data.qpos[self.arm_joint_ids],   # 6
            self.data.qvel[self.arm_joint_ids],   # 6
            tip,                                  # 3
            tip_vel,                              # 3
            HOLE_CENTER,                          # 3
        ]).astype(np.float32)

    def _insertion_depth(self):
        """peg 尖端低于孔面的深度(m)，未插入时为负。"""
        tip = self.data.site_xpos[self.tip_id]
        return HOLE_CENTER[2] - tip[2]

    def step(self, action):
        a = np.clip(np.asarray(action, dtype=float).ravel()[:3], -MAX_EE_DELTA, MAX_EE_DELTA)
        v_ee = np.concatenate([a / self.T, np.zeros(3)])
        # 速度执行：速度级 IK + 速度 PID + 重力/惯性前馈（复用抓取环境语义）
        M_full = np.zeros((self.model.nv, self.model.nv))
        mujoco.mj_fullM(self.model, self.data, M_full)
        M_eff = M_full[np.ix_(self.arm_joint_ids, self.arm_joint_ids)]
        dq_prev = np.zeros(6)
        for _ in range(ACTION_REPEAT):
            dq = ik.velocity_ik(self.model, self.data, self.ee_id, v_ee,
                                self.arm_joint_ids, LAM)
            for _ in range(SUBSTEPS):
                dq_ff = (dq - dq_prev) / DT
                dq_prev = dq.copy()
                tau = (KP * (dq - self.data.qvel[self.arm_joint_ids])
                       - KD * self.data.qvel[self.arm_joint_ids]
                       + M_eff @ dq_ff + self.data.qfrc_bias[self.arm_joint_ids])
                self.data.ctrl[self.arm_joint_ids] = np.clip(tau, -LIM, LIM)
                mujoco.mj_step(self.model, self.data)

        self.steps += 1
        tip = self.data.site_xpos[self.tip_id]
        insertion = self._insertion_depth()
        d_xy = float(np.linalg.norm(tip[:2] - HOLE_CENTER[:2]))

        # 奖励：水平对准（势能）+ 垂直接近 + 插入深度（线性，一旦插入即正收益）+ 成功 + 步罚
        r = -5.0 * min(d_xy / 0.3, 1.0)
        if insertion <= 0.0:
            r += -3.0 * min((HOLE_CENTER[2] - tip[2]) / 0.3, 1.0)
        else:
            r += 10.0 * min(insertion, 0.02) / 0.02
        r += -0.01

        success = insertion >= SUCCESS_DEPTH
        if success:
            r += 100.0
        terminated = success
        truncated = self.steps >= self.max_steps
        info = {'insertion': float(insertion), 'd_xy': d_xy,
                'success': bool(success), 'episode_length': self.steps}
        return self._get_obs(), float(r), terminated, truncated, info

def make_env():
    return PegHoleEnv()


def evaluate(model, n=30):
    env = PegHoleEnv()
    succ = 0
    depths = []
    for _ in range(n):
        obs, _ = env.reset()
        done = False
        while not done:
            action, _ = model.predict(obs, deterministic=True)
            obs, _, term, trunc, info = env.step(action)
            done = term or trunc
        succ += int(info['success'])
        depths.append(info['insertion'])
    env.close()
    return succ / n, float(np.mean(depths))


def main():
    n_envs = 8
    env = DummyVecEnv([make_env] * n_envs)
    model = PPO('MlpPolicy', env,
                n_steps=512, batch_size=512, n_epochs=10,
                ent_coef=0.01, learning_rate=3e-4, gamma=0.99,
                verbose=0)
    print('=== 训练前评估 ===')
    sr0, d0 = evaluate(model)
    print(f'success_rate={sr0*100:.1f}%  avg_insertion={d0*1000:.1f}mm')
    print('=== 开始训练 20k 步 ===')
    model.learn(total_timesteps=60000)
    model.save(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'models',
                            'peg_hole_rl.zip'))
    print('模型已保存: models/peg_hole_rl.zip')
    print('=== 训练后评估 ===')
    sr1, d1 = evaluate(model)
    print(f'success_rate={sr1*100:.1f}%  avg_insertion={d1*1000:.1f}mm')
    print(f'结论：插入信号{"可学（成功率上升）" if sr1 > sr0 else "未学出（需调奖励/加步数）"}')


if __name__ == '__main__':
    main()
