"""对照实验：冻结 VecNormalize 后迁移 fine-tune（2026-08-24）

假设：迁移训练 0% 是因为训练中 VecNormalize(training=True) 持续更新 obs_rms，
位置随机导致观测分布漂移，已定型策略输入分布被改变 → 策略崩坏。
（诊断：固定 stats 下 ±0.03 成功率 36.7%）

本脚本：迁移 final_model_fixed → ±0.03，冻结 VecNormalize.training=False，learn 5 rollout，
保存 /tmp/freeze_vn_test.zip。之后用 verify_position_migration.py 评估对比。

用法：
    cd rl_grasping_system && python scripts/freeze_vn_test.py
"""
import os
import sys
import logging

os.environ.setdefault('MUJOCO_GL', 'glfw')
import matplotlib
matplotlib.use('Agg')

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config import get_config
from agent import GraspingAgent
from environment import GraspingEnv

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL = os.path.join(BASE_DIR, "models", "final_model_fixed.zip")
OUT = "/tmp/freeze_vn_test.zip"


def main():
    logging.basicConfig(level=logging.INFO,
                        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')

    config = get_config()
    cx, cy = config.grasping.object_fixed_pos
    r = 0.03
    z = config.grasping.workspace_bounds[2]
    config.grasping.render_gui = False
    config.grasping.use_fixed_position = False
    config.grasping.workspace_bounds = ((cx - r, cx + r), (cy - r, cy + r), z)

    n_envs = max(1, int(config.system.num_envs))

    def make_train_env():
        return GraspingEnv(grasping_config=config.grasping, reward_config=config.reward)

    agent = GraspingAgent(config.network, config.training, model_path=MODEL)
    agent.set_environment(make_train_env, n_envs=n_envs)

    vn = agent.agent.get_vec_normalize_env()
    if vn is not None:
        vn.training = False
        print(">>> VecNormalize.training 已冻结（观测统计不再随训练更新）")
    else:
        print(">>> WARNING: 未找到 VecNormalize")

    steps = 5 * int(config.training.n_steps) * n_envs
    print(f">>> 训练 {steps} 步（5 rollout x {n_envs} env），保存 {OUT}")
    agent.train(total_timesteps=steps, save_path=OUT)
    print(f">>> 完成：{OUT}")


if __name__ == "__main__":
    main()
