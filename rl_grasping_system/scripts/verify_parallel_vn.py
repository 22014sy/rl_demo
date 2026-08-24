"""验证：并行 set_environment（train_with_monitor 同款）迁移加载后 VecNormalize 归一化是否生效（2026-08-24）

单机 compare_predict_paths 证明：迁移后 vn.norm_obs=True，归一化路径成功率 50-55%，
不归一化（原始 obs 喂策略）0%。训练 rollout 0% → 怀疑并行模式下 VecNormalize 归一化失效
（vec_env.step 返回原始 obs，同路径 C）。
"""
import os
import sys
import logging

os.environ.setdefault('MUJOCO_GL', 'glfw')
import matplotlib
matplotlib.use('Agg')

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
from config import get_config
from agent import GraspingAgent
from environment import GraspingEnv

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL = os.path.join(BASE_DIR, "models", "final_model_fixed.zip")


def main():
    logging.basicConfig(level=logging.WARNING)
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
    print(f"并行模式 vn.norm_obs={vn.norm_obs}  norm_reward={vn.norm_reward}  training={vn.training}")
    print(f"  obs_rms.mean[:3]={np.round(vn.obs_rms.mean[:3],4)}")
    print(f"  obs_rms.var[:3]={np.round(vn.obs_rms.var[:3],6)}")

    # 采样一个 rollout，检查 vec_env 返回 obs 是否归一化（原始物理量 vs N(0,1)）
    obs = agent.agent.env.reset()
    print(f"\nreset 后 obs[0, :3] = {np.round(obs[0, :3],4)}  (关节位置 rad, 原始量级 ~±3)")
    print(f"reset 后 obs[0, 35:38] = {np.round(obs[0, 35:38],4)}  (物体位置 m, 原始 ~0.07-0.45)")

    # 手动采样一个 rollout 统计成功率（训练同款 stochastic）
    ep_count, ok, lens = 0, 0, []
    total = 4096
    for t in range(total):
        actions, _ = agent.agent.predict(obs, deterministic=False)
        obs, rewards, dones, infos = agent.agent.env.step(actions)
        for info in infos:
            if info.get('episode') is not None:
                ep_count += 1
                if bool(info.get('grasp_success', False)):
                    ok += 1
                lens.append(info['episode'].get('l', 0))
    print(f"\n并行 + vec_env.step 采样 {total} 步：")
    print(f"  完成 episode {ep_count} | 成功 {ok} | 成功率 {ok/max(ep_count,1):.1%}")
    print(f"  平均步数 {np.mean(lens):.1f}" if lens else "  无完成 episode")


if __name__ == "__main__":
    main()
