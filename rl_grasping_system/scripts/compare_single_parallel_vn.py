"""全面对比：单机 vs 并行 set_environment 迁移加载后，归一化状态与成功率（2026-08-24）

单机 compare：归一化路径 55%，不归一化 0%。
并行 verify：reset 后 obs 被 clip_obs 截断到 ±10（饱和）→ 0%。
差异可能在 obs_rms（尤其 target_position 维度）或 reset 原始 obs。
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


def inspect(label, env_or_factory, n_envs):
    config = get_config()
    cx, cy = config.grasping.object_fixed_pos
    r = 0.03
    z = config.grasping.workspace_bounds[2]
    config.grasping.render_gui = False
    config.grasping.use_fixed_position = False
    config.grasping.workspace_bounds = ((cx - r, cx + r), (cy - r, cy + r), z)

    agent = GraspingAgent(config.network, config.training, model_path=MODEL)
    agent.set_environment(env_or_factory(config), n_envs=n_envs)
    vn = agent.agent.get_vec_normalize_env()

    var = vn.obs_rms.var
    mean = vn.obs_rms.mean
    print(f"\n{'='*60}\n[{label}] 并行={n_envs>1}  n_envs={n_envs}")
    print(f"  vn.norm_obs={vn.norm_obs} training={vn.training}")
    print(f"  obs_rms.var: joint[:3]={np.round(var[:3],4)} "
          f"ee_pos[21:24]={np.round(var[21:24],4)} "
          f"target_pos[35:38]={np.round(var[35:38],6)}")
    print(f"  obs_rms.mean: target_pos[35:38]={np.round(mean[35:38],4)}")

    # 原始 obs（vn 包装前？用 venv 内部 reset）——用 vec_env.reset 前先拿 vn.venv
    raw_obs = vn.venv.reset()
    print(f"  原始 reset obs: joint[:3]={np.round(raw_obs[0,:3],3)} "
          f"target_pos[35:38]={np.round(raw_obs[0,35:38],3)}")

    # vn 归一化后 obs（reset 会更新 stats）
    obs = vn.reset()
    print(f"  vn.reset 后 obs[0,:3]={np.round(obs[0,:3],2)}  "
          f"obs[0,35:38]={np.round(obs[0,35:38],2)}  (clip=±10)")
    print(f"  vn.reset 后 obs[0,21:24]={np.round(obs[0,21:24],2)}  (ee_pos)")

    # 采样统计成功率
    ep_count, ok, lens = 0, 0, []
    total = 2048
    for t in range(total):
        actions, _ = agent.agent.predict(obs, deterministic=False)
        obs, rewards, dones, infos = vn.step(actions)
        for info in infos:
            if info.get('episode') is not None:
                ep_count += 1
                if bool(info.get('grasp_success', False)):
                    ok += 1
                lens.append(info['episode'].get('l', 0))
    print(f"  采样 {total} 步: 完成 episode {ep_count} | 成功 {ok} | 成功率 {ok/max(ep_count,1):.1%}")
    if lens:
        print(f"  平均步数 {np.mean(lens):.1f}")
    return ok / max(ep_count, 1)


def main():
    logging.basicConfig(level=logging.WARNING)

    # 单机
    inspect("单机 DummyVecEnv",
            lambda cfg: GraspingEnv(grasping_config=cfg.grasping, reward_config=cfg.reward), 1)
    # 并行
    inspect("并行 SubprocVecEnv",
            lambda cfg: (lambda: GraspingEnv(grasping_config=cfg.grasping, reward_config=cfg.reward)), 12)


if __name__ == "__main__":
    main()
