"""探测（精简版）：训练同款并行环境下 fixed 模型在 ±0.03 的 stochastic 成功率（2026-08-24）

只跑一个 rollout（24576 步），按训练 _on_step 统计口径统计，加进度打印防卡死误判。
"""
import os
import sys
import logging
import time

os.environ.setdefault('MUJOCO_GL', 'glfw')
import matplotlib
matplotlib.use('Agg')

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
from stable_baselines3.common.vec_env import VecNormalize, SubprocVecEnv
from stable_baselines3.common.monitor import Monitor
from stable_baselines3 import PPO

from config import get_config
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

    def _make_env():
        return Monitor(GraspingEnv(grasping_config=config.grasping, reward_config=config.reward))

    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--parallel", action="store_true", help="用 SubprocVecEnv（默认 DummyVecEnv 单机）")
    ap.add_argument("--n-envs", type=int, default=0, help="覆盖 env 数（DummyVecEnv 用少些加速）")
    ap.add_argument("--freeze-vn", action="store_true", help="冻结 VecNormalize(training=False)，不更新 obs_rms")
    args, _ = ap.parse_known_args()
    if args.n_envs > 0:
        n_envs = args.n_envs

    print(f">>> 创建 {'SubprocVecEnv(并行)' if args.parallel else 'DummyVecEnv(单机)'}...", flush=True)
    if args.parallel:
        vec_env = SubprocVecEnv([_make_env for _ in range(n_envs)], start_method="fork")
    else:
        from stable_baselines3.common.vec_env import DummyVecEnv
        vec_env = DummyVecEnv([_make_env for _ in range(n_envs)])
    print(">>> VecEnv 创建完成", flush=True)

    # 在 VecNormalize 包装前，检查并行/单机环境里物体位置（观测 target_position, index 35-37，55 维 6 关节布局）
    # 是否真的在 ±radius 内随机——排除"worker 位置随机未生效"的可能
    _raw = vec_env.reset()
    print(f">>> 观测维度: {_raw.shape}", flush=True)
    _tp = _raw[:, 35:38]
    print(f">>> 物体位置分布 (观测 target_position, 期望 cx=0.1±{r}, cy=0.42±{r}):", flush=True)
    print(f"    X: [{_tp[:, 0].min():+.4f}, {_tp[:, 0].max():+.4f}]  "
          f"Y: [{_tp[:, 1].min():+.4f}, {_tp[:, 1].max():+.4f}]  "
          f"Z: [{_tp[:, 2].min():+.4f}, {_tp[:, 2].max():+.4f}]", flush=True)
    vec_env = VecNormalize(vec_env, norm_obs=True, norm_reward=True, clip_obs=10.0, clip_reward=10.0)
    print(">>> VecNormalize 包装完成", flush=True)

    norm_pkl = os.path.splitext(MODEL)[0] + "_vecnormalize.pkl"
    if os.path.exists(norm_pkl):
        vec_env = VecNormalize.load(norm_pkl, vec_env)
        print(">>> 已恢复 VecNormalize 观测统计", flush=True)

    print(">>> PPO.load...", flush=True)
    model = PPO.load(MODEL, env=vec_env)
    print(f">>> 加载模型完成: {MODEL}", flush=True)
    if args.freeze_vn:
        vec_env.training = False
        print(">>> VecNormalize.training 已冻结（不更新 obs_rms）", flush=True)

    total = 2456  # 每 env ~205 步 ≈ 1 个完整 episode，确认并行环境 stochastic 真实成功率
    obs = vec_env.reset()
    ep_count, ok_ep, gs_toggle, lens = 0, 0, 0, []
    t0 = time.time()

    for t in range(total):
        actions, _ = model.predict(obs, deterministic=False)
        obs, rewards, dones, infos = vec_env.step(actions)
        for info in infos:
            if info.get('episode') is not None:
                ep_count += 1
                if bool(info.get('grasp_success', False)):
                    ok_ep += 1
                lens.append(info['episode'].get('l', 0))
            if bool(info.get('grasp_success', False)):
                gs_toggle += 1
        if (t + 1) % 5000 == 0:
            print(f"  进度 {t+1}/{total} | 完成episode {ep_count} 成功 {ok_ep} "
                  f"({time.time()-t0:.0f}s)", flush=True)

    print(f"\n=== 训练统计口径（info['episode'] 完成时，stochastic 采样）===")
    print(f"  完成 episode: {ep_count} | 成功: {ok_ep} | 成功率: {ok_ep/max(ep_count,1):.1%}")
    print(f"  平均 episode 步数: {np.mean(lens):.1f}" if lens else "  无完成 episode")
    print(f"  grasp_success=True 出现次数(含未终止): {gs_toggle}")
    print(f"  总耗时 {time.time()-t0:.0f}s")
    vec_env.close()


if __name__ == "__main__":
    main()
