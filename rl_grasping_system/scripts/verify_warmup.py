"""验证：迁移加载后预热 obs_rms（扩展 std 覆盖位置随机分布）能否恢复策略成功率（2026-08-24）

根因：fixed 训练 pkl 的 obs_rms.std 极小（target_position 维度 var=0），迁移到位置随机后
归一化除零→clip 饱和（±10）→ 策略输入失效 → 0%。
方案：预热——用当前策略在位置随机环境跑 warmup 步（VecNormalize.training=True 更新 obs_rms），
std 扩展后再采样评估。
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

    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--floor", type=float, default=0.0,
                    help="给 obs_rms.var 加下限（防除零饱和），0=不设")
    ap.add_argument("--warmup", type=int, default=0, help="预热步数（默认 0 不预热）")
    ap.add_argument("--freeze", action="store_true", help="冻结 vn(training=False)，防 update 覆盖 floor")
    args, _ = ap.parse_known_args()

    if args.floor > 0:
        vn.obs_rms.var = np.maximum(vn.obs_rms.var, args.floor)
        print(f"obs_rms.var 加下限 {args.floor}: target_pos[35:38]={np.round(vn.obs_rms.var[35:38],8)}  "
              f"joint[:3]={np.round(vn.obs_rms.var[:3],4)}")
    if args.freeze:
        vn.training = False
        print(">>> vn.training=False（冻结，防 obs_rms update 覆盖 floor）")

    print(f"obs_rms.var 当前: target_pos[35:38]={np.round(vn.obs_rms.var[35:38],8)}  "
          f"joint[:3]={np.round(vn.obs_rms.var[:3],4)}")

    # ---- 预热（可选）----
    if args.warmup > 0:
        obs = vn.reset()
        for t in range(args.warmup):
            actions, _ = agent.agent.predict(obs, deterministic=False)
            obs, _r, _d, _i = vn.step(actions)
        print(f"预热 {args.warmup} 步后 obs_rms.var: "
              f"target_pos[35:38]={np.round(vn.obs_rms.var[35:38],8)}  "
              f"joint[:3]={np.round(vn.obs_rms.var[:3],4)}")

    print(f"reset obs[0,:3]={np.round(vn.reset()[0,:3],2)}  "
          f"target[35:38]={np.round(vn.reset()[0,35:38],2)}  (clip=±10)")

    # ---- 采样统计成功率 ----
    obs = vn.reset()
    ep_count, ok, lens = 0, 0, []
    total = 4096
    for t in range(total):
        actions, _ = agent.agent.predict(obs, deterministic=False)
        obs, _r, _d, infos = vn.step(actions)
        for info in infos:
            if info.get('episode') is not None:
                ep_count += 1
                if bool(info.get('grasp_success', False)):
                    ok += 1
                lens.append(info['episode'].get('l', 0))
    print(f"\n采样 {total} 步: 完成 episode {ep_count} | 成功 {ok} | "
          f"成功率 {ok/max(ep_count,1):.1%}")
    if lens:
        print(f"  平均步数 {np.mean(lens):.1f}")


if __name__ == "__main__":
    main()
