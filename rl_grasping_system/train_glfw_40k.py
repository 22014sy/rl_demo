"""
Task3: 完整训练（seed 42, 150k 步）——前 40k 步 GLFW 实时可视化，之后自动关闭渲染继续无头训练。
训练期间逐步采集 reward_breakdown 各分项（environment.step 已累计到 episode_breakdown），
训练结束后可用 logs/training_log_*.json 生成"每个奖励项"的事后回放图。

运行（rl_grasping_system/ 下）：
    python3 train_glfw_40k.py
"""
import os
import sys
import logging

os.environ.setdefault('MUJOCO_GL', 'glfw')   # GLFW 后端（本机有 DISPLAY）

import matplotlib
matplotlib.use('Agg')

from stable_baselines3.common.callbacks import BaseCallback

from agent import GraspingAgent
from environment import GraspingEnv
from config import get_config


class DisableGlfwAfter40k(BaseCallback):
    """num_timesteps 达 threshold 后关闭 GLFW 渲染（render_gui=False + 关 viewer 窗口），继续无头训练。"""

    def __init__(self, real_env, threshold: int = 40000, verbose: int = 0):
        super().__init__(verbose)
        self.env_ref = real_env
        self.threshold = threshold

    def _on_step(self) -> bool:
        if self.num_timesteps >= self.threshold and getattr(self.env_ref, 'render_gui', False):
            self.env_ref.render_gui = False
            vh = getattr(self.env_ref, 'viewer_handle', None)
            if vh is not None:
                try:
                    vh.close()
                except Exception:
                    pass
                self.env_ref.viewer_handle = None
            print(f"[train-glfw] 已达 {self.threshold} 步，关闭 GLFW 渲染，继续无头训练", flush=True)
        return True


def main():
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        stream=sys.stdout,
    )
    cfg = get_config()
    # Task3: 前 40k 步 GLFW 实时可视化（每 5 决策步刷新一次，平衡流畅度与速度）
    cfg.grasping.render_gui = True
    cfg.grasping.render_interval = 5
    cfg.grasping.use_fixed_position = True
    cfg.training.total_timesteps = 150000   # 完整训练（40k 可视化 + 110k 无头）
    cfg.system.seed = 42

    env = GraspingEnv(cfg.grasping, cfg.reward)
    agent = GraspingAgent(cfg.network, cfg.training)
    agent.set_environment(env)

    print("[train-glfw] 前 40000 步将打开 MuJoCo 窗口实时显示训练过程（render_interval=5），之后自动关闭。", flush=True)
    agent.train(
        total_timesteps=cfg.training.total_timesteps,
        save_path=None,
        extra_callbacks=[DisableGlfwAfter40k(env, threshold=40000)],
    )

    if agent.training_monitor is not None:
        agent.training_monitor.print_final_summary()
    print("[train-glfw] 训练完成（150k 步）。日志含 reward breakdown 分项，可用事后回放脚本绘图。", flush=True)


if __name__ == '__main__':
    main()
