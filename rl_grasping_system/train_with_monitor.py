"""
使用训练监控器的改进训练脚本（并行改造 2026-08-22）
- 12 环境 SubprocVecEnv 多进程并行（CPU 多核，无需 GPU）
- 训练 worker 全部无头（render_gui=False）
- 主进程定期弹窗播放当前策略（demo_env, render_gui=True），不拖慢训练
"""

import os
import sys
import copy
import logging

# 本地桌面渲染后端默认 GLFW（云端请用 train_cloud.py，其内部设 EGL）。
os.environ.setdefault('MUJOCO_GL', 'glfw')

# P0-3: 强制 matplotlib 非交互 Agg 后端（避免无 DISPLAY 时 Qt xcb 崩溃）。
import matplotlib
matplotlib.use('Agg')

from agent import GraspingAgent
from environment import GraspingEnv
from config import get_config
from training_monitor import TrainingMonitor


def setup_logging():
    """设置日志"""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.StreamHandler(sys.stdout),
            logging.FileHandler('training.log')
        ]
    )

def main():
    """主训练函数"""
    print("=" * 80)
    print("🤖 机械臂强化学习训练系统 (多进程并行 + 定期演示渲染)")
    print("=" * 80)

    # 设置日志
    setup_logging()
    logger = logging.getLogger(__name__)

    try:
        # 获取配置
        config = get_config()
        config.grasping.use_fixed_position = True
        config.grasping.render_gui = False  # 训练 worker 一律无头（渲染交给演示环境）
        n_envs = max(1, int(config.system.num_envs))
        total_timesteps = config.training.total_timesteps
        logger.info(f"配置加载完成：n_envs={n_envs}, total_timesteps={total_timesteps}, "
                    f"action_space_dim={config.grasping.action_space_dim}")

        # 训练环境工厂：SubprocVecEnv 每个子进程调用，返回新环境（MuJoCo 对象不跨进程传输）
        def make_train_env():
            return GraspingEnv(
                grasping_config=config.grasping,
                reward_config=config.reward
            )

        # 创建智能体（多进程并行）——必须先 fork workers 再创建演示环境：
        # demo_env 的 GLFW/X 连接若先存在，fork 后子进程继承会触发 XIO fatal error
        logger.info("创建智能体（SubprocVecEnv 并行）...")
        agent = GraspingAgent(
            network_config=config.network,
            training_config=config.training
        )
        agent.set_environment(make_train_env, n_envs=n_envs)
        logger.info("智能体创建成功")

        # 演示环境（主进程，render_gui=True）：训练中每 demo_every_n_rollouts 个 rollout
        # 用当前策略弹窗播放一个 episode（不参与训练）
        demo_grasp_cfg = copy.copy(config.grasping)
        demo_grasp_cfg.render_gui = True
        demo_grasp_cfg.render_interval = 1
        demo_env = GraspingEnv(
            grasping_config=demo_grasp_cfg,
            reward_config=config.reward
        )
        logger.info("演示环境已创建（render_gui=True，训练中定期播放当前策略）")

        # 开始训练
        logger.info("开始训练...")

        print(f"\n🎯 训练目标: {total_timesteps:,} 总步数（{n_envs} 环境并行）")
        print(f"📊 监控指标: 奖励、成功率、奇异点、训练趋势")
        print(f"🎬 演示窗口: 每 5 个 rollout 自动弹出当前策略一次")
        print(f"📈 图表保存: logs/ 目录")
        print(f"📝 日志文件: training.log")

        # 训练
        agent.train(
            total_timesteps=total_timesteps,
            demo_env=demo_env,
            demo_every_n_rollouts=5,
        )

        # 保存最终模型
        models_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "models")
        os.makedirs(models_dir, exist_ok=True)
        final_model_path = os.path.join(models_dir, "final_model.zip")
        agent.save(final_model_path)
        logger.info(f"最终模型已保存: {final_model_path}")

        # 关闭演示环境
        try:
            demo_env.close()
        except Exception:
            pass

        # 打印训练摘要
        if hasattr(agent, 'training_monitor') and agent.training_monitor:
            agent.training_monitor.print_final_summary()

        print("\n✅ 训练完成！")
        print("📊 查看训练图表: logs/ 目录")
        print("🤖 模型文件: models/final_model.zip")

    except Exception as e:
        logger.error(f"训练失败: {e}")
        raise

if __name__ == "__main__":
    main()
