"""
使用训练监控器的改进训练脚本（并行改造 2026-08-22）
- 12 环境 SubprocVecEnv 多进程并行（CPU 多核，无需 GPU）
- 训练 worker 全部无头（render_gui=False）
- 主进程定期弹窗播放当前策略（demo_env, render_gui=True），不拖慢训练
"""

import os
import sys
import copy
import argparse
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
    # 课程学习参数（2026-08-23）：
    #   --position-random: 启用物体位置随机（use_fixed_position=False，围绕 object_fixed_pos ±radius）
    #   --radius: 随机半径(m)，课程阶段1=0.03，阶段2=0.06-0.08，阶段3=完整 workspace_bounds
    #   --load-model: 迁移加载上一阶段模型继续训练（保留已学抓取技能）
    parser = argparse.ArgumentParser(description='机械臂 RL 训练（含课程学习位置随机）')
    parser.add_argument('--position-random', action='store_true',
                        help='启用物体位置随机（课程学习）；默认固定位置')
    parser.add_argument('--radius', type=float, default=0.03,
                        help='位置随机半径(m)，围绕 object_fixed_pos 的 X/Y ±radius（阶段1=0.03）')
    parser.add_argument('--load-model', type=str, default='',
                        help='迁移加载模型路径（课程学习：从上一阶段模型继续训练）')
    parser.add_argument('--save-model', type=str, default='',
                        help='模型保存路径/文件名（默认 models/final_model.zip；可指定单独文件名避免覆盖，'
                             '如 final_model_fixed.zip）')
    parser.add_argument('--total-timesteps', type=int, default=0,
                        help='训练总步数（默认 config.training.total_timesteps；复现 PPO_95 用 1500000）')
    parser.add_argument('--no-demo', action='store_true',
                        help='禁用演示窗口（后台/长训练推荐——GLFW 演示窗口 segfault 曾导致 PPO_97 '
                             '训练中途崩溃、模型未保存）')
    args = parser.parse_args()

    print("=" * 80)
    print("🤖 机械臂强化学习训练系统 (多进程并行 + 定期演示渲染)")
    print("=" * 80)

    # 设置日志
    setup_logging()
    logger = logging.getLogger(__name__)

    try:
        # 获取配置
        config = get_config()
        config.grasping.render_gui = False  # 训练 worker 一律无头（渲染交给演示环境）
        if args.position_random:
            cx, cy = config.grasping.object_fixed_pos
            r = max(0.001, float(args.radius))
            _z = config.grasping.workspace_bounds[2]  # Z 固定桌面高度，随机只影响 X/Y
            config.grasping.use_fixed_position = False
            config.grasping.workspace_bounds = (
                (cx - r, cx + r),
                (cy - r, cy + r),
                _z,
            )
            logger.info(f"📍 位置随机已启用：围绕 ({cx}, {cy}) ±{r}m（课程学习阶段）")
        else:
            config.grasping.use_fixed_position = True
            logger.info("📍 固定位置训练")
        n_envs = max(1, int(config.system.num_envs))
        if args.total_timesteps > 0:
            config.training.total_timesteps = int(args.total_timesteps)
        total_timesteps = config.training.total_timesteps
        logger.info(f"配置加载完成：n_envs={n_envs}, total_timesteps={total_timesteps}, "
                    f"action_space_dim={config.grasping.action_space_dim}, "
                    f"position={'随机±' + str(args.radius) if args.position_random else '固定'}")

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
            training_config=config.training,
            model_path=args.load_model if args.load_model else None,  # 课程学习迁移加载
        )
        agent.set_environment(make_train_env, n_envs=n_envs)
        logger.info("智能体创建成功")

        # 模型保存路径（--save-model 可指定路径/文件名，避免覆盖既有模型）
        models_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "models")
        os.makedirs(models_dir, exist_ok=True)
        if args.save_model:
            final_model_path = (args.save_model if os.path.isabs(args.save_model)
                                else os.path.join(models_dir, args.save_model))
        else:
            final_model_path = os.path.join(models_dir, "final_model.zip")

        # 演示环境（主进程，render_gui=True）：训练中每 demo_every_n_rollouts 个 rollout
        # 用当前策略弹窗播放一个 episode（不参与训练）
        # 2026-08-23: --no-demo 禁用（后台/长训练推荐——GLFW 演示窗口 segfault 曾导致 PPO_97
        # 训练中途崩溃、模型未保存）
        if args.no_demo:
            demo_env = None
            logger.info("演示窗口已禁用（--no-demo，后台训练安全）")
        else:
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
        if not args.no_demo:
            print(f"🎬 演示窗口: 每 5 个 rollout 自动弹出当前策略一次")
        print(f"📈 图表保存: logs/ 目录")
        print(f"📝 日志文件: training.log")

        # 训练（save_path 传入启用 CheckpointCallback 定期保存——崩溃可恢复，最多损失 save_freq 步）
        agent.train(
            total_timesteps=total_timesteps,
            demo_env=demo_env,
            demo_every_n_rollouts=5,
            save_path=final_model_path,
        )
        logger.info(f"最终模型已保存: {final_model_path}")

        # 关闭演示环境（若有）
        if demo_env is not None:
            try:
                demo_env.close()
            except Exception:
                pass

        # 打印训练摘要
        if hasattr(agent, 'training_monitor') and agent.training_monitor:
            agent.training_monitor.print_final_summary()

        print("\n✅ 训练完成！")
        print("📊 查看训练图表: logs/ 目录")
        print(f"🤖 模型文件: {final_model_path}")

    except Exception as e:
        logger.error(f"训练失败: {e}")
        raise

if __name__ == "__main__":
    main()
