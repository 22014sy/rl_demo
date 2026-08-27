"""
强化学习抓取评估脚本
用于测试训练好的模型性能
"""

import os
import sys
import logging
import argparse
import numpy as np

# P0-3: 无头环境强制 matplotlib 非交互 Agg 后端（默认只保存 PNG，不弹窗）。
# 需要交互窗口时设 SHOW_PLOTS=1（且要有可用 DISPLAY），否则 plot_results 里的
# plt.show() 在 DISPLAY 不可用时会因 Qt xcb 初始化失败而 Aborted (core dumped)。
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from datetime import datetime

# 添加当前目录到路径
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from config import get_config
from environment import PandaGraspingEnv
from agent import GraspingAgent

def setup_logging(log_level: str = "INFO"):
    """设置日志"""
    logging.basicConfig(
        level=getattr(logging, log_level.upper()),
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    return logging.getLogger(__name__)

def load_model(model_path: str, config):
    """加载模型"""
    logger = logging.getLogger(__name__)
    
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"模型文件不存在: {model_path}")
    
    # 创建环境
    env = PandaGraspingEnv(
        grasping_config=config.grasping,
        reward_config=config.reward
    )
    
    # 创建智能体
    agent = GraspingAgent(
        network_config=config.network,
        training_config=config.training,
        model_path=model_path
    )
    
    # 设置环境
    agent.set_environment(env)
    
    logger.info(f"模型加载成功: {model_path}")
    return agent, env

def run_episode(agent, env, render: bool = False, max_steps: int = 500,
                deterministic: bool = True):
    """运行单个episode"""
    obs, _ = env.reset()
    episode_reward = 0
    episode_length = 0
    episode_info = []
    episode_collision_count = 0
    residual_norms = []
    
    while episode_length < max_steps:
        # 预测动作（2026-08-24：加 deterministic 参数，供位置迁移诊断测量随机策略成功率）
        action, _ = agent.predict(obs, deterministic=deterministic)
        
        # 执行动作
        obs, reward, terminated, truncated, info = env.step(action)
        
        # 记录信息
        episode_reward += reward
        episode_length += 1
        # D2/评估：累计臂-障碍碰撞次数 + 收集残差幅度（§3.5-5 分工证据）
        episode_collision_count = max(episode_collision_count, info.get('obstacle_collision_count', 0))
        residual_norms.append(info.get('residual_norm', 0.0))
        episode_info.append({
            'step': episode_length,
            'reward': reward,
            'cumulative_reward': episode_reward,
            'distance_to_object': info.get('distance_to_object', 0),
            'gripper_width': info.get('gripper_width', 0),
            'is_grasped': info.get('is_grasped', False),
            'grasp_success': info.get('grasp_success', False)
        })
        
        # 渲染
        if render:
            env.render()
        
        # 检查是否结束
        if terminated or truncated:
            break
    
    return {
        'episode_reward': episode_reward,
        'episode_length': episode_length,
        'grasp_success': info.get('grasp_success', False),
        'final_distance': info.get('distance_to_object', 0),
        'collision_count': episode_collision_count,
        'avg_residual_norm': float(np.mean(residual_norms)) if residual_norms else 0.0,
        'max_residual_norm': float(np.max(residual_norms)) if residual_norms else 0.0,
        'episode_info': episode_info
    }

def evaluate_model(agent, env, n_episodes: int = 50, render: bool = False):
    """评估模型性能"""
    logger = logging.getLogger(__name__)
    
    logger.info(f"开始评估，episode数量: {n_episodes}")
    
    results = []
    success_count = 0
    
    for episode in range(n_episodes):
        logger.info(f"运行episode {episode + 1}/{n_episodes}")
        
        episode_result = run_episode(agent, env, render=render)
        results.append(episode_result)
        
        if episode_result['grasp_success']:
            success_count += 1
        
        logger.info(f"Episode {episode + 1} - 奖励: {episode_result['episode_reward']:.3f}, "
                   f"步数: {episode_result['episode_length']}, "
                   f"成功: {episode_result['grasp_success']}")
    
    # 计算统计信息
    success_rate = success_count / n_episodes
    avg_reward = np.mean([r['episode_reward'] for r in results])
    avg_length = np.mean([r['episode_length'] for r in results])
    avg_final_distance = np.mean([r['final_distance'] for r in results])

    # D2/评估：碰撞率 + 残差幅度（§3.5-5 分工证据：成功=小残差干净绕障）
    collision_episodes = sum(1 for r in results if r['collision_count'] > 0)
    avg_collision_count = float(np.mean([r['collision_count'] for r in results]))
    collision_rate = collision_episodes / n_episodes
    _success_res = [r['avg_residual_norm'] for r in results if r['grasp_success']]
    _fail_res = [r['avg_residual_norm'] for r in results if not r['grasp_success']]
    avg_residual_success = float(np.mean(_success_res)) if _success_res else 0.0
    avg_residual_fail = float(np.mean(_fail_res)) if _fail_res else 0.0
    
    # 成功episode的统计
    successful_episodes = [r for r in results if r['grasp_success']]
    if successful_episodes:
        avg_success_reward = np.mean([r['episode_reward'] for r in successful_episodes])
        avg_success_length = np.mean([r['episode_length'] for r in successful_episodes])
    else:
        avg_success_reward = 0
        avg_success_length = 0
    
    evaluation_results = {
        'n_episodes': n_episodes,
        'success_rate': success_rate,
        'avg_reward': avg_reward,
        'avg_episode_length': avg_length,
        'avg_final_distance': avg_final_distance,
        'avg_success_reward': avg_success_reward,
        'avg_success_length': avg_success_length,
        'collision_rate': collision_rate,                       # 碰撞 episode 比例（D6 验收线 = 0）
        'collision_episodes': collision_episodes,
        'avg_collision_count': avg_collision_count,
        'avg_residual_norm_success': avg_residual_success,      # §3.5-5：成功=小残差
        'avg_residual_norm_fail': avg_residual_fail,            # 失败=大残差
        'episode_results': results
    }
    
    # 打印结果
    logger.info("评估完成")
    logger.info(f"成功率: {success_rate:.3f} ({success_count}/{n_episodes})")
    logger.info(f"平均奖励: {avg_reward:.3f}")
    logger.info(f"平均步数: {avg_length:.1f}")
    logger.info(f"平均最终距离: {avg_final_distance:.3f}")
    logger.info(f"碰撞率: {collision_rate:.3f} ({collision_episodes}/{n_episodes} episodes, 平均碰撞 {avg_collision_count:.2f} 次/episode)")
    logger.info(f"残差幅度‖Δv‖: 成功 {avg_residual_success:.4f} / 失败 {avg_residual_fail:.4f}")
    if successful_episodes:
        logger.info(f"成功episode平均奖励: {avg_success_reward:.3f}")
        logger.info(f"成功episode平均步数: {avg_success_length:.1f}")
    
    return evaluation_results

def plot_results(results, save_path: str = None):
    """绘制评估结果"""
    logger = logging.getLogger(__name__)
    
    # 创建图形
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    fig.suptitle('抓取任务评估结果', fontsize=16)
    
    # 1. 奖励分布
    rewards = [r['episode_reward'] for r in results['episode_results']]
    axes[0, 0].hist(rewards, bins=20, alpha=0.7, color='blue')
    axes[0, 0].set_title('奖励分布')
    axes[0, 0].set_xlabel('奖励')
    axes[0, 0].set_ylabel('频次')
    axes[0, 0].axvline(results['avg_reward'], color='red', linestyle='--', 
                       label=f'平均: {results["avg_reward"]:.3f}')
    axes[0, 0].legend()
    
    # 2. 步数分布
    lengths = [r['episode_length'] for r in results['episode_results']]
    axes[0, 1].hist(lengths, bins=20, alpha=0.7, color='green')
    axes[0, 1].set_title('步数分布')
    axes[0, 1].set_xlabel('步数')
    axes[0, 1].set_ylabel('频次')
    axes[0, 1].axvline(results['avg_episode_length'], color='red', linestyle='--',
                       label=f'平均: {results["avg_episode_length"]:.1f}')
    axes[0, 1].legend()
    
    # 3. 成功率随时间变化
    success_rates = []
    window_size = 10
    for i in range(window_size, len(results['episode_results']) + 1):
        window = results['episode_results'][i-window_size:i]
        success_rate = sum(1 for r in window if r['grasp_success']) / len(window)
        success_rates.append(success_rate)
    
    axes[1, 0].plot(range(window_size, len(results['episode_results']) + 1), success_rates)
    axes[1, 0].set_title(f'成功率变化 (窗口大小: {window_size})')
    axes[1, 0].set_xlabel('Episode')
    axes[1, 0].set_ylabel('成功率')
    axes[1, 0].axhline(results['success_rate'], color='red', linestyle='--',
                       label=f'总体成功率: {results["success_rate"]:.3f}')
    axes[1, 0].legend()
    
    # 4. 距离分布
    distances = [r['final_distance'] for r in results['episode_results']]
    axes[1, 1].hist(distances, bins=20, alpha=0.7, color='orange')
    axes[1, 1].set_title('最终距离分布')
    axes[1, 1].set_xlabel('距离')
    axes[1, 1].set_ylabel('频次')
    axes[1, 1].axvline(results['avg_final_distance'], color='red', linestyle='--',
                       label=f'平均: {results["avg_final_distance"]:.3f}')
    axes[1, 1].legend()
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        logger.info(f"结果图已保存到: {save_path}")
    
    # P0-3: 默认不弹窗（Agg 后端 + 显式开关），避免无头环境 Qt 崩溃
    if os.environ.get('SHOW_PLOTS') == '1':
        plt.show()

def save_results(results, save_path: str):
    """保存评估结果"""
    logger = logging.getLogger(__name__)
    
    import json
    
    # 准备保存的数据
    save_data = {
        'evaluation_time': datetime.now().isoformat(),
        'n_episodes': results['n_episodes'],
        'success_rate': results['success_rate'],
        'avg_reward': results['avg_reward'],
        'avg_episode_length': results['avg_episode_length'],
        'avg_final_distance': results['avg_final_distance'],
        'avg_success_reward': results['avg_success_reward'],
        'avg_success_length': results['avg_success_length'],
        'collision_rate': results['collision_rate'],
        'collision_episodes': results['collision_episodes'],
        'avg_collision_count': results['avg_collision_count'],
        'avg_residual_norm_success': results['avg_residual_norm_success'],
        'avg_residual_norm_fail': results['avg_residual_norm_fail'],
        'episode_summaries': [
            {
                'episode_reward': r['episode_reward'],
                'episode_length': r['episode_length'],
                'grasp_success': r['grasp_success'],
                'final_distance': r['final_distance'],
                'collision_count': r['collision_count'],
                'avg_residual_norm': r['avg_residual_norm']
            }
            for r in results['episode_results']
        ]
    }
    
    # 保存到文件
    with open(save_path, 'w', encoding='utf-8') as f:
        json.dump(save_data, f, indent=2, ensure_ascii=False)
    
    logger.info(f"评估结果已保存到: {save_path}")

def main():
    """主函数"""
    parser = argparse.ArgumentParser(description="强化学习抓取评估")
    parser.add_argument("--model_path", type=str, required=True, help="模型路径")
    parser.add_argument("--n_episodes", type=int, default=50, help="评估episode数量")
    parser.add_argument("--render", action="store_true", help="是否渲染")
    parser.add_argument("--save_results", type=str, help="结果保存路径")
    parser.add_argument("--save_plot", type=str, help="图表保存路径")
    parser.add_argument("--log_level", type=str, default="INFO", help="日志级别")
    # D1/D2/D3 训练口径（2026-08-25）：评估 D2/D3 避障模型必须与训练配置一致，
    # 否则 config.py 默认（delta/无障碍）会测出与训练不符的错误结果
    parser.add_argument('--action-mode', type=str, default='', choices=['', 'delta', 'residual'],
                        help='D1 动作模式（评估 residual 模型必须传 residual）')
    parser.add_argument('--position-random', action='store_true',
                        help='启用物体位置随机（课程学习）；默认固定位置')
    parser.add_argument('--radius', type=float, default=0.03, help='位置随机半径(m)')
    parser.add_argument('--obstacle', action='store_true', help='D1 激活障碍 body')
    parser.add_argument('--obstacle-vel', type=float, default=0.0, help='D1 障碍移动速度(m/s)')
    parser.add_argument('--obstacle-on-path', action='store_true',
                        help='D2 静态障碍放标称必经之路')
    parser.add_argument('--obstacle-pos', type=str, default='',
                        help='D2 固定障碍位置 "x,y,z"（与 --obstacle-on-path 二选一）')
    parser.add_argument('--obstacle-w', type=float, default=-1.0,
                        help='D2 障碍接近惩罚权重（<0 用 config 默认）')
    parser.add_argument('--residual-reg-w', type=float, default=-1.0,
                        help='D2 残差幅度正则权重（<0 用 config 默认）')
    parser.add_argument('--obstacle-mix-ratio', type=float, default=-1.0,
                        help='D3 无障碍混合采样比例（0~1；<0 用 config 默认 0）')
    parser.add_argument('--collision-penalty', type=float, default=None,
                        help='D3 臂-障碍碰撞当步惩罚（评估口径；≤0 有效；None=用 config 默认）')
    parser.add_argument('--collision-max-streak', type=int, default=None,
                        help='D3 连续碰撞步数阈值→truncated（评估口径；0=禁用；None=用 config 默认）')
    parser.add_argument('--obstacle-path-lateral-range', type=str, default='',
                        help='D3 v8 障碍侧偏随机化区间 "min,max"（m；每-episode 随机；空=固定）')
    parser.add_argument('--obstacle-count', type=int, default=None,
                        help='v12 激活的静态障碍数量 1~3（覆盖 config.obstacle_count；None 用默认 1）。'
                             '>1 时沿必经之路不同 fraction/lateral 排布，观测仍只给最近激活障碍槽位（67 维不变）')
    parser.add_argument('--scenario-mix', type=str, default='',
                        help='P3 评估口径（2026-08-27）：per-episode 场景采样 "静态无障,动态目标无障,动态目标+动态障碍"'
                             ' 概率（如 0.4,0.3,0.3；空=旧机制全跟随全局开关）。需配合 --target-vel --obstacle')
    parser.add_argument('--target-vel', type=float, default=-1.0,
                        help='P3 动态目标速度 (m/s)（>=0 覆盖 config.target_vel_xy；<0 用默认 0）')
    parser.add_argument('--target-axis', type=str, default='',
                        help='P3 动态目标运动轴（空=用 config.target_motion_axis）')
    
    args = parser.parse_args()
    
    # 设置日志
    logger = setup_logging(args.log_level)
    
    try:
        # 获取配置
        config = get_config()

        # D1/D2/D3 训练口径（2026-08-25）：与 train_with_monitor.py CLI 语义一致。
        # 评估 D2/D3 避障模型必须与训练配置一致，否则 config.py 默认（delta/无障碍）
        # 会测出与训练不符的错误结果。
        config.grasping.render_gui = False
        if args.action_mode:
            config.grasping.action_mode = args.action_mode
        if args.obstacle or args.obstacle_on_path:
            config.grasping.obstacle_enabled = True
        if args.obstacle_on_path:
            config.grasping.obstacle_on_nominal_path = True
        if args.obstacle_vel > 0:
            config.grasping.obstacle_vel = args.obstacle_vel
        if args.obstacle_pos:
            config.grasping.obstacle_fixed_pos = tuple(float(v) for v in args.obstacle_pos.split(','))
            config.grasping.obstacle_enabled = True
        if args.obstacle_w >= 0:
            config.reward.obstacle_w = args.obstacle_w
        if args.residual_reg_w >= 0:
            config.reward.residual_reg_w = args.residual_reg_w
        if args.obstacle_mix_ratio >= 0:
            config.grasping.obstacle_mix_ratio = args.obstacle_mix_ratio
        if args.collision_penalty is not None:
            config.grasping.obstacle_collision_penalty = args.collision_penalty
        if args.collision_max_streak is not None:
            config.grasping.obstacle_collision_max_streak = args.collision_max_streak
        if args.obstacle_path_lateral_range:
            _lo, _hi = (float(v) for v in args.obstacle_path_lateral_range.split(','))
            config.grasping.obstacle_path_lateral_range = (_lo, _hi)
        if args.obstacle_count is not None:
            config.grasping.obstacle_count = args.obstacle_count
        if args.scenario_mix:
            _v = tuple(float(x) for x in args.scenario_mix.split(','))
            if len(_v) != 3:
                raise SystemExit('--scenario-mix 需要 3 个概率 "静态,动态,动态+障碍"（如 0.4,0.3,0.3）')
            config.grasping.scenario_mix = _v
            config.grasping.dynamic_target_enabled = True
            config.grasping.obstacle_enabled = True
        if args.target_vel >= 0:
            config.grasping.target_vel_xy = args.target_vel
        if args.target_axis:
            config.grasping.target_motion_axis = args.target_axis
        # 物体位置：默认固定（对齐 train_with_monitor 无 --position-random 的行为）；
        # --position-random 时围绕 object_fixed_pos ±radius 随机（收窄 workspace_bounds）
        if args.position_random:
            cx, cy = config.grasping.object_fixed_pos
            r = max(0.001, float(args.radius))
            _z = config.grasping.workspace_bounds[2]
            config.grasping.use_fixed_position = False
            config.grasping.workspace_bounds = ((cx - r, cx + r), (cy - r, cy + r), _z)
        else:
            config.grasping.use_fixed_position = True
        
        # 设置默认保存路径
        results_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results")
        os.makedirs(results_dir, exist_ok=True)
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        if args.save_results is None:
            args.save_results = os.path.join(results_dir, f"evaluation_results_{timestamp}.json")
        if args.save_plot is None:
            args.save_plot = os.path.join(results_dir, f"evaluation_plot_{timestamp}.png")
        
        # 加载模型
        agent, env = load_model(args.model_path, config)
        
        # 评估模型
        results = evaluate_model(agent, env, args.n_episodes, args.render)
        
        # 绘制结果
        plot_results(results, args.save_plot)
        
        # 保存结果
        save_results(results, args.save_results)
        
        logger.info("评估完成")
        
    except Exception as e:
        logger.error(f"评估失败: {e}")
        raise

if __name__ == "__main__":
    main()
