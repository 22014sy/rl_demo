"""
训练监控系统
提供清晰的训练进度指标和可视化
"""

import numpy as np
import matplotlib.pyplot as plt
import json
import os
from datetime import datetime
from typing import Dict, List, Optional
import logging

# 设置matplotlib中文字体
plt.rcParams['font.sans-serif'] = ['SimHei', 'DejaVu Sans', 'Arial Unicode MS', 'sans-serif']
plt.rcParams['axes.unicode_minus'] = False


def _json_default(o):
    """json.dump 的 numpy 类型转换（P1 修复：np.bool_/np.float64 等不可直接序列化，
    若不做转换 json.dump 会在写完一半时抛异常，把日志文件截断成非法 JSON）"""
    if isinstance(o, np.bool_):
        return bool(o)
    if isinstance(o, np.integer):
        return int(o)
    if isinstance(o, np.floating):
        return float(o)
    return str(o)

class TrainingMonitor:
    """训练监控器"""
    
    def __init__(self, log_dir: str = "logs", save_plots: bool = True, plot_save_freq: int = 200):
        self.log_dir = log_dir
        self.save_plots = save_plots
        self.plot_save_freq = plot_save_freq  # 图表保存频率
        self.logger = logging.getLogger(__name__)
        
        # 训练指标
        self.episode_rewards = []
        self.episode_lengths = []
        self.success_rates = []
        self.singularity_counts = []
        self.episode_times = []
        
        # 统计信息
        self.current_episode = 0
        self.best_reward = float('-inf')
        self.avg_reward_window = 100  # 平均奖励窗口大小
        
        # 创建日志目录
        os.makedirs(log_dir, exist_ok=True)
        
        # 日志文件
        self.log_file = os.path.join(log_dir, f"training_log_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json")
        
        # 初始化日志
        self._init_log()
    
    def _init_log(self):
        """初始化日志文件"""
        self._start_time = datetime.now().isoformat()
        self._logged_episodes = []  # P1: 内存持有全部 episode 记录，避免每次读回文件
        log_data = {
            "start_time": self._start_time,
            "episodes": [],
            "summary": {}
        }
        with open(self.log_file, 'w') as f:
            json.dump(log_data, f, indent=2)
    
    def log_episode(self, episode: int, reward: float, length: int, 
                   success: bool, singularity_count: int = 0, 
                   episode_time: float = 0.0):
        """
        记录一个episode的信息
        
        Args:
            episode: episode编号
            reward: 总奖励
            length: episode长度
            success: 是否成功
            singularity_count: 奇异点次数
            episode_time: episode耗时
        """
        # 更新指标
        self.episode_rewards.append(reward)
        self.episode_lengths.append(length)
        self.success_rates.append(1.0 if success else 0.0)
        self.singularity_counts.append(singularity_count)
        self.episode_times.append(episode_time)
        
        # 使用内部计数，避免外部传入的episode编号不可靠（如始终为0）
        self.current_episode = len(self.episode_rewards)
        
        # 更新最佳奖励
        if reward > self.best_reward:
            self.best_reward = reward
        
        # 计算统计信息
        stats = self._calculate_stats()
        
        # 打印进度信息（使用内部episode编号）
        self._print_progress(self.current_episode, reward, length, success, singularity_count, stats)
        
        # 保存到日志文件（使用内部episode编号）
        self._save_to_log(self.current_episode, reward, length, success, singularity_count, episode_time, stats)
        
        # 定期保存图表（基于内部计数，避免0取模导致的每次都保存）
        if self.current_episode % self.plot_save_freq == 0 and self.save_plots:
            self.save_training_plots()
    
    def _calculate_stats(self) -> Dict:
        """计算统计信息"""
        if len(self.episode_rewards) == 0:
            return {}
        
        # 最近窗口的统计
        window_size = min(self.avg_reward_window, len(self.episode_rewards))
        recent_rewards = self.episode_rewards[-window_size:]
        recent_successes = self.success_rates[-window_size:]
        recent_singularities = self.singularity_counts[-window_size:]
        
        stats = {
            "avg_reward": np.mean(recent_rewards),
            "std_reward": np.std(recent_rewards),
            "success_rate": np.mean(recent_successes) * 100,
            "avg_singularity": np.mean(recent_singularities),
            "best_reward": self.best_reward,
            "total_episodes": len(self.episode_rewards),
            "avg_episode_length": np.mean(self.episode_lengths),
            "avg_episode_time": np.mean(self.episode_times)
        }
        
        return stats
    
    def _print_progress(self, episode: int, reward: float, length: int, 
                       success: bool, singularity_count: int, stats: Dict):
        """打印训练进度"""
        if len(stats) == 0:
            return
        
        # 进度条
        progress_bar = "█" * (episode % 50) + "░" * (50 - episode % 50)
        
        print(f"\n{'='*80}")
        print(f"Episode {episode:4d} | {progress_bar} | {episode % 50:2d}/50")
        print(f"{'='*80}")
        
        # 当前episode信息
        success_symbol = "✅" if success else "❌"
        print(f"当前Episode:")
        print(f"  奖励: {reward:8.2f} | 步数: {length:3d} | 成功: {success_symbol}")
        print(f"  奇异点: {singularity_count:2d} | 耗时: {stats.get('avg_episode_time', 0):.2f}s")
        
        # 统计信息
        print(f"\n训练统计 (最近{min(self.avg_reward_window, len(self.episode_rewards))}个episodes):")
        print(f"  平均奖励: {stats['avg_reward']:8.2f} ± {stats['std_reward']:6.2f}")
        print(f"  成功率:   {stats['success_rate']:6.1f}%")
        print(f"  平均奇异点: {stats['avg_singularity']:6.1f}")
        print(f"  最佳奖励: {stats['best_reward']:8.2f}")
        
        # 训练趋势
        if len(self.episode_rewards) >= 10:
            recent_avg = np.mean(self.episode_rewards[-10:])
            if len(self.episode_rewards) >= 20:
                prev_avg = np.mean(self.episode_rewards[-20:-10])
                trend = "↗️ 上升" if recent_avg > prev_avg else "↘️ 下降" if recent_avg < prev_avg else "➡️ 稳定"
                print(f"  训练趋势: {trend} (最近10 vs 前10: {recent_avg:.2f} vs {prev_avg:.2f})")
    
    def _save_to_log(self, episode: int, reward: float, length: int, 
                    success: bool, singularity_count: int, episode_time: float, stats: Dict):
        """保存到日志文件（P1 修复：不再读回文件，直接以内存数据整体写盘，避免读到写一半的文件）"""
        try:
            self._logged_episodes.append({
                "episode": episode,
                "reward": reward,
                "length": length,
                "success": success,
                "singularity_count": singularity_count,
                "episode_time": episode_time,
                "timestamp": datetime.now().isoformat()
            })

            log_data = {
                "start_time": self._start_time,
                "episodes": self._logged_episodes,
                "summary": stats
            }
            with open(self.log_file, 'w', encoding='utf-8') as f:
                json.dump(log_data, f, indent=2, ensure_ascii=False, default=_json_default)

        except Exception as e:
            self.logger.warning(f"保存日志失败: {e}")
    
    def save_training_plots(self):
        """保存训练图表"""
        if len(self.episode_rewards) < 10:
            return
        
        try:
            # 尝试设置中文字体
            try:
                plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'DejaVu Sans', 'Arial Unicode MS', 'sans-serif']
                plt.rcParams['axes.unicode_minus'] = False
            except:
                # 如果中文字体不可用，使用英文
                plt.rcParams['font.sans-serif'] = ['DejaVu Sans', 'Arial', 'sans-serif']
            
            fig, axes = plt.subplots(2, 2, figsize=(15, 10))
            fig.suptitle('Training Progress Monitor', fontsize=16)
            
            episodes = list(range(1, len(self.episode_rewards) + 1))
            
            # 奖励曲线
            axes[0, 0].plot(episodes, self.episode_rewards, 'b-', alpha=0.6, label='Episode Reward')
            if len(episodes) >= 10:
                # 移动平均
                window = min(10, len(episodes))
                moving_avg = [np.mean(self.episode_rewards[max(0, i-window):i+1]) 
                             for i in range(len(episodes))]
                axes[0, 0].plot(episodes, moving_avg, 'r-', linewidth=2, label=f'{window}-Episode Moving Average')
            axes[0, 0].set_title('Reward Curve')
            axes[0, 0].set_xlabel('Episode')
            axes[0, 0].set_ylabel('Reward')
            axes[0, 0].legend()
            axes[0, 0].grid(True, alpha=0.3)
            
            # 成功率
            if len(self.success_rates) >= 10:
                success_window = 10
                success_moving_avg = [np.mean(self.success_rates[max(0, i-success_window):i+1]) * 100
                                     for i in range(len(self.success_rates))]
                axes[0, 1].plot(episodes, success_moving_avg, 'g-', linewidth=2)
                axes[0, 1].set_title('Success Rate (10-Episode Moving Average)')
                axes[0, 1].set_xlabel('Episode')
                axes[0, 1].set_ylabel('Success Rate (%)')
                axes[0, 1].grid(True, alpha=0.3)
            
            # 奇异点统计
            if len(self.singularity_counts) > 0:
                axes[1, 0].plot(episodes, self.singularity_counts, 'orange', alpha=0.6, label='Singularity Count')
                if len(episodes) >= 10:
                    sing_window = min(10, len(episodes))
                    sing_moving_avg = [np.mean(self.singularity_counts[max(0, i-sing_window):i+1]) 
                                      for i in range(len(episodes))]
                    axes[1, 0].plot(episodes, sing_moving_avg, 'red', linewidth=2, 
                                   label=f'{sing_window}-Episode Moving Average')
                axes[1, 0].set_title('Singularity Statistics')
                axes[1, 0].set_xlabel('Episode')
                axes[1, 0].set_ylabel('Singularity Count')
                axes[1, 0].legend()
                axes[1, 0].grid(True, alpha=0.3)
            
            # Episode长度
            axes[1, 1].plot(episodes, self.episode_lengths, 'purple', alpha=0.6, label='Episode Length')
            if len(episodes) >= 10:
                len_window = min(10, len(episodes))
                len_moving_avg = [np.mean(self.episode_lengths[max(0, i-len_window):i+1]) 
                                 for i in range(len(episodes))]
                axes[1, 1].plot(episodes, len_moving_avg, 'blue', linewidth=2, 
                               label=f'{len_window}-Episode Moving Average')
            axes[1, 1].set_title('Episode Length')
            axes[1, 1].set_xlabel('Episode')
            axes[1, 1].set_ylabel('Steps')
            axes[1, 1].legend()
            axes[1, 1].grid(True, alpha=0.3)
            
            plt.tight_layout()
            
            # 保存图表
            plot_file = os.path.join(self.log_dir, f"training_plots_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png")
            plt.savefig(plot_file, dpi=150, bbox_inches='tight')
            plt.close()
            
            self.logger.info(f"训练图表已保存: {plot_file}")
            
        except Exception as e:
            self.logger.warning(f"保存训练图表失败: {e}")
            # 尝试使用英文标签重新保存
            try:
                self._save_plots_english()
            except Exception as e2:
                self.logger.error(f"英文图表保存也失败: {e2}")
    
    def _save_plots_english(self):
        """使用英文标签保存图表（备用方案）"""
        if len(self.episode_rewards) < 10:
            return
        
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        fig.suptitle('Training Progress Monitor', fontsize=16)
        
        episodes = list(range(1, len(self.episode_rewards) + 1))
        
        # Reward curve
        axes[0, 0].plot(episodes, self.episode_rewards, 'b-', alpha=0.6, label='Episode Reward')
        if len(episodes) >= 10:
            window = min(10, len(episodes))
            moving_avg = [np.mean(self.episode_rewards[max(0, i-window):i+1]) 
                         for i in range(len(episodes))]
            axes[0, 0].plot(episodes, moving_avg, 'r-', linewidth=2, label=f'{window}-Episode Moving Average')
        axes[0, 0].set_title('Reward Curve')
        axes[0, 0].set_xlabel('Episode')
        axes[0, 0].set_ylabel('Reward')
        axes[0, 0].legend()
        axes[0, 0].grid(True, alpha=0.3)
        
        # Success rate
        if len(self.success_rates) >= 10:
            success_window = 10
            success_moving_avg = [np.mean(self.success_rates[max(0, i-success_window):i+1]) * 100
                                 for i in range(len(self.success_rates))]
            axes[0, 1].plot(episodes, success_moving_avg, 'g-', linewidth=2)
            axes[0, 1].set_title('Success Rate (10-Episode Moving Average)')
            axes[0, 1].set_xlabel('Episode')
            axes[0, 1].set_ylabel('Success Rate (%)')
            axes[0, 1].grid(True, alpha=0.3)
        
        # Singularity statistics
        if len(self.singularity_counts) > 0:
            axes[1, 0].plot(episodes, self.singularity_counts, 'orange', alpha=0.6, label='Singularity Count')
            if len(episodes) >= 10:
                sing_window = min(10, len(episodes))
                sing_moving_avg = [np.mean(self.singularity_counts[max(0, i-sing_window):i+1]) 
                                  for i in range(len(episodes))]
                axes[1, 0].plot(episodes, sing_moving_avg, 'red', linewidth=2, 
                               label=f'{sing_window}-Episode Moving Average')
            axes[1, 0].set_title('Singularity Statistics')
            axes[1, 0].set_xlabel('Episode')
            axes[1, 0].set_ylabel('Singularity Count')
            axes[1, 0].legend()
            axes[1, 0].grid(True, alpha=0.3)
        
        # Episode length
        axes[1, 1].plot(episodes, self.episode_lengths, 'purple', alpha=0.6, label='Episode Length')
        if len(episodes) >= 10:
            len_window = min(10, len(episodes))
            len_moving_avg = [np.mean(self.episode_lengths[max(0, i-len_window):i+1]) 
                             for i in range(len(episodes))]
            axes[1, 1].plot(episodes, len_moving_avg, 'blue', linewidth=2, 
                           label=f'{len_window}-Episode Moving Average')
        axes[1, 1].set_title('Episode Length')
        axes[1, 1].set_xlabel('Episode')
        axes[1, 1].set_ylabel('Steps')
        axes[1, 1].legend()
        axes[1, 1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        plot_file = os.path.join(self.log_dir, f"training_plots_english_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png")
        plt.savefig(plot_file, dpi=150, bbox_inches='tight')
        plt.close()
        
        self.logger.info(f"英文训练图表已保存: {plot_file}")
    
    def get_training_summary(self) -> Dict:
        """获取训练摘要"""
        if len(self.episode_rewards) == 0:
            return {"message": "暂无训练数据"}
        
        stats = self._calculate_stats()
        
        summary = {
            "总episodes": len(self.episode_rewards),
            "最佳奖励": self.best_reward,
            "平均奖励": stats['avg_reward'],
            "成功率": f"{stats['success_rate']:.1f}%",
            "平均奇异点": stats['avg_singularity'],
            "平均episode长度": stats['avg_episode_length'],
            "平均episode时间": f"{stats['avg_episode_time']:.2f}s"
        }
        
        return summary
    
    def print_final_summary(self):
        """打印最终训练摘要"""
        summary = self.get_training_summary()
        
        print("\n" + "="*80)
        print("🎯 训练完成摘要")
        print("="*80)
        
        for key, value in summary.items():
            print(f"{key:15s}: {value}")
        
        print("="*80)
