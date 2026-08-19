"""
强化学习抓取智能体
基于Stable-Baselines3的PPO算法，专门用于抓取任务
"""

import torch
import torch.nn as nn
import numpy as np
from typing import Dict, Tuple, Optional, Union
import logging
import os
from stable_baselines3.common.monitor import Monitor

# 尝试导入stable-baselines3
try:
    from stable_baselines3 import PPO
    from stable_baselines3.common.policies import ActorCriticPolicy
    from stable_baselines3.common.vec_env import DummyVecEnv
    from stable_baselines3.common.callbacks import BaseCallback, CheckpointCallback
    SB3_AVAILABLE = True
    logger = logging.getLogger(__name__)
    logger.info("✅ Stable-Baselines3 可用")
except ImportError:
    SB3_AVAILABLE = False
    logger = logging.getLogger(__name__)
    logger.warning("⚠️  Stable-Baselines3 不可用")

from config import NetworkConfig, TrainingConfig
from environment import PandaGraspingEnv
from training_monitor import TrainingMonitor
# 归一化功能由Stable-Baselines3内置提供

class GraspingCallback(BaseCallback):
    """抓取任务专用回调函数"""
    
    def __init__(self, verbose: int = 0, success_threshold: float = 0.8, patience: int = 30):
        super().__init__(verbose)
        self.success_count = 0
        self.total_episodes = 0
        self.episode_rewards = []
        self.episode_lengths = []
        
        # 早停参数
        self.success_threshold = success_threshold
        self.patience = patience
        self.recent_successes = []  # 记录最近的成功情况
        self.should_stop = False
        
    def _on_step(self) -> bool:
        """每步调用（P1 修复：从 info 读取 grasp_success，而不是访问 Monitor 上不存在的 env.task_state）"""
        # 如果应该停止，返回False
        if self.should_stop:
            return False

        infos = self.locals.get('infos', [])
        for info in infos:
            # 终端步若 grasp_success=True，记一次成功
            if info.get('grasp_success', False):
                self.success_count += 1
            # SB3 Monitor 在每个 episode 结束时写入 info['episode']，用于统计 episode 数
            if info.get('episode') is not None:
                self.total_episodes += 1
                self.episode_rewards.append(info['episode'].get('r', 0.0))
                self.episode_lengths.append(info['episode'].get('l', 0))

        return True
    
    def _on_rollout_end(self) -> None:
        """每个rollout结束时调用"""
        if self.total_episodes > 0:
            success_rate = self.success_count / self.total_episodes
            avg_reward = np.mean(self.episode_rewards) if self.episode_rewards else 0
            avg_length = np.mean(self.episode_lengths) if self.episode_lengths else 0
            
            logger.info(f"Rollout结束 - 成功率: {success_rate:.3f}, 平均奖励: {avg_reward:.3f}, 平均步数: {avg_length:.1f}")
            
            # 检查早停条件
            self._check_early_stopping(success_rate)
            
            # 重置计数器
            self.success_count = 0
            self.total_episodes = 0
            self.episode_rewards = []
            self.episode_lengths = []
    
    def _check_early_stopping(self, success_rate: float):
        """检查早停条件"""
        # 记录最近的成功率
        self.recent_successes.append(success_rate)
        
        # 只保留最近的patience个记录
        if len(self.recent_successes) > self.patience:
            self.recent_successes.pop(0)
        
        # 如果记录足够多，检查是否满足早停条件
        if len(self.recent_successes) >= self.patience:
            # 检查最近patience个episodes是否都达到阈值
            recent_high_success = [s >= self.success_threshold for s in self.recent_successes[-self.patience:]]
            if all(recent_high_success):
                avg_success_rate = np.mean(self.recent_successes[-self.patience:])
                logger.info(f"🎯 早停条件满足！")
                logger.info(f"   最近{self.patience}个episodes都达到成功率阈值")
                logger.info(f"   平均成功率: {avg_success_rate:.3f}")
                logger.info(f"   成功率阈值: {self.success_threshold}")
                logger.info(f"   训练将提前结束")
                self.should_stop = True

class GraspingAgent:
    """
    抓取任务智能体
    使用PPO算法，专门优化用于抓取任务
    """
    
    def __init__(self, 
                 network_config: NetworkConfig,
                 training_config: TrainingConfig,
                 model_path: str = None):
        """
        初始化智能体
        
        Args:
            network_config: 网络配置
            training_config: 训练配置
            model_path: 预训练模型路径（可选）
        """
        self.network_config = network_config
        self.training_config = training_config
        self.model_path = model_path
        
        self.agent = None
        self.env = None
        
        logger.info("抓取智能体初始化完成")
    
    def set_environment(self, env):
        """
        设置训练环境
        
        Args:
            env: Gymnasium环境
        """
        # 设置环境：Monitor 包装真实环境，在 info 中输出 episode 统计
        env = Monitor(env)
        self.env = env
        
        # 创建训练监控器
        logs_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "logs")
        self.training_monitor = TrainingMonitor(
            log_dir=logs_dir, 
            save_plots=True,
            plot_save_freq=2000  # 每2000个episodes保存一次图表
        )
        
        # P1 修复：把监控器挂到真实环境（PandaGraspingEnv）上。
        # 旧代码 `hasattr(self.env, 'training_monitor')` 检查的是 Monitor 包装层，
        # 永远为 False，导致 training_monitor 从未被挂载，JSON 里 episodes 一直为空。
        real_env = env.env if hasattr(env, 'env') else env
        real_env.training_monitor = self.training_monitor
        
        # 创建归一化向量化环境
        from vec_normalize_wrapper import create_normalized_env
        
        vec_env = create_normalized_env(
            env,
            norm_obs=self.training_config.normalize_observations,
            norm_reward=self.training_config.normalize_rewards,
            clip_obs=self.training_config.norm_obs_clip,
            clip_reward=self.training_config.norm_reward_clip
        )
        
        # 优势函数归一化由Stable-Baselines3自动处理
        self.advantage_normalizer = None
        
        if not SB3_AVAILABLE:
            raise ImportError("Stable-Baselines3不可用，无法创建智能体")
        
        # 获取logs目录路径
        logs_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "logs")
        os.makedirs(logs_dir, exist_ok=True)
        
        # 创建PPO智能体
        self.agent = PPO(
            "MlpPolicy",
            vec_env,
            verbose=1,
            learning_rate=self.training_config.learning_rate,
            n_steps=self.training_config.n_steps,
            batch_size=self.training_config.batch_size,
            n_epochs=self.training_config.n_epochs,
            gamma=self.training_config.gamma,
            gae_lambda=self.training_config.gae_lambda,
            clip_range=self.training_config.clip_range,
            ent_coef=self.training_config.ent_coef,
            vf_coef=self.training_config.vf_coef,
            max_grad_norm=self.training_config.max_grad_norm,
            tensorboard_log=logs_dir,  # 设置tensorboard日志路径
            device="cpu",  # 强制使用CPU，避免GPU警告
            policy_kwargs={
                "net_arch": {
                    "pi": self.network_config.policy_hidden_sizes,
                    "vf": self.network_config.value_hidden_sizes
                }
            }
        )
        
        # 加载预训练模型
        if self.model_path and os.path.exists(self.model_path):
            self.agent = PPO.load(self.model_path, env=vec_env)
            logger.info(f"加载预训练模型: {self.model_path}")
        
        logger.info("智能体环境设置完成")
    
    def train(self, total_timesteps: int = None, save_path: str = None):
        """
        训练智能体
        
        Args:
            total_timesteps: 总训练步数
            save_path: 模型保存路径
        """
        if self.agent is None:
            raise ValueError("智能体未初始化，请先调用set_environment")
        
        if total_timesteps is None:
            total_timesteps = self.training_config.total_timesteps
        
        # 创建回调函数
        callbacks = []
        
        # 抓取任务回调（包含早停逻辑）
        grasping_callback = GraspingCallback(
            success_threshold=self.training_config.success_threshold,
            patience=self.training_config.patience
        )
        callbacks.append(grasping_callback)
        
        # 检查点回调
        if save_path:
            checkpoint_callback = CheckpointCallback(
                save_freq=self.training_config.save_freq,
                save_path=os.path.dirname(save_path),
                name_prefix=os.path.basename(save_path).replace('.zip', '')
            )
            callbacks.append(checkpoint_callback)
        
        # 开始训练
        logger.info(f"开始训练，总步数: {total_timesteps}")
        
        # 训练（P1：移除未注册的死代码 episode_callback；episode 记录由真实环境自行完成）
        self.agent.learn(
            total_timesteps=total_timesteps,
            callback=callbacks,
            progress_bar=True
        )
        
        # 保存最终模型
        if save_path:
            self.agent.save(save_path)
            logger.info(f"模型已保存到: {save_path}")
    
    def predict(self, observation: np.ndarray, deterministic: bool = True) -> Tuple[np.ndarray, Optional[np.ndarray]]:
        """
        预测动作
        
        Args:
            observation: 观察
            deterministic: 是否使用确定性策略
            
        Returns:
            Tuple[np.ndarray, Optional[np.ndarray]]: 动作和状态
        """
        if self.agent is None:
            raise ValueError("智能体未初始化")
        
        return self.agent.predict(observation, deterministic=deterministic)
    
    def evaluate(self, env, n_eval_episodes: int = 10) -> Dict[str, float]:
        """
        评估智能体性能
        
        Args:
            env: 评估环境
            n_eval_episodes: 评估episode数量
            
        Returns:
            Dict[str, float]: 评估结果
        """
        if self.agent is None:
            raise ValueError("智能体未初始化")
        
        success_count = 0
        total_rewards = []
        episode_lengths = []
        
        for episode in range(n_eval_episodes):
            obs, _ = env.reset()
            episode_reward = 0
            episode_length = 0
            
            while True:
                action, _ = self.predict(obs, deterministic=True)
                obs, reward, terminated, truncated, info = env.step(action)
                
                episode_reward += reward
                episode_length += 1
                
                if terminated or truncated:
                    break
            
            # 检查是否成功
            if info.get('grasp_success', False):
                success_count += 1
            
            total_rewards.append(episode_reward)
            episode_lengths.append(episode_length)
        
        # 计算评估指标
        success_rate = success_count / n_eval_episodes
        avg_reward = np.mean(total_rewards)
        avg_length = np.mean(episode_lengths)
        
        results = {
            'success_rate': success_rate,
            'avg_reward': avg_reward,
            'avg_episode_length': avg_length,
            'n_episodes': n_eval_episodes
        }
        
        logger.info(f"评估结果 - 成功率: {success_rate:.3f}, 平均奖励: {avg_reward:.3f}, 平均步数: {avg_length:.1f}")
        
        return results
    
    def save(self, path: str):
        """保存模型"""
        if self.agent is None:
            raise ValueError("智能体未初始化")
        
        self.agent.save(path)
        logger.info(f"模型已保存到: {path}")
    
    def load(self, path: str):
        """加载模型"""
        if not os.path.exists(path):
            raise FileNotFoundError(f"模型文件不存在: {path}")
        
        if self.env is None:
            raise ValueError("请先设置环境")
        
        # 创建向量化环境
        vec_env = DummyVecEnv([lambda: self.env])
        
        # 加载模型
        self.agent = PPO.load(path, env=vec_env)
        logger.info(f"模型已从 {path} 加载")
    
    def get_policy(self):
        """获取策略网络"""
        if self.agent is None:
            raise ValueError("智能体未初始化")
        
        return self.agent.policy
    
    def get_value_function(self):
        """获取价值函数"""
        if self.agent is None:
            raise ValueError("智能体未初始化")
        
        return self.agent.policy.value_net
