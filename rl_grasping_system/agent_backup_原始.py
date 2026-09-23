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

# 并行改造(2026-08-22)：SB3 SubprocVecEnv 在 Linux 默认用 forkserver 启动方式，
# 要求主模块有 `if __name__ == '__main__'` 保护。为不依赖调用方写法，显式选 fork
#（Linux 直接 fork，快、COW 共享内存，且不重新 import 主模块）；无 fork 的平台用默认。
def _subproc_start_method():
    try:
        import multiprocessing as _mp
        if "fork" in _mp.get_all_start_methods():
            return "fork"
    except Exception:
        pass
    return None

# 尝试导入stable-baselines3
try:
    from stable_baselines3 import PPO
    from stable_baselines3.common.policies import ActorCriticPolicy
    from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv
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
    """抓取任务专用回调函数

    并行改造(2026-08-22)：
      1. episode 统计改用 info['episode']（Monitor 每 episode 写入一次），
         多环境并行下安全；不再用单环境专用的 _prev_grasp_success 上升沿去重（多 env 会串扰）。
      2. SubprocVecEnv 下 worker 内环境不挂 training_monitor（多进程写同一 JSON 冲突），
         由主进程在此汇总 episode 数据后记录。
      3. 新增演示播放：每 demo_every_n_rollouts 个 rollout 用当前策略在 demo_env
         （render_gui=True，主进程）跑一个 episode，弹窗看训练效果。
    """

    def __init__(self, verbose: int = 0, success_threshold: float = 0.8, patience: int = 30,
                 demo_env=None, demo_every_n_rollouts: int = 5, demo_max_steps: Optional[int] = None):
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

        # 并行改造：主进程 TrainingMonitor 引用（并行模式由这里记录 episode；单机模式为 None，环境自记）
        self.training_monitor = None

        # 演示播放参数
        self.demo_env = demo_env
        self.demo_every_n_rollouts = demo_every_n_rollouts
        self.demo_max_steps = demo_max_steps
        self._rollout_count = 0

    def _on_step(self) -> bool:
        """每步调用。从 info 读取 episode 结束标志与 grasp_success。"""
        # 如果应该停止，返回False
        if self.should_stop:
            return False

        infos = self.locals.get('infos', [])
        for info in infos:
            # Monitor 在每个 episode 结束时写入 info['episode']，此时一并统计（每 episode 恰一次，多 env 安全）
            if info.get('episode') is not None:
                self.total_episodes += 1
                ep_r = info['episode'].get('r', 0.0)
                ep_l = info['episode'].get('l', 0)
                ep_success = bool(info.get('grasp_success', False))
                self.episode_rewards.append(ep_r)
                self.episode_lengths.append(ep_l)
                if ep_success:
                    self.success_count += 1
                # 并行模式：主进程统一写 TrainingMonitor（worker 不写文件）
                if self.training_monitor is not None:
                    self.training_monitor.log_episode(
                        episode=self.total_episodes,
                        reward=ep_r,
                        length=ep_l,
                        success=ep_success,
                        singularity_count=int(info.get('singularity_count', 0)),
                        episode_time=float(info.get('episode_time', 0.0)),
                        breakdown=info.get('episode_breakdown') or None,
                    )

        return True

    def _on_rollout_end(self) -> None:
        """每个rollout结束时调用"""
        self._rollout_count += 1
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

        # 演示播放：每 demo_every_n_rollouts 个 rollout 用当前策略跑一个渲染 episode
        if (self.demo_env is not None
                and self._rollout_count % self.demo_every_n_rollouts == 0):
            self._play_demo()

    def _play_demo(self) -> None:
        """用当前策略在演示环境（render_gui=True）播放一个 episode，弹窗观察。"""
        if self.model is None or self.demo_env is None:
            return
        try:
            # 并行改造 2026-08-22：SB3 PPO.predict 不自动归一化，演示必须先把原始 obs
            # 经 VecNormalize 归一化，否则观测分布失配、策略乱走（演示会骗人）。
            vn = self.model.get_vec_normalize_env()
            use_norm = vn is not None and getattr(vn, 'norm_obs', False)
            if use_norm:
                _was_training = vn.training
                vn.training = False  # 演示不更新观测统计

            max_steps = int(self.demo_max_steps) if self.demo_max_steps else 300
            obs, _ = self.demo_env.reset()
            steps = 0
            info = {}
            logger.info("🎬 演示当前策略（渲染窗口）...")
            while steps < max_steps:
                _obs_in = vn.normalize_obs(obs) if use_norm else obs
                action, _ = self.model.predict(_obs_in, deterministic=True)
                obs, _reward, terminated, truncated, info = self.demo_env.step(action)
                steps += 1
                if terminated or truncated:
                    break
            if use_norm:
                vn.training = _was_training
            logger.info(f"🎬 演示结束：{steps} 步，grasp_success={bool(info.get('grasp_success', False))}")
        except Exception as e:
            logger.warning(f"演示播放失败: {e}")

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
    
    def set_environment(self, env, n_envs: int = 1):
        """
        设置训练环境

        并行改造(2026-08-22)：
          - env 为单环境实例且 n_envs<=1：保持原单机行为（DummyVecEnv + Monitor + 挂 training_monitor）。
          - env 为"环境工厂" callable 且 n_envs>1：SubprocVecEnv 多进程并行。
            工厂无参、每次调用返回一个新环境（MuJoCo model/data 不可 pickle，
            故由子进程各自调用工厂创建；工厂函数本身可经 cloudpickle 序列化）。
            并行模式下 training_monitor 不挂 worker（多进程写同一 JSON 冲突），
            episode 统计改由主进程 GraspingCallback 从 info 汇总后记录。

        Args:
            env: Gymnasium 环境实例（单机）或环境工厂 callable（并行）
            n_envs: 并行环境数
        """
        from stable_baselines3.common.vec_env import VecNormalize, DummyVecEnv, SubprocVecEnv

        self.n_envs = max(1, int(n_envs))

        # 创建训练监控器
        logs_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "logs")
        self.training_monitor = TrainingMonitor(
            log_dir=logs_dir,
            save_plots=True,
            plot_save_freq=2000  # 每2000个episodes保存一次图表
        )

        if self.n_envs > 1 and callable(env):
            # ---- 并行模式：SubprocVecEnv，子进程内新建环境 ----
            self._parallel = True

            def _make_env():
                # 每个 worker 内：Monitor 包装真实环境（info['episode'] 统计）
                return Monitor(env())

            vec_env = SubprocVecEnv(
                [_make_env for _ in range(self.n_envs)],
                start_method=_subproc_start_method(),  # 并行改造：显式 fork，避免 forkserver 的 __main__ 依赖
            )
            self.env = vec_env
            # 注意：worker 内环境不挂 training_monitor（多进程写同一 JSON 冲突）
            logger.info(f"已创建 {self.n_envs} 个并行环境（SubprocVecEnv）")
        else:
            # ---- 单机模式：保持原行为 ----
            self._parallel = False
            env = Monitor(env)
            self.env = env

            # P1 修复：把监控器挂到真实环境（GraspingEnv）上。
            # 旧代码 `hasattr(self.env, 'training_monitor')` 检查的是 Monitor 包装层，
            # 永远为 False，导致 training_monitor 从未被挂载，JSON 里 episodes 一直为空。
            real_env = env.env if hasattr(env, 'env') else env
            real_env.training_monitor = self.training_monitor

            vec_env = DummyVecEnv([lambda: env])

        # 创建归一化向量化环境（Task3: 官方 SB3 VecNormalize，替换自定义 VecNormalizeWrapper——
        # 修复 save/load/reset/PPO 集成缺陷，保证训练-评估-部署观测一致；见 docs/VecNormalize观测归一化替换.md）
        vec_env = VecNormalize(
            vec_env,
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

        # 加载预训练模型（迁移学习）
        if self.model_path and os.path.exists(self.model_path):
            # 2026-08-23 迁移学习恢复 VecNormalize 观测统计：若存在 <model>_vecnormalize.pkl，
            # 先恢复 obs_rms/ret_rms，否则迁移初期观测归一化失配（stats 为初始值），
            # 策略输出与训练时不一致，fine-tune 慢甚至失败。
            # 2026-08-24 修复：VecNormalize.load 的 venv 参数必须传底层 VecEnv（vec_env.venv），
            # 不能传 VecNormalize 实例本身——否则 load 后出现 "VecNormalize 嵌套 VecNormalize"，
            # vn.venv.reset()/step() 实际调用内层 VecNormalize 的归一化逻辑，
            # 返回异常（近全 0 / 双重归一化）观测 → 训练 rollout 观测错误 → 迁移训练成功率恒 0%。
            _norm_pkl = os.path.splitext(self.model_path)[0] + "_vecnormalize.pkl"
            if os.path.exists(_norm_pkl):
                _base = vec_env.venv if isinstance(vec_env, VecNormalize) else vec_env
                vec_env = VecNormalize.load(_norm_pkl, _base)
                logger.info(f"迁移学习：已恢复 VecNormalize 观测统计 {_norm_pkl}（venv={type(_base).__name__}）")
            self.agent = PPO.load(self.model_path, env=vec_env)
            logger.info(f"加载预训练模型: {self.model_path}")

        logger.info("智能体环境设置完成")
    
    def train(self, total_timesteps: int = None, save_path: str = None, extra_callbacks: list = None,
              demo_env=None, demo_every_n_rollouts: int = 5, demo_max_steps: Optional[int] = None):
        """
        训练智能体

        Args:
            total_timesteps: 总训练步数
            save_path: 模型保存路径
            extra_callbacks: 外部回调（如 40k 步后关 GLFW）
            demo_env: 演示环境（render_gui=True，主进程），训练中定期用当前策略播放（并行改造新增）
            demo_every_n_rollouts: 每隔多少个 rollout 播放一次演示
            demo_max_steps: 演示 episode 最大步数（None=300）
        """
        if self.agent is None:
            raise ValueError("智能体未初始化，请先调用set_environment")

        if total_timesteps is None:
            total_timesteps = self.training_config.total_timesteps

        # 创建回调函数
        callbacks = []

        # 抓取任务回调（包含早停逻辑 + 演示播放）
        grasping_callback = GraspingCallback(
            success_threshold=self.training_config.success_threshold,
            patience=self.training_config.patience,
            demo_env=demo_env,
            demo_every_n_rollouts=demo_every_n_rollouts,
            demo_max_steps=demo_max_steps,
        )
        # 并行模式：主进程汇总 episode 记录到 TrainingMonitor（单机模式环境自行记录，避免重复）
        grasping_callback.training_monitor = self.training_monitor if getattr(self, '_parallel', False) else None
        callbacks.append(grasping_callback)

        # 检查点回调（2026-08-23: SB3 CheckpointCallback 的 save_freq 是"callback 调用次数"，
        # 每 n_envs 步调用一次——须除以 n_envs 才是实际训练步数，否则永远不触发）
        if save_path:
            _n_envs = max(int(getattr(self, 'n_envs', 1)), 1)
            checkpoint_callback = CheckpointCallback(
                save_freq=max(int(self.training_config.save_freq) // _n_envs, 1),
                save_path=os.path.dirname(save_path),
                name_prefix=os.path.basename(save_path).replace('.zip', ''),
                # 2026-08-24: 连带保存 VecNormalize 观测统计（否则 checkpoint 无 pkl，
                # 崩溃恢复/续训时观测归一化失配 → 迁移/续训成功率暴跌）
                save_vecnormalize=True,
            )
            callbacks.append(checkpoint_callback)

        # 外部回调（Task3: 训练脚本传入，如 40k 步后关 GLFW）
        if extra_callbacks:
            callbacks.extend(extra_callbacks)

        # 开始训练
        logger.info(f"开始训练，总步数: {total_timesteps}（n_envs={getattr(self, 'n_envs', 1)}）")
        if demo_env is not None:
            logger.info(f"演示播放已启用：每 {demo_every_n_rollouts} 个 rollout 用当前策略渲染一次")

        # 训练（P1：移除未注册的死代码 episode_callback；episode 记录由真实环境/主进程 callback 完成）
        self.agent.learn(
            total_timesteps=total_timesteps,
            callback=callbacks,
            progress_bar=True
        )

        # 保存最终模型（并行改造 2026-08-22：走 self.save 以连带保存 VecNormalize 观测统计）
        if save_path:
            self.save(save_path)
            logger.info(f"最终模型已保存: {save_path}")
    
    def predict(self, observation: np.ndarray, deterministic: bool = True) -> Tuple[np.ndarray, Optional[np.ndarray]]:
        """
        预测动作

        并行改造(2026-08-22)：SB3 PPO.predict 不做观测归一化（训练时 rollout 的 obs
        由 VecNormalize 归一化后才喂给 policy）。这里统一先归一化，保证评估/部署/演示
        与训练观测分布一致（此前评估/演示喂原始 obs 导致观测失配、策略乱走）。

        Args:
            observation: 观察（原始观测）
            deterministic: 是否使用确定性策略

        Returns:
            Tuple[np.ndarray, Optional[np.ndarray]]: 动作和状态
        """
        if self.agent is None:
            raise ValueError("智能体未初始化")

        obs = np.asarray(observation, dtype=np.float32)
        vn = self.agent.get_vec_normalize_env()
        if vn is not None and getattr(vn, 'norm_obs', False):
            obs = vn.normalize_obs(obs)
        return self.agent.predict(obs, deterministic=deterministic)
    
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
        """保存模型（并行改造 2026-08-22：连同 VecNormalize 观测统计一起保存。

        SB3 的 PPO.save 的 _excluded_save_params 排除 _vec_normalize_env，
        zip 里不会有 obs_rms——必须单独存 pkl，否则评估/部署观测失配乱走。
        """
        if self.agent is None:
            raise ValueError("智能体未初始化")

        self.agent.save(path)
        vn = self.agent.get_vec_normalize_env()
        if vn is not None and getattr(vn, 'norm_obs', False):
            vn_path = path.replace('.zip', '_vecnormalize.pkl')
            vn.save(vn_path)
            logger.info(f"VecNormalize 观测统计已保存到: {vn_path}")
        else:
            logger.warning("未找到 VecNormalize env，观测统计未单独保存（评估/部署会观测失配）")
        logger.info(f"模型已保存到: {path}")

    def load(self, path: str):
        """加载模型（并行改造 2026-08-22：从 <path>_vecnormalize.pkl 恢复观测统计。

        此前误以为 PPO.load 会从 zip 自动恢复 obs_rms——实测 zip 里没有
        （_excluded_save_params 排除 _vec_normalize_env），评估/演示 stats=0/1 观测失配。
        """
        if not os.path.exists(path):
            raise FileNotFoundError(f"模型文件不存在: {path}")

        if self.env is None:
            raise ValueError("请先设置环境")

        # 创建向量化环境（Task3: 官方 SB3 VecNormalize——评估模式不更新统计）
        from stable_baselines3.common.vec_env import VecNormalize, DummyVecEnv
        vec_env = VecNormalize(
            DummyVecEnv([lambda: self.env]),
            norm_obs=getattr(self.training_config, 'normalize_observations', True),
            norm_reward=getattr(self.training_config, 'normalize_rewards', False),
            clip_obs=getattr(self.training_config, 'norm_obs_clip', 10.0),
            clip_reward=getattr(self.training_config, 'norm_reward_clip', 10.0),
            training=False,   # 评估模式：不更新统计
        )

        # 恢复训练时的观测统计（若存在）
        vn_path = path.replace('.zip', '_vecnormalize.pkl')
        if os.path.exists(vn_path):
            vec_env = VecNormalize.load(vn_path, vec_env)
            logger.info(f"已恢复 VecNormalize 观测统计: {vn_path}")
        else:
            logger.warning(f"未找到 {vn_path}，观测统计将使用初始值（评估/部署可能失配）")

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
