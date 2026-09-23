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

        # ####################################################################
        # 【2026-09-15 更正】下面这一整段（至 "修法：取 σ₀ = ..." 结束）是**上一版**的判断，
        # 结论已被实测推翻，保留在此仅作推理痕迹。正确结论：
        #   · "残差恒满幅"的**主因是 μ，不是 σ₀**。读 v25b2_gate 的 action_net 权重实测：
        #     μ 中位 3.37/3.66/2.35（初始仅 0.0034，涨了 ~1000×）、σ=1.67/1.80/1.01、
        #     |μ|/σ 三轴恒 ≈2.0、P(|μ|>cap)=1.000 → **100% 饱和**。
        #   · 只压 σ 无效：clip(3.4±0.001) 与 clip(3.4±1.8) 同为 +0.002。已跑 40k 实测：
        #     σ 强制 1.8→0.001，残差仍 0.116 m/s ≥ 满幅 0.0866，r_residual_step 仍每步 -1.0。
        #   · 真正病因是**动作空间没归一化**：env 的 np.clip(±0.002) 让 |a|>0.002 区域
        #     ∂执行动作/∂a ≡ 0 → 策略只看得见符号、看不见幅度 → 唯一可学方向变成"让 sign 稳定"
        #     → 需要 |μ|/σ 大 → 而 ent_coef 在推 σ 往上 → 只能靠 μ 上涨兑现 → 双双失控。
        #     修法在 environment.py::_setup_spaces（residual 模式动作空间改 Box(-1,1)）。
        #   · 归一化之后 SB3 默认 σ₀=1.0 终于是对的：本段下面的自动推导会看到 _a_max=1.0，
        #     走 else 分支取 0.0。所以这段逻辑现在退化为"可用的手动覆盖口子"，不是必需品。
        # ####################################################################
        #
        # ---- PPO 初始探索尺度 log_std_init：把 σ₀ 对齐到动作的真实量程 ----
        #
        # 背景：PPO 的连续动作不是网络直接吐出一个数，而是**从一个高斯分布里采样**：
        #     a = μ(s) + σ·ε,   ε ~ N(0, I)
        # 网络（action_net）只负责 μ；σ 是一个独立的可学习参数，初始值 = exp(log_std_init)，
        # 见 SB3 distributions.py：`log_std = nn.Parameter(ones(dim) * log_std_init)`。
        #
        # 问题：SB3 默认 log_std_init=0.0 → σ₀ = exp(0) = **1.0**。这个默认隐含假设
        # "动作是 O(1) 量级"（Box(-1,1) 归一化动作是通用约定）。但本环境的动作不是归一化的：
        #   · residual 模式：动作 = 末端位置增量，真正生效的量程只有 ±0.002 m/步
        #     （residual_delta_cap；见 environment.py step 里的 _pos_cap）
        #   · delta 模式：±max_ee_delta = ±0.005 m/步
        # 于是默认 σ₀=1.0 比有效量程大 200~500 倍 → 噪声项 σ₀·ε 完全淹没 μ，
        # 绝大多数采样落在 env 的 clip 边界之外。为什么这会让策略学不动：
        #   1) clip 之外，奖励对动作是**平的**（动作再大，执行结果一样）→ 策略收不到
        #      "你该往小走"的梯度。奖励只能告诉它"错了"，不能告诉它"错多少"。
        #   2) 与此同时 ent_coef 那顶熵奖励**一直在把 σ 往大推**
        #      （ent_coef>0 时 ∂loss/∂log_std < 0，梯度下降就往大的方向走）。
        # 两边一夹：σ 下不来，动作恒饱和。日志里 `res ≡ 0.0866` 正是这个症状 ——
        # √3 × 0.002 / 0.04 = 0.0866，即三个位置轴**同时**撞到 cap 的 box 角点，
        # 这不是"残差恰好需要这么大"，而是尺度失配的必然结果。
        #
        # 修法：取 σ₀ = 动作量程上限 / 2（约 95% 的初始采样落在可用区间内，尾部仍能探到边界）。
        # 起始偏小是安全方向：σ 可以由梯度**长大**（需要更多探索时），但从过大**缩小**很难
        # —— 恰恰因为饱和区没有梯度。这一条与"残差奖励整形"无关，是纯粹的动作尺度对齐问题。
        _log_std_init = getattr(self.training_config, 'log_std_init', None)
        if _log_std_init is None:
            _a_max = float(np.max(np.abs(np.asarray(vec_env.action_space.high, dtype=float))))
            if _a_max < 1.0:
                # 非归一化动作空间 → 按量程推导，别再吃 SB3 的 O(1) 默认值
                _log_std_init = float(np.log(max(_a_max / 2.0, 1e-8)))
                logger.info(
                    f"log_std_init 未显式配置 → 按动作量程 ±{_a_max:g} 自动推导："
                    f"σ₀={_a_max / 2.0:g}（log_std_init={_log_std_init:.4f}）。"
                    f"（SB3 默认 0.0→σ₀=1.0 与量程失配 {1.0 / max(_a_max / 2.0, 1e-8):.0f}×，"
                    f"会导致采样恒饱和、残差学不动）"
                )
            else:
                _log_std_init = 0.0  # 动作本就是 O(1) 量级 → SB3 默认即合适
                logger.info(f"log_std_init 未显式配置 → 动作量程 ±{_a_max:g} 属 O(1)，沿用 SB3 默认 σ₀=1.0")

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
                },
                # 关键修复：σ₀ 对齐动作真实量程（详见上方推导注释）
                "log_std_init": _log_std_init,
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
                # ################################################################
                # 【2026-09-15 修复】动作空间被 pkl 静默改回旧量程
                #
                # VecNormalize.load 在 SB3 2.9 里的实现是：
                #     vec_normalize = pickle.load(file)      # 反序列化**整个旧对象**
                #     vec_normalize.set_venv(venv)
                # 也就是 action_space / observation_space 全都跟着旧对象一起回来了；
                # 而 set_venv 只做 check_shape_equal(obs) —— **不刷新 action_space**。
                # 后果：只要 env 改了动作空间，pkl 里的旧量程就会把它悄悄改回去，
                # 而 PPO.load 的动作空间校验发生在这之后 → 校验"通过"、全程不报错。
                #
                # 实测（v27_normact_40k 那次）：env 声明 Box(-1,1)，
                # VecNormalize.load 之后变成 Box(±0.005)；策略按 ±0.005 采样、被 PPO
                # 裁到 ±0.005，env 再乘 residual_delta_cap=0.002 → 残差恒 ≈1e-5 m/步
                # （=预算 0.002 的 0.5%）→ "归一化动作空间"根本没生效，那一轮等于空跑。
                #
                # 修法：以当前 env 为唯一事实来源，pkl 只负责提供 obs 归一化统计。
                # ################################################################
                if vec_env.action_space != _base.action_space:
                    logger.warning(
                        f"⚠️ VecNormalize.load 把动作空间改回了 pkl 里的旧量程："
                        f"{_base.action_space} → {vec_env.action_space}，已强制对齐回当前 env"
                    )
                vec_env.action_space = _base.action_space
                vec_env.observation_space = _base.observation_space
                logger.info(f"迁移学习：已恢复 VecNormalize 观测统计 {_norm_pkl}（venv={type(_base).__name__}）")

            # ####################################################################
            # 【2026-09-15 修复】不再把 env 交给 PPO.load 做动作空间校验
            #
            # 两个坑叠在一起：
            # 1) PPO.load 内部 `model.__dict__.update(data)` 会把 checkpoint 里存的
            #    action_space 一起还原 —— 即使 env 声明的是 Box(-1,1)，模型仍按旧的
            #    ±0.005 采样、并按 ±0.005 裁剪（PPO.collect_rollouts 里
            #    np.clip(actions, self.action_space.low, self.action_space.high)）。
            #    也就是说：**改动作空间但依赖 checkpoint 的 spaces，等于没改。**
            # 2) check_for_correct_spaces 又要求两者一致，而 checkpoint 的量程本来
            #    就该和新的不一样（这正是我们要换的东西）→ 带上 env 只会直接抛
            #    "Action spaces do not match"。
            #
            # 所以改成：先 load（env=None），再显式把 spaces 对齐到当前 env，最后 set_env
            # （set_env 内部还会再校验一次，此时两边已一致）。这是唯一能保证
            # "env 说什么量程，策略就用什么量程"的顺序。
            # ####################################################################
            self.agent = PPO.load(self.model_path)
            _ckpt_act = self.agent.action_space
            self.agent.action_space = vec_env.action_space
            self.agent.observation_space = vec_env.observation_space
            self.agent.policy.action_space = vec_env.action_space
            self.agent.policy.observation_space = vec_env.observation_space
            if _ckpt_act != vec_env.action_space:
                logger.warning(
                    f"⚠️ checkpoint 的动作空间 {_ckpt_act} ≠ 当前 env 的 {vec_env.action_space}"
                    f"（预期之内，说明动作量程确实换了）。已强制改为 env 的量程；"
                    f"checkpoint 里像 μ 这样的**旧单位数值**需要配套处理，"
                    f"否则会被 clip 到新量程边界（见 --reinit-action-head）。"
                )
            # 等价复刻 PPO.load(env=...) 里 env 分支该做的两件事：
            #   data["n_envs"] = env.num_envs（issue #1018）与 data["_last_obs"] = None
            #   （force_reset，issue #597）。set_env(force_reset=True) 负责后者。
            if self.agent.n_envs != vec_env.num_envs:
                logger.warning(f"n_envs 由 checkpoint 的 {self.agent.n_envs} 改为 {vec_env.num_envs}")
                self.agent.n_envs = vec_env.num_envs
            self.agent.set_env(vec_env)
            logger.info(f"加载预训练模型: {self.model_path}（动作空间已对齐 {vec_env.action_space}）")

            # ---- σ 重置 / 失配告警（迁移学习专用）----
            # PPO.load 会把 checkpoint 里的 log_std **一起载入**，于是上面算出的 log_std_init
            # 在微调路径上完全不生效 —— 而现有 checkpoint 的 σ 恰恰是被熵奖励一路吹大的
            # （实测 v22→v26 收敛 σ：1.55 → 1.89，是有效动作量程 ±0.002 的 500~950 倍）。
            # 不处理就等于把病一起继承过来，所以：
            #   显式给了 log_std_init → 加载后强制覆盖 σ（这是微调路径下唯一生效的方式）
            #   没给                  → 告警并报出 checkpoint 真实 σ，避免"以为改了其实没改"
            _explicit_lsi = getattr(self.training_config, 'log_std_init', None)
            if _explicit_lsi is not None:
                with torch.no_grad():
                    self.agent.policy.log_std.data.fill_(float(_explicit_lsi))
                logger.info(
                    f"σ 已强制重置为 {float(np.exp(_explicit_lsi)):g}"
                    f"（log_std_init={float(_explicit_lsi):.4f}，覆盖 checkpoint 自带 log_std）"
                )
            else:
                _sig_loaded = float(np.exp(self.agent.policy.log_std.data.mean()))
                logger.warning(
                    f"加载了预训练模型但未显式指定 log_std_init → σ 沿用 checkpoint 的 "
                    f"{_sig_loaded:.4f}，自动推导值 {float(np.exp(_log_std_init)):g} 被覆盖、不生效。"
                    f"若想强制对齐动作量程，请显式传 --log-std-init。"
                )

            # ---- 动作头重初始化（v27 归一化动作空间专用）----
            # 归一化改变了动作的**单位**。旧 checkpoint 的 μ≈3.4 是"米"下的数，直接搬进
            # Box(-1,1) 空间会被 clip(a,±1) 压成 +1 → 残差仍恒等于 cap、仍然饱和，等于没改。
            # 所以从 v27 之前的 checkpoint 迁移时必须重init action_net；
            # features_extractor / mlp_extractor / value_net 保留 → 迁移价值不丢。
            if bool(getattr(self.training_config, 'reinit_action_head', False)):
                _head = self.agent.policy.action_net
                # 先量一下旧 μ 的量级（N(0,1) 观测代理：VecNormalize 会把 obs 标准化到 ~N(0,1)），
                # 把"病"写进日志，便于和重训后的轨迹对照
                with torch.no_grad():
                    _probe = torch.randn(2048, self.agent.observation_space.shape[0])
                    _latent = self.agent.policy.mlp_extractor.policy_net(
                        self.agent.policy.extract_features(_probe, self.agent.policy.features_extractor))
                    _mu_old = self.agent.policy.action_net(_latent).numpy()
                _sat = float((np.abs(_mu_old) > 1.0).mean())  # 归一化空间的有效量程 = ±1
                nn.init.orthogonal_(_head.weight, 0.01)
                if _head.bias is not None:
                    nn.init.constant_(_head.bias, 0.0)
                logger.warning(
                    f"action_net 已重新初始化（ortho gain=0.01, bias=0）。"
                    f"重init前 μ 中位 {np.median(np.abs(_mu_old)):.3f}、"
                    f"P(|μ|>1)={_sat:.3f}（归一化空间量程 ±1）—— 若不重init，"
                    f"这些 μ 会被 clip 成 ±1、残差仍恒等于 cap。"
                    f"trunk/value_net 保留。注意：只在从 v27 之前的 checkpoint 迁移时开一次，"
                    f"resume v27 之后的 checkpoint 必须关掉 --reinit-action-head。"
                )

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
            _live_act = vec_env.action_space
            _live_obs = vec_env.observation_space
            vec_env = VecNormalize.load(vn_path, vec_env)
            # 同 set_environment 里的修复（详见该处注释）：VecNormalize.load 是
            # pickle.load 整个旧对象，会把 pkl 里的 action_space 一起带回来，而 set_venv
            # 不刷新它 → 评估时动作量程被静默改回旧值，PPO.load 的校验也因此"通过"。
            # 评估必须按 **当前 env 的量程** 解释策略输出，否则量级全错。
            if vec_env.action_space != _live_act:
                logger.warning(
                    f"⚠️ 评估：VecNormalize.load 把动作空间改回了 pkl 旧量程 "
                    f"{vec_env.action_space}（当前 env 是 {_live_act}），已强制对齐。"
                    f"注意：v27 之前的 checkpoint 其 μ 是**旧单位**的数值，"
                    f"在新量程下语义已经变了，不能和新 checkpoint 直接同口径比较。"
                )
            vec_env.action_space = _live_act
            vec_env.observation_space = _live_obs
            logger.info(f"已恢复 VecNormalize 观测统计: {vn_path}")
        else:
            logger.warning(f"未找到 {vn_path}，观测统计将使用初始值（评估/部署可能失配）")

        # 同 set_environment：不把 env 交给 PPO.load 做空间校验（checkpoint 的 spaces
        # 会经 model.__dict__.update(data) 覆盖 env 的），改为 load → 对齐 spaces → set_env。
        self.agent = PPO.load(path)
        _ckpt_act = self.agent.action_space
        self.agent.action_space = vec_env.action_space
        self.agent.observation_space = vec_env.observation_space
        self.agent.policy.action_space = vec_env.action_space
        self.agent.policy.observation_space = vec_env.observation_space
        if _ckpt_act != vec_env.action_space:
            logger.warning(f"评估：checkpoint 动作空间 {_ckpt_act} ≠ env {vec_env.action_space}，已对齐到 env")
        if self.agent.n_envs != vec_env.num_envs:
            self.agent.n_envs = vec_env.num_envs
        self.agent.set_env(vec_env)
        logger.info(f"模型已从 {path} 加载（动作空间 {vec_env.action_space}）")
    
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
