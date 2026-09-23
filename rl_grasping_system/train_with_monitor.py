"""
使用训练监控器的改进训练脚本（并行改造 2026-08-22）
- 12 环境 SubprocVecEnv 多进程并行（CPU 多核，无需 GPU）
- 训练 worker 全部无头（render_gui=False）
- 主进程定期弹窗播放当前策略（demo_env, render_gui=True），不拖慢训练
"""

import os
import sys
import copy
import math
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
    # D1（2026-08-24，一周冲刺方案 §5）：残差策略 + 动态化环境地基
    parser.add_argument('--action-mode', type=str, default='', choices=['', 'delta', 'residual'],
                        help='D1 动作模式：delta=旧增量语义（默认）；residual=标称轨迹+残差叠加 v=v_nominal+Δv/T')
    # P2c（2026-09-07）：MPC 标称层（Stage 2：在 MPC 标称之上训练 RL 残差）
    parser.add_argument('--nominal-mode', type=str, default='', choices=['', 'velocity_field', 'mpc'],
                        help='P2c 标称层类型（residual 模式生效）：velocity_field=速度场替身（默认，O(1) 快）；'
                             'mpc=末端级滚动最优控制（scipy SLSQP，自带障碍避障软约束，见 mpc_nominal.py）')
    parser.add_argument('--ctrl-delay-max', type=int, default=-1,
                        help='模型失配域随机化：每 episode 随机执行延迟 U[0, N] 决策步（>-1 开启；'
                             '训练残差补偿 plant 执行延迟/模型误差）。默认关闭=原行为')
    # 动作尺度对齐（2026-09-15）：PPO 动作是从高斯采样 a = μ + σ·ε，σ 初始值 = exp(log_std_init)。
    # SB3 默认 log_std_init=0.0 → σ=1.0，隐含假设动作是 O(1) 量级。
    # 【更正】v27 把 residual 动作空间归一化成 Box(-1,1) 后，这个默认**终于是对的**。
    # 此前"残差恒满幅"的主因是 μ（实测 3.37/3.66/2.35，初始 0.0034），不是 σ₀ —— 只压 σ 无效。
    parser.add_argument('--log-std-init', type=float, default=None,
                        help='PPO 探索尺度 σ 的 log 值（σ = exp(该值)）。不传 → 从零训练时按动作量程'
                             '自动推导；微调（--load-model）时 PPO.load 会带入 checkpoint 的 log_std，'
                             '自动值被覆盖而**不生效**。显式传值 → 加载后强制重置 σ。'
                             'v27 归一化后正常不用传（σ₀=1.0 即为正确默认）')
    # v27 归一化动作空间迁移开关（2026-09-15）：见 config.TrainingConfig.reinit_action_head
    parser.add_argument('--reinit-action-head', action='store_true',
                        help='加载 --load-model 后重新初始化 action_net（ortho gain=0.01, bias=0）。'
                             'v27 把 residual 动作空间改成 Box(-1,1) 后，旧 checkpoint 的 μ≈3.4 是'
                             '旧单位下的数，搬进新空间会被 clip 成 ±1、残差仍恒等于 cap —— '
                             '所以从 v27 之前的 checkpoint 迁移必须开这个开关一次。'
                             'trunk/value_net 保留，迁移价值不丢。'
                             '**resume v27 之后的 checkpoint 时必须关掉**，否则抹掉已学到的策略。')
    parser.add_argument('--dynamic-target', action='store_true',
                        help='D1 动态目标（L2）：物体 per-step 沿 target_motion_axis 往返运动')
    parser.add_argument('--target-vel', type=float, default=0.0,
                        help='D1 动态目标水平速度(m/s)，>0 才运动（默认用 config.target_vel_xy）')
    parser.add_argument('--target-axis', type=str, default='', choices=['', 'x', 'y'],
                        help='D1 动态目标运动轴（默认用 config.target_motion_axis）')
    parser.add_argument('--obstacle', action='store_true',
                        help='D1 障碍物：激活障碍 body（参与碰撞，可编程运动）')
    parser.add_argument('--obstacle-vel', type=float, default=0.0,
                        help='D1 障碍移动速度(m/s)，0=静态障碍（默认用 config.obstacle_vel）')
    # D2（2026-08-24，一周冲刺方案 §5）：静态障碍绕障 + 残差幅度正则
    parser.add_argument('--obstacle-on-path', action='store_true',
                        help='D2 静态障碍放"标称必经之路"（按当前目标自动放置，位置随机化时跟随）')
    parser.add_argument('--obstacle-pos', type=str, default='',
                        help='D2 固定障碍位置 "x,y,z"（覆盖 obstacle_fixed_pos；与 --obstacle-on-path 二选一）')
    parser.add_argument('--obstacle-path-fraction', type=float, default=-1.0,
                        help='D2 on-path 沿线段比例（≥0 覆盖 config.obstacle_path_fraction；<0 用默认 0.5）')
    parser.add_argument('--obstacle-path-lateral', type=float, default=None,
                        help='D2 on-path 侧偏 m（覆盖 config.obstacle_path_lateral；None 用默认）')
    parser.add_argument('--obstacle-path-lateral-range', type=str, default='',
                        help='D3 v8 障碍侧偏随机化区间 "min,max"（m；每-episode 随机，逼策略基于障碍观测动态绕障；空=固定）')
    parser.add_argument('--obstacle-path-z-offset', type=float, default=None,
                        help='D2 on-path z 偏移 m（覆盖 config.obstacle_path_z_offset；None 用默认 -0.10）')
    parser.add_argument('--obstacle-count', type=int, default=None,
                        help='v12 激活的静态障碍数量 1~3（覆盖 config.obstacle_count；None 用默认 1）。'
                             '>1 时沿必经之路不同 fraction/lateral 排布，观测仍只给最近激活障碍槽位（67 维不变）')
    parser.add_argument('--residual-reg', type=float, default=-1.0,
                        help='D2 残差幅度正则权重（≥0 覆盖 config.reward.residual_reg_w；<0 用默认 0.5）')
    # v25 残差稀疏三件套 + 残差预算解耦（2026-09-08，见 docs/2026-09-08_残差饱和分析与奖励重塑方案.md）
    parser.add_argument('--residual-l1-w', type=float, default=-1.0,
                        help='v25 L1 稀疏权重（≥0 覆盖 config.reward.residual_l1_w；-r_residual += -w·‖Δv‖₁，'
                             '满幅残差成本与成功奖励同级，逼常态零残差；<0 用默认 0）')
    parser.add_argument('--residual-step-w', type=float, default=-1.0,
                        help='v25 残差激活步罚（≥0 覆盖 config.reward.residual_step_w；‖Δv‖>τ 当步再罚；<0 用默认 0）')
    parser.add_argument('--residual-step-threshold', type=float, default=None,
                        help='v25 残差激活判定阈值 m/s（覆盖 config.reward.residual_step_threshold；默认 0.02）')
    parser.add_argument('--residual-delta-cap', type=float, default=None,
                        help='v25 残差预算解耦：residual 分支每决策步位置增量上限 m（覆盖 config.grasping.residual_delta_cap；'
                             # 注意：argparse 会把 help 当 %-format 处理，字面量 % 必须写 %%，
                             # 否则 `%；` 会被判为非法格式符 → 整个 --help 直接抛 ValueError 崩掉
                             '0.002→v_max_res=0.05 m/s = MPC v_max 40%%；None 用默认 0.002）')
    parser.add_argument('--residual-clip-mode', type=str, default='',
                        help='残差裁剪几何（覆盖 config.grasping.residual_clip_mode；默认 modulus=不改）。'
                             'modulus=HEAD/v25+ 语义（动作层 a∈[-1,1]×cap，再对 ‖Δv‖ 做模长裁剪）。'
                             'per_axis=pre-gating/cadd00e 语义（动作空间回米制 ±max_ee_delta，逐轴裁剪）。'
                             '⚠️ 只在 residual 模式生效，且**改变了动作空间声明**——用 per_axis 训练出的 '
                             'checkpoint 自带米制动作空间，评测时也必须配 --residual-clip-mode per_axis，'
                             '否则口径失配（§9.3 实测差 33pp 以上）。')
    parser.add_argument('--residual-collision-threshold', type=float, default=None,
                        help='v25 残差碰撞双倍罚判定阈值 m/s（覆盖 config.reward.residual_collision_threshold；默认 0.02）')
    parser.add_argument('--residual-collision-penalty-extra', type=float, default=None,
                        help='v25 残差引起的碰撞双倍罚：collision 且 ‖Δv‖>τ 时在 collision-penalty 之外再叠加该罚'
                             '（≤0 有效，如 -10；None 用默认 0 关闭）')
    # v25b2 完整残差门控（2026-09-09，docs/2026-09-08 §13）：关键帧触发 + 无解放大 + 最近点触发
    parser.add_argument('--residual-gate-enabled', type=str, default='',
                        help='v25b2 关键帧门控开关（True/False 覆盖 config.grasping.residual_gate_enabled='
                             'True；空串用默认。False=回退 v25b1 基础版恒允许行为）')
    parser.add_argument('--residual-gate-unstuck-cap', type=float, default=None,
                        help='v25b2 MPC 无解时残差 cap 放大值 m/步（覆盖 config.grasping.residual_gate_unstuck_cap='
                             '0.004；0.004 → v_max 0.1 m/s = 常态 2×，补 static3 硬绕预算）')
    parser.add_argument('--residual-gate-nearest-point', type=str, default='',
                        help='v25b2 gate 触发判据（True/False 覆盖 config.grasping.residual_gate_nearest_point='
                             'True；True=臂最近碰撞体到障碍距离，覆盖整臂 link 碰撞）')
    parser.add_argument('--residual-gate-unstuck-terminal-err', type=float, default=None,
                        help='v25b2 MPC「到不了」判据阈值 m（覆盖 config.grasping.residual_gate_unstuck_terminal_err='
                             '0.10；预测序列末步到 hover 误差 > 该值 → unstuck 放大残差。冒烟实证：SLSQP 报 '
                             'success=True 但到不了，res.success 不能判无解，用 terminal_err）')
    parser.add_argument('--obstacle-w', type=float, default=-1.0,
                        help='D2 障碍接近惩罚权重（≥0 覆盖 config.reward.obstacle_w；<0 用默认 0.5）')
    parser.add_argument('--obstacle-clear-bonus', type=float, default=-1.0,
                        help='v16 奖励经济学：干净成功奖励（≥0 覆盖 config.reward.obstacle_clear_bonus；'
                             '抓取成功当步全程 0 碰撞 → +bonus，让绕障抓取优于穿障抓取；<0 用默认 0）')
    parser.add_argument('--max-steps', type=int, default=-1,
                        help='v17+ 每 episode 抬升前最大步数（覆盖 config.grasping.max_steps；'
                             '双障碍绕行需更多时间，200→250 可救超时失败；<0 用默认 200）')
    parser.add_argument('--obstacle-mix-ratio', type=float, default=-1.0,
                        help='D3 §11.4 无障碍混合采样比例（0~1；每 episode 该概率障碍隐藏做纯抓取训练；<0 用 config 默认 0）')
    parser.add_argument('--arm-aware', action='store_true',
                        help='arm-aware MPC：标称层代价纳入臂身碰撞球（默认关；开启后臂身不再对障碍失明）')
    parser.add_argument('--w-arm', type=float, default=None,
                        help='覆盖 mpc_nominal_w_arm（臂身惩罚权重，默认 config 120.0）')
    parser.add_argument('--arm-horizon', type=int, default=None,
                        help='覆盖 mpc_nominal_arm_horizon（臂身项生效步数，默认 5）')
    parser.add_argument('--scenario-mix', type=str, default='',
                        help='P3 训练分布（2026-08-27）：per-episode 场景采样 "静态无障,动态目标无障,动态目标+动态障碍"'
                             ' 概率（如 0.4,0.3,0.3；空=旧机制全跟随全局开关）。选中场景强制覆盖全局开关，'
                             '需配合 --dynamic-target --obstacle 提供 target_vel_xy/obstacle_vel')
    parser.add_argument('--collision-penalty', type=float, default=None,
                        help='D3 臂-障碍碰撞当步惩罚（有效值 ≤0；0=关闭惩罚；None=用 config 默认 0）')
    parser.add_argument('--collision-max-streak', type=int, default=None,
                        help='D3 连续碰撞步数阈值→truncated（0=禁用；>0 启用；None=用 config 默认 0）')
    parser.add_argument('--perception-noise-std', type=float, default=-1.0,
                        help='D6 感知噪声（2026-08-28）：训练时目标位置观测加噪 σ(m)（鲁棒化训练；<0 用 config 默认 0）')
    parser.add_argument('--perception-dropout', type=float, default=-1.0,
                        help='D6 感知噪声：训练时随机漏检概率（鲁棒化训练；<0 用 config 默认 0）')
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

        # D1（2026-08-24，一周冲刺方案 §5）：残差策略 + 动态化环境地基
        if args.action_mode:
            config.grasping.action_mode = args.action_mode
        if args.nominal_mode:
            config.grasping.nominal_mode = args.nominal_mode
            logger.info(f"🎛️ P2c 标称层：nominal_mode={config.grasping.nominal_mode}"
                        f"（mpc=末端级滚动最优控制，velocity_field=速度场替身）")
        if args.ctrl_delay_max >= 0:
            config.grasping.ctrl_delay_randomize = args.ctrl_delay_max > 0
            config.grasping.ctrl_delay_max_steps = max(0, args.ctrl_delay_max)
            if args.ctrl_delay_max > 0:
                logger.info(f"🌀 模型失配域随机化：每 episode 执行延迟 U[0,{args.ctrl_delay_max}] 决策步")
        if args.dynamic_target:
            config.grasping.dynamic_target_enabled = True
            if args.target_vel > 0:
                config.grasping.target_vel_xy = args.target_vel
            if args.target_axis:
                config.grasping.target_motion_axis = args.target_axis
        if args.obstacle:
            config.grasping.obstacle_enabled = True
            if args.obstacle_vel > 0:
                config.grasping.obstacle_vel = args.obstacle_vel
        # D2: 静态障碍绕障 + 残差幅度正则（一周冲刺方案 §5）
        if args.obstacle_on_path:
            config.grasping.obstacle_enabled = True
            config.grasping.obstacle_on_nominal_path = True
            # P3 修复（2026-08-27）：on-path 布局 + obstacle_vel>0 时障碍沿轴运动（动态障碍扫过路径），
            # 否则障碍被钉住成静态——--obstacle-vel 原只在 --obstacle 分支生效（v13_p3 碰撞率 78% 根因之一）。
            if args.obstacle_vel > 0:
                config.grasping.obstacle_vel = args.obstacle_vel
        if args.obstacle_pos:
            _xyz = tuple(float(v) for v in args.obstacle_pos.split(','))
            config.grasping.obstacle_fixed_pos = _xyz
            config.grasping.obstacle_enabled = True
        if args.obstacle_path_fraction >= 0:
            config.grasping.obstacle_path_fraction = args.obstacle_path_fraction
        if args.obstacle_path_lateral is not None:
            config.grasping.obstacle_path_lateral = args.obstacle_path_lateral
        if args.obstacle_path_lateral_range:
            _lo, _hi = (float(v) for v in args.obstacle_path_lateral_range.split(','))
            config.grasping.obstacle_path_lateral_range = (_lo, _hi)
        if args.obstacle_path_z_offset is not None:
            config.grasping.obstacle_path_z_offset = args.obstacle_path_z_offset
        if args.residual_reg >= 0:
            config.reward.residual_reg_w = args.residual_reg
        # v25 残差稀疏三件套 + 残差预算解耦（覆盖逻辑）
        if args.residual_l1_w >= 0:
            config.reward.residual_l1_w = args.residual_l1_w
        if args.residual_step_w >= 0:
            config.reward.residual_step_w = args.residual_step_w
        if args.residual_step_threshold is not None:
            config.reward.residual_step_threshold = args.residual_step_threshold
        if args.residual_delta_cap is not None:
            config.grasping.residual_delta_cap = args.residual_delta_cap
        if args.residual_clip_mode:
            config.grasping.residual_clip_mode = args.residual_clip_mode
        if args.residual_collision_threshold is not None:
            config.reward.residual_collision_threshold = args.residual_collision_threshold
        if args.residual_collision_penalty_extra is not None:
            config.reward.residual_collision_penalty_extra = args.residual_collision_penalty_extra
        # arm-aware 标称层（臂身碰撞球进 MPC 代价）
        if args.arm_aware:
            config.grasping.mpc_nominal_arm_aware = True
        if args.w_arm is not None:
            config.grasping.mpc_nominal_w_arm = args.w_arm
        if args.arm_horizon is not None:
            config.grasping.mpc_nominal_arm_horizon = args.arm_horizon
        # v25b2 完整残差门控（关键帧触发 + 无解放大 + 最近点触发）
        if args.residual_gate_enabled:
            config.grasping.residual_gate_enabled = args.residual_gate_enabled.lower() == 'true'
        if args.residual_gate_unstuck_cap is not None:
            config.grasping.residual_gate_unstuck_cap = args.residual_gate_unstuck_cap
        if args.residual_gate_nearest_point:
            config.grasping.residual_gate_nearest_point = args.residual_gate_nearest_point.lower() == 'true'
        if args.residual_gate_unstuck_terminal_err is not None:
            config.grasping.residual_gate_unstuck_terminal_err = args.residual_gate_unstuck_terminal_err
        if args.obstacle_w >= 0:
            config.reward.obstacle_w = args.obstacle_w
        if args.obstacle_clear_bonus >= 0:
            config.reward.obstacle_clear_bonus = args.obstacle_clear_bonus
        if args.max_steps >= 0:
            config.grasping.max_steps = int(args.max_steps)
        if args.obstacle_mix_ratio >= 0:
            config.grasping.obstacle_mix_ratio = args.obstacle_mix_ratio
        if args.scenario_mix:
            _v = tuple(float(x) for x in args.scenario_mix.split(','))
            # v25（2026-09-08）：支持 5 场景 mix（静态无障/动态目标/动态+障碍/static3/mixed3）；旧 3 值保持兼容
            if len(_v) not in (3, 5):
                raise SystemExit('--scenario-mix 需要 3 或 5 个概率 "静态,动态,动态+障碍[,静态3障碍,混合3障碍]"'
                                 '（如 0.4,0.3,0.3 或 0.25,0.2,0.35,0.1,0.1）')
            config.grasping.scenario_mix = _v
            config.grasping.dynamic_target_enabled = True   # 提供 target_vel_xy 语义；per-episode 采样决定实际激活
            config.grasping.obstacle_enabled = True
            logger.info(f"🎯 P3 训练分布已启用：scenario_mix={_v}（静态/动态/动态+障碍）")
        if args.collision_penalty is not None:
            config.grasping.obstacle_collision_penalty = args.collision_penalty
        if args.collision_max_streak is not None:
            config.grasping.obstacle_collision_max_streak = args.collision_max_streak
        if args.obstacle_count is not None:
            config.grasping.obstacle_count = args.obstacle_count
        # D6 感知噪声（2026-08-28）：训练时加噪鲁棒化（观测通道目标位置加噪 + 漏检）
        if args.perception_noise_std >= 0:
            config.grasping.perception_noise_std = args.perception_noise_std
        if args.perception_dropout >= 0:
            config.grasping.perception_dropout = args.perception_dropout
        if config.grasping.perception_noise_std > 0 or config.grasping.perception_dropout > 0:
            logger.info(f"🎭 D6 感知噪声训练已启用：noise_std={config.grasping.perception_noise_std} m, "
                        f"dropout={config.grasping.perception_dropout}（鲁棒化）")
        if args.action_mode or args.dynamic_target or args.obstacle or args.obstacle_on_path:
            logger.info(f"🆕 D1/D2 已启用：action_mode={config.grasping.action_mode}, "
                        f"dynamic_target={config.grasping.dynamic_target_enabled}"
                        f"(v={config.grasping.target_vel_xy} m/s, axis={config.grasping.target_motion_axis}), "
                        f"obstacle={config.grasping.obstacle_enabled}"
                        f"(v={config.grasping.obstacle_vel} m/s, "
                        f"on_path={config.grasping.obstacle_on_nominal_path}, "
                        f"path_lateral={config.grasping.obstacle_path_lateral}, "
                        f"path_fraction={config.grasping.obstacle_path_fraction}, "
                        f"path_z_offset={config.grasping.obstacle_path_z_offset}), "
                        f"reward: obstacle_w={config.reward.obstacle_w}, "
                        f"residual_reg_w={config.reward.residual_reg_w}")

        n_envs = max(1, int(config.system.num_envs))
        if args.total_timesteps > 0:
            config.training.total_timesteps = int(args.total_timesteps)
        # 动作尺度对齐（2026-09-15）：显式传值 → 覆盖。微调路径下 agent.py 会据此强制重置 σ
        # （PPO.load 会带入 checkpoint 的 log_std，不重置则本参数不生效）。
        if args.log_std_init is not None:
            config.training.log_std_init = float(args.log_std_init)
            logger.info(
                f"🎯 log_std_init = {args.log_std_init:.4f} → σ = {math.exp(args.log_std_init):g}"
                f"（若 --load-model，加载后会覆盖 checkpoint 自带 log_std）"
            )
        # v27：从旧 checkpoint 迁移进归一化动作空间 → 重init action_net（详见 --reinit-action-head）
        if args.reinit_action_head:
            config.training.reinit_action_head = True
            logger.warning("🔧 reinit-action-head：加载后会重新初始化 action_net"
                           "（旧 μ≈3.4 在 Box(-1,1) 下会被 clip 成 ±1，不重init 等于没改）")
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
