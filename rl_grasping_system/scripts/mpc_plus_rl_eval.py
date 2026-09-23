#!/usr/bin/env python3
"""Stage 2 评估：residual 架构 4 象限对比（velocity_field|MPC × 无RL|PPO Δv）（2026-09-07）。

同 scripts/mpc_nominal_verify.py 环境口径（GraspingEnv + action_mode=residual + nominal_mode），
区别：本脚本可加载 PPO 策略（--model，含 _vecnormalize.pkl 自动恢复）输出残差动作；
--zero-residual 时动作恒 0 = 纯标称（等价 mpc_nominal_verify.py 的 action=0）。

四象限回填（2026-09-02 对比表缺失的第 4 格 = MPC + PPO Δv，即 Stage 2 目标）：
  (velocity_field, 0)      = 速度场标称纯标称
  (velocity_field, PPO)    = v23_mix（velocity_field + PPO 残差）
  (MPC, 0)                 = v_max=0.125 纯 MPC（mpc_nominal_verify.py 同口径）
  (MPC, PPO)               = v24_mpcres（MPC + PPO 残差）→ --model ... --nominal-mode mpc

判定线（Stage 2，n=30）：dyn_both success ≥80% 且 coll ≤5%；dyn_target →83.3%；
static3 >0%；mixed_z coll <10%（部分成功可接受）。残差幅度监控：成功=小残差
（correct-not-take-over）、失败=大残差。

用法：
  # MPC+RL（Stage 2 目标，回填第 4 格）
  python3 scripts/mpc_plus_rl_eval.py --scene dyn_both \
      --model models/final_model_stage2_d2_v24_mpcres.zip \
      --n_episodes 30 --nominal-mode mpc --save-results results/mpc_plus_rl/dyn_both.json
  # 纯 MPC 标称对照（等价 mpc_nominal_verify.py action=0）
  python3 scripts/mpc_plus_rl_eval.py --scene dyn_both --n_episodes 30 \
      --nominal-mode mpc --zero-residual
  # v23 对照（velocity_field + PPO）
  python3 scripts/mpc_plus_rl_eval.py --scene dyn_both \
      --model models/final_model_stage2_d2_v23_mix.zip \
      --n_episodes 30 --nominal-mode velocity_field
"""
import os, sys, argparse, json, time
import numpy as np

sys.path.append(os.path.dirname(os.path.abspath(__file__)) + "/..")
from config import get_config
from environment import GraspingEnv
from agent import GraspingAgent
import ik   # Stage2: IK 调用/失败计数（模块级全局量，run_episode 里取差分）


def _opt_f(v):
    """可选浮点：None 原样返回（JSON 里为 null），否则转 float。"""
    return None if v is None else float(v)


def _mean_opt(results, key):
    """对「可能为 None」的列取均值；一集都没值时返回 None（与 0.0 区分开：0 是"余量 0 cm"）。"""
    vals = [r[key] for r in results if r.get(key) is not None]
    return float(np.mean(vals)) if vals else None


def _min_opt(results, key):
    """对「可能为 None」的列取最小值；一集都没值时返回 None。"""
    vals = [r[key] for r in results if r.get(key) is not None]
    return float(np.min(vals)) if vals else None


def _fmt_m(v):
    """余量格式化成毫米（None → 'n/a'，不写 0——0 mm 是"擦到"，与"没规划"是两回事）。"""
    return "n/a" if v is None else f"{v * 1000:+.1f}mm"

SCENE_CFG = {
    "static":     dict(dyn_target=False, obstacle=("off", 0, 0.0),    specs=None, z=False),
    "dyn_target": dict(dyn_target=True,  obstacle=("off", 0, 0.0),    specs=None, z=False),
    "dyn_both":   dict(dyn_target=True,  obstacle=("legacy", 1, 0.05), specs=None, z=False),
    # 训练一致口径（2026-09-15 新增）：v18/v24/v25 的训练配方是
    # `--obstacle-on-path --obstacle-count 2`，而 `dyn_both` 传 count=1 →
    # 下面第 236 行 `obstacle_on_nominal_path = (count > 1)` = False，障碍被摆到
    # off-path 固定位 (0.06,0.40,0.35)（球顶 0.40，夹爪穿越带 0.45 → 挡不住）。
    # 实测：训练口径整集最小臂-球表面距中位 **−0.0002 m**（7/10 集 <1cm，真挡路），
    # 评测口径中位 **+0.0391 m**（仅 1/10 集 <1cm）。两者不是同一个场景，
    # 这正是「不避障的速度场标称反而最好」的根因。count=2 时本场景 == 训练摆法。
    "dyn_both_train": dict(dyn_target=True, obstacle=("legacy", 2, 0.05), specs=None, z=False),
    "static3":    dict(dyn_target=False, obstacle=("legacy", 3, 0.0), specs=None, z=False),
    "mixed3":     dict(dyn_target=False, obstacle=("specs", 3, 0.0),
                       specs=[{"type": "static",  "pos": (0.10, 0.40, 0.36)},
                              {"type": "dynamic", "vel": 0.05, "mode": "random",
                               "pos": (0.06, 0.46, 0.36)},
                              {"type": "dynamic", "vel": 0.04, "mode": "roundtrip", "axis": "x",
                               "half_range": 0.10, "period": 4.0, "pos": (0.16, 0.38, 0.36)}],
                       z=False),
    "mixed_z":    dict(dyn_target=False, obstacle=("specs", 3, 0.0),
                       specs=[{"type": "static",  "pos": (0.10, 0.40, 0.36)},
                              {"type": "dynamic", "vel": 0.05, "mode": "random",
                               "z_motion": True, "z_amp": 0.10, "z_period": 3.0,
                               "pos": (0.06, 0.46, 0.36)},
                              {"type": "dynamic", "vel": 0.04, "mode": "roundtrip", "axis": "x",
                               "half_range": 0.10, "period": 4.0,
                               "z_motion": True, "z_amp": 0.12, "z_period": 5.0,
                               "pos": (0.16, 0.38, 0.36)}],
                       z=True),
}


N_PROFILE_BINS = 100   # 每步奖励剖面：归一化时间轴 [0,1] 上的格点数


def _normalized_profile(trace, n_bins=N_PROFILE_BINS):
    """把一集的分步奖励序列重采样到归一化时间轴（0=起始，1=结束）。

    episode 长度不一（成功会提前 terminated，超时是 200 步），直接按步号平均会把
    "100 步的成功集"和"200 步的超时集"错位。归一化后每个 episode 等权贡献一条曲线。

    返回 (rate, cum)：
      rate[k] = 该归一化格点上的**每决策步**奖励（连续项据此可比；一格跨 L/(n_bins-1) 步，
                所以要除以格宽而不是直接取差分，否则值会被格宽放大 ~1.6 倍）
      cum[k]  = 到该格点为止的累计奖励（事件项用它——单步尖峰画成 rate 是根细针，
                画成累计曲线才是"到这个阶段为止拿到了多少"，且末点 = 该集该项总和）

    实现上先对**累计曲线**插值再差分：单步尖峰落在网格点之间时逐点插值会丢面积，
    对累计量插值则求和精确守恒（实测 r_grasp 求和偏差 13% → 0）。
    """
    L = len(trace)
    if L == 0:
        return None, None
    keys = list(trace[0].keys())
    src = (np.arange(L) + 0.5) / L
    # 网格必须跨满 [0,1]：用格心 (j+0.5)/n 时端点处的贡献会被截掉
    dst = np.linspace(0.0, 1.0, n_bins)
    bin_steps = L * (dst[1] - dst[0])       # 一个归一化格点折合多少决策步
    rate, cum = {}, {}
    for k in keys:
        c = np.cumsum(np.array([float(p.get(k, 0.0)) for p in trace]))
        c_i = np.interp(dst, src, c)
        cum[k] = c_i
        rate[k] = np.diff(np.concatenate([[0.0], c_i])) / bin_steps
    return rate, cum


def _step_cum_profile(trace, n_bins):
    """把一集的分步奖励序列放到**绝对步号**轴上（0..n_bins），返回累计曲线。

    为什么事件项必须用这个而不是 _normalized_profile：episode 在抓取成功当步**立即终止**，
    按各自长度归一化后每一次成功都落在 t=1.0 —— 累计曲线永远是一根末尾竖线，退化。
    放到绝对步号上，成功集在它真正成功的步号处跳 +100，之后保持不变。

    短于 n_bins 的 episode 在终止步之后补 0（"该集已结束"），这正是要表达的含义：
    "到第 k 步为止，这一集已经攒了多少该项奖励"。
    """
    L = len(trace)
    if L == 0:
        return None
    keys = list(trace[0].keys())
    out = {}
    for k in keys:
        v = np.zeros(n_bins)
        n = min(L, n_bins)
        v[:n] = [float(p.get(k, 0.0)) for p in trace[:n]]
        out[k] = np.cumsum(v)
    return out


class ResidualEvaluator:
    """residual 架构闭环评估：--model 策略残差 / --zero-residual 纯标称（指标与 mpc_nominal_verify 同口径）。"""

    def __init__(self, env, agent, scene="dyn_both", max_steps=200, zero_residual=False,
                 per_episode_seed=False, base_seed=None, contact_trace=False):
        self.env = env
        self.agent = agent
        self.scene = scene
        self.max_steps = max_steps
        self.zero_residual = zero_residual
        self.per_episode_seed = per_episode_seed
        self.base_seed = base_seed
        self._ep_seed = None
        # --contact-trace：额外上报「接触发生在整集的哪个阶段」。默认关，避免给
        # 既有跑批的结果文件加键（口径一致性）。诊断「避障还有没有救」时打开：
        # 若接触集中在末段 → 是抓取/下降阶段必碰（同 static3，规划层救不了）；
        # 若散布在中段 → 运输途中可绕/可等，有时间维度的余量。
        self.contact_trace = bool(contact_trace)

    def run_episode(self):
        env = self.env
        obs, _ = env.reset(seed=self._ep_seed) if self._ep_seed is not None else env.reset()
        if env.nominal_trajectory is not None:
            env.nominal_trajectory.reset()        # 清 MPC warm-start / 速度场状态（跨 episode 防护）
        ep_len = 0
        ep_coll = 0
        contact_steps = []       # 本集发生接触的步号（1-based）——--contact-trace 时上报
        cnt_prev = 0             # obstacle_collision_count 是**累计**量，且每步最多 +1
        residual_norms = []
        info_last = {}
        bd_prev = {}
        step_trace = []          # 每步的奖励分项增量（env 只给累计值，这里做差）
        # IK 统计取差分（ik.py 的计数器是进程级全局量，跨 episode 累加）
        ik.reset_ik_stats()
        mpc0 = getattr(env, 'nominal_trajectory', None)
        solve_t0 = float(getattr(mpc0, 'solve_time_total', 0.0) or 0.0)
        solve_n0 = int(getattr(mpc0, 'solve_cnt', 0) or 0)
        via_t0 = float(getattr(mpc0, 'plan_time_total', 0.0) or 0.0)
        screen_n0 = int(getattr(mpc0, 'screen_cnt', 0) or 0)
        screen_t0 = float(getattr(mpc0, 'screen_time_total', 0.0) or 0.0)
        adopt_n0 = int(getattr(mpc0, 'adopt_cnt', 0) or 0)
        while ep_len < self.max_steps:
            if self.zero_residual or self.agent is None:
                action = np.zeros(env.action_space.shape[0], dtype=np.float32)
            else:
                action, _ = self.agent.predict(obs, deterministic=True)
            obs, _, terminated, truncated, info = env.step(action)
            ep_len += 1
            ep_coll = max(ep_coll, info.get('obstacle_collision_count', 0))
            # 接触**时刻**：累计量每步最多 +1（environment.py:910 的 `+= 1`），故增量>0 即本步接触。
            # 不需要动 environment.py —— 这一步信息在 info 里已经够了。
            _cn = int(info.get('obstacle_collision_count', 0))
            if _cn > cnt_prev:
                contact_steps.append(ep_len)
            cnt_prev = _cn
            residual_norms.append(info.get('residual_norm', 0.0))
            bd_now = info.get('episode_breakdown', {})
            step_trace.append({k: float(v) - float(bd_prev.get(k, 0.0))
                               for k, v in bd_now.items()})
            bd_prev = bd_now
            info_last = info
            if terminated or truncated:
                break
        # MPC 标称侧求解统计（速度场标称无 solve_cnt，跳过）
        # ⚠️ 原实现读的是 `mpc.solve_time`，而它是**最后一次**调用的耗时（mpc_nominal.py 里是赋值
        # 不是累加）→ 报出来的「平均规划时间」其实是「最后一步的规划时间」。现改为
        # solve_time_total 的**本集差分** / solve_cnt 的**本集差分**，才是真均值；原字段保留
        # `last_mpc_solve_s` 以备追溯。
        mpc = getattr(env, 'nominal_trajectory', None)
        solve_time, fallback = 0.0, 0
        solve_total_ep, solve_n_ep = 0.0, 0
        if mpc is not None and hasattr(mpc, 'solve_cnt'):
            fallback = int(getattr(mpc, 'fallback_cnt', 0))
            solve_total_ep = float(getattr(mpc, 'solve_time_total', 0.0) or 0.0) - solve_t0
            solve_n_ep = int(getattr(mpc, 'solve_cnt', 0) or 0) - solve_n0
            solve_time = (solve_total_ep / solve_n_ep) if solve_n_ep > 0 else 0.0
        ik_calls, ik_fails, _ik_rate = ik.ik_stats()
        via_plan_time = float(getattr(mpc, 'plan_time_total', 0.0) or 0.0) - via_t0 \
            if mpc is not None and hasattr(mpc, 'plan_time_total') else 0.0
        _prof_rate, _prof_cum = _normalized_profile(step_trace)
        _stepcum = _step_cum_profile(step_trace, self.max_steps)
        # 接触时刻诊断（--contact-trace 时才加键）：把接触步号按**本集归一化时间**分箱。
        # 读法：`first_contact_frac` 接近 1 ⇒ 接触发生在收尾/下降段（同 static3，规划层救不了）；
        # 明显小于 1（如 0.3~0.6）⇒ 途中就碰上了，有时间维度的余量（可绕/可等窗口）。
        _contact_trace_keys = {}
        if self.contact_trace:
            _NPH = 10
            _hist = [0] * _NPH
            for _s in contact_steps:
                _b = min(_NPH - 1, int((_s - 1) / max(1, ep_len) * _NPH))
                _hist[_b] += 1
            _contact_trace_keys = {
                'first_contact_step': int(contact_steps[0]) if contact_steps else -1,
                'first_contact_frac': (float(contact_steps[0]) / ep_len) if contact_steps else -1.0,
                'last_contact_step': int(contact_steps[-1]) if contact_steps else -1,
                'contact_phase_hist': _hist,
            }
        return {
            'episode_length': ep_len,
            'grasp_success': bool(info_last.get('grasp_success', False)),
            'final_distance': float(info_last.get('distance_to_object', 0.0)),
            'collision_count': int(ep_coll),
            'avg_residual_norm': float(np.mean(residual_norms)) if residual_norms else 0.0,
            'max_residual_norm': float(np.max(residual_norms)) if residual_norms else 0.0,
            'avg_mpc_solve_s': solve_time,          # 真均值 = 本集 solve_time_total / solve_cnt
            'mpc_solve_s_total': solve_total_ep,    # 本集 MPC 求解总耗时（可回溯）
            'mpc_solve_cnt': int(solve_n_ep),       # 本集 MPC 求解次数
            'last_mpc_solve_s': float(getattr(mpc, 'solve_time', 0.0) or 0.0)
            if mpc is not None else 0.0,            # 旧口径（最后一次耗时），仅供追溯
            'mpc_fallback': int(fallback),
            # --- IK 失败数（Stage2）---
            # 口径边界：只覆盖调用 `ik.solve_ik` 的地方（plan-time 目标 IK、抬升段位姿 IK）；
            # 执行期跟踪走 velocity_ik（逐帧伺服，无成功/失败语义），**不在计数内**。
            # 失败判定 = solve_ik 返回的位置误差 > tol（函数本身不返回失败标志）。
            'ik_call_count': int(ik_calls),
            'ik_fail_count': int(ik_fails),
            # --- 绕行路点层（global_planner='via_point' 时才有值，否则全 0/None）---
            # ⚠️ via_plan_time_s 是**每步穿过 via 包装器的总耗时，含内层 MPC 求解**
            #    （模块的 plan_time_total 从 reference_velocity 入口计时，内层调用在区间内）。
            #    它**不是**「via 层自己的规划耗时」——那个是下面 screen_time。
            'via_plan_time_s': float(via_plan_time),
            # via 层自己的候选筛选耗时（只在真的触发筛选时累计，与 screen_cnt 同口径）
            'via_screen_time_s': (float(getattr(mpc, 'screen_time_total', 0.0) or 0.0) - screen_t0)
            if mpc is not None else 0.0,
            # screen_cnt 是**跨集累计**计数器（模块内只加不减）→ 必须差分，否则 aggregate
            # 里按集求和会把「第 i 集时的累计值」当成「本集次数」，越靠后越离谱。
            'via_screen_cnt': (int(getattr(mpc, 'screen_cnt', 0) or 0) - screen_n0)
            if mpc is not None else 0,
            'via_cand_total': int(getattr(mpc, 'cand_total', 0) or 0) if mpc is not None else 0,
            'via_cand_pass': int(getattr(mpc, 'cand_pass', 0) or 0) if mpc is not None else 0,
            # 「本集是否采纳过路点」= 模块的**闩锁**，不是集末读 `_via_ee`
            # （到达路点后 `_via_ee` 会被清空 → 集末读法恒 False，会把采纳率报成 0）。
            'via_planned': bool(getattr(mpc, 'adopted_any', False)),
            'via_adopt_cnt': (int(getattr(mpc, 'adopt_cnt', 0) or 0) - adopt_n0)
            if mpc is not None else 0,
            # 余量三段分开报（都是 mj_geomDistance 精确口径、全 geom 对、distmax=1.0）：
            #   via   = 路点位形自身余量（无代理路径，臂必须能站上去）→ 表头口径
            #   seg1  = 第一段 q_now→q_via 路径最小余量（已硬门控 >0）
            #   seg2  = 第二段 q_via→q_hover 路径最小余量（只排序，未门控）
            #   total = min(seg1, seg2)；⚠️ 都是**关节插值代理路径**上的值，不是执行轨迹
            'via_margin_via_m': _opt_f(getattr(mpc, 'best_margin_m', None) if mpc else None),
            'via_margin_seg1_m': _opt_f(getattr(mpc, 'margin_seg1', None) if mpc else None),
            'via_margin_seg2_m': _opt_f(getattr(mpc, 'margin_seg2', None) if mpc else None),
            'via_margin_total_m': _opt_f(getattr(mpc, 'margin_total', None) if mpc else None),
            # 奖励分项（env 侧 episode_breakdown 累计，见 environment.py:926-929）。
            # 只在本集内累加，跟 episode_length 同口径，便于换算"每项每步均值"。
            'reward_breakdown': dict(info_last.get('episode_breakdown', {})),
            'episode_reward': float(sum(info_last.get('episode_breakdown', {}).values())),
            'reward_profile': _prof_rate,
            'reward_profile_cum': _prof_cum,
            'reward_stepcum': _stepcum,
            **_contact_trace_keys,
        }

    def run(self, n_episodes):
        results = []
        for i in range(n_episodes):
            if self.per_episode_seed:
                self._ep_seed = int(self.base_seed) + i if self.base_seed is not None else i
            r = self.run_episode()
            results.append(r)
            print(f"[res-eval] ep {i + 1}/{n_episodes}: len={r['episode_length']} "
                  f"success={r['grasp_success']} coll={r['collision_count']} "
                  f"res={r['avg_residual_norm']:.4f} solve={r['avg_mpc_solve_s']}s "
                  f"fallback={r['mpc_fallback']}")
        n = len(results)
        _succ = [r for r in results if r['grasp_success']]
        _fail = [r for r in results if not r['grasp_success']]
        # 奖励分项聚合：每集均值（事件项 grasp/close/avoid 是一集一次，看这一列）
        # + 每步均值（电平项 obstacle/residual/step/dist 看这一列）
        _bd_keys = sorted({k for r in results for k in r.get('reward_breakdown', {})})
        bd_mean, bd_per_step = {}, {}
        for k in _bd_keys:
            vals = [float(r.get('reward_breakdown', {}).get(k, 0.0)) for r in results]
            bd_mean[k] = float(np.mean(vals))
            # 每步均值 = 总奖励 / 总步数（ratio of means），不是"每集比值的均值"——
            # 后者会给短 episode（成功提前结束的）过高权重，且与 bd_mean/平均时长 对不上。
            bd_per_step[k] = float(np.sum(vals) / max(1, sum(r['episode_length'] for r in results)))
        # 每步奖励剖面：归一化时间轴上的平均曲线（全样本 / 仅成功子集）
        def _mean_profile(subset, field):
            profs = [r[field] for r in subset if r.get(field)]
            if not profs:
                return {}
            return {k: np.mean([p[k] for p in profs], axis=0).tolist() for k in profs[0]}
        prof_rate = _mean_profile(results, 'reward_profile')
        prof_rate_s = _mean_profile(_succ, 'reward_profile')
        prof_cum = _mean_profile(results, 'reward_profile_cum')
        prof_cum_s = _mean_profile(_succ, 'reward_profile_cum')
        stepcum = _mean_profile(results, 'reward_stepcum')
        stepcum_s = _mean_profile(_succ, 'reward_stepcum')
        return {
            'scene': self.scene,
            'n_episodes': n,
            'success_rate': sum(1 for r in results if r['grasp_success']) / n,
            # 复合判据（2026-09-15）：避障成功 **且** 抓取成功——零臂身碰撞才算这一集成功。
            # 单看 success_rate 会把「撞着障碍硬穿过去抓到了」也算成功（velocity_field 标称在
            # static3 上正是如此：成功率 76.7% 但平均每集 40 次碰撞）。
            'success_nc_rate': sum(1 for r in results
                                   if r['grasp_success'] and r['collision_count'] == 0) / n,
            'collision_rate': sum(1 for r in results if r['collision_count'] > 0) / n,
            'avg_collision_count': float(np.mean([r['collision_count'] for r in results])),
            'avg_episode_length': float(np.mean([r['episode_length'] for r in results])),
            'avg_residual_norm_success': float(np.mean([r['avg_residual_norm'] for r in _succ])) if _succ else 0.0,
            'avg_residual_norm_fail': float(np.mean([r['avg_residual_norm'] for r in _fail])) if _fail else 0.0,
            'max_residual_norm_overall': float(np.max([r['max_residual_norm'] for r in results])) if results else 0.0,
            # --- 规划时间（真均值 = Σ本集 solve_time_total / Σ本集 solve_cnt，ratio of sums）---
            # 不用「逐集均值的均值」：那会给求解次数少的集（早停成功的）过高权重。
            'avg_mpc_solve_s': (float(np.sum([r['mpc_solve_s_total'] for r in results]))
                                / max(1, int(np.sum([r['mpc_solve_cnt'] for r in results])))),
            'mpc_solve_s_total_sum': float(np.sum([r['mpc_solve_s_total'] for r in results])),
            'mpc_solve_cnt_sum': int(np.sum([r['mpc_solve_cnt'] for r in results])),
            'avg_last_mpc_solve_s': float(np.mean([r['last_mpc_solve_s'] for r in results])),
            'mpc_fallback_sum': int(np.sum([r['mpc_fallback'] for r in results])),
            # --- IK 失败数（见 run_episode 的口径边界说明）---
            'ik_call_count_sum': int(np.sum([r['ik_call_count'] for r in results])),
            'ik_fail_count_sum': int(np.sum([r['ik_fail_count'] for r in results])),
            'ik_fail_rate': (float(np.sum([r['ik_fail_count'] for r in results]))
                             / max(1, int(np.sum([r['ik_call_count'] for r in results])))),
            # --- 绕行路点层 ---
            'via_planned_rate': sum(1 for r in results if r['via_planned']) / n,
            'via_adopt_cnt_sum': int(np.sum([r['via_adopt_cnt'] for r in results])),
            'via_screen_cnt_sum': int(np.sum([r['via_screen_cnt'] for r in results])),
            'via_plan_time_s_sum': float(np.sum([r['via_plan_time_s'] for r in results])),
            'avg_via_plan_time_s': float(np.mean([r['via_plan_time_s'] for r in results])),
            # via 层自己的筛选耗时（不含内层 MPC）；规划时间指标里两者必须分开报，
            # 否则「加绕行」的机时代价会被内层 MPC 的耗时淹没。
            'via_screen_time_s_sum': float(np.sum([r['via_screen_time_s'] for r in results])),
            'avg_via_screen_time_s': float(np.mean([r['via_screen_time_s'] for r in results])),
            'avg_via_cand_total': float(np.mean([r['via_cand_total'] for r in results])),
            'avg_via_cand_pass': float(np.mean([r['via_cand_pass'] for r in results])),
            # --- 接触时刻诊断（--contact-trace 才有；其余情况不出键）---
            # `avg_first_contact_frac` 是本组最该看的一个数：接近 1 ⇒ 碰在收尾段（规划救不了）；
            # 明显小于 1 ⇒ 途中就碰，有时间维度的余量。`contact_phase_ep_rate[i]` =
            # 在第 i 个十分位内**至少碰过一次**的集数占比（比步数直方图更好读，不被长接触段主导）。
            **(dict(
                avg_first_contact_frac=(
                    float(np.mean([r['first_contact_frac'] for r in results
                                   if r['first_contact_frac'] >= 0]))
                    if any(r['first_contact_frac'] >= 0 for r in results) else -1.0),
                n_episodes_with_contact=int(sum(1 for r in results if r['first_contact_step'] >= 0)),
                contact_phase_hist_sum=[int(np.sum([r['contact_phase_hist'][i] for r in results]))
                                        for i in range(10)],
                contact_phase_ep_rate=[float(np.mean([1.0 if r['contact_phase_hist'][i] > 0 else 0.0
                                                      for r in results])) for i in range(10)],
            ) if self.contact_trace else {}),
            # 达成余量：只对**真的规划出路点**的集取均值/分布，否则均值会被 None 稀释
            'avg_via_margin_via_m': _mean_opt(results, 'via_margin_via_m'),
            'avg_via_margin_seg1_m': _mean_opt(results, 'via_margin_seg1_m'),
            'avg_via_margin_seg2_m': _mean_opt(results, 'via_margin_seg2_m'),
            'avg_via_margin_total_m': _mean_opt(results, 'via_margin_total_m'),
            'min_via_margin_via_m': _min_opt(results, 'via_margin_via_m'),
            'min_via_margin_seg1_m': _min_opt(results, 'via_margin_seg1_m'),
            'reward_breakdown_per_episode': bd_mean,
            'reward_breakdown_per_step': bd_per_step,
            'avg_episode_reward': float(np.mean([r.get('episode_reward', 0.0) for r in results])),
            'reward_profile_bins': N_PROFILE_BINS,
            'reward_profile_mean': prof_rate,
            'reward_profile_mean_success': prof_rate_s,
            'reward_profile_cum_mean': prof_cum,
            'reward_profile_cum_mean_success': prof_cum_s,
            'reward_stepcum_mean': stepcum,
            'reward_stepcum_mean_success': stepcum_s,
            # 逐集剖面只用于上面的聚合，不进 JSON（200 集 × 100 格点 × 12 项会把文件撑到几十 MB）
            'episode_results': [{k: v for k, v in r.items()
                                 if k not in ('reward_profile', 'reward_profile_cum',
                                              'reward_stepcum')} for r in results],
        }


def main():
    ap = argparse.ArgumentParser(description="Stage 2 评估：residual 架构 4 象限（velocity_field|MPC × 无RL|PPO Δv）")
    ap.add_argument("--scene", default="dyn_both", choices=list(SCENE_CFG))
    ap.add_argument("--n_episodes", type=int, default=30)
    ap.add_argument("--model", type=str, default="",
                    help="PPO 模型 zip（同目录 _vecnormalize.pkl 自动恢复观测统计）")
    ap.add_argument("--nominal-mode", default="mpc", choices=["mpc", "velocity_field"])
    ap.add_argument("--action-mode", default="residual", choices=["residual", "delta"],
                    help="residual=标称+Δv（四象限）；delta=端到端（无标称，标称层返回 None）")
    ap.add_argument("--residual-gate", default=None, choices=["on", "off"],
                    help="覆盖 residual_gate_enabled。HEAD 默认 on：无障场景 gate 恒 0 → 残差被清零。"
                         "评 pre-gating 年代训练的模型（v18/v11）必须置 off，否则残差是惰性的")
    ap.add_argument("--residual-clip-mode", default=None, choices=["modulus", "per_axis"],
                    help="残差裁剪几何。modulus=HEAD/v25+ 默认（‖Δv‖ 裁到 residual_delta_cap/T）。"
                         "per_axis=pre-gating/cadd00e 口径（逐轴裁 ±max_ee_delta/T，动作声明量程也回到米）。"
                         # ⚠️ argparse 会用 % 做格式化 → 帮助文本里的百分号必须写成 %%
                        # （否则 `--help` 直接 ValueError 崩掉；2026-09-20 踩到）
                        "评 v18_p3f / v11 等 pre-gating 模型必须置 per_axis，否则预算被压到约 23%%（§6）。")
    ap.add_argument("--zero-residual", action="store_true",
                    help="动作恒 0 → v_servo=v_nominal（纯标称对照）")
    ap.add_argument("--target-vel", type=float, default=0.05)
    ap.add_argument("--max-steps", type=int, default=200)
    ap.add_argument("--residual-delta-cap", type=float, default=None,
                    help="v25 残差预算解耦：residual 分支位置增量上限 m（默认 config.residual_delta_cap=0.002；"
                         "重跑 v24 同口径对照传 0.005 恢复旧 clip）")
    ap.add_argument("--d-safe", type=float, default=None,
                    help="覆盖 mpc_nominal_d_safe（默认 config 0.20；验证实验：调小验证 static3 物理可绕）")
    ap.add_argument("--ctrl-delay-steps", type=int, default=0,
                    help="模型失配扰动：执行速度延后 D 个决策步（plant 传输延迟；D=0 原行为）。"
                         "验证实验：纯 MPC vs MPC+RL 谁对执行延迟更鲁棒")
    ap.add_argument("--seed", type=int, default=None,
                    help="固定环境随机源（障碍随机游走/场景采样）→ 跨模型 A/B 可复现；默认不固定")
    ap.add_argument("--seed-per-episode", action="store_true",
                    help="逐 episode 重播种（seed+i）：使各配置的 30 个初始场景**逐集一致**，"
                         "消除随机游走消耗 RNG 造成的跨配置发散（A/B 配对对照）")
    ap.add_argument("--arm-aware", action="store_true",
                    help="arm-aware MPC：标称层代价纳入臂身碰撞球（默认关；开启后臂身不再对障碍失明）")
    ap.add_argument("--ori-servo", default=None, choices=["on", "off"],
                    help="姿态伺服：补上速度模式缺失的角速度通道（默认 off=现状，姿态不受控）。"
                         "on 时 ω=clamp(k·rotvec(q_grasp⊗q_ee⁻¹), ω_max)，对标 moveit_servo 跟完整 6 维位姿")
    ap.add_argument("--w-arm", type=float, default=None,
                    help="覆盖 mpc_nominal_w_arm（臂身惩罚权重，默认 config 120.0）")
    ap.add_argument("--arm-margin", type=float, default=None,
                    help="覆盖 mpc_nominal_arm_margin（臂身**目标**留量 m，默认 0.05）："
                         "d < 此值才进铰链代价。这是直接管「成功避障率」的保守度旋钮")
    ap.add_argument("--arm-pad", type=float, default=None,
                    help="覆盖 mpc_nominal_arm_pad（臂身碰撞球**半径**留量 m，默认 0.01）："
                         "把障碍在几何上放大，补球面采样/线性化误差")
    ap.add_argument("--contact-trace", action="store_true",
                    help="额外上报「接触发生在整集的哪个阶段」（first_contact_frac + 十分位分箱）。"
                         "诊断用：贴近 1 说明碰在收尾段（规划层救不了），明显小于 1 说明途中有余量。"
                         "默认关，避免给既有跑批的结果文件加键")
    ap.add_argument("--obs-raise", type=float, default=0.0,
                    help="障碍整体抬高(m)：obstacle_fixed_pos.z 与 obstacle_path_z_offset 同加该值，"
                         "默认 0=现状。背景(2026-09-15)：dyn_both 障碍球心 z=0.35、球顶 0.40，"
                         "而夹爪穿越高度带在 0.45 上下 → 障碍根本挡不住（速度场标称 9/10 集零碰撞），"
                         "且球还戳进物体 2.8cm(mj_geomDistance −0.0276)。抬 0.13(球心→0.48)后"
                         "速度场掉到 4/10 零碰撞，穿透同时归零")
    ap.add_argument("--perception-noise-std", type=float, default=None,
                    help="目标位置感知高斯噪声 σ(m)：污染观测通道的 target_position（默认 None=不改）。"
                         "配 --seed-per-episode 时 A/B 两臂噪声序列逐集一致（_perc_rng 在 reset 里按 seed 重播种）")
    ap.add_argument("--perception-dropout", type=float, default=None,
                    help="随机漏检概率(0~1)：漏检时目标估计零阶保持上次值（tracker 丢帧语义）")
    ap.add_argument("--perception-bias", type=float, default=None,
                    help="恒偏置感知误差(m)：每集抽一次方向、集内恒定的目标估计偏移（默认 None=不改）。"
                         "与 --perception-noise-std 的区别是时间结构——白噪声每步重抽会被闭环平均掉，"
                         "恒偏不会。配 --seed-per-episode 时 A/B 两臂偏置方向一致（_perc_rng 按 seed 重播种）")
    ap.add_argument("--perception-affects-nominal", action="store_true",
                    help="让标称层也消费同一份带噪估计（默认关=标称读真值）。开启后测的是"
                         "「感知误差经控制器传播后残差能否补回」，而不是「策略被喂垃圾时掉多少分」")
    ap.add_argument("--global-planner", default=None, choices=["off", "via_point"],
                    help="标称层外的全局规划开关（Stage2 2026-09-19）：'via_point' 启用末端绕行"
                         "路点层（见 via_point_nominal.py）。默认 None=不改（config 的 'off' → "
                         "与改造前逐位一致）。仅在 --nominal-mode mpc 下生效")
    ap.add_argument("--via-margin-target", type=float, default=None,
                    help="绕行路点的余量软目标(m)，默认 config 0.01。**软目标**：只用于上报口径，"
                         "不作硬门控（1 cm 硬余量在本场景族几何不可行，见 docs 数字真值表 §10）")
    ap.add_argument("--via-step", type=float, default=None,
                    help="候选网格间距(m)，默认 0.05。与 --via-grid 一起决定候选集大小")
    ap.add_argument("--via-grid", type=int, default=None,
                    help="候选网格半宽，默认 6 → (2*6+1)^2*3=507 个候选（与既有测量同口径）")
    ap.add_argument("--via-replan-steps", type=int, default=None,
                    help="重规划间隔（步），默认 25。静态障碍靠「按障碍位置缓存」实际每集只筛一次")
    ap.add_argument("--save-results", type=str, default="")
    ap.add_argument("--tag", type=str, default="",
                    help="臂标识（写进结果 config.tag，供出表脚本分组；不参与任何计算）")
    args = ap.parse_args()
    sc = SCENE_CFG[args.scene]

    cfg = get_config()
    g = cfg.grasping
    g.render_gui = False
    g.control_mode = "velocity"
    g.action_mode = args.action_mode    # residual=残差架构 / delta=端到端
    g.nominal_mode = args.nominal_mode  # 'mpc' 或 'velocity_field'
    g.nominal_enabled = True
    g.action_space_dim = 3
    g.use_fixed_position = True         # 固定物体位置（对齐消融口径）
    g.max_steps = args.max_steps
    if args.residual_delta_cap is not None:
        g.residual_delta_cap = args.residual_delta_cap   # v25：残差预算覆盖（默认 config 0.002）
    if args.d_safe is not None:
        g.mpc_nominal_d_safe = args.d_safe               # v25b2 验证：D_SAFE 覆盖
    if args.residual_gate is not None:
        g.residual_gate_enabled = (args.residual_gate == "on")
    # Stage2 全局规划（绕行路点）：全部带默认值，不传就逐位等同改造前
    if args.global_planner is not None:
        g.global_planner = args.global_planner
    if args.via_margin_target is not None:
        g.via_margin_target = args.via_margin_target
    if args.via_step is not None:
        g.via_step = args.via_step
    if args.via_grid is not None:
        g.via_grid = args.via_grid
    if args.via_replan_steps is not None:
        g.via_replan_steps = args.via_replan_steps
    if args.residual_clip_mode is not None:
        g.residual_clip_mode = args.residual_clip_mode      # 残差裁剪几何（口径显式化）
    if args.arm_aware:
        g.mpc_nominal_arm_aware = True                   # arm-aware MPC（臂身碰撞球进代价）
    if args.ori_servo is not None:
        g.orientation_servo_enabled = (args.ori_servo == "on")
    if args.w_arm is not None:
        g.mpc_nominal_w_arm = args.w_arm                 # 臂身惩罚权重覆盖
    if args.arm_margin is not None:
        g.mpc_nominal_arm_margin = args.arm_margin       # 臂身目标留量覆盖（保守度旋钮）
    if args.arm_pad is not None:
        g.mpc_nominal_arm_pad = args.arm_pad             # 臂身碰撞球半径留量覆盖
    if args.obs_raise:
        _ox, _oy, _oz = g.obstacle_fixed_pos             # 障碍抬高（球心 z）
        g.obstacle_fixed_pos = (_ox, _oy, _oz + args.obs_raise)
        g.obstacle_path_z_offset = g.obstacle_path_z_offset + args.obs_raise
    if args.ctrl_delay_steps is not None:
        g.ctrl_delay_steps = args.ctrl_delay_steps       # 模型失配：执行延迟扰动
    if args.perception_noise_std is not None:
        g.perception_noise_std = args.perception_noise_std
    if args.perception_dropout is not None:
        g.perception_dropout = args.perception_dropout
    if args.perception_bias is not None:
        g.perception_bias = args.perception_bias
    if args.perception_affects_nominal:
        g.perception_affects_nominal = True

    # ---- 目标/障碍场景（与 mpc_nominal_verify.py 同口径） ----
    g.dynamic_target_enabled = sc["dyn_target"]
    g.target_vel_xy = args.target_vel if g.dynamic_target_enabled else 0.0
    kind, count, vel = sc["obstacle"]
    if kind == "off":
        g.obstacle_enabled = False
        g.obstacle_count = 0
        g.obstacle_specs = None
    elif kind == "legacy":
        g.obstacle_enabled = True
        g.obstacle_count = count
        g.obstacle_vel = vel
        g.obstacle_on_nominal_path = (count > 1)   # 单障碍非 on-path（对齐基线 dyn_both 口径）
        g.obstacle_specs = None
        g.obstacle_z_motion = sc["z"]
    else:  # specs
        g.obstacle_enabled = True
        g.obstacle_specs = sc["specs"]
        g.obstacle_on_nominal_path = True

    env = GraspingEnv(g, cfg.reward)
    if args.seed is not None:
        env.reset(seed=int(args.seed))   # 固定 RNG（跨模型 A/B 同随机源）；后续 episode reset 不再重播种
    agent = None
    if args.model and not args.zero_residual:
        agent = GraspingAgent(cfg.network, cfg.training, model_path=args.model)
        agent.set_environment(env, n_envs=1)      # DummyVecEnv + VecNormalize（自动恢复 pkl 统计）
        print(f"[res-eval] 已加载策略模型: {args.model} (nominal_mode={args.nominal_mode})")

    ev = ResidualEvaluator(env, agent, scene=args.scene, max_steps=args.max_steps,
                           zero_residual=args.zero_residual,
                           per_episode_seed=args.seed_per_episode, base_seed=args.seed,
                           contact_trace=bool(args.contact_trace))
    res = ev.run(args.n_episodes)
    res['config'] = dict(
        scene=args.scene, n_episodes=args.n_episodes, nominal_mode=args.nominal_mode,
        # 臂标识（由 runner 脚本传入）：让结果文件自证「是四条臂里的哪一条」，
        # 出表脚本据此分组，不必靠文件名猜（tag 本身含下划线，切分不可靠）。
        tag=str(getattr(args, 'tag', '') or ''),
        action_mode=args.action_mode, model=os.path.basename(args.model) if args.model else "",
        zero_residual=bool(args.zero_residual), d_safe=g.mpc_nominal_d_safe,
        arm_aware=bool(g.mpc_nominal_arm_aware), w_arm=g.mpc_nominal_w_arm,
        # 臂身保守度两旋钮（2026-09-20 新增）——**口径事实**：它们直接决定臂身离障碍多远，
        # 调大即更保守。arm_margin 是代价里的目标距离，arm_pad 是把障碍几何放大。
        # 注意：主跑批（results/via_compare）是这两键加入**之前**启动的，早期文件里没有这两键；
        # 默认值全程未变（0.05 / 0.01），故缺键等价于默认值，指标不受影响。
        arm_margin=float(getattr(g, 'mpc_nominal_arm_margin', 0.05)),
        arm_pad=float(getattr(g, 'mpc_nominal_arm_pad', 0.01)),
        # 接触时刻诊断是否开启（口径事实：开了才有 first_contact_* 键）
        contact_trace=bool(getattr(args, 'contact_trace', False)),
        seed=args.seed, seed_per_episode=bool(args.seed_per_episode),
        residual_gate=(args.residual_gate or "default"),
        ori_servo=(args.ori_servo or "off"),
        obs_raise=float(args.obs_raise),
        max_steps=args.max_steps, ctrl_delay_steps=args.ctrl_delay_steps,
        # 残差裁剪几何是**口径事实**，必须进结果文件——否则同一份 JSON 无法自证
        # 它是在模长口径还是逐轴口径下产生的（v18/v11 在两种口径下差 33pp 以上）。
        residual_clip_mode=str(getattr(g, 'residual_clip_mode', 'modulus')),
        residual_delta_cap=float(getattr(g, 'residual_delta_cap', 0.002)),
        max_ee_delta=float(getattr(g, 'max_ee_delta', 0.005)),
        perception_noise_std=float(getattr(g, 'perception_noise_std', 0.0)),
        perception_bias=float(getattr(g, 'perception_bias', 0.0)),
        perception_affects_nominal=bool(getattr(g, 'perception_affects_nominal', False)),
        # Stage2 全局规划（绕行路点）——**口径事实**，必须自描述：global_planner 一开，
        # 标称层就不再是纯 MPC，而是「筛选路点 + MPC」，两者的成功率不可直接互推。
        global_planner=str(getattr(g, 'global_planner', 'off')),
        via_step=float(getattr(g, 'via_step', 0.05)),
        via_grid=int(getattr(g, 'via_grid', 6)),
        via_z_step=float(getattr(g, 'via_z_step', 0.06)),
        via_edge_res=float(getattr(g, 'via_edge_res', 0.01)),
        via_ik_tol=float(getattr(g, 'via_ik_tol', 1.5e-3)),
        via_advance_tol=float(getattr(g, 'via_advance_tol', 0.03)),
        via_replan_steps=int(getattr(g, 'via_replan_steps', 25)),
        via_rescreen_tol=float(getattr(g, 'via_rescreen_tol', 0.01)),
        via_margin_target=float(getattr(g, 'via_margin_target', 0.01)),
        via_score_points=int(getattr(g, 'via_score_points', 12)),
        via_topk=int(getattr(g, 'via_topk', 20)),
    )

    mode = ("端到端 PPO" if args.action_mode == "delta"
            else ("纯标称(Δv=0)" if (args.zero_residual or not args.model) else "PPO Δv"))
    print("\n========== Stage2 residual 评估结果 ==========")
    # 2026-09-20：**成功避障率提到头条**。此前口径围绕「1 cm 安全余量」展开，
    # 但该要求已被实测判定几何不可行并取消（见 docs 数字真值表 §11 / §10 失败链条）。
    # 现目标 = 在给定场景与候选集内尽量提高「无碰撞成功率」，故它排第一。
    print(f"scene={res['scene']} nominal={args.nominal_mode} model={mode}  "
          f"【成功避障率 succ_nc={res['success_nc_rate'] * 100:.1f}%】 "
          f"success={res['success_rate'] * 100:.1f}% "
          f"coll={res['collision_rate'] * 100:.1f}% "
          f"avg_coll={res['avg_collision_count']:.2f} len={res['avg_episode_length']:.1f} "
          f"res_succ={res['avg_residual_norm_success']:.4f} res_fail={res['avg_residual_norm_fail']:.4f}")
    # 规划时间 / IK 失败数 / 绕行路点达成余量（用户的四项指标里后三项）
    print(f"  规划: mpc_avg={res['avg_mpc_solve_s'] * 1000:.1f}ms "
          f"(总 {res['mpc_solve_s_total_sum']:.1f}s / {res['mpc_solve_cnt_sum']} 次) "
          f"via筛选均={res['avg_via_screen_time_s']:.2f}s/集 "
          f"(合计 {res['via_screen_time_s_sum']:.1f}s, {res['via_screen_cnt_sum']} 次) | "
          f"IK 失败 {res['ik_fail_count_sum']}"
          f"/{res['ik_call_count_sum']} = {res['ik_fail_rate'] * 100:.2f}%")
    if args.contact_trace:
        # 接触时刻：这是判「避障还有没有救」的关键数。接近 1 = 碰在收尾段（同 static3，规划救不了）；
        # 明显小于 1 = 途中就碰，有时间维度的余量。十分位给的是「该段内至少碰过一次的集数占比」。
        _ph = res.get('contact_phase_ep_rate') or []
        _bar = " ".join(f"{int(round(x * 100)):3d}" for x in _ph)
        print(f"  接触时刻: 有接触 {res.get('n_episodes_with_contact', 0)}/{res['n_episodes']} 集  "
              f"首次接触位置中位 {res.get('avg_first_contact_frac', -1) * 100:.0f}% of 整集")
        print(f"    十分位「碰过的集数占比(%)」: [{_bar}]  (左=开局 右=收尾)")
    if res['via_planned_rate'] > 0:
        # ⚠️ 不再拿达成余量去比任何「目标值」——1 cm 目标已取消。这里只报**实测达成值**，
        #    且必须**分段**：via/seg1 是本层可归因的；seg2 的终点是内层 hover（本身在墙里），
        #    本层对它只是排序、没有硬门控，故 seg2 小并不等于「本层没绕开」。
        print(f"  绕行路点: 采纳率 {res['via_planned_rate'] * 100:.1f}%  "
              f"候选均 {res['avg_via_cand_total']:.0f} 过 {res['avg_via_cand_pass']:.0f} | "
              f"实测达成余量(均值) via={_fmt_m(res['avg_via_margin_via_m'])} "
              f"seg1={_fmt_m(res['avg_via_margin_seg1_m'])} "
              f"seg2={_fmt_m(res['avg_via_margin_seg2_m'])}")

    if args.save_results:
        path = args.save_results if os.path.isabs(args.save_results) else \
            os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", args.save_results)
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            json.dump(res, f, indent=2, ensure_ascii=False)
        print(f"saved -> {path}")


if __name__ == "__main__":
    main()

