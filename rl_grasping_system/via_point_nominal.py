"""末端绕行路点标称层（Stage2 2026-09-19）——给末端级 MPC 补一层「绕过去」。

## 为什么需要它

`MpcNominal` 是**末端单积分器**滚动最优控制：决策变量是未来 N 步的末端速度，代价只惩罚
「预测末端轨迹挨障碍太近」。它没有「绕过去」的概念——障碍挡在必经之路上时，唯一能降代价
的方向被堵死，于是停在安全位不动（static3 失败机制，见 docs/数字真值表_20260910.md §7）。

本模块在末端空间找 1 个偏离直连的中间点（绕行路点 / via point），让内层 MPC 先跟踪它，
绕过障碍带后交回内层 hover / 下降逻辑。即把 `docs/混合控制架构设计.md` 里 MoveIt2 的角色，
用一个极简替身补上。

## 为什么不做关节空间 RRT（前身方案，已否决）

实测（static3 族，`mj_geomDistance` 精确口径）：
- 零余量 RRT 最好调参 6 seed 只成功 **2/6**，失败四次各跑满 42–45 s，平均 **29.8 s/次**；
- 要求 ≥1 cm 余量：5 墙位 × 3 seed 只有 1/15；507 候选 3D 网格扫描 0 命中。
故改为**确定性候选筛选 + 打分**，不做采样规划。详见 docs 同节与 memory。

## 流程（惰性，首次 `reference_velocity` 时才规划）

1. 无障碍 / 障碍未激活 → **立即退化**为内层 MPC（零规划开销，逐位同一结果）。
2. 候选集：以 hover 为中心，在 (lat_dir, path_dir) 平面 + z 上取网格（默认 507 个）。
3. IK 筛选：`ik.solve_ik` 逐个解，位置误差 > `via_ik_tol` 的丢弃并计数。
4. **第一段**（`q_now → q_via`）可执行性检验：按 `via_edge_res` 关节插值，逐点用
   `mj_geomDistance` 精确表面距判**不许穿透**（min > 0）——**硬门控**。
   ⚠️ 不用窄相接触表：mesh 深穿透会漏报，实测首集「接触表判无接触」的路径精确余量是
   −55 mm。门控与上报必须同源（见 `_seg_min_gap` docstring）。
5. **第二段**（`q_via → q_hover`）同样检验，但**只作排序与上报，不作硬门控**。
   理由（2026-09-19 实测，n=30 集）：raw hover 位形本身有 **17/30 集**与墙接触（中位余量
   −7.0 mm），且用 24 个随机扰动种子重启 IK 也搜不到 hover 处的干净位形（所有解的 EE 偏差恒
   0.2 mm，即都收敛回同一分支）。若把第二段设成硬门控，via 层会在 2/3 的集里直接空转。
   语义上这也更正确：到达位姿后**臂身**怎么避让，是内层（arm-aware 代价 + 零空间自由度）
   的职责，末端路点层只负责把 EE 带到别处。该分野如实写进文档。
6. 打分选优：粗排（路点位形余量）→ 前 `via_topk` 名沿两段路径精扫，取「最小余量最大」者。
   最终上报的余量用 `mj_geomDistance` **精确口径**（全 geom 对、distmax=1.0）独立复算。
7. 执行：选中候选经 FK 得到末端点，作为 `hover_override` 传给内层 MPC；末端距该点
   < `via_advance_tol` 时交回内层。路点正好落在脚下（≤ `via_advance_tol`）则不接管，避免自锁。
8. 重规划：首次 + 每 `via_replan_steps` 步。**候选筛选结果按障碍位置缓存**，仅当障碍最大位移
   > `via_rescreen_tol` 才重跑完整筛选（static3 的墙不动 → 每集只筛一次）；缓存命中时仍用
   **当前**位形对最优候选复检第一段（最多 `_RECHECK_MAX` 个），复检不过就顺延到下一个候选。

## 指标口径边界（如实交代）

- **「1 cm」是软目标**：本模块只能在给定候选集内最大化最小余量，达成值逐集上报
  （`best_margin_m`）。不得写成「给出了 1 cm 安全路径」，也不得外推为「全局最优绕行」。
- **余量只在「两段关节插值」这条代理路径上测**：真实执行是 EE 伺服，关节轨迹不同。该代理
  与内层接触判据同源、可比，但它不是执行轨迹本身。
- **桌面不在判据内**：沿用 `env._arm_obstacle_geoms()`（排除 table），与既有口径一致。
- **障碍是 freejoint**：臂可以推开球（碰撞照计数），所以「余量为负」不等于任务必然失败。
- **缓存的语义**：静态场景每集只筛一次是缓存的结果，不是「重规划免费」。
"""
import time

import mujoco
import numpy as np

from mpc_nominal import MpcNominal

# 缓存命中时对候选做位形复检的上限个数（每次复检 ≈ 两段插值扫描，~7 ms）。设上限是为了让
# 「采纳」这步耗时可控；超过上限仍无候选通过 → 本步不接管（交回内层）。
_RECHECK_MAX = 8


class ViaPointNominal:
    """绕行路点标称层：内层持有 `MpcNominal`，对外接口与其逐字相同。

    暴露给环境（`bind` / `attach_env` / `reset` / `reference_velocity`）与评测器
    （`solve_cnt` / `solve_time` / `solve_time_total` / `fallback_cnt` / `min_obstacle_dist` /
    `terminal_err` / `solver_ok`）的字段全部透传，故 `environment.py` 与
    `nominal_trajectory.py` 无需为它做任何特判。
    """

    def __init__(self, cfg):
        self.cfg = cfg
        self.inner = MpcNominal(cfg)
        # 开关：'via_point' 才启用；'off' 时本类退化为纯透传壳（逐位等同内层）
        self.enabled = str(getattr(cfg, 'global_planner', 'off')).lower() == 'via_point'
        # --- 规划超参（全部来自 cfg 的 GlobalPlanner 块，见 config.py）---
        self.step = float(getattr(cfg, 'via_step', 0.05))
        self.grid = int(getattr(cfg, 'via_grid', 6))
        self.z_step = float(getattr(cfg, 'via_z_step', 0.06))
        self.edge_res = float(getattr(cfg, 'via_edge_res', 0.01))
        self.ik_tol = float(getattr(cfg, 'via_ik_tol', 1.5e-3))
        self.advance_tol = float(getattr(cfg, 'via_advance_tol', 0.03))
        self.replan_steps = int(getattr(cfg, 'via_replan_steps', 25))
        self.rescreen_tol = float(getattr(cfg, 'via_rescreen_tol', 0.01))
        self.margin_target = float(getattr(cfg, 'via_margin_target', 0.01))
        self.score_points = int(getattr(cfg, 'via_score_points', 12))
        self.topk = int(getattr(cfg, 'via_topk', 20))
        self.hover_z_offset = float(getattr(cfg, 'nominal_hover_z_offset', 0.134))
        # --- 模型句柄（bind 时拿到）---
        self.model = None
        self.data = None
        self.arm_joint_ids = None
        self.ee_id = None
        self._env = None
        self._scratch = None            # 独立 MjData：筛选期反复写 qpos 做窄相，不污染真实仿真
        self._gsets_cache = None
        # --- 规划缓存（按障碍位置）---
        self._cache_key = None          # 上次筛选时的激活障碍位置 (n,3)
        self._cache_passers = None      # 通过第一段的候选（已排序）
        self._cache_q_hover = None      # 上次筛选时的 hover 位形（第二段终点）
        # --- 累计指标（跨 episode 累加；评测器取差分）---
        self.plan_time_total = 0.0      # 本层自身开销合计（筛选 + 复检 + 步内簿记）
        self.screen_cnt = 0             # 真正跑了完整筛选的次数
        self.screen_time_total = 0.0
        self.ik_fail_cnt = 0            # 筛选期 IK 失败累计
        self.cand_total = 0             # 上次筛选的候选总数（过了预筛的）
        self.cand_pass = 0              # 上次筛选通过第一段的候选数
        self.replan_cnt = 0             # 触发规划的步数
        self.adopt_cnt = 0              # 采纳路点的次数（跨集累计；一集可采纳多次）
        # 注意：本层自己的退化次数叫 via_fallback_cnt；`fallback_cnt` 是内层 MPC 的 SLSQP
        # 降级计数（评测器的 mpc_fallback 读的是那个），两者语义不同，不要合并。
        self.via_fallback_cnt = 0       # 规划无可用路点的次数（含无障碍退化）
        # --- 当前集状态：三段精确余量（m；None=本集未规划或无障碍）---
        # best_margin_m 是**表头口径** = 路点位形自身的余量（与代理路径无关，臂必须能站上去）；
        # seg1 是第一段路径最小余量（已硬门控，>0）；seg2 是第二段（只排序、不上报门控）；
        # total = min(seg1, seg2)。拆开报的原因见 _finalize docstring。
        self.best_margin_m = None
        self.margin_seg1 = None
        self.margin_seg2 = None
        self.margin_total = None
        self.margin_fast = None         # 打分排序用的快速口径值（与上面几个口径不同，勿混用）
        self.seg2_contact = None        # 采纳路点的第二段是否穿透（软指标）
        self._reset_episode_state()

    # ------------------------------------------------------------------ 接口
    def bind(self, model, data, arm_joint_ids, body_id):
        """绑定模型句柄（与 MpcNominal.bind 同签名，但这里**真的**要用到它们）。"""
        self.model = model
        self.data = data
        self.arm_joint_ids = list(arm_joint_ids)
        self.ee_id = int(body_id)
        self._scratch = mujoco.MjData(model)
        self.inner.bind(model, data, arm_joint_ids, body_id)
        return self

    def attach_env(self, env):
        """注入环境引用（读激活障碍位置 / 臂几何 / 接触判据）。"""
        self._env = env
        self.inner.attach_env(env)

    def reset(self):
        """清内层 warm-start 与**本集**路点状态；保留跨集缓存与累计指标。"""
        self.inner.reset()
        self._reset_episode_state()

    def _reset_episode_state(self):
        self._via_ee = None
        self._step_cnt = 0
        self._need_plan = True          # 本集第一次调用即规划
        # ⚠️ 「本集是否采纳过路点」必须用**闩锁**，不能在集末读 `_via_ee`：
        # 末端到达路点后 `_via_ee` 会被清空交回内层（_maybe_plan），成功集的集末几乎恒为 None
        # → 集末读法会把采纳率系统性报成 0（2026-09-20 冒烟实测踩到：2/2 都采纳了却报 0/2）。
        self.adopted_any = False
        self.best_margin_m = None
        self.margin_seg1 = None
        self.margin_seg2 = None
        self.margin_total = None
        self.margin_fast = None
        self.seg2_contact = None

    # 内层字段直接透传（评测器 / v25b2 gate 读这些）
    @property
    def solve_cnt(self):
        return self.inner.solve_cnt

    @property
    def solve_time(self):
        return self.inner.solve_time

    @property
    def solve_time_total(self):
        return self.inner.solve_time_total

    @property
    def fallback_cnt(self):
        return self.inner.fallback_cnt

    @property
    def solver_ok(self):
        return self.inner.solver_ok

    @property
    def min_obstacle_dist(self):
        return self.inner.min_obstacle_dist

    @property
    def terminal_err(self):
        return self.inner.terminal_err

    def reference_velocity(self, ee_pos, target_pos, dt=None, target_vel=None):
        """每决策步入口：必要时挑一个绕行路点，作为 hover_override 交给内层 MPC。"""
        t0 = time.time()
        override = None
        if self.enabled and self.model is not None:
            override = self._maybe_plan(ee_pos, target_pos)
        v = self.inner.reference_velocity(ee_pos, target_pos, dt, target_vel,
                                         hover_override=override)
        self.plan_time_total += time.time() - t0
        return v

    # ------------------------------------------------------------------ 规划主流程
    def _maybe_plan(self, ee_pos, target_pos):
        """返回本步要用的绕行路点（3 维）或 None（交回内层）。

        None 的语义：内层按 `target_pos + hover_z_offset` 自行算 hover（含两阶段抬 z），
        与本层未启用时逐位一致——这是「无障碍场景回归护栏」的基础。
        """
        self._step_cnt += 1
        ee = np.asarray(ee_pos, dtype=float).ravel()[:3]
        if self._via_ee is not None:
            if float(np.linalg.norm(ee - self._via_ee)) <= self.advance_tol:
                self._via_ee = None         # 到了 → 交回内层
                self._need_plan = True      # 过了路点可能还需要下一个（同一步内立即重筛）
            else:
                return self._via_ee         # 没到 → 继续跟踪
        if not (self._need_plan or (self.replan_steps > 0
                                    and self._step_cnt % self.replan_steps == 0)):
            return None
        self._need_plan = False
        self.replan_cnt += 1
        plan = self._screen(target_pos)
        if plan is None:
            self.via_fallback_cnt += 1
            return None
        if float(np.linalg.norm(ee - plan['ee'])) <= self.advance_tol:
            return None                     # 路点就在脚下 → 不接管（防「到达→重筛→再到达」自锁）
        self._via_ee = plan['ee']
        self.adopted_any = True          # 本集采纳过（闩锁，见 _reset_episode_state）
        self.adopt_cnt += 1              # 跨集累计（评测器差分）
        self.best_margin_m = plan['margin']
        self.margin_seg1 = plan['margin_seg1']
        self.margin_seg2 = plan['margin_seg2']
        self.margin_total = plan['margin_total']
        self.margin_fast = plan['margin_fast']
        self.seg2_contact = plan['seg2_contact']
        return self._via_ee

    def _screen(self, target_pos):
        """候选筛选（带障碍位置缓存）。返回选中路点 dict 或 None。"""
        env = self._env
        if env is None or not getattr(env, '_obstacle_active', False):
            return None                      # 无障碍：本层退化
        obs = np.asarray(env._get_obstacle_positions(), dtype=float)
        if obs.size == 0:
            return None
        key = obs.reshape(-1, 3)
        if (self._cache_passers is not None and self._cache_key is not None
                and self._cache_key.shape == key.shape
                and float(np.max(np.abs(key - self._cache_key))) <= self.rescreen_tol):
            return self._adopt_from_cache()
        t0 = time.time()
        passers, cand_total, ik_fail = self._enumerate_and_check(target_pos, key)
        self.screen_time_total += time.time() - t0
        self.screen_cnt += 1
        self.cand_total, self.cand_pass = cand_total, len(passers)
        self.ik_fail_cnt += ik_fail
        self._cache_key, self._cache_passers = key.copy(), passers
        if not passers:
            return None
        return self._finalize(passers[0])

    def _adopt_from_cache(self):
        """缓存命中：用**当前**位形复检第一段（硬门控），顺延取第一个仍可行的候选。

        缓存记的是「障碍不动时哪些候选当初可行」，但第一段的起点是臂的当前位形——臂一动，
        这一段是否无接触就可能变。故采纳前必须复检，不能直接复用当初的结论。
        """
        for cand in self._cache_passers[:_RECHECK_MAX]:
            if self._seg_min_gap(self._q_now(), cand['q'], early_stop=0.0) > 0.0:
                return self._finalize(cand)
        return None

    # ------------------------------------------------------------------ 候选筛选
    def _enumerate_and_check(self, target_pos, obs):
        """枚举候选 → IK → 第一段硬门控 → 粗排。返回 (passers 降序, 候选总数, IK 失败数)。"""
        hover = np.asarray(target_pos, float).ravel()[:3] + np.array([0.0, 0.0, self.hover_z_offset])
        lat_dir, path_dir = self._search_dirs(hover)
        q_now = self._q_now()
        q_hover, _ = self._ik(hover)                 # 第二段终点（软指标用）
        self._cache_q_hover = q_hover
        n_used, n_fail, passers = 0, 0, []
        for i in range(-self.grid, self.grid + 1):
            for j in range(-self.grid, self.grid + 1):
                for k in (0, -1, 1):
                    cand = (hover + lat_dir * (i * self.step)
                            + path_dir * (j * self.step)
                            + np.array([0.0, 0.0, k * self.z_step]))
                    if not self._cand_sane(cand, obs, hover):
                        continue
                    n_used += 1
                    q_via, err = self._ik(cand)
                    if q_via is None or err > self.ik_tol:
                        n_fail += 1
                        continue
                    if self._seg_min_gap(q_now, q_via, early_stop=0.0) <= 0.0:
                        continue                       # 第一段硬门控：不许穿透
                    seg2 = True
                    if q_hover is not None:
                        seg2 = self._seg_min_gap(q_via, q_hover, early_stop=0.0) <= 0.0
                    passers.append({'q': q_via, 'target': cand,
                                    'coarse': self._gap_fast(q_via),
                                    'seg2_contact': bool(seg2)})
        passers.sort(key=lambda c: -c['coarse'])
        for cand in passers[:self.topk]:               # 前 topk 名沿两段路径精扫
            cand['score'] = self._path_gap_fast(q_now, cand['q'], q_hover)
        for cand in passers[self.topk:]:
            cand['score'] = cand['coarse']
        # 软目标 = 最大化最小余量；同分优先「第二段无接触」（终点更容易交回内层）
        passers.sort(key=lambda c: (c['seg2_contact'], -c['score']))
        return passers, n_used, n_fail

    def _finalize(self, cand):
        """把候选落成路点：FK 求末端点 + 精确口径复算三段余量。

        为什么拆三段而不是只报「整条路径最小值」（2026-09-19 实测倒逼）：
        - 第一段 q_now→q_via 是本层**能控制、且已硬门控**的部分 → `seg1` 才是可归因于
          「绕行选点」的余量。
        - 第二段 q_via→q_hover 的终点是内层自己的 hover；raw hover 位形在 17/30 集里本来就
          泡在墙里（中位 −7.0 mm），**不是本层的决策**。若把两段合起来报，数字被第二段主导，
          归因失真。
        - `via` = 路点位形自身的余量：**与代理路径无关**的硬事实（臂必须能站到那个位形），
          故取它作为表头「达成余量」。
        三个数都是 `mj_geomDistance` 精确口径、全 geom 对、distmax=1.0，与
        `env._min_arm_obstacle_gap()` 同源可比。⚠️ 仍是**关节插值代理路径**上的余量，不是
        执行轨迹（执行是 EE 伺服，关节轨迹不同）——文档必须带这句边界。
        """
        q_via = cand['q']
        m = self._margins_exact(self._q_now(), q_via, self._cache_q_hover)
        return {
            'ee': self._fk(q_via),
            'q': q_via,
            'margin': m['via'],                      # 表头口径：路点位形自身余量（无代理）
            'margin_seg1': m['seg1'],
            'margin_seg2': m['seg2'],
            'margin_total': m['total'],
            'margin_fast': float(cand.get('score', cand.get('coarse', 0.0))),
            'seg2_contact': bool(cand['seg2_contact']),
        }

    def _margins_exact(self, q_now, q_via, q_hover):
        """三段精确余量：{via, seg1, seg2, total}（全 geom 对、distmax=1.0）。"""
        env = self._env
        if not env.obstacle_geom_ids_all:
            return {'via': None, 'seg1': None, 'seg2': None, 'total': None}
        geoms = (list(env.obstacle_geom_ids_all), list(env._arm_obstacle_geoms()))
        n = self.score_points * 2
        seg1 = min(self._gap_at(q, geoms, 1.0)
                   for q in _sample_seg(q_now, q_via, n))
        seg2 = None
        if q_hover is not None:
            seg2 = min(self._gap_at(q, geoms, 1.0)
                       for q in _sample_seg(q_via, q_hover, n))
        return {
            'via': self._gap_at(q_via, geoms, 1.0),
            'seg1': float(seg1),
            'seg2': None if seg2 is None else float(seg2),
            'total': float(seg1 if seg2 is None else min(seg1, seg2)),
        }

    # ------------------------------------------------------------------ 几何与判据工具
    def _q_now(self):
        return self.data.qpos[self.arm_joint_ids].copy()

    def _ik(self, target_pos):
        """DLS IK（以当前 data 位形为起点 → 取同一分支）。返回 (q, 位置误差)。"""
        import ik as _ik
        quat = self.data.xquat[self.ee_id].copy()
        q, err = _ik.solve_ik(self.model, self.data, self.ee_id,
                              np.asarray(target_pos, float), quat, self.arm_joint_ids,
                              tol=self.ik_tol)
        return q, float(err)

    def _fk(self, q):
        """FK：给定关节角求末端世界系位置（走 scratch，不扰动真实仿真）。"""
        self._scratch.qpos[:] = self.data.qpos[:]
        self._scratch.qpos[self.arm_joint_ids] = q
        mujoco.mj_kinematics(self.model, self._scratch)
        return self._scratch.xpos[self.ee_id].copy()

    def _search_dirs(self, hover):
        """候选搜索平面：(lat_dir, path_dir)。

        lat_dir 取环境的墙法向（= 墙自身长轴），沿它平移等于让墙「滑自己」，对余量帮助很小；
        真正有绕行意义的是 path_dir（绕过墙端）与 z（越过墙顶）。仍按方案保留两轴网格，是为了
        与「507 候选」的既有测量同口径。
        """
        env = self._env
        d = hover - self.data.xpos[self.ee_id]
        d[2] = 0.0
        n = float(np.linalg.norm(d))
        path_dir = d / n if n > 1e-6 else np.array([1.0, 0.0, 0.0])
        lat = getattr(env, '_obstacle_lat_dir', None)
        lat = (np.array([-path_dir[1], path_dir[0], 0.0]) if lat is None
               else np.asarray(lat, float).ravel()[:3])
        lat = np.array([lat[0], lat[1], 0.0])
        ln = float(np.linalg.norm(lat))
        return (lat / ln if ln > 1e-6 else np.array([-path_dir[1], path_dir[0], 0.0])), path_dir

    def _cand_sane(self, cand, obs, hover):
        """候选预筛：不落进障碍球、不钻到桌面下、不离谱远离 hover（限定搜索盒）。"""
        r = float(getattr(self._env.grasping_config, 'obstacle_radius', 0.05))
        table_top = float(getattr(self._env.grasping_config, 'table_top_z', 0.30))
        if cand[2] < table_top + 0.03:
            return False
        for p in obs:
            if float(np.linalg.norm(cand - p)) <= r:
                return False
        box = (self.grid * self.step) * 1.5 + self.z_step
        return float(np.linalg.norm(cand - hover)) <= box + 1e-9

    def _seg_min_gap(self, q_a, q_b, early_stop=0.0):
        """段上最小臂-障碍表面距：q_a → q_b 按 `via_edge_res` 关节插值，逐点精确算。

        **为什么不用窄相接触表**（原设计）：MuJoCo 窄相只对凸包做检测，mesh（夹爪）与球**深穿透**
        时会漏报。2026-09-19 实测 static3 首集：接触表判「无接触」的路径，`mj_geomDistance` 给出
        最深 **−55 mm**。用接触表做门控 = 门控口径（松）与上报口径（严）不一致，会放行一堆
        实际 5 cm 插进墙里的「合格候选」。故门控改用与上报同源的精确表面距。

        代价控制：只算**可碰撞 geom 对**（69 对），逐点收缩 distmax、一旦 ≤ `early_stop` 立即
        退出，且用 `mj_kinematics`（5.9 µs）而非 `mj_forward`（127 µs）——`mj_geomDistance` 只
        依赖 `geom_xpos/geom_xmat`，实测两者结果**逐位相同**，本层也不再需要接触表。
        `early_stop=0.0` → 「不许穿透」门控；返回值为该段最小余量（≥5 cm 一律截到 5 cm）。
        """
        env = self._env
        if not env.obstacle_geom_ids_all:
            return float('inf')
        O, A = self._geom_sets()
        dq = np.asarray(q_b, float) - np.asarray(q_a, float)
        n = max(1, int(np.ceil(float(np.linalg.norm(dq)) / self.edge_res)))
        ft = np.zeros(6)
        best = 0.05
        # 障碍是 freejoint，位置存在 qpos 里 → 扫描前同步一次（扫描期间障碍不动，故只需一次）
        self._scratch.qpos[:] = self.data.qpos[:]
        for t in np.linspace(0.0, 1.0, n + 1):
            self._scratch.qpos[self.arm_joint_ids] = q_a + t * dq
            mujoco.mj_kinematics(self.model, self._scratch)
            for a in O:
                for b in A:
                    d = mujoco.mj_geomDistance(self.model, self._scratch, a, b, best, ft)
                    if d < best:
                        best = float(d)
            if best <= early_stop:
                return best
        return best

    def _geom_sets(self):
        """(可碰撞障碍 geom, 可碰撞臂 geom)：把 mj_geomDistance 的对数从 348 降到 69。

        实测 348 对 6.2 ms / 69 对 0.11 ms（≈90×）。contype 与 conaffinity 都为 0 的纯显示 geom
        永远不产生接触，对余量没有贡献，剔除后**数值不变**（已与 env 的全量口径交叉核对）。
        """
        if self._gsets_cache is None:
            m = self._env.model
            live = lambda g: (m.geom_contype[g] > 0) or (m.geom_conaffinity[g] > 0)
            self._gsets_cache = ([g for g in self._env.obstacle_geom_ids_all if live(g)],
                                 [g for g in self._env._arm_obstacle_geoms() if live(g)])
        return self._gsets_cache

    def _gap_at(self, q, geoms, distmax):
        """给定位形的最小臂-障碍表面距。`geoms=(O,A)` 决定用什么 geom 集合与 distmax。

        走 `mj_kinematics`（不跑碰撞/动力学）——只依赖位姿，且实测与 `mj_forward` 结果一致。
        """
        O, A = geoms
        self._scratch.qpos[:] = self.data.qpos[:]
        self._scratch.qpos[self.arm_joint_ids] = q
        mujoco.mj_kinematics(self.model, self._scratch)
        ft = np.zeros(6)
        best = float(distmax)
        for a in O:
            for b in A:
                d = mujoco.mj_geomDistance(self.model, self._scratch, a, b, best, ft)
                if d < best:
                    best = float(d)
        return best

    def _gap_fast(self, q):
        """快速余量：可碰撞 geom 对 + distmax 收缩（≥5 cm 一律记 5 cm，够用即止）。"""
        if not self._env.obstacle_geom_ids_all:
            return float('inf')
        return self._gap_at(q, self._geom_sets(), 0.05)

    def _path_gap_fast(self, q_now, q_via, q_hover):
        """两段路径上的快速最小余量（**打分排序**用；上报不用它，见 `_margins_exact`）。"""
        qs = _sample_two_seg(q_now, q_via, q_hover, self.score_points)
        return min(self._gap_fast(q) for q in qs)


def _sample_seg(q_a, q_b, n_points):
    """单段按点数均匀采样（含两端）。用于精确余量：点数是「精度预算」，不是几何分辨率。"""
    q_a = np.asarray(q_a, float)
    q_b = np.asarray(q_b, float)
    m = max(1, int(n_points))
    return [q_a + t * (q_b - q_a) for t in np.linspace(0.0, 1.0, m + 1)]


def _sample_two_seg(q_now, q_via, q_hover, n_points):
    """把 q_now→q_via（+ q_via→q_hover）按关节弧长均匀采 n_points 个位形。

    q_hover 为 None（第二段终点 IK 失败）时只采第一段。采样点含两端。
    用于**快速**口径：总点数固定（`via_score_points`），按弧长在两段间分配。
    """
    q_now = np.asarray(q_now, float)
    q_via = np.asarray(q_via, float)
    segs = [(q_now, q_via)]
    if q_hover is not None:
        segs.append((q_via, np.asarray(q_hover, float)))
    lens = [float(np.linalg.norm(b - a)) for a, b in segs]
    total = sum(lens)
    n_points = max(2, int(n_points))
    if total <= 1e-12:
        return [q_via]
    qs = []
    for (a, b), L in zip(segs, lens):
        m = max(1, int(round(n_points * L / total)))
        qs.extend(a + t * (b - a) for t in np.linspace(0.0, 1.0, m + 1))
    return qs
