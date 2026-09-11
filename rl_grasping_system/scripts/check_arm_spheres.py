#!/usr/bin/env python3
"""arm-aware MPC 的几何 + 模型保真度自检（2026-09-11，覆盖口径修正后）。

三件事：
  ① 覆盖：列出所有可碰撞 geom 的采样球数量与半径 → 人工可核（夹爪必须在内）。
  ② 保真度（关键）：把随机末端速度 u 的一阶预测 `A_s @ u · dt`
     vs 真实执行链（`solve_ik` 到 `cur_pos + u·dt`、锁姿态）后的球位移。
     二者应吻合（误差 ≪ 球半径 r_s），否则 DLS 口径 / 雅可比取列 / 坐标系写错。
  ③ 包住性：MESH/BOX 用 `geom_rbound` 包围球覆盖，必须**真包住** mesh 顶点
     （逐顶点验证 dist(v, 球心) ≤ 半径）。
  ④ 有效性：把障碍移到夹爪球心 → `_check_arm_obstacle_contact()` 必须为 True
     （证明我们优化的部位确实是会被判碰撞的部位）。

⚠️ 不要用 `env._get_obstacle_distance()` 当 ground truth（AABB 近似、忽略朝向）。

用法：
    cd rl_grasping_system
    MUJOCO_GL=egl python3 scripts/check_arm_spheres.py
"""
import os
import sys

import numpy as np

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import mujoco
from config import get_config
from environment import GraspingEnv
from ik import solve_ik


def build_env(arm_aware=True):
    cfg = get_config()
    g = cfg.grasping
    g.render_gui = False
    g.control_mode = 'velocity'
    g.action_mode = 'residual'
    g.nominal_mode = 'mpc'
    g.nominal_enabled = True
    g.action_space_dim = 3
    g.use_fixed_position = True
    g.max_steps = 200
    g.obstacle_enabled = True            # ④ 需要 _obstacle_active/obstacle_geom_ids_all 生效
    g.obstacle_count = 1
    g.obstacle_on_nominal_path = False
    g.mpc_nominal_arm_aware = arm_aware
    return cfg, GraspingEnv(g, cfg.reward)


def main():
    cfg, env = build_env()
    env.reset(seed=12345)
    mpc = env.nominal_trajectory
    if mpc is None or not hasattr(mpc, '_sample_geom_spheres'):
        raise SystemExit('nominal_mode 不是 mpc / 无 arm context')

    model, data = env.model, env.data
    ee_body = int(env.end_effector_id)
    arm_joints = list(env.arm_joint_ids)

    dt = 0.04
    mpc.dt = dt
    ctx = mpc._build_arm_context()
    if ctx is None:
        raise SystemExit('arm context 构建失败（None）——检查 env/model/arm_joint_ids')

    T = {int(mujoco.mjtGeom.mjGEOM_MESH): 'MESH', int(mujoco.mjtGeom.mjGEOM_CAPSULE): 'CAPSULE',
         int(mujoco.mjtGeom.mjGEOM_CYLINDER): 'CYL', int(mujoco.mjtGeom.mjGEOM_BOX): 'BOX',
         int(mujoco.mjtGeom.mjGEOM_SPHERE): 'SPHERE'}

    # ---------- ① 覆盖 ----------
    print(f"==== ① 覆盖：可碰撞 geom 的采样球（总计 n={ctx['n']}）====")
    scratch = mujoco.MjData(model)
    scratch.qpos[:] = data.qpos[:]
    mujoco.mj_forward(model, scratch)
    n_total = 0
    for g in sorted(env._arm_obstacle_geoms()):
        if model.geom_contype[g] == 0:
            continue
        b = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, int(model.geom_bodyid[g])) or '?'
        sp = mpc._sample_geom_spheres(model, scratch, int(g))
        n_total += len(sp)
        rr = f"{sp[0][1]:.3f}" if sp else "-"
        print(f"  geom {g:>3} {T.get(int(model.geom_type[g]), int(model.geom_type[g])):>8} "
              f"{b:>22}  球数={len(sp):>3}  半径={rr}")
    print(f"  → 采样球合计 {n_total}（应与 ctx['n']={ctx['n']} 一致，含 pad={mpc.arm_pad}）")
    ee = np.asarray(env._get_end_effector_position(), dtype=float)
    print(f"  末端={np.round(ee, 3)}  障碍半径 r_obs={ctx['r_obs']:.3f}  留量 margin={mpc.arm_margin:.3f}")

    # ---------- ③ 包住性：包围球必须真包住 mesh 顶点 ----------
    print(f"\n==== ③ 包住性：geom_rbound 球 vs mesh 全部顶点 ====")
    worst = 0.0
    for g in sorted(env._arm_obstacle_geoms()):
        if model.geom_contype[g] == 0 or int(model.geom_type[g]) != int(mujoco.mjtGeom.mjGEOM_MESH):
            continue
        di = int(model.geom_dataid[g])
        a, k = int(model.mesh_vertadr[di]), int(model.mesh_vertnum[di])
        # ⚠️ mesh_vert 已是 **geom 局部系**（编译器已烘入 mesh_pos/quat/scale）：
        # 既不能减世界系 geom_xpos，也不能再加 mesh_pos（重复计入）。
        V = np.asarray(model.mesh_vert[a:a + k], dtype=float)
        R = float(model.geom_rbound[g])
        dmax = float(np.max(np.linalg.norm(V, axis=1)))   # 到 geom 局部原点（= 包围球心）
        b = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, int(model.geom_bodyid[g])) or '?'
        ok = dmax <= R + 1e-6
        worst = max(worst, dmax - R)
        print(f"  geom {g:>3} {b:>22} 顶点最远={dmax:.4f} rbound={R:.4f} "
              f"{'✅ 包住' if ok else '❌ 未包住'}")
    print(f"  → 最大超出量 {worst * 1000:+.3f} mm（应 ≤0）")

    # ---------- ② 保真度 ----------
    print(f"\n==== ② 保真度：一阶 dp_pred = A_s@u·dt  vs  真实 solve_ik ====")
    cur_pos = np.asarray(data.xpos[ee_body], dtype=float).copy()
    cur_quat = np.asarray(data.xquat[ee_body], dtype=float).copy()
    vmax = mpc.v_max
    A = ctx['A_s']

    def spheres_now(sc):
        pts = []
        for g in env._arm_obstacle_geoms():
            if model.geom_contype[g] == 0:
                continue
            pts += [p for p, _r in mpc._sample_geom_spheres(model, sc, int(g))]
        return np.asarray(pts, dtype=float)

    assert len(spheres_now(scratch)) == ctx['n'], '采样顺序/数量不一致'

    rng = np.random.default_rng(0)
    errs = []
    for trial in range(8):
        u = rng.uniform(-1.0, 1.0, size=3)
        u = u / max(np.linalg.norm(u), 1e-9) * vmax * rng.uniform(0.5, 1.0)
        dp_pred = (A @ u) * dt                       # (n,3)
        q_tgt, err = solve_ik(model, data, ee_body, cur_pos + u * dt, cur_quat, arm_joints)
        sc = mujoco.MjData(model)
        sc.qpos[:] = data.qpos[:]
        sc.qpos[arm_joints] = q_tgt
        mujoco.mj_forward(model, sc)
        dp_real = spheres_now(sc) - ctx['pos0']
        e = np.linalg.norm(dp_pred - dp_real, axis=1)
        errs.append(e)
        print(f"trial{trial}: |u|={np.linalg.norm(u):.3f} ik_err={err:.2e}  "
              f"max|dp_pred|={np.linalg.norm(dp_pred, axis=1).max() * 1000:.2f}mm  "
              f"max|预测-真实|={e.max() * 1000:.3f}mm  相对球半径={e.max() / ctx['rad'].min():.3f}")

    errs = np.asarray(errs)
    print(f"\n汇总：最大预测误差 {errs.max() * 1000:.3f} mm  "
          f"vs 最小球半径 {ctx['rad'].min() * 1000:.1f} mm  "
          f"→ {'✅ 通过（≪ 球半径）' if errs.max() < 0.2 * ctx['rad'].min() else '❌ 不合格（口径/雅可比/坐标有误）'}")
    print(f"arm_skip_cnt={mpc.arm_skip_cnt}（奇异跳过次数，应很少）")

    # ---------- ④ 有效性：夹爪球心放障碍 → 必须被判碰撞 ----------
    print(f"\n==== ④ 有效性：把障碍移到采样球心 → _check_arm_obstacle_contact() 应为 True ====")
    env2 = build_env(arm_aware=False)[1]
    env2.reset(seed=12345)
    for _ in range(3):                      # 让臂动起来（非初始位姿）
        env2.step(np.zeros(env2.action_space.shape[0], dtype=np.float32))
    mujoco.mj_forward(env2.model, env2.data)
    sc2 = mujoco.MjData(env2.model)
    sc2.qpos[:] = env2.data.qpos[:]
    mujoco.mj_forward(env2.model, sc2)
    mpc2 = env2.nominal_trajectory
    mpc2.dt = dt
    hits, tried = 0, 0
    ee2 = int(env2.end_effector_id)
    for g in sorted(env2._arm_obstacle_geoms()):
        if env2.model.geom_contype[g] == 0:
            continue
        for pt, r in mpc2._sample_geom_spheres(env2.model, sc2, int(g)):
            tried += 1
            env2._write_obstacle_pos(env2.obstacle_body_ids[0], np.asarray(pt, dtype=float))
            mujoco.mj_forward(env2.model, env2.data)
            if env2._check_arm_obstacle_contact():
                hits += 1
    print(f"  在 {tried} 个采样球心各放一个障碍：{hits} 个立刻判为接触 "
          f"→ {'✅ 采样球确实覆盖可被判碰撞的部位' if hits >= 0.5 * tried else '❌ 覆盖部位与碰撞判据不符'}")
    env.close()
    env2.close()
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
