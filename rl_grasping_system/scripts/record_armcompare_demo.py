#!/usr/bin/env python3
"""同一障碍、同一初始状态下，多臂并排对比 demo（GIF）。

为什么单独写一个：`record_static3_demo.py` 场景写死 static3、只吃单个 --model；
`record_mpc_demo.py` 只录纯 MPC。要排查「同一条障碍下各臂行为差在哪」，
需要**跨臂逐帧对齐**的录像，且障碍摆放必须逐臂一致。

同一障碍怎么保证：episode 编号 k 固定 → 每臂都用 `env.reset(seed=base+k)` 重播种，
障碍摆放只依赖该 RNG 与初始臂位形 → 逐臂逐位一致（摆放发生在任何动作之前）。
脚本会把各臂的障碍坐标打印出来并校验一致，不一致直接报错。

口径与 `scripts/mpc_plus_rl_eval.py` 完全对齐（复用其 SCENE_CFG）。

用法：
    cd rl_grasping_system
    # 默认 5 臂 / dyn_both_train（训练一致口径）/ 第 0 集 / 训练时不存在姿态伺服
    python3 scripts/record_armcompare_demo.py --episode 0
    # 开姿态伺服那一轮
    python3 scripts/record_armcompare_demo.py --episode 0 --ori-servo on
    # 只看某几个臂 / 换场景
    python3 scripts/record_armcompare_demo.py --arms mpc_nominal mpc_res25b2 --scene dyn_both_train
"""
import os
import sys
import argparse

os.environ.setdefault('MUJOCO_GL', 'egl')

import numpy as np
from PIL import Image, ImageDraw

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from config import get_config
from environment import GraspingEnv
from show_grasp_demo import _set_oblique_camera, _set_topdown_camera
from mpc_plus_rl_eval import SCENE_CFG

# 与 run_unified7_trainmatch.sh / run_unified7_offpathds020.sh 的 ARGS 逐条对应
ARMS = {
    'vf_nominal':            ('velocity_field', '',                                              ['--zero-residual']),
    'vf_res18':              ('velocity_field', 'models/final_model_stage2_d2_v18_p3f.zip',        ['--residual-gate', 'off']),
    'vf_res25b2':            ('velocity_field', 'models/final_model_stage2_d2_v25b2_gate.zip',     ['--residual-gate', 'on']),
    'mpc_nominal':           ('mpc',            '',                                               ['--zero-residual']),
    'mpc_res25b2':           ('mpc',            'models/final_model_stage2_d2_v25b2_gate.zip',     ['--residual-gate', 'on']),
    'mpc_armaware':          ('mpc',            '',                                               ['--zero-residual', '--arm-aware']),
    'mpc_armaware_res25b2':  ('mpc',            'models/final_model_stage2_d2_v25b2_gate.zip',     ['--arm-aware', '--residual-gate', 'on']),
    'e2e_v11':               ('mpc',            'models/final_model_stage2_d2_v11_500k.zip',       ['--action-mode', 'delta']),
}
DEFAULT_ARMS = ['vf_nominal', 'vf_res18', 'mpc_nominal', 'mpc_res25b2', 'e2e_v11']


def build_env(cfg, scene, arm):
    """按 mpc_plus_rl_eval.main() 的同一条路径配置环境。

    不在此处碰 d_safe / orientation_servo：那两个由 main() 统一设好（单一来源），
    之前在这里用 `ori_servo == 'on'` 接 bool 传参，`True == 'on'` 恒 False，
    把 main 设好的姿态伺服又覆盖回关（servo on/off 两轮 GIF 逐字节相同即此故）。
    """
    g = cfg.grasping
    nominal_mode, model, extra = ARMS[arm]
    action_mode = 'delta' if '--action-mode' in extra else 'residual'

    g.render_gui = False
    g.control_mode = 'velocity'
    g.action_mode = action_mode
    g.nominal_mode = nominal_mode
    g.nominal_enabled = True
    g.action_space_dim = 3
    g.use_fixed_position = True
    g.mpc_nominal_arm_aware = '--arm-aware' in extra
    if '--residual-gate' in extra:
        g.residual_gate_enabled = (extra[extra.index('--residual-gate') + 1] == 'on')

    sc = SCENE_CFG[scene]
    g.dynamic_target_enabled = sc['dyn_target']
    g.target_vel_xy = 0.05 if sc['dyn_target'] else 0.0
    kind, count, vel = sc['obstacle']
    if kind == 'off':
        g.obstacle_enabled = False
        g.obstacle_count = 0
        g.obstacle_specs = None
    elif kind == 'legacy':
        g.obstacle_enabled = True
        g.obstacle_count = count
        g.obstacle_vel = vel
        g.obstacle_on_nominal_path = (count > 1)
        g.obstacle_specs = None
        g.obstacle_z_motion = sc['z']
    else:
        g.obstacle_enabled = True
        g.obstacle_specs = sc['specs']
        g.obstacle_on_nominal_path = True

    env = GraspingEnv(g, cfg.reward)
    agent = None
    if model:
        from agent import GraspingAgent
        agent = GraspingAgent(cfg.network, cfg.training, model_path=model)
        agent.set_environment(env, n_envs=1)
    return env, agent


def obstacle_positions(env):
    try:
        return np.asarray(env._get_obstacle_positions(), dtype=float).round(4)
    except Exception:
        return np.zeros((0, 3))


def record_arm_with_render(cfg, scene, arm, seed, max_steps, frame_step,
                           lookat_xy, topdown, width, height):
    """每臂自建 renderer（各臂 model 实例不同，不能共用）。"""
    import mujoco
    env, agent = build_env(cfg, scene, arm)
    renderer = mujoco.Renderer(env.model, height, width)
    try:
        obs, _ = env.reset(seed=seed)
        if env.nominal_trajectory is not None:
            env.nominal_trajectory.reset()
        obs_pos = obstacle_positions(env)
        frames, info = [], {}
        step = 0
        for step in range(1, max_steps + 1):
            if agent is None:
                action = np.zeros(env.action_space.shape[0], dtype=np.float32)
            else:
                action, _ = agent.predict(obs, deterministic=True)
            obs, _r, term, trunc, info = env.step(action)
            if step % frame_step == 0 or term or trunc:
                renderer.update_scene(env.data)
                if topdown:
                    _set_topdown_camera(renderer, lookat_xy=lookat_xy)
                else:
                    _set_oblique_camera(renderer, lookat_xy=lookat_xy)
                frames.append(renderer.render().copy())
            if term or trunc:
                break
        stat = dict(arm=arm, steps=step,
                    success=bool(info.get('grasp_success', False)),
                    collisions=int(info.get('obstacle_collision_count', 0)),
                    final_d=float(info.get('distance_to_object', 0.0)))
        return frames, obs_pos, stat
    finally:
        renderer.close()
        env.close()


def label(img, text):
    im = Image.fromarray(img)
    d = ImageDraw.Draw(im)
    d.rectangle([0, 0, im.width, 16], fill=(0, 0, 0))
    d.text((4, 3), text, fill=(255, 255, 255))
    return im


def main():
    ap = argparse.ArgumentParser(description='同一障碍下多臂并排对比 demo')
    ap.add_argument('--scene', default='dyn_both_train', choices=list(SCENE_CFG))
    ap.add_argument('--arms', nargs='+', default=DEFAULT_ARMS, choices=list(ARMS))
    ap.add_argument('--episode', type=int, default=0, help='episode 编号 k → seed = --seed + k（各臂一致）')
    ap.add_argument('--seed', type=int, default=12345)
    ap.add_argument('--d-safe', type=float, default=None)
    ap.add_argument('--ori-servo', default=None, choices=['on', 'off'])
    ap.add_argument('--max-steps', type=int, default=200)
    ap.add_argument('--frame-step', type=int, default=4)
    ap.add_argument('--width', type=int, default=480)
    ap.add_argument('--height', type=int, default=360)
    ap.add_argument('--topdown', action='store_true')
    ap.add_argument('--outdir', default='results/demo_armcompare')
    args = ap.parse_args()

    import mujoco  # noqa: F401  （确保 MUJOCO_GL 生效后再导入）

    cfg = get_config()
    cfg.grasping.max_steps = args.max_steps
    if args.d_safe is not None:
        cfg.grasping.mpc_nominal_d_safe = args.d_safe
    if args.ori_servo is not None:
        cfg.grasping.orientation_servo_enabled = (args.ori_servo == 'on')

    seed = args.seed + args.episode
    lookat_xy = tuple(cfg.grasping.object_fixed_pos)
    os.makedirs(args.outdir, exist_ok=True)

    servo_tag = 'servo' + ('on' if cfg.grasping.orientation_servo_enabled else 'off')
    dsafe = cfg.grasping.mpc_nominal_d_safe

    per_arm = {}
    ref_obs = None
    stats = []
    for arm in args.arms:
        frames, obs_pos, stat = record_arm_with_render(
            cfg, args.scene, arm, seed, args.max_steps, args.frame_step,
            lookat_xy, args.topdown, args.width, args.height)
        per_arm[arm] = frames
        stats.append(stat)
        if ref_obs is None:
            ref_obs = obs_pos
        elif obs_pos.shape == ref_obs.shape and not np.allclose(obs_pos, ref_obs, atol=1e-6):
            print(f'[WARN] {arm} 的障碍摆放与首臂不一致：\n  首臂 {ref_obs.tolist()}\n  本臂 {obs_pos.tolist()}')
        print(f"[cmp] {arm:<22} steps={stat['steps']:<4} success={stat['success']} "
              f"collisions={stat['collisions']:<3} final_d={stat['final_d']:.3f}")

    if ref_obs is not None and len(ref_obs):
        print(f'[cmp] 本次障碍坐标（各臂共用）:\n      {ref_obs.tolist()}')

    n = max(len(v) for v in per_arm.values())
    panels = []
    for arm in args.arms:
        fr = per_arm[arm]
        fr = fr + [fr[-1]] * (n - len(fr))          # 短 episode 用末帧静止补足
        s = next(x for x in stats if x['arm'] == arm)
        panels.append([label(f, f"{arm} | {'OK' if s['success'] else 'FAIL'} | coll={s['collisions']}")
                       for f in fr])

    for i in range(n):
        strip = Image.new('RGB', (panels[0][i].width * len(args.arms), panels[0][i].height))
        for j in range(len(args.arms)):
            strip.paste(panels[j][i], (j * panels[0][i].width, 0))
        per_arm.setdefault('_strip', []).append(strip)

    tag = f"{args.scene}_{servo_tag}_dsafe{dsafe:g}_ep{args.episode}"
    for arm in args.arms:
        p = os.path.join(args.outdir, f'{tag}_{arm}.gif')
        imgs = [label(f, arm) for f in per_arm[arm]]
        imgs[0].save(p, save_all=True, append_images=imgs[1:], duration=60, loop=0)
        print(f'      -> {p}')
    strip = per_arm['_strip']
    p = os.path.join(args.outdir, f'{tag}_COMPARE.gif')
    strip[0].save(p, save_all=True, append_images=strip[1:], duration=60, loop=0)
    print(f'      -> {p}  ({len(strip)} 帧, {strip[0].width}x{strip[0].height})')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
