#!/usr/bin/env python3
"""把每个场景的「场景设置」拍成一张对照总表（场景 × 视角/时刻）。

为什么单独写：文档里描述场景全靠文字（"3 个障碍、1 静止 2 运动"），读的人
（包括面试前的自己）很难在脑子里拼出画面；而 dyn_both 与 dyn_both_train 的差别
**全在摆位上**（off-path vs on-path）、mixed3 与 mixed_z 的差别**全在 z 向运动**上，
一句话说不清，一张图就清楚。static3 那面墙为什么必然挡路，也是看图一眼就明白的事。

口径：复用 record_armcompare_demo.build_env（它与 mpc_plus_rl_eval.SCENE_CFG 逐条对齐），
不加载任何策略，**不改任何指标、不写任何结果 JSON**，纯出图。

`--arm` 两种取值，默认 `hold`：
  hold    = `action_mode='delta'`（标称层返回 None，见 nominal_trajectory.py:191）+ 零动作
            → 机械臂停在起始位形不动（速度控制带重力前馈 qfrc_bias，不会塌，见 environment.py:817）。
  nominal = 标称层驱动机械臂去抓（更"真实"，但**不是所有场景都能跑满**：实测 static / dyn_target
            在 seed 12345 上第 1 步就终止，晚快照点只能复用末帧）。
**默认 hold 的理由**：这张图要的是"场景长什么样"，静止的臂让障碍/目标的运动成为画面主体，
且每集必然跑满 200 步，不会出现"某一列只是重复上一列"。要看臂怎么动请用 --arm nominal。

文字一律英文：本机没装 CJK 字体，中文会被渲成豆腐块（图内文字与代码标识符一样保持英文）。

列布局（每行一个场景）：第 1 列是**几何俯视示意图**（layout_schematic，纯 matplotlib，
不经 MuJoCo 相机），后面是 EGL 照片（front@首快照 + oblique@每个快照）。

输出：
    results/scene_setup/scene_setup_sheet.png       总表（场景 × 列）
    results/scene_setup/<scene>__layout.png         该场景的俯视示意图（单独引用）
    results/scene_setup/<scene>__<view>_at<NNN>.png 单帧照片

用法：
    cd rl_grasping_system
    python3 scripts/render_scene_setup.py                              # 全部 7 个场景
    python3 scripts/render_scene_setup.py --scenes static3 mixed3
    python3 scripts/render_scene_setup.py --snapshot-steps 0 80 160    # 默认
"""
import os
import sys
import subprocess
import argparse

os.environ.setdefault('MUJOCO_GL', 'egl')

import numpy as np
from PIL import Image, ImageDraw, ImageFont

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.dirname(_HERE))   # -> rl_grasping_system
sys.path.append(_HERE)                    # -> rl_grasping_system/scripts

from config import get_config
from show_grasp_demo import _set_oblique_camera
from mpc_plus_rl_eval import SCENE_CFG
from record_armcompare_demo import build_env, ARMS

# 用哪条臂**只影响机械臂怎么动**，不影响障碍摆放（摆放只依赖 RNG 与初始位形）。
# 「臂不动」档照 e2e_v11 的写法给 ARMS 加一条，让 build_env 走同一条已被验证的代码路径。
# 为什么不用 g.nominal_enabled：该键在 environment.py 里**根本没被读过**（grep 无命中），
# 真正决定标称层有无的是 action_mode —— nominal_trajectory.py:191 非 'residual' 直接返回 None。
ARMS['mpc_hold'] = ('mpc', '', ['--zero-residual', '--action-mode', 'delta'])
ARM_BY_MODE = {'hold': 'mpc_hold', 'nominal': 'mpc_nominal'}
TARGET_VEL = 0.05    # 与 record_armcompare_demo.build_env 的 g.target_vel_xy 同源


def _set_front_camera(renderer, lookat_xy=(0.0, 0.0), distance=0.8, elevation=0.15):
    """从 -y 方向看（正对末端接近路径），最能把"墙横在必经之路上"拍清楚。

    与 _set_oblique_camera 同构，只把水平方位角从 45° 改成 0°（方向 (0,-1,elevation)）。
    为什么需要它：static3 那面墙的长轴约在 +y 方向（`_obstacle_lat_dir`），从 45° 斜看
    会被墙的侧棱挡住，"墙挡住通道"这件事得正对着看才直观。
    """
    cam = renderer.scene.camera[0]
    lookat = np.array([lookat_xy[0], lookat_xy[1], 0.32], dtype=float)
    direction = np.array([0.0, -1.0, elevation], dtype=float)
    direction /= np.linalg.norm(direction)
    cam.pos[:] = lookat + distance * direction
    cam.forward[:] = -direction
    up = np.array([0.0, 0.0, 1.0], dtype=float)
    up -= cam.forward * float(up @ cam.forward)
    up /= np.linalg.norm(up)
    cam.up[:] = up


# 只保留这两个 EGL 视角。**没有 topdown**：本模型 XML 已定义 eye_to_hand 相机，
# `renderer.scene.camera` 因此变成两槽，实测按 fovy=45 反推的俯拍高度与实际成像差约
# 1.7 倍、桌面顶边总被切出画面（渲染确定、位姿也确实生效，但取景对不上，三轮未定位根因）。
# 俯视布局改由 layout_schematic() 纯几何绘制——确定、可标注、且不必开 MuJoCo。
VIEW_SETTERS = {
    'front': _set_front_camera,
    'oblique': _set_oblique_camera,
}


def layout_schematic(scene, g, obj_xy, obs_xyz, w, h):
    """该场景的俯视布局示意图（纯几何绘制，不经 MuJoCo 相机）。

    画的是 reset 时刻的**真值**：桌面矩形、物体、**激活的**障碍球（半径按 0.05 m 等比），
    另加 roundtrip 障碍的往返双箭头（half_range/period 来自 SCENE_CFG）。
    未激活障碍不在 obs_xyz 里（`_get_obstacle_positions` 只返回激活的），
    所以 static / dyn_target 画出来就是空桌面——与文字口径一致。
    """
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D
    from matplotlib.patches import Circle, FancyArrowPatch, Rectangle

    gc = g.grasping
    cx, cy = gc.table_center
    hs = gc.table_half_size
    R_OBS = 0.05          # 障碍球半径：与 XML sphere size 一致（config 未单独导出）

    fig = plt.figure(figsize=(w / 100.0, h / 100.0), dpi=100)
    ax = fig.add_axes([0.15, 0.16, 0.82, 0.74])
    ax.grid(True, lw=0.3, color='0.92', zorder=0)
    ax.add_patch(Rectangle((cx - hs, cy - hs), 2 * hs, 2 * hs, facecolor='0.80',
                           edgecolor='0.45', zorder=1))
    ax.plot([0], [0], marker='s', ms=7, color='k', zorder=5)
    ax.annotate('base', (0, 0), textcoords='offset points', xytext=(-6, -13),
                fontsize=7, color='k', ha='right', zorder=6)
    ax.plot([obj_xy[0]], [obj_xy[1]], marker='s', ms=9, mfc='#ff7f0e', mec='k',
            mew=0.6, zorder=5)
    ax.annotate('target', obj_xy[:2], textcoords='offset points', xytext=(6, 4),
                fontsize=7, color='#a04000', zorder=6)

    pts = [(cx - hs, cy - hs), (cx + hs, cy + hs), (0.0, 0.0), (obj_xy[0], obj_xy[1])]
    for p in obs_xyz:
        ax.add_patch(Circle((p[0], p[1]), R_OBS, facecolor='m', edgecolor='0.25',
                            lw=0.7, zorder=4))
        pts.append((p[0], p[1]))
    if scene in SCENE_CFG and SCENE_CFG[scene]['obstacle'][0] == 'specs':
        for sp in SCENE_CFG[scene]['specs']:
            if sp.get('mode') != 'roundtrip':
                continue
            ax_, hr = sp.get('axis', 'x'), sp['half_range']
            x0, y0 = sp['pos'][0], sp['pos'][1]
            d = (hr, 0.0) if ax_ == 'x' else (0.0, hr)
            ax.add_patch(FancyArrowPatch((x0 - d[0], y0 - d[1]), (x0 + d[0], y0 + d[1]),
                                         arrowstyle='<->', ls='--', color='m', lw=1.2,
                                         mutation_scale=8, zorder=4))
            pts += [(x0 - d[0], y0 - d[1]), (x0 + d[0], y0 + d[1])]

    pad = R_OBS + 0.03
    xs = [p[0] for p in pts]
    ys = [p[1] for p in pts]
    ax.set_xlim(min(xs) - pad, max(xs) + pad)
    ax.set_ylim(min(ys) - pad, max(ys) + pad)
    ax.set_aspect('equal', adjustable='box')
    ax.set_xlabel('x [m]', fontsize=8)
    ax.set_ylabel('y [m]', fontsize=8)
    ax.tick_params(labelsize=6)
    ax.set_title(scene, fontsize=10)
    ax.legend(handles=[
        Line2D([], [], ls='', marker='s', ms=7, color='k', label='robot base'),
        Line2D([], [], ls='', marker='s', ms=8, mfc='#ff7f0e', mec='k', label='target'),
        Line2D([], [], ls='', marker='o', ms=8, mfc='m', mec='0.25', label='obstacle (r=0.05)'),
    ], fontsize=6.5, loc='lower left', framealpha=0.95, borderpad=0.4, labelspacing=0.35)
    fig.canvas.draw()
    buf = np.asarray(fig.canvas.buffer_rgba())[:, :, :3].copy()
    plt.close(fig)
    return buf


def scene_lines(scene, obs_xyz=None, tgt=None):
    """场景描述文字。**运动规格**取自 SCENE_CFG，**位置/目标参数**用 reset 真值。

    为什么位置不照抄 SCENE_CFG 里 specs 的 `pos`：评测跑在 `--obstacle-on-nominal-path` 下，
    位置会被 _place_obstacles_on_path() 覆盖（config.py:404 自己写明"位置由必经之路布局覆盖，
    运动行为仍按 specs 生效"）。实测 mixed3/mixed_z 的 reset 位置与 static3 的墙**逐位相同**、
    与 specs 写的 `pos` 完全不同；照抄 specs 会让文字与图互相矛盾。
    逐障碍的下标 i 与 specs 下标一致（_obstacle_spec(i) 按同一序取）。

    tgt = (v, T, axis) 取自 env 自己的 grasping_config（不是外层 cfg——目标的
    target_vel_xy 是 build_env 按场景写进去的）。动态目标是沿轴的**三角波往返**
    （environment.py:1105），不是单向平移：峰峰幅 = v·T，t=0 与 t=T 同位置、
    半周期处最远。只写"0.05 m/s"会让人以为 6.4 s 后该跑出 0.32 m，实际它又回到起点。
    """
    sc = SCENE_CFG[scene]
    obs_xyz = list(obs_xyz or [])
    v_t, T_t, ax_t = tgt if tgt else (TARGET_VEL, 6.0, 'x')
    if sc['dyn_target']:
        lines = ["moving target: %s, +-%.2f m, T=%.1f s" % (ax_t, v_t * T_t / 2.0, T_t),
                 "  (triangular round trip, back at start at t=T)"]
    else:
        lines = ["fixed target"]
    kind, count, vel = sc['obstacle']

    def xy(i):
        return "(%.2f,%.2f)" % (obs_xyz[i][0], obs_xyz[i][1]) if i < len(obs_xyz) else "(?)"

    if kind == 'off':
        lines.append("no obstacle")
    elif kind == 'legacy':
        # count == 1 时 obstacle_on_nominal_path 为 False（见 SCENE_CFG 里 dyn_both 的注释），
        # 障碍被摆到 off-path 固定位；count > 1 才是 on-path（= 训练摆法）。
        lines.append("%d obstacle(s), legacy, %s" % (count, "ON nominal path" if count > 1 else "off-path"))
        lines.append("obstacle speed %.2f m/s" % vel)
        if obs_xyz:
            lines.append("reset xy: " + " ".join(xy(i) for i in range(len(obs_xyz))))
    else:
        lines.append("%d obstacles (on-path placement):" % count)
        for i, sp in enumerate(sc['specs']):
            if sp['type'] == 'static':
                lines.append("  [%d] static %s" % (i, xy(i)))
            elif sp.get('mode') == 'roundtrip':
                lines.append("  [%d] roundtrip %s +-%.2f m T=%.1f s %s"
                             % (i, sp.get('axis', 'x'), sp['half_range'], sp['period'], xy(i)))
            else:
                lines.append("  [%d] random walk %.2f m/s %s" % (i, sp['vel'], xy(i)))
            if sp.get('z_motion'):
                lines.append("        + z osc %.2f m T=%.1f s" % (sp['z_amp'], sp['z_period']))
    return lines


def _load_fonts():
    """用 matplotlib 自带的 DejaVu Sans（纯 ASCII 字形，绕开本机无 CJK 字体的问题）。"""
    try:
        from matplotlib import font_manager
        reg = font_manager.findfont('DejaVu Sans', fallback_to_default=False)
        bold = os.path.join(os.path.dirname(reg), 'DejaVuSans-Bold.ttf')
        if not os.path.exists(bold):
            bold = reg
        return (ImageFont.truetype(reg, 14), ImageFont.truetype(bold, 18),
                ImageFont.truetype(reg, 12))
    except Exception:
        f = ImageFont.load_default()
        return f, f, f


def _grab(renderer, data, view, lookat_xy):
    """取一帧。必须先 update_scene 再改相机——update_scene 每次都会重算相机位姿。"""
    renderer.update_scene(data)
    VIEW_SETTERS[view](renderer, lookat_xy=lookat_xy)
    return renderer.render().copy()


def capture(cfg, scene, seed, max_steps, snapshot_steps, views, lookat_xy, w, h,
            arm_key, debug=False):
    """跑一集（动作恒 0），在指定决策步上取快照。

    返回 (frames, meta)：frames[(view, step)] = HxWx3；step 是**实际**发生的决策步。
    若某集提前结束（抓取成功/掉桌/困于奇异），后面够不着的快照点复用最后一帧并标 reused。
    """
    import mujoco
    env, _agent = build_env(cfg, scene, arm_key)
    renderer = mujoco.Renderer(env.model, h, w)
    try:
        obs, _ = env.reset(seed=seed)
        if env.nominal_trajectory is not None:
            env.nominal_trajectory.reset()
        # 「隐藏」障碍只到物理层：`_set_obstacle_active` 把未激活的球停到 obstacle_hidden_pos(1,1,1)
        # 并置 contype=0（不参与碰撞），**但渲染器照画**（contype 与渲染无关）。于是"无障碍"的
        # static/dyn_target 场景里会漂着一个球、看图的人会以为它是场景的一部分。
        # 这里把未激活障碍的 alpha 置 0——只改渲染、不动物理，与文字口径（"no obstacle"）一致。
        # ⚠️ 必须在 reset() 之后：`_active_obstacle_body_ids` 是 reset 里才填的（构造时为空，
        #    放前面会把本该激活的球一起抹掉）。
        # ⚠️ 与感知无关：眼在手外相机在 (0.1,0.42,1.2) 俯拍，停放点在光轴外约 79°（半视场约 30°），
        #    本来就不进深度图。这一条纯粹是让出图不误导。
        _act = set(getattr(env, '_active_obstacle_body_ids', []))
        for _g in getattr(env, 'obstacle_geom_ids_all', []):
            if int(env.model.geom_bodyid[_g]) not in _act:
                env.model.geom_rgba[_g, 3] = 0.0
        dt = env.substeps * max(1, getattr(env, 'action_repeat', 1)) * env.model.opt.timestep
        # reset 时刻的真值位置：画布局示意图要用**起始**摆位（终态的位置已经被运动带走了，
        # 尤其 mixed3/mixed_z 的随机游走障碍）。
        obs_xyz0 = [list(np.asarray(p, dtype=float).round(4)) for p in env._get_obstacle_positions()]
        obj_xy0 = np.asarray(getattr(env, 'target_pos', [0.0, 0.0, 0.0]),
                             dtype=float).round(4).tolist()

        targets = sorted(set(snapshot_steps))
        frames, actual = {}, {}
        last_step = 0
        last_frames = None
        need = set(targets)

        def take(step):
            nonlocal last_frames, last_step
            fr = {v: _grab(renderer, env.data, v, lookat_xy) for v in views}
            last_frames, last_step = fr, step
            for v in views:
                frames[(v, step)] = fr[v]
                actual[(v, step)] = (step, False)

        zero = np.zeros(env.action_space.shape[0], dtype=np.float32)
        take(0)
        term_step = max_steps          # 默认：loop 走满（只在 truncated 时到此）
        for step in range(1, max_steps + 1):
            obs, _r, term, trunc, info = env.step(zero)
            if step in need:
                take(step)
            if term or trunc:
                term_step = step
                if debug:
                    ts = getattr(env, 'task_state', {}) or {}
                    print("\n      [debug] step=%d term=%s trunc=%s | knocked_off=%s is_grasped=%s "
                          "lift_active=%s ep_steps=%s max_steps=%s | nominal=%s"
                          % (step, term, trunc, ts.get('knocked_off_table'),
                             ts.get('is_grasped'), ts.get('lift_active'),
                             ts.get('episode_steps'), getattr(env.grasping_config, 'max_steps', '?'),
                             env.nominal_trajectory is not None))
                break

        # 补齐够不着的快照点（复用末帧，如实标注）
        for t in targets:
            for v in views:
                if (v, t) not in frames:
                    frames[(v, t)] = last_frames[v]
                    actual[(v, t)] = (last_step, True)

        meta = dict(dt=dt, actual=actual, last_step=last_step, term_step=term_step,
                    ended_early=term_step < max(targets),
                    obs_pos0=obs_xyz0, obj_pos0=obj_xy0,
                    tgt=(float(getattr(env.grasping_config, 'target_vel_xy', 0.0)),
                         float(getattr(env.grasping_config, 'target_period', 6.0)),
                         str(getattr(env.grasping_config, 'target_motion_axis', 'x'))),
                    obs_pos=[p.round(4).tolist() for p in env._get_obstacle_positions()],
                    obj_pos=np.asarray(getattr(env, 'target_pos', [0, 0, 0])).round(4).tolist())
        return frames, meta
    finally:
        renderer.close()
        env.close()


def _panel_label(img, text, font):
    """面板左上角黑条 + 白字，标该帧的实际决策步（复用的帧带 * ）。"""
    im = Image.fromarray(img)
    d = ImageDraw.Draw(im)
    d.rectangle([0, 0, min(im.width, 8 + int(d.textlength(text, font=font)) + 6), 18],
                fill=(0, 0, 0))
    d.text((4, 2), text, fill=(255, 255, 255), font=font)
    return im


def compose_sheet(rows, cols, panels, panel_w, panel_h, out_path, footer_lines):
    """rows = [(scene, [描述行...])]；cols = [(view, step)]；panels[(scene, view, step)] = PIL Image"""
    f_reg, f_bold, f_small = _load_fonts()
    LM, HD, FT = 360, 26, 22 * len(footer_lines) + 12
    W = LM + len(cols) * panel_w
    H = HD + len(rows) * panel_h + FT
    sheet = Image.new('RGB', (W, H), (255, 255, 255))
    d = ImageDraw.Draw(sheet)

    for j, (view, step) in enumerate(cols):
        hdr = ("layout (top view), reset truth" if view == 'layout'
               else "%s  @ step %d" % (view, step))
        d.text((LM + j * panel_w + 8, 6), hdr, fill=(0, 0, 0), font=f_reg)
    for i, (scene, desc) in enumerate(rows):
        y0 = HD + i * panel_h
        d.line([(0, y0), (W, y0)], fill=(170, 170, 170), width=1)
        d.text((8, y0 + 8), scene, fill=(0, 0, 0), font=f_bold)
        for k, line in enumerate(desc):
            d.text((8, y0 + 32 + k * 16), line, fill=(60, 60, 60), font=f_small)
        for j, (view, step) in enumerate(cols):
            sheet.paste(panels[(scene, view, step)], (LM + j * panel_w, y0))
    y0 = HD + len(rows) * panel_h
    d.line([(0, y0), (W, y0)], fill=(170, 170, 170), width=1)
    for k, line in enumerate(footer_lines):
        d.text((8, y0 + 6 + k * 22), line, fill=(90, 90, 90), font=f_small)
    sheet.save(out_path)
    return sheet.size


def _assert_no_concurrent_mujoco(allow):
    """本机并发跑 MuJoCo 会把墙钟指标污染到离谱（单集 9 s 曾被记成 18536 s）。
    本脚本自己也要开 MuJoCo + EGL 上下文，所以跑之前先看有没有评测在跑。"""
    if allow:
        return
    try:
        out = subprocess.run(['pgrep', '-af', 'mpc_plus_rl_eval.py'],
                             capture_output=True, text=True).stdout.strip()
    except Exception:
        return
    if out:
        raise SystemExit("[拒绝运行] 检测到正在跑的 MuJoCo 评测：\n" + out +
                         "\n本机不要并发跑 MuJoCo（见 memory: fact_machine_serialize_runs）。"
                         "\n确实要并发加 --allow-concurrent。")


def main():
    ap = argparse.ArgumentParser(description='按场景拍「场景设置」对照图')
    ap.add_argument('--scenes', nargs='+', default=list(SCENE_CFG), choices=list(SCENE_CFG))
    ap.add_argument('--arm', default='hold', choices=list(ARM_BY_MODE),
                    help='hold=臂停在起始位（默认；场景图要的是摆位，且每集必然跑满）；'
                         'nominal=标称层驱动机械臂去抓（部分场景会提前终止）')
    ap.add_argument('--debug-info', action='store_true',
                    help='提前终止时打印 task_state 的判据，用于定位"为什么这一集没跑满"')
    ap.add_argument('--seed', type=int, default=12345, help='重置种子（障碍摆放由它决定）')
    ap.add_argument('--max-steps', type=int, default=200)
    ap.add_argument('--snapshot-steps', nargs='+', type=int, default=[0, 80, 160],
                    help='要在哪些决策步取快照（默认 0/80/160 ≈ 0.0/3.2/6.4 s）')
    ap.add_argument('--views', nargs='+', default=['front'], choices=list(VIEW_SETTERS),
                    help='除 oblique 外还要哪些 EGL 视角（oblique 总在快照步上出现；'
                         '第一列固定为几何俯视示意图，不受此项影响）')
    ap.add_argument('--width', type=int, default=440)
    ap.add_argument('--height', type=int, default=330)
    ap.add_argument('--outdir', default='results/scene_setup')
    ap.add_argument('--allow-concurrent', action='store_true')
    args = ap.parse_args()

    _assert_no_concurrent_mujoco(args.allow_concurrent)

    import mujoco  # noqa: F401  确保 MUJOCO_GL 生效

    cfg = get_config()
    cfg.grasping.max_steps = args.max_steps
    lookat_xy = tuple(cfg.grasping.object_fixed_pos)
    os.makedirs(args.outdir, exist_ok=True)

    snap = sorted(args.snapshot_steps)
    cols = [('layout', 0)]          # 第一列固定是俯视布局示意图（不占 EGL 渲染）
    for v in args.views:
        cols.append((v, snap[0]))
    for s in snap:
        cols.append(('oblique', s))

    rows, panels, metas = [], {}, {}
    arm_key = ARM_BY_MODE[args.arm]
    for scene in args.scenes:
        print("[拍] %-14s ..." % scene, end='', flush=True)
        frames, meta = capture(cfg, scene, args.seed, args.max_steps, snap,
                               list(VIEW_SETTERS), lookat_xy, args.width, args.height,
                               arm_key, debug=args.debug_info)
        metas[scene] = meta
        rows.append((scene, scene_lines(scene, meta['obs_pos0'], meta['tgt'])))
        f_reg, _fb, f_small = _load_fonts()
        for (view, step) in cols:
            if view == 'layout':
                # 示意图自带标题与图例，不再压黑条（黑条会盖住标题）
                p = Image.fromarray(layout_schematic(scene, cfg, meta['obj_pos0'],
                                                     meta['obs_pos0'],
                                                     args.width, args.height))
                panels[(scene, view, step)] = p
                p.save(os.path.join(args.outdir, "%s__layout.png" % scene))
                continue
            real, reused = meta['actual'][(view, step)]
            tag = "step %d  t=%.1fs%s" % (real, real * meta['dt'], "  *ended" if reused else "")
            p = _panel_label(frames[(view, step)], tag, f_small)
            panels[(scene, view, step)] = p
            # 文件名用**列**的步号（唯一），不用实际帧的步号——帧被复用时实际步号会重复，
            # 多个列会写同一个文件、互相覆盖（上一版 7×5=35 个面板只落了 31 个文件即此故）。
            p.save(os.path.join(args.outdir, "%s__%s_at%03d%s.png"
                                % (scene, view, step, "_reused" if reused else "")))
        print(" 障碍位 %s%s" % (meta['obs_pos'],
                               "  本集在 step %d 终止（快照点 %s 之外，已复用末帧）"
                               % (meta['term_step'], max(snap)) if meta['ended_early'] else ''))

    dt = metas[args.scenes[0]]['dt']
    arm_txt = ("arm HELD at its start pose (action_mode=delta, no nominal layer) - the moving obstacles/target are the subject; "
               "no policy is loaded and this figure documents scene LAYOUT, not policy performance."
               if args.arm == 'hold' else
               "arm driven by the NOMINAL layer only (zero action, no policy loaded) - documents scene layout, not policy performance.")
    footer = [
        arm_txt,
        "decision step dt = %.3f s (substeps x action_repeat x timestep); seed = %d; * = episode ended earlier, last frame reused."
        % (dt, args.seed),
        "column 1 = geometric top-view layout at reset (table / target / ACTIVE obstacles; dashed arrow = roundtrip range from SCENE_CFG); "
        "columns 2+ = EGL photos of that same episode.",
        "obstacle positions in the text/panel are RESET TRUTH, not SCENE_CFG's specs['pos']: with --obstacle-on-nominal-path the "
        "layout is auto-placed and overrides specs['pos'] (config.py:404); mixed3/mixed_z reset xy == static3's wall.",
    ]
    out = os.path.join(args.outdir, 'scene_setup_sheet.png')
    size = compose_sheet(rows, cols, panels, args.width, args.height, out, footer)
    print("总表 -> %s  %dx%d" % (out, size[0], size[1]))


if __name__ == '__main__':
    main()
