# 集成 menagerie Robotiq 2F-85 到 ur5e_robotiq.xml（替换手写简化版）
import re

ROOT = '/home/zrq/githubprojects/robotic_arm_control/models/universal_robots_ur5e'
U = open(ROOT + '/ur5e.xml').read()
R = open(ROOT + '/robotiq_2f85/2f85.xml').read()


def balanced_extract(text, start):
    """从 start 处提取平衡 XML 块（直到对应 </tag> 之后）。"""
    i = text.index(start)
    tag = re.search(r'<\s*([a-zA-Z]+)', start).group(1)
    depth = 0
    j = i
    while True:
        m = re.search(r'</?' + tag + r'\b', text[j:])
        if not m:
            raise RuntimeError('unbalanced ' + tag)
        tok = m.group(0)
        j += m.end()
        gt = text.index('>', j)      # 跳过当前标签的 '>'
        j = gt + 1
        depth += 1 if not tok.startswith('</') else -1
        if depth == 0:
            return text[i:j]


# 需要加 rq_ 前缀的 2F-85 标识符（按长度降序，避免子串误伤）
TOKENS = ['right_silicone_pad', 'left_silicone_pad',
          'right_spring_link', 'left_spring_link',
          'right_follower', 'left_follower',
          'right_driver', 'left_driver',
          'right_coupler', 'left_coupler',
          'right_pad', 'left_pad',
          'silicone_pad', 'spring_link',
          'pad_box1', 'pad_box2',
          'base_mount', 'follower', 'driver', 'coupler', 'pad',
          'metal', 'silicone', 'gray', 'black', 'visual', 'collision',
          '2f85', 'base', 'split']


def rename(s):
    """body/joint/mesh/class/material 名统一加 rq_ 前缀（防与 ur5e 冲突）。"""
    for t in TOKENS:
        # A: 词边界替换（name=/mesh=/file=/class=/childclass=/material=/body1= 等）
        s = re.sub(r'(?<![A-Za-z0-9_])' + t + r'(?![A-Za-z0-9_])', 'rq_' + t, s)
        # B: 关节名上下文（right_driver_joint 等）
        s = re.sub(r'(?<![A-Za-z0-9_])' + t + r'(?=_joint\b)', 'rq_' + t, s)
    return s


# ---- 从 2f85.xml 提取并改名各段 ----
asset2_raw = balanced_extract(R, '  <asset>').strip()
_am = re.match(r'<asset>(.*)</asset>', asset2_raw, re.S)
assert _am, 'asset inner'
asset2 = rename(_am.group(1).strip())
def2 = rename(balanced_extract(R, '    <default class="2f85">').strip())
body2 = rename(balanced_extract(R, '    <body name="base_mount"').strip())
contact2 = rename(balanced_extract(R, '  <contact>').strip())
tendon2 = rename(balanced_extract(R, '  <tendon>').strip())
equality2 = rename(balanced_extract(R, '  <equality>').strip())
act_finger_m = re.search(r'    <general class="2f85" name="fingers_actuator".*?/>', R, re.S)
assert act_finger_m, 'fingers_actuator not found'
act_finger = rename(act_finger_m.group(0))

# 给左右 pad 加测量 site（用于抓取宽度 = 两 site 距离）
pad_r = '<body name="rq_right_pad" pos="0 -0.0189 0.01352">'
pad_l = '<body name="rq_left_pad" pos="0 -0.0189 0.01352">'
assert pad_r in body2 and pad_l in body2, 'pad body tags not found'
body2 = body2.replace(pad_r, pad_r + '\n                <site name="rq_pad_right_site" pos="0 -0.0025 0.0185" size="0.002" rgba="1 0 0 0.6"/>')
body2 = body2.replace(pad_l, pad_l + '\n                <site name="rq_pad_left_site" pos="0 -0.0025 0.0185" size="0.002" rgba="0 1 0 0.6"/>')

# ---- 组装 ur5e_robotiq.xml ----
# 1) actuator：6 个 velocity + fingers_actuator
old_act = '''  <actuator>
    <general class="size3" name="shoulder_pan" joint="shoulder_pan_joint"/>
    <general class="size3" name="shoulder_lift" joint="shoulder_lift_joint"/>
    <general class="size3_limited" name="elbow" joint="elbow_joint"/>
    <general class="size1" name="wrist_1" joint="wrist_1_joint"/>
    <general class="size1" name="wrist_2" joint="wrist_2_joint"/>
    <general class="size1" name="wrist_3" joint="wrist_3_joint"/>
  </actuator>'''
new_act = '''  <!-- ===== 臂关节速度控制 =====
       方案（Task3 最终）：arm 关节用 motor(力矩)，环境内实现"速度 PID + 重力补偿前馈"：
         tau = kp*(dq - qvel) + qfrc_bias（qfrc_bias = 重力+科氏+离心前馈）。
       纯 MuJoCo velocity-servo(biasprm=[0,0,-kp]) 在重力静载荷下会拉塌臂/力矩饱和，
       与真实 UR speedJ（内置 PID + 重力补偿）不一致；motor+PID 复刻真实行为。
       2F-85 手指用 menagerie fingers_actuator（ctrl 0-255 位置伺服, kp=100）。 -->
  <actuator>
    <motor name="shoulder_pan" joint="shoulder_pan_joint" ctrlrange="-150 150" forcerange="-150 150"/>
    <motor name="shoulder_lift" joint="shoulder_lift_joint" ctrlrange="-150 150" forcerange="-150 150"/>
    <motor name="elbow" joint="elbow_joint" ctrlrange="-150 150" forcerange="-150 150"/>
    <motor name="wrist_1" joint="wrist_1_joint" ctrlrange="-28 28" forcerange="-28 28"/>
    <motor name="wrist_2" joint="wrist_2_joint" ctrlrange="-28 28" forcerange="-28 28"/>
    <motor name="wrist_3" joint="wrist_3_joint" ctrlrange="-28 28" forcerange="-28 28"/>
    <general class="rq_2f85" name="fingers_actuator" tendon="rq_split" forcerange="-8 8" ctrlrange="0 255"
      gainprm="0.3137255 0 0" biasprm="0 -100 -10"/>
  </actuator>'''
assert old_act in U, 'actuator block not found'
U = U.replace(old_act, new_act)

# 2) asset：把 2f85 的 mesh + material 引用并入 ur5e asset
old_asset_close = '    <mesh file="wrist3.obj"/>\n  </asset>'
new_asset = '''    <mesh file="wrist3.obj"/>
''' + asset2 + '''
  </asset>'''
assert old_asset_close in U
U = U.replace(old_asset_close, new_asset)

# 3) default：把 rq_ 类并入 ur5e default
old_def_close = '''      <site size="0.001" rgba="0.5 0.5 0.5 0.3" group="4"/>
    </default>
  </default>'''
new_def = '''      <site size="0.001" rgba="0.5 0.5 0.5 0.3" group="4"/>
    </default>
''' + def2 + '''
  </default>'''
assert old_def_close in U
U = U.replace(old_def_close, new_def)

# 4) worldbody：attachment_site 后插入 2f85 body 树（挂载 quat 使指尖朝下）
anchor = '                  <site name="attachment_site" pos="0 0.1 0" quat="-1 1 0 0"/>'
assert anchor in U
# base_mount 根挂载：pos 对齐 attachment_site。quat=绕x -90° 使指尖(base_mount -z)从世界+y 转到 -z(朝下)；x 不变(开合水平)
grip_mount = body2.replace('name="rq_base_mount" pos="0 0 0.007"', 'name="rq_base_mount" pos="0 0.1 0" quat="1 -1 0 0"')
U = U.replace(anchor, anchor + '\n' + grip_mount)

# 5) 追加 contact / tendon / equality（放在 worldbody 之后、actuator 之前）
U = U.replace('  <actuator>', contact2 + '\n\n' + tendon2 + '\n\n' + equality2 + '\n\n  <actuator>', 1)

# 6) model 名
U = U.replace('<mujoco model="ur5e">', '<mujoco model="ur5e_robotiq">')

out = ROOT + '/ur5e_robotiq.xml'
open(out, 'w').write(U)

# 8 个 2F-85 STL 复制到 assets/ 并重命名 rq_*（幂等）
import shutil, os
for f in ['base', 'base_mount', 'coupler', 'driver', 'follower', 'pad', 'silicone_pad', 'spring_link']:
    src = f'{ROOT}/robotiq_2f85/assets/{f}.stl'
    dst = f'{ROOT}/assets/rq_{f}.stl'
    if not os.path.exists(dst):
        shutil.copy(src, dst)
print('written', out, len(U), 'bytes; stl copied')
print('asset2', asset2[:80].replace('\n', ' '))

