# 从 ur5e_robotiq.xml 生成 ur5e_robotiq_cube.xml：加桌面 + target_cube(freejoint)
ROOT = '/home/zrq/githubprojects/robotic_arm_control/models/universal_robots_ur5e'
U = open(ROOT + '/ur5e_robotiq.xml').read()

# 位置设计：home 位形 pinch world (-0.134, 0.492, 0.339)，pad 全开 width 0.093 沿 x
# 桌面顶面 z=0.30；cube 中心 z=0.32（半边长 0.02）；cube (x,y)=pinch 水平位置
TABLE_TOP_Z = 0.30
TH = 0.05                 # 桌厚
CX, CY = 0.1, 0.42        # cube 水平位置（换位后；桌面中心同步，保持桌面 y 下界>0.14 不碰下臂）
T_HALF = 0.24             # 桌面半边长（0.45 会伸到上臂活动区 y<0.14 导致 upper_arm-table 接触，缩至 0.24）

scene = '''
  <!-- ===== 桌面(固定) + 目标立方体（Task3: UR5e+2F-85 抓取场景） =====
       桌面为固定 body（不可加 freejoint，否则会被接触力推移导致 cube 穿桌） -->
  <body name="table" pos="%(CX)f %(CY)f %(TZ)f">
    <geom type="box" size="%(T_HALF)f %(T_HALF)f %(TH2)f" rgba="0.55 0.55 0.55 1"
      friction="0.8 0.5 0.02" solimp="0.99 0.995 0.001" solref="0.01 1"/>
  </body>
  <body name="target_cube" pos="%(CX)f %(CY)f %(CZ)f">
    <geom type="box" size="0.02 0.02 0.02" rgba="1 0.35 0.1 1"
      friction="0.6 0.4 0.02" mass="0.1" solimp="0.99 0.995 0.001" solref="0.01 1"/>
    <freejoint/>
  </body>
''' % dict(CX=CX, CY=CY, TZ=TABLE_TOP_Z - TH / 2.0, TH2=TH / 2.0,
           T_HALF=T_HALF, CZ=TABLE_TOP_Z + 0.02)

anchor = '  </worldbody>'
assert anchor in U
U = U.replace(anchor, scene + '\n' + anchor, 1)
U = U.replace('<mujoco model="ur5e_robotiq">', '<mujoco model="ur5e_robotiq_cube">')

out = ROOT + '/ur5e_robotiq_cube.xml'
open(out, 'w').write(U)
print('written', out, len(U), 'bytes')
