# 从 ur5e_robotiq.xml 生成 ur5e_robotiq_cube.xml：加桌面 + target_cube(freejoint)
ROOT = '/home/zrq/githubprojects/robotic_arm_control/models/universal_robots_ur5e'
U = open(ROOT + '/ur5e_robotiq.xml').read()

# 位置设计：home 位形 pinch world (-0.134, 0.492, 0.339)，pad 全开 width 0.093 沿 x
# 桌面顶面 z=0.30；cube 中心 z=0.32（半边长 0.02）；cube (x,y)=pinch 水平位置
# 2026-08-28 桌面扩大：半边长 0.24→0.35（0.48m→0.70m 见方，更接近真实实验台），
#   中心 Y 0.42→0.53（保持桌面 Y 下界 = 0.18 ≥ 历史安全值 0.14，不碰下臂）。
#   workspace_bounds 同步扩大（config.py），IK 可达性打点见 docs/2026-08-28_深度相机分工与桌面扩大眼在手外.md。
TABLE_TOP_Z = 0.30
TH = 0.05                 # 桌厚
CX, CY = 0.1, 0.53        # 桌面中心水平位置（cube 固定位仍 0.1,0.42，见 object_fixed_pos；中心后移保持 y 下界 0.18）
T_HALF = 0.35             # 桌面半边长（0.45 会伸到上臂活动区 y<0.14 导致 upper_arm-table 接触；0.35 中心 0.53 → 下界 0.18 安全）

scene = '''
  <!-- ===== 桌面(固定) + 目标立方体（Task3: UR5e+2F-85 抓取场景） =====
       桌面为固定 body（不可加 freejoint，否则会被接触力推移导致 cube 穿桌）
       2026-08-28 桌面扩大：半边长 0.24→0.35、中心 Y 0.42→0.53（下界保持 0.18 不碰下臂）。
       workspace_bounds 同步扩大，IK 打点见 docs/2026-08-28_深度相机分工与桌面扩大眼在手外.md -->
  <body name="table" pos="%(CX)f %(CY)f %(TZ)f">
    <geom type="box" size="%(T_HALF)f %(T_HALF)f %(TH2)f" rgba="0.55 0.55 0.55 1"
      friction="0.8 0.5 0.02" solimp="0.99 0.995 0.001" solref="0.01 1"/>
  </body>
  <body name="target_cube" pos="0.100000 0.420000 %(CZ)f">
    <geom type="box" size="0.02 0.02 0.02" rgba="1 0.35 0.1 1"
      friction="0.6 0.4 0.02" mass="0.1" solimp="0.99 0.995 0.001" solref="0.01 1"/>
    <freejoint/>
  </body>

  <!-- ===== 眼在手外深度相机（2026-08-28，eye-to-hand） =====
       相机固定在工作区正上方、垂直向下看抓取工作区中心 (0.1, 0.42, 0.30)，
       不随机械臂运动（区别于 eye-in-hand）。mode="track"：位置 pos 固定（世界系），
       朝向自动指向 target=cam_target（不可见靶标 body）——MuJoCo 3.10 已验证。
       对应部署侧 RealSense D435i 支架布局；训练端用 mujoco.Renderer 渲染等价深度图，
       两端感知特征同构（决策记录 docs/2026-08-28_深度相机分工与桌面扩大眼在手外.md）。 -->
  <body name="cam_target" pos="0.100000 0.420000 0.300000">
    <geom type="sphere" size="0.001" rgba="0 0 0 0" contype="0" conaffinity="0"/>
  </body>
  <camera name="eye_to_hand" mode="track" target="cam_target"
          pos="0.100000 0.420000 1.200000" fovy="60"/>

  <!-- ===== 动态障碍物（D1 地基：freejoint 可编程运动，step 内 qvel 赋值） =====
       一周冲刺方案 §4.1：障碍物 body 加 freejoint + step 内 qvel 赋值——
       L1 隐藏（环境把 qpos 置 obstacle_hidden_pos + 运行时改 contype=0 禁用碰撞）；
       L2/L3 激活（环境置 obstacle_fixed_pos / 必经之路，contype=1 参与碰撞，可 qvel 驱动匀速运动）。
       ⚠️ contype 必须在 XML 默认为 1：MuJoCo 编译期 contype=0 的 geom 不进碰撞检测树，
       运行时再改 1 也无效（D2 实测）；XML=1 后运行时可 0/1 切换隐藏/激活。
       开关切换只改 config，不返工。 -->
  <body name="obstacle" pos="1.000000 1.000000 1.000000">
    <geom type="sphere" size="0.05" rgba="0.1 0.6 0.9 1"
      contype="1" conaffinity="1" friction="0.5 0.3 0.01" mass="0.5"
      solimp="0.99 0.995 0.001" solref="0.01 1"/>
    <freejoint/>
  </body>

  <!-- ===== 附加静态障碍（v12 多障碍：obstacle_2 / obstacle_3） =====
       与 obstacle 同构（sphere r=0.05 + freejoint + contype=1）；
       环境按 obstacle_count 激活前 N 个，激活时置必经之路不同 fraction/lateral，
       未激活（隐藏/超出 count）置远处 obstacle_hidden_pos + contype=0 不参与碰撞。
       观测只保留"最近激活障碍"的 rel/vel 槽位（67 维不变，兼容旧模型 warm-start）。 -->
  <body name="obstacle_2" pos="1.000000 1.000000 1.000000">
    <geom type="sphere" size="0.05" rgba="0.9 0.6 0.1 1"
      contype="1" conaffinity="1" friction="0.5 0.3 0.01" mass="0.5"
      solimp="0.99 0.995 0.001" solref="0.01 1"/>
    <freejoint/>
  </body>
  <body name="obstacle_3" pos="1.000000 1.000000 1.000000">
    <geom type="sphere" size="0.05" rgba="0.6 0.1 0.9 1"
      contype="1" conaffinity="1" friction="0.5 0.3 0.01" mass="0.5"
      solimp="0.99 0.995 0.001" solref="0.01 1"/>
    <freejoint/>
  </body>

  <!-- ===== P2c 更多障碍（2026-09-07）：obstacle_4 / obstacle_5 / obstacle_6 =====
       与 obstacle 同构（sphere r=0.05 + freejoint + contype=1），per-obstacle 规格 obstacle_specs
       可静态/动态混合（type/vel/mode/axis/z_motion），按 specs 数量激活前 N 个；
       未激活置远处 + contype=0 不参与碰撞。观测仍只给"最近激活障碍"槽位（67 维不变），
       MPC 标称（nominal_mode='mpc'）用全部激活障碍做避障软约束。 -->
  <body name="obstacle_4" pos="1.000000 1.000000 1.000000">
    <geom type="sphere" size="0.05" rgba="0.1 0.9 0.9 1"
      contype="1" conaffinity="1" friction="0.5 0.3 0.01" mass="0.5"
      solimp="0.99 0.995 0.001" solref="0.01 1"/>
    <freejoint/>
  </body>
  <body name="obstacle_5" pos="1.000000 1.000000 1.000000">
    <geom type="sphere" size="0.05" rgba="0.9 0.1 0.1 1"
      contype="1" conaffinity="1" friction="0.5 0.3 0.01" mass="0.5"
      solimp="0.99 0.995 0.001" solref="0.01 1"/>
    <freejoint/>
  </body>
  <body name="obstacle_6" pos="1.000000 1.000000 1.000000">
    <geom type="sphere" size="0.05" rgba="0.9 0.1 0.9 1"
      contype="1" conaffinity="1" friction="0.5 0.3 0.01" mass="0.5"
      solimp="0.99 0.995 0.001" solref="0.01 1"/>
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
