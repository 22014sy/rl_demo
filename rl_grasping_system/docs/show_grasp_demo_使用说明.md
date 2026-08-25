# show_grasp_demo.py 使用说明（2026-08-25 整理）

> 离屏渲染展示当前模型抓取效果并保存 GIF 的工具。
> 本文档对应 `scripts/show_grasp_demo.py` 当前版本（含 D2/D3 训练口径参数、`--track` 跟随末端视角、`--dual` 双视角拼接）。

---

## 一、工具定位

- **功能**：加载指定 RL 模型，用 **deterministic 策略** 跑 episode，离屏渲染成功抓取过程并保存为 GIF。
- **无头运行**：`MUJOCO_GL=egl`（不弹 GLFW 窗口，可在服务器/SSH 环境跑）。
- **口径**：与训练/评估同口径——`use_fixed_position=False` + `workspace_bounds=±radius`（物体位置在工作区 ±radius 内随机）。

## 二、运行环境与前置条件

```bash
cd rl_grasping_system          # 必须在仓库根目录跑（脚本按相对路径引 config/evaluate）
```

- 模型文件：默认 `models/final_model_stage2.zip`，用 `--model` 指定具体权重。
- VecNormalize **无需手动处理**：`load_model` 自动恢复 `<model>_vecnormalize.pkl`，`predict` 内部自动归一化。

## 三、⚠️ 最重要的：训练口径一致性

**背景**：D2/D3 避障抓取模型（v4/v5，如 `final_model_stage2_d2_v5.zip`）是
**residual 动作（标称轨迹 + 残差叠加）+ 静态障碍放"标称必经之路"** 训练的。
而 `config.py` 默认 `action_mode='delta'`、无障碍。

> 若直接跑默认配置展示 v5，观测/动作语义与训练**失配**，展示出来的是"骗人"的假效果（绕障、残差动作全都不体现）。

因此 **v5 模型必须带齐下面 4 个参数**：

```bash
--action-mode residual --obstacle-on-path --obstacle-w 0.35 --residual-reg-w 0.3
```

对应覆盖的配置项：

| 参数 | 覆盖的 config |
|---|---|
| `--action-mode residual` | `grasping.action_mode` |
| `--obstacle-on-path` | `grasping.obstacle_enabled=True` + `grasping.obstacle_on_nominal_path=True` |
| `--obstacle-w 0.35` | `reward.obstacle_w`（v4/v5 训练值） |
| `--residual-reg-w 0.3` | `reward.residual_reg_w`（v4/v5 训练值） |

> `--obstacle-w` / `--residual-reg-w` 仅影响 **reward 数值口径**，不影响物理/策略。
> 成功 episode 的 reward ≈ **104**（`r_grasp +100` + `r_obstacle ≈ -16`），与训练成功 episode 口径一致，可用于核对配置是否对齐。

## 四、CLI 参数表

### 视角类

| 参数 | 说明 |
|---|---|
| `--topdown` | 单视角俯拍（桌面正上方垂直向下，展示布局） |
| `--track` | 单视角**跟随末端**（相机锁定夹爪特写，机械臂始终居中，突出绕障轨迹） |
| `--dual` | 双视角同屏拼接：**左(低平) + 右(俯拍)**，一张 GIF 同时看动作与布局 |
| `--dual --track` | 双视角拼接：**左(跟随末端特写) + 右(俯拍)** |

> 不传任何视角参数 = 默认 **低平 oblique**（接近桌面持平的斜视，俯角约 6°）。

### D2/D3 训练口径类（v4/v5 必须带）

| 参数 | 类型 | 说明 |
|---|---|---|
| `--action-mode` | `{delta,residual}` | 动作模式（默认空=沿用 config，即 delta） |
| `--obstacle-on-path` | flag | 静态障碍放标称必经之路 |
| `--obstacle-w` | float | 障碍接近惩罚权重（`<0` 用 config 默认；v5 用 0.35） |
| `--residual-reg-w` | float | 残差幅度正则权重（`<0` 用 config 默认；v5 用 0.3） |

### 采集/渲染类

| 参数 | 默认 | 说明 |
|---|---|---|
| `--model` | `models/final_model_stage2.zip` | 模型路径 |
| `--radius` | `0.06` | 物体位置随机范围（v5 演示建议 `0.03`，与训练一致） |
| `--n-success` | `2` | 要采集的成功案例数 |
| `--max-episodes` | `60` | 最多尝试的 episode 数 |
| `--frame-step` | `2` | 每 N 决策步采一帧 |
| `--width` / `--height` | `640` / `480` | 单视角分辨率 |
| `--outdir` | `results/demo` | GIF 输出目录 |

## 五、视角选项详解

| 视角 | 相机行为 | 适合展示 |
|---|---|---|
| 默认（低平 oblique） | 固定相机，接近桌面持平（俯角 ~6°），从 y 负侧看 | 桌面上的高度差、下降抓取 |
| `--topdown` | 桌面正上方垂直俯拍 | 桌面布局、障碍与物体相对位置 |
| `--track` | **跟随末端**：相机始终对准夹爪，固定后上方方向，距离 0.45 | 绕障时末端偏离标称路径的弧线（特写） |
| `--dual` | 左低平 + 右俯拍 水平拼接（1280×480） | 动作 + 布局两不误，推荐 |
| `--dual --track` | 左跟随末端特写 + 右俯拍（1280×480） | 绕障特写 + 布局，最完整 |

**可调参数**（改脚本里的函数默认值）：
- `_set_oblique_camera(..., elevation=0.15)`：`0`=完全平视；`0.5`≈16.8°；`1.0`≈35°
- `_set_track_camera(..., distance=0.45)`：特写距离，越大越远
- `_set_track_camera` 的 direction 决定相机方位（当前 `(-1,-1,0.8)` 末端后上方）

## 六、命令示例（完整可复制）

```bash
cd rl_grasping_system

# ① 默认低平视角（D2/D3 v5 训练口径，必带 4 个口径参数）
MUJOCO_GL=egl python3 scripts/show_grasp_demo.py --model models/final_model_stage2_d2_v5.zip \
    --radius 0.03 --action-mode residual --obstacle-on-path --obstacle-w 0.35 --residual-reg-w 0.3

# ② 俯拍布局
MUJOCO_GL=egl python3 scripts/show_grasp_demo.py --model models/final_model_stage2_d2_v5.zip \
    --radius 0.03 --action-mode residual --obstacle-on-path --obstacle-w 0.35 --residual-reg-w 0.3 \
    --topdown

# ③ 跟随末端特写（绕障弧线）
MUJOCO_GL=egl python3 scripts/show_grasp_demo.py --model models/final_model_stage2_d2_v5.zip \
    --radius 0.03 --action-mode residual --obstacle-on-path --obstacle-w 0.35 --residual-reg-w 0.3 \
    --track

# ④ 双视角拼接（推荐）：左低平 + 右俯拍
MUJOCO_GL=egl python3 scripts/show_grasp_demo.py --model models/final_model_stage2_d2_v5.zip \
    --radius 0.03 --action-mode residual --obstacle-on-path --obstacle-w 0.35 --residual-reg-w 0.3 \
    --dual

# ⑤ 双视角最完整：左跟随末端特写 + 右俯拍
MUJOCO_GL=egl python3 scripts/show_grasp_demo.py --model models/final_model_stage2_d2_v5.zip \
    --radius 0.03 --action-mode residual --obstacle-on-path --obstacle-w 0.35 --residual-reg-w 0.3 \
    --dual --track
```

> 早期 D1 模型（如 `final_model_stage2_d1.zip`）仍是 delta 语义，**不要**带 `--action-mode residual` 等 D2 口径参数，直接用默认即可。

## 七、产物

输出到 `results/demo/`：

| 产物 | 命名 | 尺寸 |
|---|---|---|
| 单视角 | `{model名}_success{N}.gif` | 640×480 |
| 双视角 | `{model名}_dual{N}.gif` | 1280×480 |

示例（v5 成功案例，reward ≈104）：
```
results/demo/final_model_stage2_d2_v5_success1.gif
results/demo/final_model_stage2_d2_v5_dual1.gif
```

## 八、注意事项 / 已知坑

1. **演示有随机性**：物体位置每 episode 随机，可能先跑几个失败才采到成功（`--max-episodes` 默认 60，足够；失败 episode 步数上限 500）。
2. **reward 数值口径**：`--obstacle-w 0.35` / `--residual-reg-w 0.3` 只影响显示的 reward 数值，对策略执行无影响；改错了只会让 reward 与训练不对齐。
3. **并发编辑风险**：本脚本被手动编辑时**容易覆盖已注册的 CLI 参数**（曾发生两次：D2/D3 参数、oblique 相机被旧缓冲覆盖）。改完建议核对：
   ```bash
   grep -c 'action-mode' scripts/show_grasp_demo.py   # 当前版本应 = 5（docstring 4 行示例 + 1 个 add_argument）
   python3 scripts/show_grasp_demo.py --help          # 确认 4 个 D2/D3 参数都在
   ```
4. **双视角耗时**：每帧渲染两次，demo 耗时约为单视角 2 倍。
5. **CUDA/无头**：若 GPU 不可用或卡在渲染，确认 `MUJOCO_GL=egl` 已设置；GIF 生成需要 `PIL`。
