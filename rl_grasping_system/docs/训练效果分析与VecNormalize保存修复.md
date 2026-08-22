# 训练效果分析 + VecNormalize 保存/评估失配修复

> 记录 2026-08-22 对「并行改造后首次完整训练」的效果分析与发现的 bug。
> 状态：**✅ 修复已完成并通过往返测试**（2026-08-22，改动见 CHANGELOG.md 顶部条目）

---

## 一、首次并行训练的成绩单

训练配置：12 环境并行、3D 动作空间、spread 0.15、max_steps 200、200k 步。
数据来源：`logs/training_log_20260822_195451.json`（1104 episodes）、`logs/PPO_83`、`training.log`。

### ✅ 提速（巨大成功）
| 项 | 改造前 | 改造后 |
|---|---|---|
| 训练时长 | 40k 步 / 87 分钟 | **200k 步 / 9 分钟**（~370 步/s，47 倍） |
| 演示窗口 | — | ✅ 正常弹出 |

### ⚠️ 成功率 0%（但被 bug 污染，见第三节）
1104 episodes 全跑满 200 步，0 成功，r_grasp/r_success 从未非零。

### 📈 训练内部确实在学
| 指标 | 早期 | 后期 | 解读 |
|---|---|---|---|
| 平均奖励 | -17.4 | -14.1 | 缓慢上升 |
| r_dist_xy | 5.19 | 5.14（峰值 6.25） | XY 对齐在学，中段最好 |
| **r_contact 出现率** | 0% | **29%** | **策略学会了碰到物体** |
| r_dist_z | 3.5 | 3.9 | **Z 方向学习明显偏弱** |
| r_grasp / r_success | 0 | 0 | 闭合从未触发 |
| entropy / std | -4.25 / 0.99 | -4.09 / 0.94 | 策略仍未收敛自信 |

**结论**：策略学到「接近并接触物体」（接触率 29%），但从未达到「闭合触发」的严格对齐
（XY<4.5cm 且 Z<3.2cm）——卡在最后一步。Z 方向是主要弱项。

---

## 二、任务硬约束（实测）

初始位形（spread 0.15）下 pad 到 cube 距离：
```
pad中心-cube 距离: mean=0.257m  min=0.146m  max=0.356m
300 个初始位形中 pad 距 cube < 0.1m 的比例 = 0%
```
初始姿态良好（z_axis_z 平均 -0.99，99%+ 朝下）。
任务 = 从平均 25cm 外精确飞到 cube 正上方 5cm 内。物理行程够（200 步 × 0.005m = 1m），
但策略只学到「靠近+碰到」，没学到「精确对齐到 5cm」。

---

## 三、核心 Bug：VecNormalize 观测统计未被保存（评估/演示全面失配）

1. **`models/final_model.zip` 里没有 `vec_normalize.pkl`**。
   SB3 `save()` 的 `_excluded_save_params` 排除 `_vec_normalize_env`（源码实锤），
   必须手动 `vec_normalize.save()` 单独存。
2. **实证**：`PPO.load(path, env=vec_env)` 后 `obs_rms.mean 非零元素数 = 0/55`（初始值）。
3. **后果链**：训练时观测被正确归一化（自洽学习）→ 评估/演示时 stats=0/1 →
   观测分布失配 → 策略输出乱动作。
4. **对照实验**（同 seed 同环境）：
   - 随机策略：d0=0.286 → d_end=0.280（稳定）
   - final_model(deterministic)：d0=0.286 → **d_end=0.460**（把末端推远 = 失配乱走铁证）
5. **`_play_demo` 演示回调同样失配**：`self.model.predict(obs)` 传原始 obs 未归一化。

**推论**：0% 成功率不能全怪策略——评估/演示看到的都是「失配的乱走」；
训练真实学到的「接触率 29%」才是策略水平。

---

## 四、VecNormalize 修复方案（已实施 ✅）

1. `agent.py save`：`self.agent.save(path)` 后追加
   `self.agent.get_vec_normalize_env().save(path.replace('.zip', '_vecnormalize.pkl'))`。
2. `agent.py load`：加载时 `VecNormalize.load(pkl, vec_env)` 恢复统计，再 `PPO.load`。
3. `agent.predict` 统一做观测归一化（评估/部署入口自动适配）。
4. `_play_demo` 演示前归一化 obs（临时关 training 避免污染统计）。
5. 验证：save/load 往返测试 PASS（obs_rms 45/55 保存→恢复）；并行端到端 PASS。

---

## 五、补充：为什么"pad 到夹取位置了却不抓、停住"（方案 A 已实施）

### 现象与解释（实验实证）
用户观察：训练演示中 pad 已到 cube 上方"夹取位置"，夹爪张开、机械臂停住不抓。

**闭合触发要求 pad 中心与 cube 中心 Z 差 < 3.2cm**（= pad 距 cube 顶面 < 1.2cm），非常严格：
```
pad 正对中心+悬停4cm : Z差 3.96cm → ❌ 不触发（差 0.76cm）
pad 悬停顶上方2cm+XY偏2cm : Z差 3.96cm → ❌ 不触发   ← 用户看到的样子
pad 悬停2cm : Z差 0.95cm → ✅ 触发成功
```
**根本原因**：Z 主势能 slope = w_height/height_scale = 2.0/0.3 = 6.67/m，下压一步收益
~0.033 < step_penalty 0.1（净亏）→ 策略停在"pad 悬停 2-4cm"处无下压动力。

### 方案 A（已实施，config.py RewardConfig）
`w_height` 2.0→8.0、`anchor_w_z` 1.0→2.0、`anchor_dist_z` 0.04→0.06。
修复后 Z 差 15cm→1cm 全程下压一步收益 +0.17~+0.33 > 0.1（量化验证）。
`verify_grasp_success_criterion` PASS（无回归）。

### 若重训后仍卡
- XY 对齐不足 → 检查闭合触发（需 XY<4.5cm）
- 闭合触发 Z 阈值受 home 位形约束（pad 距 cube 顶 0.014m → Z差 0.034m，阈值 0.032 恰好避开），
  不能直接放宽；可考虑"进入 closing 后 pad 自动微降"（方案 C）。

