#!/usr/bin/env python3
"""
P1 短训练冒烟验证：验证整条 PPO 训练管线 + 指标监控（P1-4 验收）。

- 无头模式、固定物体位置、seed 42；
- 默认只跑 4096 步（4 个 rollout），验证：
  1. 训练不报错、PPO rollout 正常；
  2. TrainingMonitor 真正挂载：logs/training_log_*.json 出现非空 episodes（含 reward/length/success）；
  3. GraspingCallback 统计 episode 数并打印成功率。

运行（在 rl_grasping_system/ 下）：
    python scripts/smoke_train_p1.py [steps]
退出码 0 = 通过。
"""
import os
import sys
import json
import glob
import logging

HERE = os.path.dirname(os.path.abspath(__file__))
PKG = os.path.dirname(HERE)
sys.path.insert(0, PKG)

from agent import GraspingAgent  # noqa: E402
from environment import GraspingEnv  # noqa: E402
from config import get_config  # noqa: E402


def main():
    steps = int(sys.argv[1]) if len(sys.argv) > 1 else 4096

    cfg = get_config()
    cfg.grasping.render_gui = False
    cfg.grasping.use_fixed_position = True
    cfg.training.total_timesteps = steps
    cfg.system.seed = 42

    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        stream=sys.stdout,
    )

    env = GraspingEnv(cfg.grasping, cfg.reward)
    agent = GraspingAgent(cfg.network, cfg.training)
    agent.set_environment(env)
    agent.train(total_timesteps=steps)
    agent.training_monitor.print_final_summary()

    # 校验监控输出
    ok = True
    logs_dir = cfg.system.logs_dir
    jsons = sorted(glob.glob(os.path.join(logs_dir, 'training_log_*.json')))
    if not jsons:
        print('FAIL: 未找到训练日志 JSON')
        ok = False
    else:
        jf = jsons[-1]
        data = json.load(open(jf, encoding='utf-8'))
        n_ep = len(data.get('episodes', []))
        print(f'最新日志: {jf}')
        print(f'episodes 数量: {n_ep}')
        ok &= n_ep > 0
        if n_ep > 0:
            ep0 = data['episodes'][0]
            print(f'首条 episode 字段: {list(ep0.keys())}')
            print(f'  reward={ep0.get("reward")}  length={ep0.get("length")}  '
                  f'success={ep0.get("success")}')
            has_success = all('success' in e for e in data['episodes'])
            ok &= has_success

    print('SMOKE PASS' if ok else 'SMOKE FAIL')
    sys.exit(0 if ok else 1)


if __name__ == '__main__':
    main()
