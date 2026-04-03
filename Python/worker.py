# parallel_env/worker.py
import matlab.engine
import numpy as np
import torch
import time
import os
import json
from collections import deque
from satellite_env import SatelliteRoutingEnv
from dqn_agent import DuelingDQN

def worker(rank, queue, num_episodes, code_path, num_domains, state_dim, action_dim,
           shared_epsilon, reward_config_path=None, model_path='dqn_model_latest.pth'):
    N_STEP = 3
    GAMMA = 0.99

    original_path = os.environ.get('PATH', '')
    eng = None
    try:
        eng = matlab.engine.start_matlab()
        eng.addpath(eng.genpath(code_path), nargout=0)
        eng.init_environment(nargout=0)
        # 尝试从项目输出目录读取仿真配置，以对齐 MATLAB 与 Python 的队列/延迟参数
        sim_cfg = {}
        sim_config_path = os.path.join(code_path, 'simulation_results', 'config.json')
        if os.path.exists(sim_config_path):
            try:
                with open(sim_config_path, 'r', encoding='utf-8') as scf:
                    sim_cfg = json.load(scf)
            except Exception:
                sim_cfg = {}

        qmod = sim_cfg.get('queue_model', {}) if isinstance(sim_cfg, dict) else {}
        enable_q = qmod.get('enable', True)
        queue_capacity = qmod.get('capacity', 500)
        queue_service_rate = qmod.get('service_rate', 1.0)
        queue_delay_weight = qmod.get('delay_weight', 0.01)
        queue_drop_penalty = qmod.get('drop_penalty', 20.0)
        queue_time_per_step_ms = qmod.get('time_per_step_ms', 10.0)

        # 创建环境，优先使用仿真配置中的队列参数
        env = SatelliteRoutingEnv(eng, num_satellites=648, num_domains=num_domains,
                      enable_queue_model=enable_q, queue_capacity=queue_capacity, queue_service_rate=queue_service_rate,
                      queue_delay_weight=queue_delay_weight, queue_drop_penalty=queue_drop_penalty, queue_time_per_step_ms=queue_time_per_step_ms)
    finally:
        os.environ['PATH'] = original_path
    if reward_config_path is not None:
        if os.path.exists(reward_config_path):
            try:
                env.load_reward_weights(reward_config_path)
                print(f'Worker {rank}: loaded reward weights from {reward_config_path}')
            except Exception as ex:
                print(f'Warning: failed to load reward weights from {reward_config_path}: {ex}')
        else:
            print(f'Warning: reward config path does not exist: {reward_config_path}')
    else:
        # 如果没有提供专门的 reward config，尝试从 simulation_results/config.json 中提取 reward 或 reward_norm 设置
        sim_config_path = os.path.join(code_path, 'simulation_results', 'config.json')
        if os.path.exists(sim_config_path):
            try:
                with open(sim_config_path, 'r', encoding='utf-8') as scf:
                    sim_cfg = json.load(scf)
                # 提取 reward 权重（若以顶层字段存在）
                reward_keys = {k: sim_cfg[k] for k in sim_cfg if k in env.reward_weights}
                if reward_keys:
                    # 写入临时文件并通过已有逻辑加载
                    import tempfile
                    tf = tempfile.NamedTemporaryFile(mode='w', delete=False, suffix='.json', encoding='utf-8')
                    json.dump(reward_keys, tf, indent=2, ensure_ascii=False)
                    tf.flush(); tf.close()
                    try:
                        env.load_reward_weights(tf.name)
                        print(f'Worker {rank}: loaded reward keys from simulation_results/config.json')
                    finally:
                        try:
                            os.unlink(tf.name)
                        except Exception:
                            pass

                # reward 归一化参数位于 sim_cfg['reward_norm']（如果存在），应用到 env
                rnorm = sim_cfg.get('reward_norm', {}) if isinstance(sim_cfg, dict) else {}
                if isinstance(rnorm, dict):
                    if 'min_count' in rnorm:
                        try:
                            env.reward_norm_min_count = int(rnorm['min_count'])
                        except Exception:
                            pass
                    if 'ema_alpha' in rnorm:
                        try:
                            env.reward_ema_alpha = float(rnorm['ema_alpha'])
                        except Exception:
                            pass
                    if 'clip' in rnorm:
                        try:
                            env.reward_norm_clip = float(rnorm['clip'])
                        except Exception:
                            pass
            except Exception:
                pass

    # 快速验证阶段性趋势（仅用于调试）
    env.eng.workspace['reward_fast_phase'] = 0

    # 初始化本地网络，NoisyNet 用于探索
    net = DuelingDQN(state_dim, action_dim, use_noisy=True)
    net.eval()
    last_load_time = 0

    for ep in range(num_episodes):
        env.eng.workspace['training_episode'] = ep
        state, _ = env.reset()
        done = False
        steps = 0
        episode_reward = 0.0
        episode_steps = 0
        transition_buffer = deque()

        # 每个 episode 开始时，加载最新模型（确保使用最新策略）
        if os.path.exists(model_path):
            try:
                net.load_state_dict(torch.load(model_path, map_location='cpu'))
            except:
                pass

        while not done and steps < 200:
            action_mask, invalid = env.get_action_mask()
            if invalid:
                break
            valid_actions = [i for i, valid in enumerate(action_mask) if valid]
            if len(valid_actions) == 0:
                break

            # 若使用 NoisyNet，则用参数化噪声作为探索，不使用 epsilon-greedy
            if getattr(net, 'use_noisy', False):
                net.reset_noise()
                with torch.no_grad():
                    state_t = torch.FloatTensor(state).unsqueeze(0)
                    q = net(state_t).squeeze(0)
                    mask_tensor = torch.tensor(action_mask, dtype=torch.bool)
                    q = q.masked_fill(~mask_tensor, -1e9)
                    action = q.argmax().item()
            else:
                eps_val = shared_epsilon.value if shared_epsilon is not None else 1.0
                if np.random.rand() < eps_val:
                    action = np.random.choice(valid_actions)
                else:
                    with torch.no_grad():
                        state_t = torch.FloatTensor(state).unsqueeze(0)
                        q = net(state_t).squeeze(0)
                        mask_tensor = torch.tensor(action_mask, dtype=torch.bool)
                        q = q.masked_fill(~mask_tensor, -1e9)
                        action = q.argmax().item()

            next_state, reward, done, _, _ = env.step(int(action))
            episode_reward += reward
            episode_steps += 1
            transition_buffer.append((state, action, reward, next_state, done))

            if len(transition_buffer) >= N_STEP:
                total_reward = 0.0
                for idx in range(N_STEP):
                    total_reward += (GAMMA ** idx) * transition_buffer[idx][2]
                n_state = transition_buffer[N_STEP - 1][3]
                n_done = transition_buffer[N_STEP - 1][4]
                queue.put({
                    'type': 'transition',
                    'state': transition_buffer[0][0],
                    'action': transition_buffer[0][1],
                    'reward': total_reward,
                    'next_state': n_state,
                    'done': n_done
                })
                transition_buffer.popleft()

            state = next_state
            steps += 1

        # 期末刷新剩余的多步缓存，构建短序列 return
        while len(transition_buffer) > 0:
            n = len(transition_buffer)
            total_reward = 0.0
            for idx in range(n):
                total_reward += (GAMMA ** idx) * transition_buffer[idx][2]
            n_state = transition_buffer[-1][3]
            n_done = transition_buffer[-1][4]
            queue.put({
                'type': 'transition',
                'state': transition_buffer[0][0],
                'action': transition_buffer[0][1],
                'reward': total_reward,
                'next_state': n_state,
                'done': n_done
            })
            transition_buffer.popleft()

        queue.put({
            'type': 'episode_end',
            'episode_reward': episode_reward,
            'episode_steps': episode_steps,
            'episode_success': bool(env.episode_success),
            'episode_delay': float(getattr(env, 'episode_total_delay', 0.0))
        })

    eng.quit()
