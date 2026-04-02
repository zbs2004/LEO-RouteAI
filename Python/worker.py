# parallel_env/worker.py
import matlab.engine
import numpy as np
import torch
import time
import os
from collections import deque
from satellite_env import SatelliteRoutingEnv
from dqn_agent import DuelingDQN

def worker(rank, queue, num_episodes, code_path, num_domains, state_dim, action_dim,
           epsilon_start=1.0, epsilon_end=0.05, epsilon_decay=0.995,
           reward_config_path=None, model_path='dqn_model_latest.pth'): 
    N_STEP = 3
    GAMMA = 0.99

    eng = matlab.engine.start_matlab()
    eng.addpath(eng.genpath(code_path), nargout=0)
    eng.init_environment(nargout=0)
    env = SatelliteRoutingEnv(eng, num_satellites=648, num_domains=num_domains)
    if reward_config_path is not None:
        if os.path.exists(reward_config_path):
            try:
                env.load_reward_weights(reward_config_path)
                print(f'Worker {rank}: loaded reward weights from {reward_config_path}')
            except Exception as ex:
                print(f'Warning: failed to load reward weights from {reward_config_path}: {ex}')
        else:
            print(f'Warning: reward config path does not exist: {reward_config_path}')

    # 初始化本地网络，NoisyNet 用于探索
    net = DuelingDQN(state_dim, action_dim, use_noisy=True)
    net.eval()
    last_load_time = 0

    epsilon = epsilon_start
    for ep in range(num_episodes):
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

            if np.random.rand() < epsilon:
                action = np.random.choice(valid_actions)
            else:
                net.reset_noise()
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
        epsilon = max(epsilon_end, epsilon * epsilon_decay)
    eng.quit()