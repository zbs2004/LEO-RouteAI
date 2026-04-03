import argparse
import matlab.engine
from satellite_env import SatelliteRoutingEnv
from dqn_agent import Agent, NoisyLinear
import torch
import numpy as np
import csv
import json
import os
import tempfile
import subprocess
import sys

# 预先初始化 matplotlib，以避免 MATLAB 引擎启动后出现 pyexpat DLL 载入失败
_try_matplotlib = False
try:
    import matplotlib
    matplotlib.use('Agg')
    _try_matplotlib = True
except Exception as ex:
    print(f'Warning: 初始化 matplotlib 失败: {ex}')

import scipy.io as sio
import multiprocessing as mp
import time
from worker import worker
from plot_training import plot_training_results

def parse_args():
    parser = argparse.ArgumentParser(description='Train DQN for satellite routing with configurable reward weights.')
    parser.add_argument('--reward-config', type=str, default=None,
                        help='Optional reward weight JSON file to use for this training session.')
    parser.add_argument('--num-workers', type=int, default=8,
                        help='Number of parallel worker processes.')
    parser.add_argument('--epsilon-start', type=float, default=1.0,
                        help='Starting epsilon for epsilon-greedy exploration.')
    parser.add_argument('--epsilon-end', type=float, default=0.01,
                        help='Minimum epsilon for epsilon-greedy exploration.')
    parser.add_argument('--epsilon-decay', type=float, default=0.995,
                        help='Per-episode epsilon decay factor.')
    parser.add_argument('--learning-rate', type=float, default=0.0001,
                        help='Learning rate for the DQN optimizer.')
    parser.add_argument('--batch-size', type=int, default=64,
                        help='Batch size for DQN updates.')
    parser.add_argument('--target-update', type=int, default=20,
                        help='Number of learn steps between hard target network updates.')
    parser.add_argument('--num-episodes-per-worker', type=int, default=150,
                        help='Number of episodes each worker collects.')
    parser.add_argument('--log-path', type=str, default='training_logs.csv',
                        help='CSV path to append training episode logs.')
    parser.add_argument('--model-path', type=str, default='dqn_model_latest.pth',
                        help='Path to save the trained PyTorch model.')
    parser.add_argument('--metadata-path', type=str, default='training_metadata.json',
                        help='Path to save training metadata.')
    parser.add_argument('--plot-interval', type=int, default=20,
                        help='Plot training metrics every N episodes')
    parser.add_argument('--no-plot', action='store_true',
                        help='Skip plotting results after training.')
    return parser.parse_args()


def safe_torch_save(state_dict, model_path, max_retries=5, retry_delay=0.5):
    model_dir = os.path.dirname(model_path)
    if model_dir:
        os.makedirs(model_dir, exist_ok=True)
    tmp_path = os.path.join(model_dir or '.', f'.{os.path.basename(model_path)}.tmp')
    for attempt in range(1, max_retries + 1):
        try:
            torch.save(state_dict, tmp_path)
            os.replace(tmp_path, model_path)
            return
        except Exception:
            if os.path.exists(tmp_path):
                try:
                    os.remove(tmp_path)
                except Exception:
                    pass
            if attempt == max_retries:
                raise
            time.sleep(retry_delay)


def make_absolute_path(path, root):
    if os.path.isabs(path):
        return path
    return os.path.abspath(os.path.join(root, path))


def train(args=None):
    if args is None:
        args = parse_args()

    # 主进程获取参数，确保 MATLAB 路径包含所有子目录
    project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
    eng = None
    original_path = os.environ.get('PATH', '')
    try:
        try:
            eng = matlab.engine.start_matlab('-nodesktop', '-nosplash')
        except Exception:
            eng = matlab.engine.start_matlab()
        if eng is None:
            raise RuntimeError('MATLAB 引擎启动失败：start_matlab() 返回 None。')
        eng.addpath(eng.genpath(project_root), nargout=0)
        eng.init_environment(nargout=0)
        num_domains = int(eng.workspace['params']['cross_domain_params']['num_domains'])
    except Exception as exc:
        raise RuntimeError('无法启动 MATLAB 引擎，请检查 MATLAB 安装、matlab.engine 配置以及 MATLAB 是否可用。') from exc
    finally:
        os.environ['PATH'] = original_path
        if eng is not None:
            try:
                eng.quit()
            except Exception:
                pass

    default_reward_weights = {
        'congestion': 0.25,
        'hop': 4.0,
        'delay': 6.0,
        'stability': 0.5,
        'balance': 0.25,
        'e2e': 2.5,
        'success': 1.0,
        'failure': 0.1,
        'step': 0.02,
        'progress': 0.05
    }
    if args.reward_config is None:
        candidate_paths = [
            os.path.join(project_root, 'best_reward_weights.json'),
            os.path.join(project_root, 'reward_weights.json')
        ]
        reward_config_path = None
        for path in candidate_paths:
            if os.path.exists(path):
                reward_config_path = path
                break
        if reward_config_path is None:
            reward_config_path = os.path.join(project_root, 'reward_weights.json')
            with open(reward_config_path, 'w', encoding='utf-8') as config_file:
                json.dump(default_reward_weights, config_file, indent=2, ensure_ascii=False)
            print(f'No reward config found. Created default: {reward_config_path}')
        else:
            print(f'Loading reward config: {reward_config_path}')
    else:
        reward_config_path = os.path.abspath(args.reward_config)
        if not os.path.exists(reward_config_path):
            raise FileNotFoundError(f'Reward config not found: {reward_config_path}')
        print(f'Loading reward config: {reward_config_path}')

    with open(reward_config_path, 'r', encoding='utf-8') as f:
        loaded_reward_weights = json.load(f)
    print(f'Loaded reward weights: {loaded_reward_weights}')

    # 尝试读取来自 MATLAB 的仿真配置（simulation_results/config.json），并将其中的 RL 参数对齐到训练设置
    sim_config_path = os.path.join(project_root, 'simulation_results', 'config.json')
    if os.path.exists(sim_config_path):
        try:
            with open(sim_config_path, 'r', encoding='utf-8') as sf:
                sim_cfg = json.load(sf)
            rl_cfg = sim_cfg.get('rl_params', {}) if isinstance(sim_cfg, dict) else {}
            if isinstance(rl_cfg, dict) and rl_cfg:
                if 'learning_rate' in rl_cfg:
                    try:
                        args.learning_rate = float(rl_cfg['learning_rate'])
                    except Exception:
                        pass
                if 'batch_size' in rl_cfg:
                    try:
                        args.batch_size = int(rl_cfg['batch_size'])
                    except Exception:
                        pass
                if 'target_update' in rl_cfg:
                    try:
                        args.target_update = int(rl_cfg['target_update'])
                    except Exception:
                        pass
                if 'num_episodes_per_worker' in rl_cfg:
                    try:
                        args.num_episodes_per_worker = int(rl_cfg['num_episodes_per_worker'])
                    except Exception:
                        pass
                print(f'Applied RL params from {sim_config_path}: {rl_cfg}')
        except Exception as ex:
            print(f'Warning: failed to load simulation config {sim_config_path}: {ex}')

    # 增加3个队列特征（mean,max,current domain）
    state_dim = 3 + num_domains * 2 + num_domains * num_domains + 13  # 新增2个负载特征 +3 队列特征
    action_dim = num_domains
    # Create agent
    agent = Agent(state_dim, action_dim,
                  lr=args.learning_rate, gamma=0.99,
                  epsilon_start=args.epsilon_start, epsilon_end=args.epsilon_end,
                  epsilon_decay=args.epsilon_decay, memory_size=100000, batch_size=args.batch_size,
                  target_update=args.target_update, n_step=3, use_noisy=True)

    # 使用 Manager 共享 epsilon，由主进程控制并下发给 worker
    manager = mp.Manager()
    shared_epsilon = manager.Value('d', float(agent.epsilon))

    model_path = make_absolute_path(args.model_path, project_root)
    model_dir = os.path.dirname(model_path)
    if model_dir:
        os.makedirs(model_dir, exist_ok=True)
    log_path = make_absolute_path(args.log_path, project_root)
    log_dir = os.path.dirname(log_path)
    if log_dir:
        os.makedirs(log_dir, exist_ok=True)
    metadata_path = make_absolute_path(args.metadata_path, project_root)
    metadata_dir = os.path.dirname(metadata_path)
    if metadata_dir:
        os.makedirs(metadata_dir, exist_ok=True)
    # 每次训练重新写入日志头，确保当前可视化只包含本次训练数据
    with open(log_path, 'w', newline='', encoding='utf-8') as log_file:
        writer = csv.writer(log_file)
        writer.writerow(['episode', 'episode_reward', 'episode_steps', 'episode_success', 'episode_delay', 'total_samples', 'epsilon', 'loss', 'timestamp'])

    queue = mp.Queue(maxsize=10000)
    num_workers = args.num_workers
    num_episodes_per_worker = args.num_episodes_per_worker   # 每个子进程采集的 episode 数
    workers = []
    epsilon_start = agent.epsilon
    epsilon_end = agent.epsilon_end
    epsilon_decay = agent.epsilon_decay
    for i in range(num_workers):
        p = mp.Process(target=worker, args=(i, queue, num_episodes_per_worker, project_root, num_domains, state_dim, action_dim, shared_epsilon, reward_config_path, model_path))
        p.start()
        workers.append(p)

    total_samples = 0
    episode_count = 0
    last_save_time = time.time()
    last_loss = None

    plot_interval = args.plot_interval
    while True:
        try:
            msg = queue.get(timeout=5)
        except:
            # 队列空，检查子进程是否还活着
            if all(not p.is_alive() for p in workers):
                break
            continue

        if msg.get('type') == 'transition':
            agent.store_transition(msg['state'], msg['action'], msg['reward'], msg['next_state'], msg['done'])
            loss = agent.learn()
            if loss is not None:
                last_loss = loss
            total_samples += 1
        elif msg.get('type') == 'episode_end':
            episode_count += 1
            agent.update_epsilon()
            # 将主进程的 epsilon 值写回共享变量，供 worker 使用
            try:
                shared_epsilon.value = float(agent.epsilon)
            except Exception:
                pass
            episode_reward = msg.get('episode_reward', 0.0)
            episode_steps = msg.get('episode_steps', 0)
            print(f"Episode {episode_count}: reward={episode_reward:.2f}, steps={episode_steps}, samples={total_samples}, epsilon={agent.epsilon:.3f}, loss={(last_loss if last_loss is not None else float('nan')):.4f}")
            episode_success = msg.get('episode_success', False)
            episode_delay = msg.get('episode_delay', 0.0)
            with open(log_path, 'a', newline='', encoding='utf-8') as log_file:
                writer = csv.writer(log_file)
                writer.writerow([episode_count, episode_reward, episode_steps, int(episode_success), episode_delay, total_samples, f"{agent.epsilon:.4f}", f"{last_loss:.6f}" if last_loss is not None else '', time.strftime('%Y-%m-%d %H:%M:%S')])
            try:
                safe_torch_save(agent.policy_net.state_dict(), model_path)
            except Exception as ex:
                print(f"Warning: failed to save model to {model_path}: {ex}")

            if not args.no_plot and plot_interval > 0 and episode_count % plot_interval == 0:
                try:
                    plot_script = os.path.join(project_root, 'Python', 'plot_training.py')
                    subprocess.run([sys.executable, plot_script, '--log', log_path, '--out', os.path.join(project_root, 'simulation_results', 'figures')], check=True)
                    plot_output = os.path.join(project_root, 'simulation_results', 'figures', 'training_metrics.png')
                    print(f"实时绘图已更新: {plot_output}")
                except Exception as ex:
                    print(f"Warning: 实时绘图失败: {ex}")
        else:
            continue

        

    # 等待所有子进程结束
    for p in workers:
        p.join()

   # 训练完成后保存模型权重（适配 Dueling DQN）
    policy_net = agent.policy_net

    def extract_linear_weights(module):
        weights = []
        biases = []
        for m in module.modules():
            if isinstance(m, NoisyLinear):
                weights.append(m.weight_mu.data.cpu().numpy())
                biases.append(m.bias_mu.data.cpu().numpy())
        return weights, biases

    feature_weights, feature_biases = extract_linear_weights(policy_net.feature)
    value_weights, value_biases = extract_linear_weights(policy_net.value)
    advantage_weights, advantage_biases = extract_linear_weights(policy_net.advantage)

    if len(feature_weights) < 2 or len(value_weights) < 2 or len(advantage_weights) < 2:
        raise RuntimeError('Unexpected DQN network structure when exporting weights.')

    weights_path = os.path.join(project_root, 'dqn_weights.mat')
    sio.savemat(weights_path, {
        'feature_w1': feature_weights[0], 'feature_b1': feature_biases[0],
        'feature_w2': feature_weights[1], 'feature_b2': feature_biases[1],
        'value_w1': value_weights[0], 'value_b1': value_biases[0],
        'value_w2': value_weights[1], 'value_b2': value_biases[1],
        'advantage_w1': advantage_weights[0], 'advantage_b1': advantage_biases[0],
        'advantage_w2': advantage_weights[1], 'advantage_b2': advantage_biases[1],
        'state_dim': state_dim, 'action_dim': action_dim
    })
    matlab_rl_weights_path = os.path.join(project_root, 'matlab', 'rl', 'dqn_weights.mat')
    os.makedirs(os.path.dirname(matlab_rl_weights_path), exist_ok=True)
    sio.savemat(matlab_rl_weights_path, {
        'feature_w1': feature_weights[0], 'feature_b1': feature_biases[0],
        'feature_w2': feature_weights[1], 'feature_b2': feature_biases[1],
        'value_w1': value_weights[0], 'value_b1': value_biases[0],
        'value_w2': value_weights[1], 'value_b2': value_biases[1],
        'advantage_w1': advantage_weights[0], 'advantage_b1': advantage_biases[0],
        'advantage_w2': advantage_weights[1], 'advantage_b2': advantage_biases[1],
        'state_dim': state_dim, 'action_dim': action_dim
    })
    print(f"Dueling DQN 权重已导出为 {weights_path} 和 {matlab_rl_weights_path}")

    metadata = {
        'timestamp': time.strftime('%Y-%m-%d %H:%M:%S'),
        'state_dim': state_dim,
        'action_dim': action_dim,
        'num_domains': num_domains,
        'use_noisy': agent.use_noisy,
        'n_step': agent.n_step,
        'gamma': agent.gamma,
        'batch_size': agent.batch_size,
        'target_update': agent.target_update,
        'model_path': model_path,
        'log_path': log_path,
        'reward_config_path': reward_config_path
    }
    with open(args.metadata_path, 'w', encoding='utf-8') as meta_file:
        json.dump(metadata, meta_file, indent=2, ensure_ascii=False)
    print('训练元数据已保存为 training_metadata.json')

    if not args.no_plot:
        try:
            plot_script = os.path.join(project_root, 'Python', 'plot_training.py')
            subprocess.run([sys.executable, plot_script, '--log', log_path, '--out', os.path.join(project_root, 'simulation_results', 'figures')], check=True)
            output_path = os.path.join(project_root, 'simulation_results', 'figures', 'training_metrics.png')
            print(f"训练结果可视化已保存到: {output_path}")
        except Exception as ex:
            print(f"训练结果可视化失败: {ex}")

if __name__ == '__main__':
    train()
