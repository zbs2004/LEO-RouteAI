import argparse
import json
import os
import random
import shutil
import subprocess
import sys
import tempfile
import time
from pathlib import Path

DEFAULT_WEIGHTS = {
  "congestion": 0.25,
  "hop": 4.0,
  "delay": 6.0,
  "stability": 0.5,
  "balance": 0.25,
  "e2e": 2.5,
  "success": 1.0,
  "failure": 0.1,
  "step": 0.02,
  "progress": 0.05
}

# 新增可优化的队列/仿真参数（用于压力场景）
DEFAULT_EXTRA = {
    'enable_queue_model': True,
    # 调整为更贴近卫星/链路处理的默认值：更高容量、较低每包处理时间（ms）
    'queue_capacity': 500,
    'queue_service_rate': 1.0,
    'queue_delay_weight': 0.01,
    'queue_drop_penalty': 20.0,
    'queue_time_per_step_ms': 10.0
}


def parse_args():
    parser = argparse.ArgumentParser(
        description='Optimize reward weight configuration via short training evaluations.')
    parser.add_argument('--code-dir', type=str, default=os.path.dirname(os.path.abspath(__file__)),
                        help='Project root containing train_rl.py and reward_weights.json.')
    parser.add_argument('--base-config', type=str, default=None,
                        help='Optional base reward weight JSON file to start from.')
    parser.add_argument('--population', type=int, default=8,
                        help='Number of reward weight candidates per generation.')
    parser.add_argument('--generations', type=int, default=4,
                        help='Number of evolution generations.')
    parser.add_argument('--num-workers', type=int, default=2,
                        help='Workers used during each evaluation run.')
    parser.add_argument('--episodes-per-worker', type=int, default=40,
                        help='Episode count per worker during evaluation.')
    parser.add_argument('--output', type=str, default='best_reward_weights.json',
                        help='Output path for the best reward weights.')
    parser.add_argument('--timeout', type=int, default=3600,
                        help='Timeout in seconds for each candidate training run.')
    return parser.parse_args()


def load_base_weights(config_path):
    # 返回包含 reward weights 与附加队列参数的初始配置字典
    base = dict(DEFAULT_WEIGHTS)
    extras = dict(DEFAULT_EXTRA)
    if config_path is None:
        combined = {**base, **extras}
        return combined
    with open(config_path, 'r', encoding='utf-8') as reader:
        cfg = json.load(reader)
    # 覆盖 reward keys
    for k in base:
        if k in cfg:
            try:
                base[k] = float(cfg[k])
            except Exception:
                pass
    # 覆盖 extras
    for k in extras:
        if k in cfg:
            extras[k] = cfg[k]
    combined = {**base, **extras}
    return combined


def perturb(config, scale=0.3):
    # 对不同参数采取不同的扰动策略并保持在合理范围内
    mutated = {}
    for name, value in config.items():
        if name == 'enable_queue_model':
            # 小概率翻转布尔值
            if random.random() < 0.05:
                mutated[name] = not bool(value)
            else:
                mutated[name] = bool(value)
            continue

        # 队列参数的合理范围
        if name == 'queue_capacity':
            low, high = 50, 2000
            base = int(max(1, float(value)))
            delta = int(round(random.uniform(-scale, scale) * max(10, base * 0.5)))
            v = max(low, min(high, base + delta))
            mutated[name] = int(v)
            continue
        if name == 'queue_service_rate':
            low, high = 0.1, 5.0
            base = float(value)
            v = base * (1.0 + random.uniform(-scale, scale))
            mutated[name] = float(max(low, min(high, v)))
            continue
        if name == 'queue_delay_weight':
            low, high = 0.0, 2.0
            base = float(value)
            v = base + random.uniform(-scale, scale) * max(0.01, base)
            mutated[name] = float(max(low, min(high, v)))
            continue
        if name == 'queue_drop_penalty':
            low, high = 0.0, 1000.0
            base = float(value)
            v = base * (1.0 + random.uniform(-scale, scale))
            mutated[name] = float(max(low, min(high, v)))
            continue
        if name == 'queue_time_per_step_ms':
            low, high = 0.1, 500.0
            base = float(value)
            v = base * (1.0 + random.uniform(-scale, scale))
            mutated[name] = float(max(low, min(high, v)))
            continue

        # 默认处理 reward 权重
        try:
            val = float(value)
        except Exception:
            mutated[name] = value
            continue
        delta = random.uniform(-scale, scale) * max(0.05, abs(val))
        mutated[name] = round(max(0.0, val + delta), 6)
    return mutated


def crossover(parent_a, parent_b):
    child = {}
    for key in parent_a:
        child[key] = parent_a[key] if random.random() < 0.5 else parent_b[key]
    return child


def normalize_weights(weights):
    out = {}
    for k, v in weights.items():
        if k == 'enable_queue_model':
            out[k] = bool(v)
        elif k == 'queue_capacity':
            out[k] = int(max(1, int(round(float(v)))))
        elif k == 'queue_service_rate':
            out[k] = float(v)
        elif k == 'queue_delay_weight':
            out[k] = float(v)
        elif k == 'queue_drop_penalty':
            out[k] = float(v)
        elif k == 'queue_time_per_step_ms':
            out[k] = float(v)
        else:
            out[k] = max(0.0, float(v))
    return out


def evaluate(weights, code_dir, num_workers, episodes_per_worker, timeout):
    code_dir = Path(code_dir)
    if not code_dir.exists():
        raise FileNotFoundError(f'Code directory not found: {code_dir}')

    with tempfile.TemporaryDirectory(prefix='reward_tuner_') as tmpdir:
        tmpdir_path = Path(tmpdir)
        config_path = tmpdir_path / 'reward_weights.json'
        log_path = tmpdir_path / 'training_logs.csv'
        metadata_path = tmpdir_path / 'training_metadata.json'
        model_path = tmpdir_path / 'candidate_model.pth'

        with open(config_path, 'w', encoding='utf-8') as writer:
            json.dump(normalize_weights(weights), writer, indent=2, ensure_ascii=False)

        cmd = [sys.executable,
               'train_rl.py',
               '--reward-config', str(config_path),
               '--num-workers', str(num_workers),
               '--num-episodes-per-worker', str(episodes_per_worker),
               '--log-path', str(log_path),
               '--model-path', str(model_path),
               '--metadata-path', str(metadata_path),
               '--no-plot']

        start_time = time.time()
        result = subprocess.run(cmd, cwd=str(code_dir), capture_output=True, text=True, timeout=timeout)
        elapsed = time.time() - start_time
        if result.returncode != 0:
            raise RuntimeError(f'Candidate evaluation failed after {elapsed:.1f}s: {result.returncode}\n{result.stdout}\n{result.stderr}')
        print(f'    Candidate evaluation completed in {elapsed:.1f}s')

        if not log_path.exists():
            raise RuntimeError('Expected log file not created during evaluation.')

        with open(log_path, 'r', encoding='utf-8') as reader:
            lines = [line.strip() for line in reader.readlines() if line.strip()]

        if len(lines) < 2:
            return float('-inf')

        headers = lines[0].split(',')
        reward_idx = headers.index('episode_reward')
        steps_idx = headers.index('episode_steps')
        success_idx = headers.index('episode_success') if 'episode_success' in headers else None
        delay_idx = headers.index('episode_delay') if 'episode_delay' in headers else None

        rewards = [float(line.split(',')[reward_idx]) for line in lines[1:]]
        steps = [float(line.split(',')[steps_idx]) for line in lines[1:]]
        successes = [int(line.split(',')[success_idx]) if success_idx is not None else 0 for line in lines[1:]]
        delays = [float(line.split(',')[delay_idx]) if delay_idx is not None else 0.0 for line in lines[1:]]

        n = len(rewards)
        trim = max(1, n // 10)
        if n > 2 * trim:
            trimmed = sorted(rewards)[trim:-trim]
        else:
            trimmed = rewards

        success_rate = sum(successes) / n
        avg_steps = sum(steps) / n
        avg_delay = sum(delays) / n
        # 训练日志中以毫秒记录延迟，将其换算为秒以避免在评分中产生过大惩罚
        avg_delay_s = avg_delay / 1000.0

        packet_loss_rate = (len(lines[1:]) - sum(successes)) / len(lines[1:]) if len(lines[1:]) > 0 else 0.0  # 丢包率 = 失败次数 / 总次数

        # 复合评分：优先保证成功率，再考虑路径长度和延迟，加入丢包率惩罚
        score = success_rate * 1000.0
        score -= 3.0 * avg_steps
        score -= 0.75 * avg_delay_s
        score -= 500.0 * packet_loss_rate  # 丢包率惩罚
        if success_rate < 0.90:
            score -= (0.90 - success_rate) * 1000.0
        if len(trimmed) > 0:
            score += sum(trimmed) / len(trimmed) * 0.2
        return score


def select_elite(population, scores, retain=0.4):
    paired = sorted(zip(scores, population), key=lambda x: x[0], reverse=True)
    retain_count = max(1, int(len(paired) * retain))
    return [candidate for _, candidate in paired[:retain_count]]


def main():
    args = parse_args()
    base_weights = load_base_weights(args.base_config)
    population = [perturb(base_weights, scale=0.2) for _ in range(args.population)]
    population[0] = base_weights

    best_candidate = base_weights
    best_score = float('-inf')

    for generation in range(1, args.generations + 1):
        print(f'Generation {generation}/{args.generations} evaluating {len(population)} candidates...')
        scores = []
        for idx, candidate in enumerate(population, start=1):
            print(f'  Candidate {idx}/{len(population)} weights={candidate}')
            try:
                score = evaluate(candidate, args.code_dir, args.num_workers, args.episodes_per_worker, args.timeout)
            except Exception as ex:
                print(f'    Evaluation failed: {ex}')
                score = float('-inf')
            scores.append(score)
            print(f'    Score={score:.4f}')

            if score > best_score:
                best_score = score
                best_candidate = candidate

        elite = select_elite(population, scores, retain=0.4)
        children = []
        while len(children) + len(elite) < args.population:
            if len(elite) == 1:
                parents = [elite[0], elite[0]]
            else:
                parents = random.sample(elite, 2)
            child = crossover(parents[0], parents[1])
            child = perturb(child, scale=0.15)
            children.append(child)

        population = elite + children

    output_path = Path(args.output)
    with open(output_path, 'w', encoding='utf-8') as writer:
        json.dump(normalize_weights(best_candidate), writer, indent=2, ensure_ascii=False)

    print(f'Best candidate saved to {output_path}')
    print(f'Best score: {best_score:.4f}')
    print(f'Best weights: {best_candidate}')


if __name__ == '__main__':
    main()
