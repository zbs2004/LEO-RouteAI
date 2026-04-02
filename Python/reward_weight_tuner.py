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
  "congestion": 0.7339,
  "hop": 1.499,
  "delay": 2.7253,
  "stability": 0.6445,
  "balance": 0.7989,
  "e2e": 1.4617,
  "success": 0.5563,
  "failure": 0.0047,
  "step": 0.0533,
  "progress": 0.0852
}


def parse_args():
    parser = argparse.ArgumentParser(
        description='Optimize reward weight configuration via short training evaluations.')
    parser.add_argument('--code-dir', type=str, default=os.path.dirname(os.path.abspath(__file__)),
                        help='Project root containing train_rl.py and reward_weights.json.')
    parser.add_argument('--base-config', type=str, default=None,
                        help='Optional base reward weight JSON file to start from.')
    parser.add_argument('--population', type=int, default=4,
                        help='Number of reward weight candidates per generation.')
    parser.add_argument('--generations', type=int, default=2,
                        help='Number of evolution generations.')
    parser.add_argument('--num-workers', type=int, default=1,
                        help='Workers used during each evaluation run.')
    parser.add_argument('--episodes-per-worker', type=int, default=20,
                        help='Episode count per worker during evaluation.')
    parser.add_argument('--output', type=str, default='best_reward_weights.json',
                        help='Output path for the best reward weights.')
    parser.add_argument('--timeout', type=int, default=3600,
                        help='Timeout in seconds for each candidate training run.')
    return parser.parse_args()


def load_base_weights(config_path):
    if config_path is None:
        return dict(DEFAULT_WEIGHTS)
    with open(config_path, 'r', encoding='utf-8') as reader:
        weights = json.load(reader)
    return {k: float(weights.get(k, DEFAULT_WEIGHTS[k])) for k in DEFAULT_WEIGHTS}


def perturb(weights, scale=0.3):
    mutated = {}
    for name, value in weights.items():
        delta = random.uniform(-scale, scale) * max(0.05, abs(value))
        mutated[name] = max(0.0, round(value + delta, 4))
    return mutated


def crossover(parent_a, parent_b):
    child = {}
    for key in parent_a:
        child[key] = parent_a[key] if random.random() < 0.5 else parent_b[key]
    return child


def normalize_weights(weights):
    return {k: max(0.0, float(v)) for k, v in weights.items()}


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

        # 复合评分：优先保证成功率，再考虑路径长度和延迟。
        # 成功率被放大到 1000 量级，避免大延迟/步数惩罚把好策略压成负数。
        score = success_rate * 1000.0
        score -= 3.0 * avg_steps
        score -= 0.75 * avg_delay
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
