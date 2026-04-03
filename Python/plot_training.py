import os
import csv
import math
import json

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
DEFAULT_LOG_PATH = os.path.join(PROJECT_ROOT, 'training_logs.csv')
DEFAULT_OUTPUT_DIR = os.path.join(PROJECT_ROOT, 'simulation_results', 'figures')


def _normalize_row(row):
    try:
        episode = int(row.get('episode', '') or 0)
    except ValueError:
        episode = 0
    try:
        reward = float(row.get('episode_reward', 'nan') or math.nan)
    except ValueError:
        reward = math.nan
    try:
        steps = int(row.get('episode_steps', '0') or 0)
    except ValueError:
        steps = 0
    try:
        samples = int(row.get('total_samples', '0') or 0)
    except ValueError:
        samples = 0
    try:
        epsilon = float(row.get('epsilon', 'nan') or math.nan)
    except ValueError:
        epsilon = math.nan
    try:
        loss = float(row.get('loss', 'nan') or math.nan)
    except ValueError:
        loss = math.nan
    return {
        'episode': episode,
        'reward': reward,
        'steps': steps,
        'samples': samples,
        'epsilon': epsilon,
        'loss': loss,
    }


def _select_latest_run(rows):
    if not rows:
        return []

    segments = []
    current = []
    prev_episode = None
    prev_epsilon = None

    for row in rows:
        try:
            episode = int(row.get('episode', '') or 0)
        except ValueError:
            episode = 0
        try:
            epsilon = float(row.get('epsilon', 'nan') or math.nan)
        except ValueError:
            epsilon = math.nan

        reset = False
        if prev_episode is not None:
            if episode <= prev_episode:
                reset = True
            elif not math.isnan(epsilon) and not math.isnan(prev_epsilon) and epsilon > prev_epsilon + 0.1:
                reset = True

        if reset and current:
            segments.append(current)
            current = []

        current.append(row)
        prev_episode = episode
        prev_epsilon = epsilon

    if current:
        segments.append(current)

    rows = segments[-1]

    # Keep only the last occurrence of each episode number in the latest segment.
    deduped = {}
    for row in rows:
        try:
            episode = int(row.get('episode', '') or 0)
        except ValueError:
            episode = 0
        if episode > 0:
            deduped[episode] = row

    return [deduped[ep] for ep in sorted(deduped)]


def read_training_log(log_path):
    if not os.path.exists(log_path):
        raise FileNotFoundError(f"Training log not found: {log_path}")

    raw_rows = []
    with open(log_path, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for row in reader:
            raw_rows.append(row)

    rows = _select_latest_run(raw_rows)
    episodes = []
    rewards = []
    steps = []
    total_samples = []
    epsilons = []
    losses = []

    for row in rows:
        normalized = _normalize_row(row)
        episodes.append(normalized['episode'])
        rewards.append(normalized['reward'])
        steps.append(normalized['steps'])
        total_samples.append(normalized['samples'])
        epsilons.append(normalized['epsilon'])
        losses.append(normalized['loss'])

    total_reward = []
    cumsum = 0.0
    for r in rewards:
        if math.isnan(r):
            total_reward.append(cumsum)
        else:
            cumsum += r
            total_reward.append(cumsum)

    return {
        'episode': episodes,
        'reward': rewards,
        'total_reward': total_reward,
        'steps': steps,
        'samples': total_samples,
        'epsilon': epsilons,
        'loss': losses,
    }


def smooth_curve(values, window=20):
    if window <= 1 or len(values) < window:
        return values
    smoothed = []
    cum = 0.0
    for i, v in enumerate(values):
        cum += v
        if i >= window:
            cum -= values[i - window]
            smoothed.append(cum / window)
        else:
            smoothed.append(cum / (i + 1))
    return smoothed


def save_html_chartjs(output_path, episodes, reward, reward_smooth, loss, loss_smooth, epsilon, steps, samples):
    template = f'''<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<title>Training Metrics</title>
<script src="https://cdn.jsdelivr.net/npm/chart.js"></script>
<style>
  body {{ font-family: Arial, sans-serif; margin: 20px; }}
  .chart-container {{ width: 100%; max-width: 1000px; margin-bottom: 40px; }}
</style>
</head>
<body>
<h1>Training Metrics</h1>
<div class="chart-container"><canvas id="rewardChart"></canvas></div>
<div class="chart-container"><canvas id="lossChart"></canvas></div>
<div class="chart-container"><canvas id="epsilonChart"></canvas></div>
<div class="chart-container"><canvas id="stepsChart"></canvas></div>
<script>
const labels = {json.dumps(episodes)};
const rewardData = {json.dumps(reward)};
const rewardSmoothData = {json.dumps(reward_smooth)};
const lossData = {json.dumps(loss)};
const lossSmoothData = {json.dumps(loss_smooth)};
const epsilonData = {json.dumps(epsilon)};
const stepsData = {json.dumps(steps)};
const samplesData = {json.dumps(samples)};

function createLineChart(ctx, title, datasets, yLabel) {{
  new Chart(ctx, {{
    type: 'line',
    data: {{ labels, datasets }},
    options: {{
      responsive: true,
      plugins: {{
        title: {{ display: true, text: title, font: {{ size: 18 }} }},
        legend: {{ position: 'bottom' }}
      }},
      scales: {{
        x: {{ title: {{ display: true, text: 'Episode' }} }},
        y: {{ title: {{ display: true, text: yLabel }}, beginAtZero: true }}
      }}
    }}
  }});
}}

createLineChart(document.getElementById('rewardChart'), 'Episode Reward', [
  {{ label: 'Reward', data: rewardData, borderColor: 'rgba(54, 162, 235, 0.7)', backgroundColor: 'rgba(54, 162, 235, 0.2)', tension: 0.2 }},
  {{ label: 'Reward (smoothed)', data: rewardSmoothData, borderColor: 'rgba(33, 150, 243, 1)', backgroundColor: 'rgba(33, 150, 243, 0.2)', tension: 0.2 }}
], 'Reward');

createLineChart(document.getElementById('lossChart'), 'Loss', [
  {{ label: 'Loss', data: lossData, borderColor: 'rgba(255, 99, 132, 0.7)', backgroundColor: 'rgba(255, 99, 132, 0.2)', tension: 0.2 }},
  {{ label: 'Loss (smoothed)', data: lossSmoothData, borderColor: 'rgba(229, 57, 53, 1)', backgroundColor: 'rgba(229, 57, 53, 0.2)', tension: 0.2 }}
], 'Loss');

createLineChart(document.getElementById('epsilonChart'), 'Epsilon', [
  {{ label: 'Epsilon', data: epsilonData, borderColor: 'rgba(75, 192, 192, 0.8)', backgroundColor: 'rgba(75, 192, 192, 0.2)', tension: 0.2 }}
], 'Epsilon');

createLineChart(document.getElementById('stepsChart'), 'Steps and Samples', [
  {{ label: 'Episode Steps', data: stepsData, borderColor: 'rgba(153, 102, 255, 0.8)', backgroundColor: 'rgba(153, 102, 255, 0.2)', tension: 0.2 }},
  {{ label: 'Total Samples', data: samplesData, borderColor: 'rgba(255, 159, 64, 0.8)', backgroundColor: 'rgba(255, 159, 64, 0.2)', tension: 0.2 }}
], 'Count');
</script>
</body>
</html>'''
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write(template)
    return output_path


def _pillow_plot_placeholder(output_path):
    from PIL import Image, ImageDraw, ImageFont
    width, height = 1200, 800
    im = Image.new('RGB', (width, height), 'white')
    draw = ImageDraw.Draw(im)
    try:
        font = ImageFont.truetype('arial.ttf', 24)
    except Exception:
        font = ImageFont.load_default()
    msg = 'No training data available'
    tw, th = draw.textsize(msg, font=font)
    draw.text(((width - tw) / 2, (height - th) / 2), msg, fill='black', font=font)
    im.save(output_path)
    return output_path


def _pillow_plot(episodes, reward, reward_smooth, loss, loss_smooth, epsilon, steps, samples, output_path):
    from PIL import Image, ImageDraw, ImageFont
    width, height = 1400, 1000
    im = Image.new('RGB', (width, height), 'white')
    draw = ImageDraw.Draw(im)
    try:
        font = ImageFont.truetype('arial.ttf', 16)
        title_font = ImageFont.truetype('arial.ttf', 24)
    except Exception:
        font = ImageFont.load_default()
        title_font = ImageFont.load_default()

    def plot_series(x, y, bbox, color='blue', label=None):
        x0, y0, x1, y1 = bbox
        data = [(xi, yi) for xi, yi in zip(x, y) if not (isinstance(yi, float) and math.isnan(yi))]
        if len(data) < 2:
            return
        xs, ys = zip(*data)
        if len(xs) < 2:
            return
        min_x = min(xs)
        max_x = max(xs)
        min_y = min(ys)
        max_y = max(ys)
        if max_x == min_x:
            max_x = min_x + 1
        if max_y == min_y:
            max_y = min_y + 1
        for i in range(len(xs) - 1):
            x_start = x0 + (xs[i] - min_x) / (max_x - min_x) * (x1 - x0)
            y_start = y1 - (ys[i] - min_y) / (max_y - min_y) * (y1 - y0)
            x_end = x0 + (xs[i+1] - min_x) / (max_x - min_x) * (x1 - x0)
            y_end = y1 - (ys[i+1] - min_y) / (max_y - min_y) * (y1 - y0)
            draw.line((x_start, y_start, x_end, y_end), fill=color, width=2)
        if label:
            draw.text((x0 + 5, y0 + 5), label, fill=color, font=font)

    def draw_panel(title, x, ys, colors, labels, bbox):
        x0, y0, x1, y1 = bbox
        draw.rectangle(bbox, outline='black')
        draw.text((x0 + 5, y0 + 5), title, fill='black', font=title_font)
        for y, c, l in zip(ys, colors, labels):
            plot_series(x, y, (x0+40, y0+40, x1-20, y1-30), color=c, label=l)

    draw_panel('Reward', episodes, [reward, reward_smooth], ['blue', 'red'], ['Reward', 'Reward smooth'], (20, 20, 680, 480))
    draw_panel('Loss', episodes, [loss, loss_smooth], ['orange', 'red'], ['Loss', 'Loss smooth'], (720, 20, 1380, 480))
    draw_panel('Epsilon', episodes, [epsilon], ['green'], ['Epsilon'], (20, 520, 680, 980))
    draw_panel('Steps/Samples', episodes, [steps, samples], ['purple', 'brown'], ['Steps', 'Samples'], (720, 520, 1380, 980))

    im.save(output_path)
    return output_path


def plot_training_results(log_path=DEFAULT_LOG_PATH, output_dir=DEFAULT_OUTPUT_DIR):
    stats = read_training_log(log_path)
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, 'training_metrics.png')

    if len(stats['episode']) == 0:
        # 空数据时仍用 matplotlib 生成空白图，不再使用 Pillow
        episodes = []
        reward = []
        total_reward = []
        loss = []
        epsilon = []
        steps = []
        samples = []
        reward_smooth = []
        loss_smooth = []
    else:
        episodes = stats['episode']
        reward = stats['reward']
        total_reward = stats['total_reward']
        loss = stats['loss']
        epsilon = stats['epsilon']
        steps = stats['steps']
        samples = stats['samples']
        reward_smooth = smooth_curve([x for x in reward if not math.isnan(x)], window=20)
        loss_smooth = smooth_curve([x for x in loss if not math.isnan(x)], window=20)

    total_reward_smooth = smooth_curve([x for x in total_reward if not math.isnan(x)], window=20)

    try:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
    except Exception as exc:
        raise RuntimeError(f'无法绘制 PNG：matplotlib 不可用或初始化失败：{exc}')
    except Exception as exc:
        raise RuntimeError(f'无法绘制 PNG：matplotlib 不可用或初始化失败：{exc}')

    fig, axs = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle('Training Results', fontsize=18)

    if any(not math.isnan(r) for r in reward):
        axs[0, 0].plot(episodes, reward, color='tab:blue', alpha=0.3, label='Reward')
        axs[0, 0].plot(episodes[:len(reward_smooth)], reward_smooth, color='tab:blue', label='Reward (smoothed)')
        axs[0, 0].set_ylabel('Episode Reward')
        axs[0, 0].legend()
        axs[0, 0].grid(True, alpha=0.3)
    else:
        axs[0, 0].text(0.5, 0.5, 'No reward data', ha='center', va='center')
        axs[0, 0].set_axis_off()

    if any(not math.isnan(l) for l in loss):
        axs[0, 1].plot(episodes, loss, color='tab:red', alpha=0.3, label='Loss')
        axs[0, 1].plot(episodes[:len(loss_smooth)], loss_smooth, color='tab:red', label='Loss (smoothed)')
        axs[0, 1].set_ylabel('Loss')
        axs[0, 1].legend()
        axs[0, 1].grid(True, alpha=0.3)
    else:
        axs[0, 1].text(0.5, 0.5, 'No loss data', ha='center', va='center')
        axs[0, 1].set_axis_off()

    if any(not math.isnan(e) for e in epsilon):
        axs[1, 0].plot(episodes, epsilon, color='tab:green', label='Epsilon')
        axs[1, 0].set_ylabel('Epsilon')
        axs[1, 0].legend()
        axs[1, 0].grid(True, alpha=0.3)
    else:
        axs[1, 0].text(0.5, 0.5, 'No epsilon data', ha='center', va='center')
        axs[1, 0].set_axis_off()

    if len(total_reward) > 0:
        axs[1, 1].plot(episodes, total_reward, color='tab:green', alpha=0.3, label='Total Reward')
        axs[1, 1].plot(episodes[:len(total_reward_smooth)], total_reward_smooth, color='tab:green', label='Total Reward (smoothed)')
        axs[1, 1].set_ylabel('Total Reward')
        axs[1, 1].legend()
        axs[1, 1].grid(True, alpha=0.3)
    else:
        axs[1, 1].text(0.5, 0.5, 'No total reward data', ha='center', va='center')
        axs[1, 1].set_axis_off()

    for ax in axs.flat:
        ax.set_xlabel('Episode')

    fig.tight_layout(rect=[0, 0.03, 1, 0.95])
    fig.savefig(output_path)
    plt.close(fig)
    return output_path


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(description='Plot training metrics from training_logs.csv')
    parser.add_argument('--log', default=DEFAULT_LOG_PATH, help='Training log CSV path')
    parser.add_argument('--out', default=DEFAULT_OUTPUT_DIR, help='Output directory for plots')
    args = parser.parse_args()

    os.makedirs(args.out, exist_ok=True)
    image_file = plot_training_results(log_path=args.log, output_dir=args.out)
    print(f'Training plot saved to: {image_file}')
