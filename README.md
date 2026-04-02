# LEO RouteAI

LEO RouteAI 是一个混合型的 LEO 卫星跨域路由研究项目。它将强化学习用于域间决策，并结合经典路由算法来提升低轨卫星网络中的路由性能。

LEO RouteAI is a hybrid LEO satellite cross-domain routing research project. It uses reinforcement learning for inter-domain decision-making and combines classical routing algorithms to improve routing performance in low Earth orbit satellite networks.

## Project Overview / 项目概览

- `matlab/core/`: 主仿真框架，拓扑生成、域划分和性能评估。
- `matlab/rl/`: 基于 RL 的路由封装、状态构建和本地推理支持。
- `matlab/routing/`: 传统路由方法，包括 Dijkstra、跨域路由和统计工具。
- `Python/`: RL 训练、模型定义、预测工具，以及 MATLAB 集成。
- `simulation_results/`: 仿真输出、图表和日志存储目录。

- `matlab/core/`: main simulation framework, topology generation, domain partition, and performance evaluation.
- `matlab/rl/`: RL-based routing wrappers, state generation, and local inference support.
- `matlab/routing/`: traditional routing methods including Dijkstra, cross-domain routing, and statistics utilities.
- `Python/`: RL training, model definition, prediction tools, and MATLAB integration.
- `simulation_results/`: generated outputs, figures and logs.

## Features / 功能描述

- 混合路由：RL 做域级选择，经典算法完成路径构建。
- 跨域路由性能评估：延迟、跳数、负载等指标。
- 兼容 MATLAB 仿真与 Python RL 训练。

- Hybrid routing: RL selects domain-level decisions, classical algorithms complete the path.
- Cross-domain routing performance evaluation with delay/hop/load metrics.
- Support for both MATLAB simulation and Python-based RL training.

## Quick Start / 快速开始

1. 在 MATLAB 中将当前文件夹切换到项目根目录（`d:\Documents\Project`）。
2. 运行 `run_simulation.m` 启动仿真。
3. 在 Python 中运行 `Python/train_rl.py` 训练 RL 模型。
4. 使用 `Python/plot_training.py` 可视化训练结果。

English:

1. Open MATLAB and set the current folder to the project root (`d:\Documents\Project`).
2. Run `run_simulation.m` to start the simulation.
3. In Python, use `Python/train_rl.py` to train the RL model.
4. Use `Python/plot_training.py` to visualize training results.

## Requirements / 运行环境

- MATLAB（含基本图和最短路径函数）
- Python 3.x
- Python 包：`numpy`, `torch`（以及 Python 脚本所需的其他包）
- 可选：如果需要 Python 与 MATLAB 直接集成，则安装 MATLAB Engine for Python

English:

- MATLAB (with basic graph and path functions)
- Python 3.x
- Python packages: `numpy`, `torch` (and any other packages required by the Python scripts)
- Optional: MATLAB Engine for Python if integrating Python and MATLAB directly

## Repository Upload / 仓库上传

To upload this project to GitHub from a machine with Git installed:

```bash
cd /d d:\Documents\Project
git init
git add .
git commit -m "Initial commit"
git branch -M main
git remote add origin https://github.com/your-other-account/repo.git
git push -u origin main
```

如果仓库已经存在本地或者需要更新远程地址：

```bash
git remote set-url origin https://github.com/your-other-account/repo.git
git push -u origin main
```

If the repository already exists locally or you need to update the remote URL:

```bash
git remote set-url origin https://github.com/your-other-account/repo.git
git push -u origin main
```

