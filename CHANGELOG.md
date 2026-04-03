# CHANGELOG

**日期**: 2026-04-03

**概要**
- 稳定化强化学习训练：引入 EMA + warm-up 的奖励归一化、调整 DQN 默认超参（学习率、batch、软更新tau）以降低震荡。
- 加入轻量级域队列（queue）模型：Python 环境中实现队列参数、状态扩展与奖励惩罚；MATLAB 中增加队列动力学与状态向量扩展并导出 `config.json` 以与 Python 同步。
- 将 reward 调参器（GA）扩展为包含队列参数，调整评分函数以使用延迟（秒）尺度。
- 修复了 MATLAB ↔ Python 的 state_dim 不匹配（为模型输入添加了 3 个队列特征）。
- 增加监控脚本，自动根据训练日志生成和更新训练指标图表。

**重要说明（已排除）**
- 未将大体积模型权重或训练日志推送到仓库（例如 `.pth`、`.mat`、`training_logs.csv`），以避免仓库膨胀；这些文件仍保留在本地。

**包含在本次提交的主要文件**

- Python 源代码
  - Python/dqn_agent.py
  - Python/plot_training.py
  - Python/reward_weight_tuner.py
  - Python/rl_predict.py
  - Python/satellite_env.py
  - Python/train_rl.py
  - Python/worker.py
  - Python/monitor_simulation_and_plot.py
  - run_eval_candidate.py
  - eval_candidate.json
  - inspect_model.py
  - best_reward_weights.json

- MATLAB 源代码
  - matlab/core/build_state_vector.m
  - matlab/core/load_config_params.m
  - matlab/core/main.m
  - matlab/core/routing_performance_test.m
  - matlab/core/run_performance_test.m
  - matlab/core/visualize_all_results.m
  - matlab/rl/apply_rl_action.m
  - matlab/rl/rl_cross_domain_routing.m
  - matlab/routing/cross_domain_routing.m
  - matlab/routing/cross_domain_routing_stats.m
  - matlab/routing/route_to_domain.m

- 配置/元数据
  - simulation_results/config.json
  - training_metadata.json

- 其他
  - 删除文件: run_simulation.m (已从仓库中删除)

---

如需我把模型权重或日志也推送到 GitHub（不推荐直接放在主仓库），我可以建议并创建一个 Git LFS / release 上传流程，或把它们上传到云存储并在仓库中放置下载脚本/链接。