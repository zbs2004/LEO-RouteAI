import json
import numpy as np
import gymnasium as gym
from gymnasium import spaces

class SatelliteRoutingEnv(gym.Env):
    def __init__(self, eng, num_satellites=66, num_domains=8,
                 enable_queue_model=False, queue_capacity=200, queue_service_rate=1.0,
                 queue_delay_weight=0.1, queue_drop_penalty=50.0, queue_time_per_step_ms=100.0):
        super().__init__()
        self.eng = eng
        self.num_satellites = num_satellites
        self.num_domains = num_domains
        self.action_space = spaces.Discrete(num_domains)
        # observation 增加 3 个队列相关特征（mean, max, current domain queue）
        self.observation_space = spaces.Box(
            low=0.0,
            high=1.0,
            shape=(3 + num_domains * 2 + num_domains * num_domains + 13,),  # 增加2个负载特征 +3 队列特征
            dtype=np.float32
        )
        self.reward_weights = {
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
        self.episode_total_delay = 0.0
        self.episode_success = False
        # 在线 reward 归一化统计（Welford），用于稳定 Q 目标分布
        self.reward_running_mean = 0.0
        self.reward_running_M2 = 0.0
        self.reward_running_count = 0
        self.reward_normalize = True
        self.reward_norm_eps = 1e-6
        self.reward_norm_clip = 5.0
        self.reward_norm_scale = 1.0
        # 归一化改进：EMA 统计与最小样本数（warm-up）以减少单步噪声
        self.reward_norm_min_count = 200
        self.reward_ema_alpha = 0.005
        self.reward_ema_mean = 0.0
        self.reward_ema_var = 1.0
        # ===== 队列模型（域级，轻量） =====
        self.enable_queue_model = bool(enable_queue_model)
        self.domain_queue_capacity = int(queue_capacity)
        # 支持标量或数组的服务率（包/步）
        if hasattr(queue_service_rate, '__iter__'):
            self.domain_service_rate = np.array(queue_service_rate, dtype=np.float32)
        else:
            self.domain_service_rate = np.full(self.num_domains, float(queue_service_rate), dtype=np.float32)
        self.queue_delay_weight = float(queue_delay_weight)
        self.queue_drop_penalty = float(queue_drop_penalty)
        self.queue_time_per_step_ms = float(queue_time_per_step_ms)
        self.domain_queues = np.zeros(self.num_domains, dtype=np.float32)
        self.sim_time_steps = 0

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        src = np.random.randint(1, self.num_satellites + 1)
        dst = np.random.randint(1, self.num_satellites + 1)
        while dst == src:
            dst = np.random.randint(1, self.num_satellites + 1)

        self.src = src
        self.dst = dst

        self.eng.workspace['current_src'] = src
        self.eng.workspace['current_dst'] = dst
        self.eng.workspace['current_satellite'] = src
        self.episode_total_delay = 0.0
        self.episode_success = False

        domains = self.eng.workspace['domains']
        load_matrix = self.eng.workspace['load_matrix']
        domain_graph = self.eng.workspace['domain_graph']

        self.domain_graph = np.array(self.eng.eval('full(domain_graph)', nargout=1))
        state_struct = self.eng.get_state_for_rl(src, dst, src, domains, load_matrix, domain_graph, nargout=1)
        self.eng.workspace['current_domain'] = state_struct.get('current_domain', 1)
        self.current_domain = int(self.eng.workspace['current_domain'])
        self.state = self._struct_to_array(state_struct)

        self.prev_max_load = None
        # reset domain queues when starting fresh (if enabled)
        if self.enable_queue_model:
            self.domain_queues[:] = 0
            self.sim_time_steps = 0
        return self.state, {}

    def _validate_reward_weights(self, weights):
        if not isinstance(weights, dict):
            raise TypeError('reward_weights must be a dict')
        required_keys = {'congestion', 'hop', 'delay', 'stability', 'balance', 'e2e'}
        optional_defaults = {
            'success': 0.0,
            'failure': 0.0,
            'step': 0.1,
            'progress': 0.3
        }
        missing = required_keys - set(weights.keys())
        if missing:
            raise ValueError(f'Missing reward weights: {sorted(missing)}')
        validated = {}
        for key in required_keys:
            validated[key] = float(weights[key])
        for key, default in optional_defaults.items():
            validated[key] = float(weights.get(key, default))
        return validated

    def set_reward_weights(self, weights):
        validated = self._validate_reward_weights(weights)
        self.reward_weights.update(validated)

    def load_reward_weights(self, config_path):
        with open(config_path, 'r', encoding='utf-8') as config_file:
            cfg = json.load(config_file)

        # 提取 reward 权重并设置
        reward_keys = set(self.reward_weights.keys())
        reward_subset = {k: cfg[k] for k in cfg if k in reward_keys}
        if reward_subset:
            self.set_reward_weights(reward_subset)

        # 将 reward 权重下发到 MATLAB workspace，便于混合仿真一致性
        if hasattr(self, 'eng') and self.eng is not None:
            for key, value in self.reward_weights.items():
                try:
                    self.eng.workspace[f'reward_{key}'] = float(value)
                except Exception:
                    pass

        # 可选：加载队列/仿真相关参数（如果提供）
        if 'enable_queue_model' in cfg:
            self.enable_queue_model = bool(cfg['enable_queue_model'])
        if 'queue_capacity' in cfg:
            try:
                self.domain_queue_capacity = int(cfg['queue_capacity'])
            except Exception:
                self.domain_queue_capacity = int(self.domain_queue_capacity)
            # 重置队列状态以应用新容量
            self.domain_queues = np.zeros(self.num_domains, dtype=np.float32)
        if 'queue_service_rate' in cfg:
            v = cfg['queue_service_rate']
            if hasattr(v, '__iter__'):
                try:
                    arr = np.array(v, dtype=np.float32)
                    if arr.size == self.num_domains:
                        self.domain_service_rate = arr
                except Exception:
                    pass
            else:
                try:
                    self.domain_service_rate = np.full(self.num_domains, float(v), dtype=np.float32)
                except Exception:
                    pass
        if 'queue_delay_weight' in cfg:
            try:
                self.queue_delay_weight = float(cfg['queue_delay_weight'])
            except Exception:
                pass
        if 'queue_drop_penalty' in cfg:
            try:
                self.queue_drop_penalty = float(cfg['queue_drop_penalty'])
            except Exception:
                pass
        if 'queue_time_per_step_ms' in cfg:
            try:
                self.queue_time_per_step_ms = float(cfg['queue_time_per_step_ms'])
            except Exception:
                pass
        # 也可接受 reward-normalization 的调整参数
        if 'reward_norm_min_count' in cfg:
            try:
                self.reward_norm_min_count = int(cfg['reward_norm_min_count'])
            except Exception:
                pass
        if 'reward_ema_alpha' in cfg:
            try:
                self.reward_ema_alpha = float(cfg['reward_ema_alpha'])
            except Exception:
                pass
        if 'reward_norm_clip' in cfg:
            try:
                self.reward_norm_clip = float(cfg['reward_norm_clip'])
            except Exception:
                pass

    def save_reward_weights(self, config_path):
        with open(config_path, 'w', encoding='utf-8') as config_file:
            json.dump(self.reward_weights, config_file, indent=2, ensure_ascii=False)

    def get_reward_weights(self):
        return dict(self.reward_weights)

    def _workspace_get(self, key, default=None):
        try:
            return self.eng.workspace[key]
        except Exception:
            return default

    def get_action_mask(self):
        # 默认所有 action 都可用；如果有更严格可行域规则，可在这里扩展
        return [True] * self.num_domains, False

    def step(self, action):
        prev_domain = int(self.current_domain)
        try:
            next_state_struct, reward, done = self.eng.rl_step(int(action), nargout=3)
        except Exception as ex:
            raise RuntimeError(f'MATLAB rl_step 调用失败: {ex}') from ex

        self._validate_state_struct(next_state_struct)

        self.eng.workspace['load_matrix'] = next_state_struct.get('load_matrix', self._workspace_get('load_matrix'))
        self.eng.workspace['smooth_max_load'] = next_state_struct.get('smooth_max_load', self._workspace_get('smooth_max_load', 0.0))
        self.eng.workspace['current_satellite'] = next_state_struct.get('current_satellite', self._workspace_get('current_satellite'))
        self.eng.workspace['current_domain'] = next_state_struct.get('current_domain', self._workspace_get('current_domain'))

        self.state = self._struct_to_array(next_state_struct)
        self.current_domain = int(next_state_struct.get('current_domain', self.current_domain))

        step_delay = float(next_state_struct.get('end_to_end_delay', 0.0))
        self.episode_total_delay += step_delay

        dst_domain = int(next_state_struct.get('dst_domain', 1))
        if done and self.current_domain == dst_domain:
            self.episode_success = True

        weights = self.reward_weights
        raw_reward = float(reward)
        next_domain = self.current_domain

        # 已由 MATLAB 端完成 success/failure/progress 统一计算，避免重复叠加

        # 先对原始 reward 做宽尺度裁剪，避免极值影响统计量计算
        raw_reward = float(np.clip(raw_reward, -20.0, 20.0))

        # 如果启用队列模型，更新域队列并把额外的排队/处理延时计入 reward 与延时
        if self.enable_queue_model:
            # advance simulation time by one step
            self.sim_time_steps += 1

            # 每步先处理服务（每域按服务率出队）
            for d in range(self.num_domains):
                proc = self.domain_service_rate[d]
                if proc > 0:
                    self.domain_queues[d] = max(0.0, self.domain_queues[d] - proc)

            # 到达域索引（将 MATLAB 返回的 1-based 转为 0-based）
            try:
                arrival_domain = int(next_state_struct.get('current_domain', self.current_domain)) - 1
            except Exception:
                arrival_domain = int(self.current_domain) - 1
            if arrival_domain < 0 or arrival_domain >= self.num_domains:
                arrival_domain = max(0, min(self.num_domains - 1, arrival_domain))

            # 入队
            self.domain_queues[arrival_domain] += 1.0

            # 检查是否溢出（丢包）
            dropped = False
            if self.domain_queues[arrival_domain] > self.domain_queue_capacity:
                dropped = True

            if dropped:
                # 丢包惩罚（同时不再进一步计算排队延时）
                extra_delay_ms = self.queue_time_per_step_ms * (self.domain_queue_capacity / max(1.0, self.domain_service_rate[arrival_domain]))
                raw_reward -= self.queue_drop_penalty
            else:
                # 估计等待位置（队列位置，从1开始）
                pos = self.domain_queues[arrival_domain]
                wait_steps = max(0.0, (pos - 1.0) / max(1e-6, self.domain_service_rate[arrival_domain]))
                proc_steps = 1.0 / max(1e-6, self.domain_service_rate[arrival_domain])
                delay_steps = wait_steps + proc_steps
                extra_delay_ms = delay_steps * self.queue_time_per_step_ms
                # 按权重惩罚 reward
                raw_reward -= (self.queue_delay_weight * (delay_steps))

            # 把额外延时计入环境的统计延时（以 ms 为单位）
            try:
                self.episode_total_delay += float(extra_delay_ms)
            except Exception:
                pass

        # 在线统计：更新 Welford（精确）与 EMA（鲁棒、平滑）并计算归一化 reward
        if self.reward_normalize:
            # 更新 Welford 统计量（保留以便诊断）
            self.reward_running_count += 1
            delta = raw_reward - self.reward_running_mean
            self.reward_running_mean += delta / self.reward_running_count
            delta2 = raw_reward - self.reward_running_mean
            self.reward_running_M2 += delta * delta2

            # 更新 EMA 统计，EMA 对单步异常值不那么敏感
            if self.reward_running_count == 1:
                self.reward_ema_mean = raw_reward
                self.reward_ema_var = 1.0
            else:
                d_ema = raw_reward - self.reward_ema_mean
                self.reward_ema_mean += self.reward_ema_alpha * d_ema
                self.reward_ema_var = (1.0 - self.reward_ema_alpha) * self.reward_ema_var + self.reward_ema_alpha * (d_ema * d_ema)

            # warm-up：样本不足时先不做归一化，避免初期统计不稳定
            if self.reward_running_count < self.reward_norm_min_count:
                final_reward = raw_reward
            else:
                std = np.sqrt(max(self.reward_ema_var, self.reward_running_M2 / max(1, (self.reward_running_count - 1))))
                normalized = (raw_reward - self.reward_ema_mean) / (std + self.reward_norm_eps)
                final_reward = float(np.clip(normalized * self.reward_norm_scale, -self.reward_norm_clip, self.reward_norm_clip))
        else:
            final_reward = raw_reward

        done = bool(done)
        return self.state, final_reward, done, False, {}

    def _validate_state_struct(self, struct):
        required_keys = ['src_domain', 'dst_domain', 'current_domain', 'load_matrix']
        for key in required_keys:
            if key not in struct:
                raise ValueError(f'rl_step returned struct missing key: {key}')

    def _struct_to_array(self, struct):
        domain_count = float(max(1, self.num_domains - 1))
        src_dom = (int(struct.get('src_domain', 1)) - 1) / domain_count
        dst_dom = (int(struct.get('dst_domain', 1)) - 1) / domain_count
        curr_dom = (int(struct.get('current_domain', 1)) - 1) / domain_count

        distances = self.domain_graph[int(round(curr_dom * domain_count)), :].astype(np.float32)
        max_dist = np.max(distances)
        if max_dist > 0:
            distances = distances / max_dist

        candidate_mask = (self.domain_graph[int(round(curr_dom * domain_count)), :] > 0).astype(np.float32)

        load_mat = struct.get('load_matrix', np.zeros((self.num_domains, self.num_domains), dtype=np.float32))
        try:
            load = np.array(load_mat, dtype=np.float32)
        except Exception:
            load = np.zeros((self.num_domains, self.num_domains), dtype=np.float32)
        if load.ndim == 0:
            load = np.zeros((self.num_domains, self.num_domains), dtype=np.float32)

        load_flat = load.flatten()
        max_load = np.max(load_flat) if load_flat.size > 0 else 1.0
        mean_load = np.mean(load_flat) if load_flat.size > 0 else 0.0
        var_load = np.var(load_flat) if load_flat.size > 0 else 0.0
        avg_load = mean_load
        load_flat_norm = load_flat / (max_load + 1e-6)

        inter_domain_loads = load[np.triu_indices(self.num_domains, k=1)]
        inter_load_mean = np.mean(inter_domain_loads) if inter_domain_loads.size > 0 else 0.0
        inter_load_max = np.max(inter_domain_loads) if inter_domain_loads.size > 0 else 0.0
        inter_load_mean_norm = inter_load_mean / (max_load + 1e-6)
        inter_load_max_norm = inter_load_max / (max_load + 1e-6)

        load_balance = np.std(load_flat) / (max_load + 1e-6)
        load_variance = var_load  # 新增负载方差特征
        max_load_feature = max_load  # 新增最大负载特征

        # 队列特征（若未启用则返回 0）
        if getattr(self, 'enable_queue_model', False):
            mean_queue = float(np.mean(self.domain_queues))
            max_queue = float(np.max(self.domain_queues))
            curr_idx = int(round(curr_dom * domain_count))
            curr_idx = max(0, min(self.num_domains - 1, curr_idx))
            curr_queue = float(self.domain_queues[curr_idx])
        else:
            mean_queue = 0.0
            max_queue = 0.0
            curr_queue = 0.0

        mean_queue_norm = mean_queue / (self.domain_queue_capacity + 1e-6) if getattr(self, 'domain_queue_capacity', 1) > 0 else 0.0
        max_queue_norm = max_queue / (self.domain_queue_capacity + 1e-6) if getattr(self, 'domain_queue_capacity', 1) > 0 else 0.0
        curr_domain_queue_norm = curr_queue / (self.domain_queue_capacity + 1e-6) if getattr(self, 'domain_queue_capacity', 1) > 0 else 0.0

        end_to_end_delay_norm = 0.0
        return np.concatenate([
            [src_dom, dst_dom, curr_dom],
            distances,
            candidate_mask,
            load_flat_norm,
            [mean_load, var_load, inter_load_mean_norm, inter_load_max_norm, load_balance, avg_load, float(struct.get('smooth_max_load', 0.0)) / (max_load + 1e-6), end_to_end_delay_norm, load_variance, max_load_feature, mean_queue_norm, max_queue_norm, curr_domain_queue_norm]  # 新增特征
        ], axis=0)

