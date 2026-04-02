import json
import numpy as np
import gymnasium as gym
from gymnasium import spaces

class SatelliteRoutingEnv(gym.Env):
    def __init__(self, eng, num_satellites=66, num_domains=8):
        super().__init__()
        self.eng = eng
        self.num_satellites = num_satellites
        self.num_domains = num_domains
        self.action_space = spaces.Discrete(num_domains)
        self.observation_space = spaces.Box(
            low=0.0,
            high=1.0,
            shape=(3 + num_domains * 2 + num_domains * num_domains + 8,),
            dtype=np.float32
        )
        self.reward_weights = {
            'congestion': 0.7339,
            'hop': 1.499,
            'delay': 2.7253,
            'stability': 0.6445,
            'balance': 0.7989,
            'e2e': 1.4617,
            'success': 0.5563,
            'failure': 0.0047,
            'step': 0.0533,
            'progress': 0.0852
        }
        self.episode_total_delay = 0.0
        self.episode_success = False

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
            weights = json.load(config_file)
        self.set_reward_weights(weights)
        if hasattr(self, 'eng') and self.eng is not None:
            for key, value in self.reward_weights.items():
                self.eng.workspace[f'reward_{key}'] = float(value)

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
        reward = float(reward)
        next_domain = self.current_domain

        if done:
            if self.current_domain == dst_domain:
                reward += weights.get('success', 0.0)
            else:
                reward -= weights.get('failure', 0.0)

        if prev_domain is not None and 1 <= prev_domain <= self.num_domains and 1 <= dst_domain <= self.num_domains:
            old_dist = float(self.domain_graph[prev_domain - 1, dst_domain - 1])
            new_dist = float(self.domain_graph[next_domain - 1, dst_domain - 1])
            progress_delta = max(0.0, (old_dist - new_dist) / (abs(old_dist) + 1e-6))
        else:
            progress_delta = 0.0
        reward += weights.get('progress', 0.3) * progress_delta

        reward = float(np.clip(reward, -20.0, 20.0))
        done = bool(done)
        return self.state, reward, done, False, {}

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

        end_to_end_delay_norm = 0.0
        return np.concatenate([
            [src_dom, dst_dom, curr_dom],
            distances,
            candidate_mask,
            load_flat_norm,
            [mean_load, var_load, inter_load_mean_norm, inter_load_max_norm, load_balance, avg_load, float(struct.get('smooth_max_load', 0.0)) / (max_load + 1e-6), end_to_end_delay_norm]
        ], axis=0)

