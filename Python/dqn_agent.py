import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import random
import math
from collections import deque


class NoisyLinear(nn.Module):
    def __init__(self, in_features, out_features, sigma_zero=0.5, use_noisy=True):
        super(NoisyLinear, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.use_noisy = use_noisy
        self.weight_mu = nn.Parameter(torch.empty(out_features, in_features))
        self.weight_sigma = nn.Parameter(torch.empty(out_features, in_features))
        self.register_buffer('weight_epsilon', torch.zeros(out_features, in_features))
        self.bias_mu = nn.Parameter(torch.empty(out_features))
        self.bias_sigma = nn.Parameter(torch.empty(out_features))
        self.register_buffer('bias_epsilon', torch.zeros(out_features))
        self.sigma_zero = sigma_zero
        self.reset_parameters()
        self.reset_noise()

    def reset_parameters(self):
        mu_range = 1.0 / math.sqrt(self.in_features)
        self.weight_mu.data.uniform_(-mu_range, mu_range)
        self.bias_mu.data.uniform_(-mu_range, mu_range)
        sigma_init = self.sigma_zero / math.sqrt(self.in_features)
        self.weight_sigma.data.fill_(sigma_init)
        self.bias_sigma.data.fill_(sigma_init)

    def _scale_noise(self, size):
        x = torch.randn(size)
        return x.sign().mul_(x.abs().sqrt_())

    def reset_noise(self):
        if self.use_noisy:
            epsilon_in = self._scale_noise(self.in_features)
            epsilon_out = self._scale_noise(self.out_features)
            self.weight_epsilon.copy_(epsilon_out.ger(epsilon_in))
            self.bias_epsilon.copy_(epsilon_out)
        else:
            self.weight_epsilon.zero_()
            self.bias_epsilon.zero_()

    def forward(self, input):
        if self.use_noisy:
            weight = self.weight_mu + self.weight_sigma * self.weight_epsilon
            bias = self.bias_mu + self.bias_sigma * self.bias_epsilon
        else:
            weight = self.weight_mu
            bias = self.bias_mu
        return F.linear(input, weight, bias)

class DuelingDQN(nn.Module):
    def __init__(self, state_dim, action_dim, use_noisy=False):
        super(DuelingDQN, self).__init__()
        self.use_noisy = use_noisy
        #  LayerNorm
        self.feature = nn.Sequential(
            NoisyLinear(state_dim, 512, use_noisy=use_noisy),
            nn.LayerNorm(512),
            nn.LeakyReLU(0.01),
            NoisyLinear(512, 512, use_noisy=use_noisy),
            nn.LayerNorm(512),
            nn.LeakyReLU(0.01)
        )
        
        self.value = nn.Sequential(
            NoisyLinear(512, 256, use_noisy=use_noisy),
            nn.LeakyReLU(0.01),
            NoisyLinear(256, 1, use_noisy=use_noisy)
        )
        
        self.advantage = nn.Sequential(
            NoisyLinear(512, 256, use_noisy=use_noisy),
            nn.LeakyReLU(0.01),
            NoisyLinear(256, action_dim, use_noisy=use_noisy)
        )

    def forward(self, x):
        features = self.feature(x)
        value = self.value(features)
        advantage = self.advantage(features)
        q = value + advantage - advantage.mean(dim=1, keepdim=True)
        return q

    def reset_noise(self):
        for module in self.modules():
            if isinstance(module, NoisyLinear):
                module.reset_noise()

class DQN(nn.Module):
    def __init__(self, state_dim, action_dim):
        super(DQN, self).__init__()
        self.fc = nn.Sequential(
            nn.Linear(state_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Linear(256, action_dim)
        )
    
    def forward(self, x):
        return self.fc(x)

import numpy as np

class SumTree:
    def __init__(self, capacity):
        self.capacity = capacity
        self.tree = np.zeros(2 * capacity - 1) 
        self.data = np.zeros(capacity, dtype=object)
        self.write = 0
        self.n_entries = 0

    def _propagate(self, idx, change):
        parent = (idx - 1) // 2
        self.tree[parent] += change
        if parent != 0:
            self._propagate(parent, change)

    def _retrieve(self, idx, s):
        left = 2 * idx + 1
        right = left + 1
        if left >= len(self.tree):
            return idx
        if s <= self.tree[left]:
            return self._retrieve(left, s)
        else:
            return self._retrieve(right, s - self.tree[left])

    def total(self):
        return self.tree[0]

    def add(self, p, data):
        idx = self.write + self.capacity - 1
        self.data[self.write] = data
        self.update(idx, p)
        self.write = (self.write + 1) % self.capacity
        self.n_entries = min(self.n_entries + 1, self.capacity)

    def update(self, idx, p):
        change = p - self.tree[idx]
        self.tree[idx] = p
        self._propagate(idx, change)

    def get(self, s):
        idx = self._retrieve(0, s)
        data_idx = idx - self.capacity + 1
        return idx, self.tree[idx], self.data[data_idx]

class ReplayBuffer:
    def __init__(self, capacity, alpha=0.6, beta=0.4, beta_increment=0.005):
        self.tree = SumTree(capacity)
        self.alpha = alpha          
        self.beta = beta           
        self.beta_increment = beta_increment
        self.max_priority = 1.0     
        self.capacity = capacity

    def push(self, state, action, reward, next_state, done):
        p = self.max_priority ** self.alpha
        self.tree.add(p, (state, action, reward, next_state, done))

    def sample(self, batch_size):
        batch = []
        idxs = []
        segment = self.tree.total() / batch_size
        priorities = []

        for i in range(batch_size):
            a = segment * i
            b = segment * (i + 1)
            s = np.random.uniform(a, b)
            idx, p, data = self.tree.get(s)
            batch.append(data)
            idxs.append(idx)
            priorities.append(p)

        
        total_p = self.tree.total()
        probs = np.array(priorities) / total_p
        is_weights = (self.capacity * probs) ** (-self.beta)
        is_weights /= is_weights.max()  
        self.beta = min(1.0, self.beta + self.beta_increment)

       
        states, actions, rewards, next_states, dones = zip(*batch)
        states = np.array(states)
        actions = np.array(actions)
        rewards = np.array(rewards, dtype=np.float32)
        next_states = np.array(next_states)
        dones = np.array(dones, dtype=np.float32)

        return states, actions, rewards, next_states, dones, idxs, is_weights

    def update_priority(self, idxs, td_errors):
       
        for idx, td in zip(idxs, td_errors):
            p = (abs(td) + 1e-6) ** self.alpha
            self.tree.update(idx, p)
            self.max_priority = max(self.max_priority, p)
    def __len__(self):
        return self.tree.n_entries   

class Agent:
    def __init__(self, state_dim, action_dim, lr=0.0001, gamma=0.99,
                 epsilon_start=1.0, epsilon_end=0.01, epsilon_decay=0.993,
                 memory_size=150000, batch_size=64, target_update=50,
                 tau=0.001, n_step=3, use_noisy=False):
        self.action_dim = action_dim
        self.gamma = gamma
        self.batch_size = batch_size
        self.target_update = target_update
        self.tau = tau
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.n_step = n_step
        self.use_noisy = use_noisy
        self.learn_steps = 0
        
        self.policy_net = DuelingDQN(state_dim, action_dim, use_noisy=use_noisy).to(self.device)
        self.target_net = DuelingDQN(state_dim, action_dim, use_noisy=use_noisy).to(self.device)
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.optimizer = optim.Adam(self.policy_net.parameters(), lr=lr, weight_decay=1e-4)
        self.memory = ReplayBuffer(memory_size)
        
        self.epsilon = epsilon_start
        self.epsilon_end = epsilon_end
        self.epsilon_decay = epsilon_decay
        self.steps_done = 0
    
    def select_action(self, state, action_mask=None):
        if self.use_noisy:
            self.policy_net.reset_noise()
        if not self.use_noisy and random.random() < self.epsilon:
            if action_mask is not None:
                valid_actions = [i for i, m in enumerate(action_mask) if m]
                if valid_actions:
                    return random.choice(valid_actions)
            return random.randint(0, self.action_dim - 1)
        with torch.no_grad():
            state_t = torch.FloatTensor(state).unsqueeze(0).to(self.device)
            q_values = self.policy_net(state_t).squeeze(0)
            if action_mask is not None:
                mask = torch.tensor(action_mask, dtype=torch.bool, device=self.device)
                q_values = q_values.masked_fill(~mask, -1e9)
            return q_values.argmax().item()
    
    def update_epsilon(self):
        self.epsilon = max(self.epsilon_end, self.epsilon * self.epsilon_decay)
    
    def store_transition(self, state, action, reward, next_state, done):
        self.memory.push(state, action, reward, next_state, done)
    
    def learn(self):
        if len(self.memory) < self.batch_size:
            return
        states, actions, rewards, next_states, dones, idxs, is_weights = self.memory.sample(self.batch_size)

        states = torch.FloatTensor(states).to(self.device)
        actions = torch.LongTensor(actions).unsqueeze(1).to(self.device)
        rewards = torch.FloatTensor(rewards).to(self.device)
        next_states = torch.FloatTensor(next_states).to(self.device)
        dones = torch.FloatTensor(dones).to(self.device)
        is_weights = torch.FloatTensor(is_weights).to(self.device)

        # Double DQN
        with torch.no_grad():
            next_actions = self.policy_net(next_states).argmax(1, keepdim=True)
            next_q = self.target_net(next_states).gather(1, next_actions).squeeze(1)
            target_q = rewards + self.gamma * next_q * (1 - dones)

        current_q = self.policy_net(states).gather(1, actions).squeeze(1)
        td_errors = target_q - current_q
        loss = (is_weights * (td_errors ** 2)).mean()

        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.policy_net.parameters(), 5.0)
        self.optimizer.step()

        self.memory.update_priority(idxs, td_errors.detach().cpu().numpy())

        self.learn_steps += 1

        # 仅使用软更新以避免硬更新与软更新冲突引发目标网络震荡
        self.soft_update_target_network()
        if self.use_noisy:
            self.target_net.reset_noise()

        return loss.item()

    def soft_update_target_network(self):
        for target_param, policy_param in zip(self.target_net.parameters(), self.policy_net.parameters()):
            target_param.data.copy_(self.tau * policy_param.data + (1.0 - self.tau) * target_param.data)

    
    def reset_noise(self):
        if self.use_noisy:
            self.policy_net.reset_noise()
            self.target_net.reset_noise()

    def update_target_network(self):
        self.target_net.load_state_dict(self.policy_net.state_dict())
