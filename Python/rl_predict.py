import sys
import json
import torch
import numpy as np
from dqn_agent import DuelingDQN

def main():
    if len(sys.argv) < 2:
        print("Usage: python rl_predict.py state.json")
        return
    state_file = sys.argv[1]
    with open(state_file, 'r') as f:
        state_dict = json.load(f)
    
    load = np.array(state_dict['load_matrix'])
    domain_graph = np.array(state_dict['domain_graph'])
    num_domains = load.shape[0]
    state_dim = 3 + num_domains * 2 + num_domains * num_domains + 8
    action_dim = num_domains

    domain_count = float(max(1, num_domains - 1))
    src_dom = (state_dict['src_domain'] - 1) / domain_count
    dst_dom = (state_dict['dst_domain'] - 1) / domain_count
    curr_dom = (state_dict['current_domain'] - 1) / domain_count

    curr_idx = int(round(curr_dom * domain_count))
    distances = domain_graph[curr_idx, :].astype(np.float32)
    max_dist = np.max(distances)
    if max_dist > 0:
        distances = distances / max_dist

    candidate_mask = (domain_graph[curr_idx, :] > 0).astype(np.float32)

    load_flat = load.flatten()
    max_load = np.max(load_flat) if load_flat.size > 0 else 1.0
    avg_load = np.mean(load_flat) if load_flat.size > 0 else 0.0
    load_norm = load_flat / (max_load + 1e-6)

    inter_domain_loads = load[np.triu_indices(num_domains, k=1)]
    inter_load_mean = np.mean(inter_domain_loads) if inter_domain_loads.size > 0 else 0.0
    inter_load_max = np.max(inter_domain_loads) if inter_domain_loads.size > 0 else 0.0
    inter_load_mean_norm = inter_load_mean / (max_load + 1e-6)
    inter_load_max_norm = inter_load_max / (max_load + 1e-6)

    connectivity = np.sum(candidate_mask) / num_domains
    load_variance_norm = np.var(load_flat) / ((max_load + 1e-6) ** 2)
    degree_norm = np.sum(candidate_mask) / max(1.0, num_domains - 1)
    avg_load_norm = avg_load / (max_load + 1e-6)
    smooth_max = float(state_dict.get('smooth_max_load', 0.0))
    smooth_max_norm = smooth_max / (max_load + 1e-6)
    load_balance = np.std(load_flat) / (max_load + 1e-6)

    state = np.concatenate([
        [src_dom, dst_dom, curr_dom],
        distances,
        candidate_mask,
        load_norm,
        [connectivity, load_variance_norm, inter_load_mean_norm, inter_load_max_norm,
         degree_norm, avg_load_norm, smooth_max_norm, load_balance]
    ]).astype(np.float32)
    
    # 加载模型（需确保模型文件维度匹配）
    model = DuelingDQN(state_dim=state_dim, action_dim=action_dim)
    model.load_state_dict(torch.load('dqn_model_latest.pth', map_location='cpu'))
    model.eval()
    
    with torch.no_grad():
        state_t = torch.FloatTensor(state).unsqueeze(0)
        q_values = model(state_t).squeeze(0)
    
    action = int(torch.argmax(q_values).item())
    
    result = {'action': action}
    print(json.dumps(result))

if __name__ == '__main__':
    main()