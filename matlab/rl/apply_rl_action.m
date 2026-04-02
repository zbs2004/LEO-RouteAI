function [next_state, reward, done] = apply_rl_action(action, src, dst, current_satellite, ...
    adjacency, domains, load_matrix, domain_graph, domain_nodes, intra_dists, intra_hops, inter_links, ...
    load_penalty_coef, smooth_max_load, alpha)
    
    current_domain = domains.domain_assignment(current_satellite);
    next_domain = action + 1;  % 转为 MATLAB 1-based

    function val = get_reward_weight(var_name, default_val)
        if evalin('base', ['exist(''' var_name ''',''var'')'])
            val = evalin('base', var_name);
        else
            val = default_val;
        end
    end

    delay_weight = get_reward_weight('reward_delay', 2.4);
    congestion_weight = get_reward_weight('reward_congestion', 0.8);
    hop_weight = get_reward_weight('reward_hop', 1.5);
    stability_weight = get_reward_weight('reward_stability', 0.8);
    e2e_weight = get_reward_weight('reward_e2e', 1.5);
    balance_weight = get_reward_weight('reward_balance', 0.8);
    success_reward = get_reward_weight('reward_success', 8.0);
    failure_penalty = get_reward_weight('reward_failure', 50.0);
    step_penalty = get_reward_weight('reward_step', 0.05);
    
    function d = shortest_domain_distance(start_dom, end_dom, graph)
        if start_dom == end_dom
            d = 0;
            return;
        end
        n = size(graph, 1);
        dist = inf(1, n);
        visited = false(1, n);
        dist(start_dom) = 0;
        graph2 = graph;
        graph2(graph2 == 0) = inf;
        while true
            temp = dist;
            temp(visited) = inf;
            [minval, u] = min(temp);
            if isinf(minval)
                break;
            end
            visited(u) = true;
            if u == end_dom
                break;
            end
            for v = 1:n
                if ~visited(v) && graph2(u,v) < inf
                    dist(v) = min(dist(v), dist(u) + graph2(u,v));
                end
            end
        end
        d = dist(end_dom);
    end

    % 辅助函数：构造失败时的 next_state
    function fail_state = make_fail_state()
        fail_state.src_domain = domains.domain_assignment(src);
        fail_state.dst_domain = domains.domain_assignment(dst);
        fail_state.current_domain = current_domain;
        fail_state.current_satellite = current_satellite;
        fail_state.load_matrix = load_matrix;
        fail_state.smooth_max_load = smooth_max_load;
        fail_state.domain_graph = domain_graph;
    end
    
    % 合法性检查
    if domain_graph(current_domain, next_domain) == 0
        reward = -failure_penalty;
        done = false;
        next_state = make_fail_state();
        return;
    end
    
    % 获取当前域到下一域的所有直接链路
    links = inter_links{current_domain, next_domain};
    if isempty(links)
        reward = -failure_penalty;
        done = true;
        next_state = make_fail_state();
        return;
    end
    
    % 选择最优出口/入口
    nodes_curr = domain_nodes{current_domain};
    [~, i_curr] = ismember(current_satellite, nodes_curr);
    if i_curr == 0
        reward = -failure_penalty;
        done = true;
        next_state = make_fail_state();
        return;
    end
    
    best_intra = inf;
    best_exit = -1;
    best_entry = -1;
    best_isl = 0;
    for k = 1:size(links,1)
        exit_node = links(k,1);
        entry_node = links(k,2);
        isl_dist = links(k,3);
        [~, i_exit] = ismember(exit_node, nodes_curr);
        if i_exit == 0
            continue;
        end
        intra = intra_dists{current_domain}(i_curr, i_exit);
        if isinf(intra)
            continue;
        end
        if intra < best_intra
            best_intra = intra;
            best_isl = isl_dist;
            best_exit = exit_node;
            best_entry = entry_node;
        end
    end
    
    if best_exit == -1
        reward = -failure_penalty;
        done = true;
        next_state = make_fail_state();
        return;
    end
    
    % 计算时延
    light_speed = 3e8;
    propagation_delay = (best_intra + best_isl) / light_speed * 1000;
    delay = propagation_delay + 0.1;
    delay_norm = min(delay / 50.0, 1.0);  % 归一化到 [0,1]

    % 负载惩罚：使用当前边上负载，同样归一化
    fixed_max_load = 100;
    congestion = min(load_matrix(current_domain, next_domain) / fixed_max_load, 1);
    load_penalty = -load_penalty_coef * congestion_weight * congestion;

    % 最短跳数惩罚
    [~, best_hops] = route_to_domain_stats(current_satellite, current_domain, next_domain, ...
        adjacency, domains, domain_nodes, intra_dists, intra_hops, inter_links);
    if isinf(best_hops)
        best_hops = 1;
    end
    max_hops = 20;
    hop_cost = hop_weight * min(best_hops, max_hops) / max_hops;

    % 稳定性奖励：选择连通度更高的下一个域
    domain_degrees = sum(domain_graph > 0, 2);
    stability_bonus = 0;
    if max(domain_degrees) > 0
        stability_bonus = stability_weight * (domain_degrees(next_domain) / max(domain_degrees));
    end

    % 端到端距离估计奖励
    remaining_dist = shortest_domain_distance(next_domain, domains.domain_assignment(dst), domain_graph);
    max_domain_dist = max(domain_graph(domain_graph > 0));
    if isempty(max_domain_dist) || max_domain_dist == 0
        max_domain_dist = 1;
    end
    if isinf(remaining_dist)
        e2e_cost = e2e_weight;
    else
        e2e_cost = e2e_weight * (remaining_dist / max_domain_dist);
    end

    % 基础奖励（越小越好）
    reward = -1.4 * delay_weight * delay_norm + load_penalty - 1.2 * hop_cost + stability_bonus - e2e_cost;

    % 目标到达奖励和拥塞奖励机制
    if next_domain == domains.domain_assignment(dst)
        done = true;
        reward = reward + success_reward;  % 强烈奖励终点到达
    else
        done = false;
        reward = reward + balance_weight * (1 - congestion);  % 鼓励低拥塞路径
    end

    % 统一步长惩罚
    reward = reward - step_penalty;
    
    % 更新负载矩阵
    load_matrix(current_domain, next_domain) = load_matrix(current_domain, next_domain) + 1;
    load_matrix(next_domain, current_domain) = load_matrix(next_domain, current_domain) + 1;
    
    % 更新平滑最大负载
    new_max = max(load_matrix(:));
    smooth_max_load = alpha * new_max + (1 - alpha) * smooth_max_load;
    
    next_state.end_to_end_delay = delay;
    next_state.current_satellite = best_entry;
    next_state.current_domain = next_domain;
    
    % 9. 构造下一个状态结构体
    next_state.src_domain = domains.domain_assignment(src);
    next_state.dst_domain = domains.domain_assignment(dst);
    next_state.current_domain = next_domain;
    next_state.current_satellite = best_entry;   % 新增
    next_state.load_matrix = load_matrix;
    next_state.smooth_max_load = smooth_max_load; % 新增
    next_state.domain_graph = domain_graph;
end