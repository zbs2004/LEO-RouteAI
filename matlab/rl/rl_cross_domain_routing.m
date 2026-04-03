function [path, distance, overhead] = rl_cross_domain_routing(src, dst, adjacency, domains, params, domain_cache, load_matrix, smooth_max_load)
    % 使用训练好的 RL 模型选择第一跳域，直接评估 RL 决策效果，不回退到经典跨域路由
    overhead = 1;

    if nargin < 8
        smooth_max_load = 0;
    end

    % 从 domain_cache 中提取必要数据
    domain_nodes = domain_cache.domain_nodes;
    intra_dists = domain_cache.intra_dists;
    inter_links = domain_cache.inter_links;

    % 初始当前卫星就是源卫星
    current_sat = src;
    current_domain = domains.domain_assignment(current_sat);

    % 获取当前域到各域的连通性掩码（从 domain_graph 提取）
    % 获取当前域到各域的连通性掩码（从 domain_cache 中提取 domain_graph）
    domain_graph = domain_cache.domain_graph;   % 添加这一行
    action_mask = domain_graph(current_domain, :) > 0;   % 逻辑向量，1 表示有链路
    % 构建状态向量时需传入 domain_graph
    state_vec = build_state_vector(src, dst, current_sat, load_matrix, domains, domain_nodes, domain_graph, smooth_max_load);

    % 调用本地推理函数
    %action = dqn_predict(state_vec);
    %next_domain = action + 1;    % MATLAB 1-based
    
    action = dqn_predict(state_vec, action_mask);
    %fprintf('dqn_predict 返回 action = %s\n', mat2str(action));
    num_domains = size(domain_graph, 1);
    if isempty(action) || ~isscalar(action) || isnan(action) || action < 0 || action >= num_domains
        fprintf('错误：无效的动作 action\n');
        path = []; distance = Inf; return;
    end
    next_domain = action + 1;
    %fprintf('next_domain = %d\n', next_domain);

    % 第一步：从当前卫星到 next_domain 的入口节点
    dst_domain = domains.domain_assignment(dst);
    [first_seg, first_dist] = route_to_domain(current_sat, current_domain, next_domain, ...
        adjacency, domains, domain_nodes, intra_dists, inter_links, domain_graph, dst_domain, load_matrix, params.cross_domain_params.load_balancing_factor);

    if isempty(first_seg)
        path = [];
        distance = Inf;
        return;
    end

    % 更新当前卫星和当前域
    current_sat = first_seg(end);
    current_domain = next_domain;

    % 如果已经到达目的域，只需最后一段域内路由
    if current_domain == domains.domain_assignment(dst)
        [final_seg, final_dist] = dijkstra_algorithm(current_sat, dst, adjacency);
        if isempty(final_seg)
            path = [];
            distance = Inf;
            return;
        end
        path = [first_seg(1:end-1), final_seg];
        distance = first_dist + final_dist;
    else
        % 否则，调用原有的跨域路由函数（使用预计算缓存）完成剩余路径
        [rest_path, rest_dist, ~] = cross_domain_routing(current_sat, dst, adjacency, domains, params, domain_cache, load_matrix);
        if isempty(rest_path)
            path = [];
            distance = Inf;
            return;
        end
        path = [first_seg(1:end-1), rest_path];
        distance = first_dist + rest_dist;
    end

end