function [path, distance, overhead] = cross_domain_routing(src, dst, adjacency, domains, params, domain_cache, load_matrix)
    % 跨域路由 - 使用预计算缓存
    % 输入：
    %   src, dst, adjacency, domains, params, domain_cache
    %   load_matrix (可选) - 当前负载矩阵，用于动态权重调整
    overhead = 0;

    src_domain = domains.domain_assignment(src);
    dst_domain = domains.domain_assignment(dst);

    if src_domain == dst_domain
        [path, distance] = dijkstra_algorithm(src, dst, adjacency);
        overhead = 1;
        return;
    end

    % 提取缓存
    domain_nodes = domain_cache.domain_nodes;
    intra_dists = domain_cache.intra_dists;
    inter_links = domain_cache.inter_links;
    k_shortest_paths = domain_cache.k_shortest_paths;

    % 动态权重系数（可调）
    alpha = params.cross_domain_params.load_balancing_factor;  % 已在 load_config_params 中定义

    % 构建域级图（考虑负载）
    num_domains = domains.num_domains;
    domain_graph = zeros(num_domains, num_domains);

    % 计算最大负载，用于归一化（避免除零）
    if nargin >= 7 && ~isempty(load_matrix)
        max_load = max(load_matrix(:));
        if max_load == 0
            max_load = 1;
        end
    end

    % 预先计算域间最短距离和负载
    for d1 = 1:num_domains
        nodes_d1 = domain_nodes{d1};
        for d2 = d1+1:num_domains
            nodes_d2 = domain_nodes{d2};
            min_dist = Inf;
            min_load = 0;  % 记录最小距离对应的负载（暂未用）
            for i = 1:length(nodes_d1)
                for j = 1:length(nodes_d2)
                    if adjacency(nodes_d1(i), nodes_d2(j)) > 0
                        dist_ij = adjacency(nodes_d1(i), nodes_d2(j));
                        if dist_ij < min_dist
                            min_dist = dist_ij;
                        end
                    end
                end
            end
            if ~isinf(min_dist)
                % 如果有负载矩阵，计算归一化负载
                if nargin >= 7 && ~isempty(load_matrix)
                    norm_load = load_matrix(d1, d2) / max_load;
                    weight = min_dist * (1 + alpha * norm_load);
                else
                    weight = min_dist;
                end
                domain_graph(d1, d2) = weight;
                domain_graph(d2, d1) = weight;
            end
        end
    end

    % 获取候选域路径
    candidate_paths = k_shortest_paths{src_domain, dst_domain};
    if isempty(candidate_paths)
        path = []; distance = Inf; return;
    end

    best_path = [];
    best_dist = Inf;
    for p_idx = 1:length(candidate_paths)
        domain_path = candidate_paths{p_idx};
        [full_path, total_distance] = try_domain_path(domain_path, src, dst, adjacency, domains, ...
            domain_nodes, inter_links, intra_dists);
        if ~isempty(full_path) && total_distance < best_dist
            best_path = full_path;
            best_dist = total_distance;
        end
    end

    if isempty(best_path)
        path = []; distance = Inf;
    else
        path = best_path;
        distance = best_dist;
    end
    overhead = length(candidate_paths);
end
% try_domain_path 函数（与原版基本相同，但已使用预计算矩阵）
function [full_path, total_distance] = try_domain_path(domain_path, src, dst, adjacency, domains, ...
    domain_nodes, inter_links, intra_dists)
    L = length(domain_path);
    % 逆向动态规划
    dp = cell(L,1);
    prev_entry = cell(L,1);
    
    % 最后阶段
    last_domain = domain_path(L);
    nodes_last = domain_nodes{last_domain};
    dp{L} = inf(length(nodes_last),1);
    prev_entry{L} = cell(length(nodes_last),1);
    for i = 1:length(nodes_last)
        node = nodes_last(i);
        [~, d] = dijkstra_algorithm(node, dst, adjacency);  % 仍需一次Dijkstra（到目的）
        dp{L}(i) = d;
    end
    
    % 逆向递推
    for stage = L-1:-1:1
        curr_domain = domain_path(stage);
        next_domain = domain_path(stage+1);
        nodes_curr = domain_nodes{curr_domain};
        nodes_next = domain_nodes{next_domain};
        n_curr = length(nodes_curr);
        n_next = length(nodes_next);
        
        dp{stage} = inf(n_curr,1);
        prev_entry{stage} = cell(n_curr,1);
        
        links = inter_links{curr_domain, next_domain};
        if isempty(links)
            full_path = []; total_distance = Inf; return;
        end
        
        for i = 1:n_curr
            u = nodes_curr(i);
            best_dist = inf;
            best_exit = -1; best_entry = -1; best_next_node = -1;
            for k = 1:size(links,1)
                exit_node = links(k,1);
                entry_node = links(k,2);
                isl_dist = links(k,3);
                
                if domains.domain_assignment(exit_node) ~= curr_domain, continue; end
                
                [~, i_u] = ismember(u, nodes_curr);
                [~, i_exit] = ismember(exit_node, nodes_curr);
                intra_dist = intra_dists{curr_domain}(i_u, i_exit);
                if isinf(intra_dist), continue; end
                
                [~, i_entry] = ismember(entry_node, nodes_next);
                if i_entry == 0, continue; end
                
                next_dist = dp{stage+1}(i_entry);
                if isinf(next_dist), continue; end
                
                total = intra_dist + isl_dist + next_dist;
                if total < best_dist
                    best_dist = total;
                    best_exit = exit_node;
                    best_entry = entry_node;
                    best_next_node = entry_node;
                end
            end
            if ~isinf(best_dist)
                dp{stage}(i) = best_dist;
                prev_entry{stage}{i} = [best_next_node, best_exit, best_entry];
            end
        end
    end
    
    % 正向回溯
    src_domain_nodes = domain_nodes{domain_path(1)};
    [~, src_idx] = ismember(src, src_domain_nodes);
    if src_idx == 0 || isinf(dp{1}(src_idx))
        full_path = []; total_distance = Inf; return;
    end
    total_distance = dp{1}(src_idx);
    
    full_path = src;
    current_node = src;
    for stage = 1:L-1
        nodes_curr = domain_nodes{domain_path(stage)};
        [~, i_curr] = ismember(current_node, nodes_curr);
        next_info = prev_entry{stage}{i_curr};
        if isempty(next_info)
            full_path = []; total_distance = Inf; return;
        end
        exit_node = next_info(2);
        entry_node = next_info(3);
        
        if current_node ~= exit_node
            [seg_path, ~] = dijkstra_algorithm(current_node, exit_node, adjacency);
            if isempty(seg_path)
                full_path = []; total_distance = Inf; return;
            end
            full_path = [full_path(1:end-1), seg_path];
        end
        full_path = [full_path, entry_node];
        current_node = entry_node;
    end
    
    if current_node ~= dst
        [final_seg, ~] = dijkstra_algorithm(current_node, dst, adjacency);
        if isempty(final_seg)
            full_path = []; total_distance = Inf; return;
        end
        full_path = [full_path(1:end-1), final_seg];
    end
end