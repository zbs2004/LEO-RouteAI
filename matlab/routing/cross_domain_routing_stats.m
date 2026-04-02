function [distance, hops, overhead] = cross_domain_routing_stats(src, dst, adjacency, domains, params, domain_cache, load_matrix)
    % 仅返回距离和跳数，不返回路径，用于性能统计
    overhead = 0;
    src_domain = domains.domain_assignment(src);
    dst_domain = domains.domain_assignment(dst);
    
    intra_hops = domain_cache.intra_hops;

    if src_domain == dst_domain
        [~, distance] = dijkstra_algorithm(src, dst, adjacency);
        hops = 0; % 同域内跳数可通过距离估算，但这里简化
        overhead = 1;
        return;
    end

    % 提取缓存
    domain_nodes = domain_cache.domain_nodes;
    intra_dists = domain_cache.intra_dists;
    inter_links = domain_cache.inter_links;
    k_shortest_paths = domain_cache.k_shortest_paths;

    % 构建域级图（考虑负载）
    num_domains = domains.num_domains;
    domain_graph = zeros(num_domains);
    if nargin >= 7 && ~isempty(load_matrix)
        max_load = max(load_matrix(:));
        if max_load == 0, max_load = 1; end
    end

    for d1 = 1:num_domains
        nodes_d1 = domain_nodes{d1};
        for d2 = d1+1:num_domains
            nodes_d2 = domain_nodes{d2};
            min_dist = Inf;
            for i = 1:length(nodes_d1)
                for j = 1:length(nodes_d2)
                    if adjacency(nodes_d1(i), nodes_d2(j)) > 0
                        dist = adjacency(nodes_d1(i), nodes_d2(j));
                        if dist < min_dist, min_dist = dist; end
                    end
                end
            end
            if ~isinf(min_dist)
                if nargin >= 7 && ~isempty(load_matrix)
                    norm_load = load_matrix(d1,d2) / max_load;
                    weight = min_dist * (1 + params.cross_domain_params.load_balancing_factor * norm_load);
                else
                    weight = min_dist;
                end
                domain_graph(d1,d2) = weight;
                domain_graph(d2,d1) = weight;
            end
        end
    end

    % 获取候选域路径
    candidate_paths = k_shortest_paths{src_domain, dst_domain};
    if isempty(candidate_paths)
        distance = Inf; hops = Inf; return;
    end

    best_dist = Inf;
    best_hops = Inf;
    for p_idx = 1:length(candidate_paths)
        domain_path = candidate_paths{p_idx};
        [dist, h] = try_domain_path_stats(domain_path, src, dst, adjacency, domains, domain_nodes, inter_links, intra_dists, intra_hops);
        if dist < best_dist
            best_dist = dist;
            best_hops = h;
        end
    end

    distance = best_dist;
    hops = best_hops;
    overhead = length(candidate_paths);
end

function [distance, hops] = try_domain_path_stats(domain_path, src, dst, adjacency, domains, domain_nodes, inter_links, intra_dists, intra_hops)
    L = length(domain_path);
    dp_dist = cell(L,1);
    dp_hops = cell(L,1);
    
    % 最后阶段
    last_domain = domain_path(L);
    nodes_last = domain_nodes{last_domain};
    dp_dist{L} = inf(length(nodes_last),1);
    dp_hops{L} = inf(length(nodes_last),1);
    for i = 1:length(nodes_last)
        node = nodes_last(i);
        [~, d] = dijkstra_algorithm(node, dst, adjacency);
        dp_dist{L}(i) = d;
        % 跳数：从 node 到 dst 的最少跳数（无权图）
        hop_adj = double(adjacency > 0);
        [~, h] = dijkstra_algorithm(node, dst, hop_adj);
        dp_hops{L}(i) = h;
    end
    
    for stage = L-1:-1:1
        curr_domain = domain_path(stage);
        next_domain = domain_path(stage+1);
        nodes_curr = domain_nodes{curr_domain};
        nodes_next = domain_nodes{next_domain};
        n_curr = length(nodes_curr);
        n_next = length(nodes_next);
        dp_dist{stage} = inf(n_curr,1);
        dp_hops{stage} = inf(n_curr,1);
        links = inter_links{curr_domain, next_domain};
        if isempty(links), distance = Inf; hops = Inf; return; end
        
        for i = 1:n_curr
            u = nodes_curr(i);
            best_dist = inf;
            best_hops = inf;
            for k = 1:size(links,1)
                exit_node = links(k,1);
                entry_node = links(k,2);
                isl_dist = links(k,3);
                [~, i_exit] = ismember(exit_node, nodes_curr);
                if i_exit == 0, continue; end
                intra_dist = intra_dists{curr_domain}(i, i_exit);
                if i > size(intra_hops{curr_domain},1) || i_exit > size(intra_hops{curr_domain},2)
                    continue;
                end
                intra_h = intra_hops{curr_domain}(i, i_exit);
                if isinf(intra_dist), continue; end
                [~, i_entry] = ismember(entry_node, nodes_next);
                if i_entry == 0, continue; end
                next_dist = dp_dist{stage+1}(i_entry);
                next_hops = dp_hops{stage+1}(i_entry);
                if isinf(next_dist), continue; end
                total_dist = intra_dist + isl_dist + next_dist;
                total_hops = intra_h + 1 + next_hops;   % 域内跳数 + 1（域间链路） + 后续跳数
                if total_dist < best_dist
                    best_dist = total_dist;
                    best_hops = total_hops;
                end
            end
            if ~isinf(best_dist)
                dp_dist{stage}(i) = best_dist;
                dp_hops{stage}(i) = best_hops;
            end
        end
    end
    
    src_domain_nodes = domain_nodes{domain_path(1)};
    [~, src_idx] = ismember(src, src_domain_nodes);
    if src_idx == 0 || isinf(dp_dist{1}(src_idx))
        distance = Inf; hops = Inf;
    else
        distance = dp_dist{1}(src_idx);
        hops = dp_hops{1}(src_idx);
    end
end