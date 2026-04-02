function [distance, hops, overhead] = rl_cross_domain_routing_stats(src, dst, adjacency, domains, params, domain_cache, load_matrix, smooth_max_load)
    overhead = 0;
    domain_nodes = domain_cache.domain_nodes;
    intra_dists = domain_cache.intra_dists;
    intra_hops = domain_cache.intra_hops;
    inter_links = domain_cache.inter_links;
    domain_graph = domain_cache.domain_graph;
    
    current_sat = src;
    current_domain = domains.domain_assignment(current_sat);
    action_mask = domain_graph(current_domain, :) > 0;
    state_vec = build_state_vector(src, dst, current_sat, load_matrix, domains, domain_nodes, domain_graph, smooth_max_load);
    action = dqn_predict(state_vec, action_mask);
    next_domain = action + 1;
    
    [first_dist, first_hops] = route_to_domain_stats(current_sat, current_domain, next_domain, ...
    adjacency, domains, domain_nodes, intra_dists, intra_hops, inter_links);
    if isinf(first_dist)
        distance = Inf;
        hops = Inf;
        overhead = 1;
        return;
    end
    
    current_domain = next_domain;
    if current_domain == domains.domain_assignment(dst)
        [final_dist, final_hops] = dijkstra_stats(current_sat, dst, adjacency);
        distance = first_dist + final_dist;
        hops = first_hops + final_hops;
    else
        [rest_dist, rest_hops, ~] = cross_domain_routing_stats(current_sat, dst, adjacency, domains, params, domain_cache, load_matrix);
        distance = first_dist + rest_dist;
        hops = first_hops + rest_hops;
    end

    overhead = 1;
end

function [distance, hops] = try_domain_path_stats(domain_path, src, dst, adjacency, domains, ...
    domain_nodes, inter_links, intra_dists, intra_hops)
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
        if isempty(links)
            distance = Inf; hops = Inf; return;
        end
        
        % 确保 intra_hops{curr_domain} 尺寸正确
        if size(intra_hops{curr_domain},1) ~= n_curr || size(intra_hops{curr_domain},2) ~= n_curr
            % 重新计算该域的跳数矩阵
            nodes = domain_nodes{curr_domain};
            n = length(nodes);
            if n == 0
                intra_hops{curr_domain} = [];
            else
                sub_adj = adjacency(nodes, nodes);
                hops_mat = inf(n);
                for ii = 1:n
                    for jj = 1:n
                        if ii == jj
                            hops_mat(ii,jj) = 0;
                        else
                            sub_adj_hops = double(sub_adj > 0);
                            [~, h_tmp] = dijkstra_algorithm(ii, jj, sub_adj_hops);
                            hops_mat(ii,jj) = h_tmp;
                        end
                    end
                end
                intra_hops{curr_domain} = hops_mat;
            end
        end
        
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
                if isinf(intra_dist), continue; end
                % 检查索引范围
                if i > size(intra_hops{curr_domain},1) || i_exit > size(intra_hops{curr_domain},2)
                    continue;
                end
                intra_h = intra_hops{curr_domain}(i, i_exit);
                [~, i_entry] = ismember(entry_node, nodes_next);
                if i_entry == 0, continue; end
                next_dist = dp_dist{stage+1}(i_entry);
                next_hops = dp_hops{stage+1}(i_entry);
                if isinf(next_dist), continue; end
                total_dist = intra_dist + isl_dist + next_dist;
                total_hops = intra_h + 1 + next_hops;
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