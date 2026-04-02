function [dist, hops] = route_to_domain_stats(current_sat, current_domain, next_domain, ...
    adjacency, domains, domain_nodes, intra_dists, intra_hops, inter_links)
    dist = Inf; hops = Inf;
    links = inter_links{current_domain, next_domain};
    if isempty(links), return; end
    nodes_curr = domain_nodes{current_domain};
    [~, i_curr] = ismember(current_sat, nodes_curr);
    if i_curr == 0, return; end
    
    % 确保 intra_hops{current_domain} 尺寸正确
    n_curr = length(nodes_curr);
    if size(intra_hops{current_domain},1) ~= n_curr || size(intra_hops{current_domain},2) ~= n_curr
        % 重新计算该域的跳数矩阵
        if n_curr == 0
            intra_hops{current_domain} = [];
        else
            sub_adj = adjacency(nodes_curr, nodes_curr);
            hops_mat = inf(n_curr);
            for ii = 1:n_curr
                for jj = 1:n_curr
                    if ii == jj
                        hops_mat(ii,jj) = 0;
                    else
                        sub_adj_hops = double(sub_adj > 0);
                        [~, h_tmp] = dijkstra_algorithm(ii, jj, sub_adj_hops);
                        hops_mat(ii,jj) = h_tmp;
                    end
                end
            end
            intra_hops{current_domain} = hops_mat;
        end
    end
    
    best_dist = inf; best_hops = inf;
    for k = 1:size(links,1)
        exit_node = links(k,1);
        entry_node = links(k,2);
        isl_dist = links(k,3);
        [~, i_exit] = ismember(exit_node, nodes_curr);
        if i_exit == 0, continue; end
        if i_curr > size(intra_dists{current_domain},1) || i_exit > size(intra_dists{current_domain},2)
            continue;
        end
        intra_dist = intra_dists{current_domain}(i_curr, i_exit);
        if isinf(intra_dist), continue; end
        if i_curr > size(intra_hops{current_domain},1) || i_exit > size(intra_hops{current_domain},2)
            continue;
        end
        intra_h = intra_hops{current_domain}(i_curr, i_exit);
        total_dist = intra_dist + isl_dist;
        total_hops = intra_h + 1;
        if total_dist < best_dist
            best_dist = total_dist;
            best_hops = total_hops;
        end
    end
    if ~isinf(best_dist)
        dist = best_dist;
        hops = best_hops;
    end
end