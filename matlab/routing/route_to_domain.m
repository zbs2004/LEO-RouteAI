function [path, distance] = route_to_domain(current_sat, current_domain, next_domain, adjacency, domains, domain_nodes, intra_dists, inter_links, domain_graph, dst_domain)
    path = [];
    distance = Inf;
    use_heuristic = nargin >= 10 && ~isempty(domain_graph) && nargin >= 11 && ~isempty(dst_domain);
    if use_heuristic
        if size(domain_graph,1) < max(next_domain, dst_domain) || size(domain_graph,2) < max(next_domain, dst_domain)
            use_heuristic = false;
        else
            try
                G_dom = graph(domain_graph);
                domain_to_dst = distances(G_dom, (1:size(domain_graph,1))', dst_domain);
            catch
                use_heuristic = false;
            end
        end
    end
    

    if current_domain < 1 || current_domain > length(domain_nodes)
        fprintf('错误：current_domain 超出范围\n');
        return;
    end
    if next_domain < 1 || next_domain > length(domain_nodes)
        fprintf('错误：next_domain 超出范围\n');
        return;
    end

    % 尝试访问 inter_links
    try
        links = inter_links{current_domain, next_domain};
    catch ME
        fprintf('访问 inter_links 失败: %s\n', ME.message);
        return;
    end

    if isempty(links)
        fprintf('无链路\n');
        return;
    end
    % 检查输入合法性
    if current_domain < 1 || current_domain > length(domain_nodes) || next_domain < 1 || next_domain > length(domain_nodes)
        return;
    end

    links = inter_links{current_domain, next_domain};
    if isempty(links)
        return;
    end

    nodes_curr = domain_nodes{current_domain};
    [~, i_curr] = ismember(current_sat, nodes_curr);
    if i_curr == 0
        return;
    end

    best_score = inf;
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
        score = intra + isl_dist;
        if use_heuristic
            next_domain_dist = domain_to_dst(next_domain);
            if isinf(next_domain_dist)
                continue;
            end
            score = score + next_domain_dist;
        end
        if score < best_score
            best_score = score;
            best_intra = intra;
            best_isl = isl_dist;
            best_exit = exit_node;
            best_entry = entry_node;
        end
    end

    if best_exit == -1
        return;
    end

    [path_to_exit, ~] = dijkstra_algorithm(current_sat, best_exit, adjacency);
    if isempty(path_to_exit)
        return;
    end

    path = [path_to_exit, best_entry];
    distance = best_intra + best_isl;
end