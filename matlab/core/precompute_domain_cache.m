function domain_cache = precompute_domain_cache(adjacency, domains, params)
    % 预计算跨域路由所需数据结构
    num_domains = domains.num_domains;
    K = params.cross_domain_params.K_paths;
    
    % 1. 各域节点列表
    domain_nodes = cell(num_domains, 1);
    for d = 1:num_domains
        domain_nodes{d} = find(domains.domain_assignment == d);
    end
    
    % 2. 域内最短距离矩阵
    intra_dists = cell(num_domains, 1);
    for d = 1:num_domains
        nodes = domain_nodes{d};
        n = length(nodes);
        if n == 0
            intra_dists{d} = [];
            continue;
        end
        sub_adj = adjacency(nodes, nodes);
        dist_mat = inf(n);
        for i = 1:n
            for j = 1:n
                if i == j
                    dist_mat(i,j) = 0;
                else
                    [~, d_tmp] = dijkstra_algorithm(i, j, sub_adj);
                    dist_mat(i,j) = d_tmp;
                end
            end
        end
        intra_dists{d} = dist_mat;
    end
    % 2.5 域内跳数矩阵
    intra_hops = cell(num_domains, 1);
    for d = 1:num_domains
        nodes = domain_nodes{d};
        n = length(nodes);
        if n == 0
            intra_hops{d} = [];
            continue;
        end
        sub_adj_hops = double(adjacency(nodes, nodes) > 0);
        hops_mat = inf(n);
        for i = 1:n
            for j = 1:n
                if i == j
                    hops_mat(i,j) = 0;
                else
                    [~, d_tmp] = dijkstra_algorithm(i, j, sub_adj_hops);
                    hops_mat(i,j) = d_tmp;
                end
            end
        end
        intra_hops{d} = hops_mat;
    end
    domain_cache.intra_hops = intra_hops;
    
    % 3. 域间链路和域级图
    inter_links = cell(num_domains, num_domains);
    domain_graph = zeros(num_domains);
    for d1 = 1:num_domains
        nodes_d1 = domain_nodes{d1};
        for d2 = d1+1:num_domains
            nodes_d2 = domain_nodes{d2};
            min_dist = Inf;
            for i = 1:length(nodes_d1)
                for j = 1:length(nodes_d2)
                    if adjacency(nodes_d1(i), nodes_d2(j)) > 0
                        dist_ij = adjacency(nodes_d1(i), nodes_d2(j));
                        inter_links{d1,d2} = [inter_links{d1,d2}; nodes_d1(i), nodes_d2(j), dist_ij];
                        inter_links{d2,d1} = [inter_links{d2,d1}; nodes_d2(j), nodes_d1(i), dist_ij];
                        if dist_ij < min_dist
                            min_dist = dist_ij;
                        end
                    end
                end
            end
            if ~isinf(min_dist)
                domain_graph(d1,d2) = min_dist;
                domain_graph(d2,d1) = min_dist;
            end
        end
    end
    
    % 4. 所有域对的K条最短路径
    k_shortest_paths = cell(num_domains, num_domains);
    for s = 1:num_domains
        for t = 1:num_domains
            if s == t
                k_shortest_paths{s,t} = {[s]};
            else
                k_shortest_paths{s,t} = k_shortest_paths_yen(s, t, domain_graph, K);
            end
        end
    end
    
    domain_cache.domain_nodes = domain_nodes;
    domain_cache.intra_dists = intra_dists;
    domain_cache.inter_links = inter_links;
    domain_cache.k_shortest_paths = k_shortest_paths;
    domain_cache.domain_graph = domain_graph; 
    domain_cache.intra_hops = intra_hops;
end
