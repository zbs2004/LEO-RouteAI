function init_environment()
    % 设置随机种子（可选）
    rng(100);
    
    % 加载参数
    params = load_config_params();
    
    % 生成卫星星座
    [satellites, params] = generate_constellation(params);
    
    % 初始拓扑
    topology = create_topology(satellites, params);
    
    % 域划分
    domains = perform_domain_partition(satellites, topology, params);
    
    % 域级图构建（基于域间最短距离）
    num_domains = domains.num_domains;
    domain_graph = zeros(num_domains);
    for d1 = 1:num_domains
        for d2 = d1+1:num_domains
            nodes1 = find(domains.domain_assignment == d1);
            nodes2 = find(domains.domain_assignment == d2);
            min_dist = inf;
            for i = 1:length(nodes1)
                for j = 1:length(nodes2)
                    if topology.adjacency_matrix(nodes1(i), nodes2(j)) > 0
                        dist = topology.adjacency_matrix(nodes1(i), nodes2(j));
                        if dist < min_dist
                            min_dist = dist;
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
    
    % 初始化负载矩阵
    load_matrix = zeros(num_domains);
    
    % 预计算 RL 缓存
    % 1. 域节点列表
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
        sub_adj = full(topology.adjacency_matrix(nodes, nodes));
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
        sub_adj_hops = double(full(topology.adjacency_matrix(nodes, nodes)) > 0);
        hops_mat = inf(n);
        for i = 1:n
            for j = 1:n
                if i == j
                    hops_mat(i,j) = 0;
                else
                    [~, h_tmp] = dijkstra_algorithm(i, j, sub_adj_hops);
                    hops_mat(i,j) = h_tmp;
                end
            end
        end
        intra_hops{d} = hops_mat;
    end

    % 3. 域间链路列表
    inter_links = cell(num_domains, num_domains);
    for d1 = 1:num_domains
        nodes_d1 = domain_nodes{d1};
        for d2 = d1+1:num_domains
            nodes_d2 = domain_nodes{d2};
            for i = 1:length(nodes_d1)
                for j = 1:length(nodes_d2)
                    if topology.adjacency_matrix(nodes_d1(i), nodes_d2(j)) > 0
                        dist = topology.adjacency_matrix(nodes_d1(i), nodes_d2(j));
                        inter_links{d1,d2} = [inter_links{d1,d2}; nodes_d1(i), nodes_d2(j), dist];
                        inter_links{d2,d1} = [inter_links{d2,d1}; nodes_d2(j), nodes_d1(i), dist];
                    end
                end
            end
        end
    end
    
    % 将数据保存到 base 工作区
    assignin('base', 'satellites', satellites);
    assignin('base', 'topology', topology);
    assignin('base', 'domains', domains);
    assignin('base', 'domain_graph', domain_graph);
    assignin('base', 'load_matrix', load_matrix);
    assignin('base', 'params', params);
    assignin('base', 'domain_nodes', domain_nodes);
    assignin('base', 'intra_dists', intra_dists);
    assignin('base', 'inter_links', inter_links);
    
    % 初始当前卫星和域（占位）
    assignin('base', 'current_src', []);
    assignin('base', 'current_dst', []);
    assignin('base', 'current_satellite', []);
    assignin('base', 'current_domain', []);
    
    fprintf('环境初始化完成\n');
    % 预计算跨域路由缓存（与 routing_performance_test 中相同）
    domain_cache = struct();
    domain_cache.domain_nodes = domain_nodes;
    domain_cache.intra_dists = intra_dists;
    domain_cache.intra_hops = intra_hops;
    domain_cache.inter_links = inter_links;
    % 注意：还需要 k_shortest_paths，但这里如果不需要 K 路径，可以省略或另行计算
    % 这里我们只用到域内距离和域间链路，所以暂时这样存储
    assignin('base', 'domain_cache', domain_cache);
    % 初始当前卫星和域（占位）
    assignin('base', 'current_src', []);
    assignin('base', 'current_dst', []);
    assignin('base', 'current_satellite', []);
    assignin('base', 'current_domain', []);
    
    % 初始化上一时间步的状态（用于增量更新）
    assignin('base', 'prev_satellites', satellites);
    assignin('base', 'prev_adjacency', full(topology.adjacency_matrix));  % 转为满矩阵，便于比较
    assignin('base', 'prev_intra_dists', intra_dists);
    assignin('base', 'prev_inter_links', inter_links);
    assignin('base', 'prev_domain_graph', domain_graph);
    % 初始化平滑最大负载（用于负载归一化）
    assignin('base', 'smooth_max_load', 0);
    % 从参数中读取平滑因子，若不存在则默认为 0.2
    alpha = 0.2;
    if isfield(params.cross_domain_params, 'load_smoothing_alpha')
        alpha = params.cross_domain_params.load_smoothing_alpha;
    end
    assignin('base', 'load_smoothing_alpha', alpha);
    
end