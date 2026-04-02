function [topology, domains, domain_nodes, intra_dists, inter_links, domain_graph, k_shortest_paths] = ...
    incremental_update_network(satellites, params, prev_satellites, prev_topology, ...
    prev_domains, prev_domain_nodes, prev_intra_dists, prev_inter_links, prev_domain_graph, prev_k_shortest_paths)
    % 增量更新网络拓扑及相关缓存
    % 输入：
    %   satellites            - 当前卫星位置
    %   params                - 仿真参数
    %   prev_satellites       - 上一时刻卫星位置
    %   prev_topology         - 上一时刻拓扑结构
    %   prev_domains          - 上一时刻域划分
    %   prev_domain_nodes     - 上一时刻域节点列表
    %   prev_intra_dists      - 上一时刻域内距离矩阵
    %   prev_inter_links      - 上一时刻域间链路列表
    %   prev_domain_graph     - 上一时刻域级图
    %   prev_k_shortest_paths - 上一时刻K条候选域路径
    % 输出：
    %   topology              - 更新后的拓扑结构
    %   domains               - 更新后的域划分
    %   domain_nodes          - 更新后的域节点列表
    %   intra_dists           - 更新后的域内距离矩阵
    %   inter_links           - 更新后的域间链路列表
    %   domain_graph          - 更新后的域级图
    %   k_shortest_paths      - 更新后的K条候选域路径

    % 阈值（相对距离变化超过此值才认为链路状态变化）
    threshold = 0.01;  % 1%
    
    num_sats = params.num_satellites;
    num_domains = prev_domains.num_domains;
    max_dist = params.max_isl_distance;
    min_elev = deg2rad(params.min_elevation);
    earth_radius = params.earth_radius;
    link_avail = params.link_availability;
    
    % 1. 更新邻接矩阵（只计算变化的链路）
    new_adjacency = zeros(num_sats, num_sats);
    old_adjacency = full(prev_topology.adjacency_matrix);
    % 记录哪些域内链路发生变化（用于后续域内距离矩阵更新）
    intra_link_changed_domains = false(num_domains,1);
    % 记录域间链路变化情况
    inter_link_changed = false(num_domains, num_domains);
    
    % 预计算卫星位置（当前和上一时刻）
    cur_pos = [satellites.position]';
    prev_pos = [prev_satellites.position]';
    
    for i = 1:num_sats
        for j = i+1:num_sats
            old_dist = old_adjacency(i,j);
            cur_dist = norm(cur_pos(i,:) - cur_pos(j,:));
            
            % 检查是否满足链路建立条件
            is_link_possible = (cur_dist <= max_dist) && ...
                (calculate_elevation_angle(cur_pos(i,:)', cur_pos(j,:)', earth_radius) >= min_elev) && ...
                (rand() <= link_avail);  % 随机性，但为了可复现，可固定种子
            
            if old_dist > 0
                % 原来有链路
                if is_link_possible
                    if abs(cur_dist - old_dist) / old_dist > threshold
                        % 距离变化超过阈值，更新链路
                        new_adjacency(i,j) = cur_dist;
                        new_adjacency(j,i) = cur_dist;
                        % 记录变化
                        d1 = prev_domains.domain_assignment(i);
                        d2 = prev_domains.domain_assignment(j);
                        if d1 == d2
                            intra_link_changed_domains(d1) = true;
                        else
                            inter_link_changed(d1,d2) = true;
                            inter_link_changed(d2,d1) = true;
                        end
                    else
                        % 距离变化小，保留原值（仍用新距离更精确，但若小于阈值可保留旧值以节省计算）
                        new_adjacency(i,j) = cur_dist;   % 还是用新值，确保精确
                        new_adjacency(j,i) = cur_dist;
                    end
                else
                    % 链路失效，不添加（保持0）
                    % 记录变化
                    d1 = prev_domains.domain_assignment(i);
                    d2 = prev_domains.domain_assignment(j);
                    if d1 == d2
                        intra_link_changed_domains(d1) = true;
                    else
                        inter_link_changed(d1,d2) = true;
                        inter_link_changed(d2,d1) = true;
                    end
                end
            else
                % 原来无链路，检查现在是否建立
                if is_link_possible
                    new_adjacency(i,j) = cur_dist;
                    new_adjacency(j,i) = cur_dist;
                    d1 = prev_domains.domain_assignment(i);
                    d2 = prev_domains.domain_assignment(j);
                    if d1 == d2
                        intra_link_changed_domains(d1) = true;
                    else
                        inter_link_changed(d1,d2) = true;
                        inter_link_changed(d2,d1) = true;
                    end
                end
            end
        end
    end
    
    % 2. 更新域划分（根据当前卫星位置重新计算）
    domains = perform_domain_partition(satellites, struct('adjacency_matrix', sparse(new_adjacency)), params);
    % 如果域分配与上一时刻相同，则 domain_nodes 不变，否则需要更新
    if isequal(domains.domain_assignment, prev_domains.domain_assignment)
        domain_nodes = prev_domain_nodes;
    else
        % 域分配变化，需要重新生成 domain_nodes
        domain_nodes = cell(num_domains,1);
        for d = 1:num_domains
            domain_nodes{d} = find(domains.domain_assignment == d);
        end
        % 域分配变化会直接影响域内和域间链路，标记所有域间链路和域内链路需要更新
        intra_link_changed_domains(:) = true;
        inter_link_changed(:,:) = true;
    end
    
    % 3. 增量更新域内距离矩阵
    intra_dists = prev_intra_dists;
    for d = 1:num_domains
    if intra_link_changed_domains(d) || ~isequal(domains.domain_assignment, prev_domains.domain_assignment)
        nodes = domain_nodes{d};
        n = length(nodes);
        if n == 0
            intra_dists{d} = [];
            intra_hops{d} = [];
            continue;
        end
        sub_adj = new_adjacency(nodes, nodes);
        dist_mat = inf(n);
        hops_mat = inf(n);
        for i = 1:n
            for j = 1:n
                if i == j
                    dist_mat(i,j) = 0;
                    hops_mat(i,j) = 0;
                else
                    [~, d_tmp] = dijkstra_algorithm(i, j, sub_adj);
                    dist_mat(i,j) = d_tmp;
                    sub_adj_hops = double(sub_adj > 0);
                    [~, h_tmp] = dijkstra_algorithm(i, j, sub_adj_hops);
                    hops_mat(i,j) = h_tmp;
                end
            end
        end
        intra_dists{d} = dist_mat;
        intra_hops{d} = hops_mat;
    end
end
    
    % 4. 增量更新域间链路列表和域级图
    inter_links = prev_inter_links;
    domain_graph = prev_domain_graph;
    for d1 = 1:num_domains
        for d2 = d1+1:num_domains
            if inter_link_changed(d1,d2) || ~isequal(domains.domain_assignment, prev_domains.domain_assignment)
                % 域间链路有变化或域分配变化，重新计算该域对
                nodes1 = domain_nodes{d1};
                nodes2 = domain_nodes{d2};
                if isempty(nodes1) || isempty(nodes2)
                    inter_links{d1,d2} = [];
                    inter_links{d2,d1} = [];
                    domain_graph(d1,d2) = 0;
                    domain_graph(d2,d1) = 0;
                else
                    links = [];
                    min_dist = inf;
                    for i = 1:length(nodes1)
                        for j = 1:length(nodes2)
                            if new_adjacency(nodes1(i), nodes2(j)) > 0
                                dist = new_adjacency(nodes1(i), nodes2(j));
                                links = [links; nodes1(i), nodes2(j), dist];
                                if dist < min_dist
                                    min_dist = dist;
                                end
                            end
                        end
                    end
                    inter_links{d1,d2} = links;
                    if ~isempty(links)
                        inter_links{d2,d1} = [links(:,2), links(:,1), links(:,3)];  % 对称
                    else
                        inter_links{d2,d1} = [];
                    end
                    if ~isinf(min_dist)
                        domain_graph(d1,d2) = min_dist;
                        domain_graph(d2,d1) = min_dist;
                    else
                        domain_graph(d1,d2) = 0;
                        domain_graph(d2,d1) = 0;
                    end
                end
            end
        end
    end
    
    % 5. 更新K条候选域路径
    K = params.cross_domain_params.K_paths;
    if isequal(domains.domain_assignment, prev_domains.domain_assignment)
        % 仅对受影响域对和邻域更新K条候选路径
        k_shortest_paths = prev_k_shortest_paths;

        % 受影响域集：发生链路变化的域及其邻居
        changed_domains = intra_link_changed_domains | any(inter_link_changed, 2) | any(inter_link_changed, 1)';
        affected_domains = changed_domains;
        if any(changed_domains)
            neighbor_domains = any(domain_graph(changed_domains, :) > 0, 1)' | any(domain_graph(:, changed_domains) > 0, 2);
            affected_domains = affected_domains | neighbor_domains;
        end

        affected_pairs = false(num_domains, num_domains);
        changed_idx = find(changed_domains);
        for s = 1:num_domains
            for t = 1:num_domains
                if s == t
                    continue;
                end
                if affected_domains(s) || affected_domains(t)
                    affected_pairs(s, t) = true;
                    continue;
                end
                old_paths = prev_k_shortest_paths{s, t};
                for p_idx = 1:length(old_paths)
                    if any(ismember(old_paths{p_idx}, changed_idx))
                        affected_pairs(s, t) = true;
                        break;
                    end
                end
            end
        end

        for s = 1:num_domains
            for t = 1:num_domains
                if s == t
                    k_shortest_paths{s, t} = {[s]};
                elseif affected_pairs(s, t)
                    k_shortest_paths{s, t} = k_shortest_paths_yen(s, t, domain_graph, K);
                end
            end
        end
    else
        % 域划分发生变化，重新计算所有域对路径
        k_shortest_paths = cell(num_domains, num_domains);
        for s = 1:num_domains
            for t = 1:num_domains
                if s == t
                    k_shortest_paths{s, t} = {[s]};
                else
                    k_shortest_paths{s, t} = k_shortest_paths_yen(s, t, domain_graph, K);
                end
            end
        end
    end
    
    % 6. 构建拓扑结构
    topology.adjacency_matrix = sparse(new_adjacency);
    topology.num_satellites = num_sats;
    topology.num_connections = nnz(new_adjacency) / 2;
    topology.connection_density = 2 * topology.num_connections / (num_sats * (num_sats - 1));
end