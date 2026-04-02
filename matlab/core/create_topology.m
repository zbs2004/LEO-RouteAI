function topology = create_topology(satellites, params)
    % 创建网络拓扑
    
    num_sats = length(satellites);
    max_dist = params.max_isl_distance;
    
    % 初始化邻接矩阵
    adjacency_matrix = zeros(num_sats, num_sats);
    
    % 建立星间链路
    connection_count = 0;
    
    for i = 1:num_sats
        pos_i = satellites(i).position;
        
        for j = i+1:num_sats
            pos_j = satellites(j).position;
            distance = norm(pos_i - pos_j);
            
            % 检查距离约束
            if distance <= max_dist
                % 检查仰角约束
                elevation = calculate_elevation_angle(pos_i, pos_j, params.earth_radius);
                
                if elevation >= deg2rad(params.min_elevation)
                    % 检查链路可用性
                    if rand() <= params.link_availability
                        adjacency_matrix(i, j) = distance;
                        adjacency_matrix(j, i) = distance;
                        connection_count = connection_count + 1;
                    end
                end
            end
        end
    end
    
    % 构建拓扑结构
    topology.adjacency_matrix = sparse(adjacency_matrix);
    topology.num_satellites = num_sats;
    topology.num_connections = connection_count;
    topology.connection_density = 2 * connection_count / (num_sats * (num_sats - 1));
    
    fprintf('     拓扑创建完成: %d条有效星间链路\n', connection_count);
end

