function [path, distance] = dijkstra_algorithm(src, dst, adjacency)
    % Dijkstra最短路径算法
    
    num_nodes = size(adjacency, 1);
    
    % 初始化
    dist = Inf(1, num_nodes);
    prev = zeros(1, num_nodes);
    visited = false(1, num_nodes);
    
    dist(src) = 0;
    
    for i = 1:num_nodes
        % 找到未访问的最小距离节点
        unvisited_dist = dist;
        unvisited_dist(visited) = Inf;
        
        [min_dist, current] = min(unvisited_dist);
        
        % 如果找不到可达节点或到达目标，结束
        if isinf(min_dist) || current == dst
            break;
        end
        
        visited(current) = true;
        
        % 更新邻居距离
        neighbors = find(adjacency(current, :) > 0);
        
        for neighbor = neighbors
            if ~visited(neighbor)
                alt = dist(current) + adjacency(current, neighbor);
                
                if alt < dist(neighbor)
                    dist(neighbor) = alt;
                    prev(neighbor) = current;
                end
            end
        end
    end
    
    % 重建路径
    if isinf(dist(dst))
        path = [];
        distance = Inf;
    else
        path = [];
        current = dst;
        
        while current ~= 0
            path = [current, path];
            current = prev(current);
        end
        
        distance = dist(dst);
    end
end