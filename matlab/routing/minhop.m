function [path, distance, overhead] = minhop(src, dst, adjacency)
    % 最少跳数路由
    hop_adj = double(adjacency > 0);
    [path, hops] = dijkstra_algorithm(src, dst, hop_adj);
    if isempty(path)
        distance = Inf;
        overhead = 0;
        return;
    end
    % 计算实际距离
    distance = 0;
    for k = 1:length(path)-1
        distance = distance + adjacency(path(k), path(k+1));
    end
    overhead = size(adjacency,1) * log(size(adjacency,1));
end
