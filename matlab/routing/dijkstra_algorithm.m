function [path, distance] = dijkstra_algorithm(src, dst, adjacency)
    % 强制对称
    adjacency = max(adjacency, adjacency');
    % 移除零值（可选）
    adjacency(adjacency == 0) = 0;
    G = graph(adjacency);
    try
        [path, distance] = shortestpath(G, src, dst);
    catch
        path = [];
        distance = Inf;
    end
    if isempty(path)
        distance = Inf;
    end
end