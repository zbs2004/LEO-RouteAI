function [distance, hops, overhead] = dijkstra_stats(src, dst, adjacency)
    % 仅返回最短距离和跳数（跳数基于无权图）
    % 距离
    [~, distance] = dijkstra_algorithm(src, dst, adjacency);
    % 跳数（使用无权图）
    hop_adj = double(adjacency > 0);
    [~, hops] = dijkstra_algorithm(src, dst, hop_adj);
    overhead = size(adjacency,1) * log(size(adjacency,1));
    end