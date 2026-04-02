% Yen算法求K条最短路径
function paths = k_shortest_paths_yen(source, target, graph, K)
    % 简单实现，返回路径元胞数组
    paths = {};
    [shortest_path, ~] = dijkstra_algorithm(source, target, graph);
    if isempty(shortest_path)
        return;
    end
    paths{1} = shortest_path;
    candidates = [];
    for k = 2:K
        prev_path = paths{k-1};
        for i = 1:length(prev_path)-1
            spur_node = prev_path(i);
            root_path = prev_path(1:i);
            % 临时修改图
            temp_graph = graph;
            % 删除已用边
            for p = 1:length(paths)
                pth = paths{p};
                if length(pth) > i && isequal(pth(1:i), root_path)
                    u = pth(i); v = pth(i+1);
                    temp_graph(u,v) = 0;
                    temp_graph(v,u) = 0;
                end
            end
            % 删除root_path中的中间节点
            for rn = root_path(1:end-1)
                temp_graph(rn,:) = 0;
                temp_graph(:,rn) = 0;
            end
            [spur_path, ~] = dijkstra_algorithm(spur_node, target, temp_graph);
            if ~isempty(spur_path)
                total_path = [root_path(1:end-1), spur_path];
                total_dist = 0;
                for j = 1:length(total_path)-1
                    total_dist = total_dist + graph(total_path(j), total_path(j+1));
                end
                candidates = [candidates; {total_path, total_dist}];
            end
        end
        if isempty(candidates)
            break;
        end
        % 选最小距离
        dists = cell2mat(candidates(:,2));
        [~, idx] = min(dists);
        new_path = candidates{idx,1};
        paths{k} = new_path;
        candidates(idx,:) = [];
    end
end