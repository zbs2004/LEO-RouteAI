function [next_state, reward, done] = rl_step(action)
    % 从 base 工作区读取当前环境状态
    src = evalin('base', 'current_src');
    dst = evalin('base', 'current_dst');
    current_satellite = evalin('base', 'current_satellite');
    current_domain = evalin('base', 'current_domain');
    adjacency = evalin('base', 'topology');
    domains = evalin('base', 'domains');
    load_matrix = evalin('base', 'load_matrix');
    domain_graph = evalin('base', 'domain_graph');
    domain_nodes = evalin('base', 'domain_nodes');
    intra_dists = evalin('base', 'intra_dists');
    intra_hops = evalin('base', 'domain_cache').intra_hops;
    inter_links = evalin('base', 'inter_links');
    load_penalty_coef = evalin('base', 'params.cross_domain_params.rl_load_penalty');
    smooth_max_load = evalin('base', 'smooth_max_load');
    alpha = evalin('base', 'load_smoothing_alpha');

    [next_state, reward, done] = apply_rl_action(action, src, dst, current_satellite, ...
        adjacency, domains, load_matrix, domain_graph, domain_nodes, intra_dists, intra_hops, inter_links, ...
        load_penalty_coef, smooth_max_load, alpha);

    % 兼容可能的稀疏矩阵：对返回结构体的所有字段逐一检查并转换
    if isstruct(next_state)
        for idx = 1:numel(next_state)
            fnames = fieldnames(next_state);
            for j = 1:numel(fnames)
                field_name = fnames{j};
                field_value = next_state(idx).(field_name);
                if issparse(field_value)
                    next_state(idx).(field_name) = full(field_value);
                end
            end
        end
    end

    % 兼容失败分支可能缺失的字段
    if ~isfield(next_state, 'smooth_max_load')
        next_state.smooth_max_load = smooth_max_load;
    end
    if ~isfield(next_state, 'current_satellite')
        next_state.current_satellite = current_satellite;
    end
    if ~isfield(next_state, 'current_domain')
        next_state.current_domain = current_domain;
    end

    % 将更新后的变量写回 base 工作区
    assignin('base', 'load_matrix', next_state.load_matrix);
    assignin('base', 'smooth_max_load', next_state.smooth_max_load);
    assignin('base', 'current_satellite', next_state.current_satellite);
    assignin('base', 'current_domain', next_state.current_domain);
end