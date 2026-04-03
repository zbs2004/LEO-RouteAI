function state_vec = build_state_vector(src, dst, current_sat, load_matrix, domains, domain_nodes, domain_graph, smooth_max_load)
    src_dom = domains.domain_assignment(src);
    dst_dom = domains.domain_assignment(dst);
    curr_dom = domains.domain_assignment(current_sat);
    num_domains = size(domain_graph, 1);
    domain_count = max(1, num_domains - 1);
    src_dom_norm = (src_dom - 1) / domain_count;
    dst_dom_norm = (dst_dom - 1) / domain_count;
    curr_dom_norm = (curr_dom - 1) / domain_count;

    load_matrix = full(load_matrix);
    load_flat = double(load_matrix(:)');
    max_load = max(load_flat);
    if max_load <= 0
        load_flat_norm = zeros(size(load_flat));
    else
        load_flat_norm = load_flat / max_load;
    end

    distances = double(full(domain_graph(curr_dom, :)));
    max_dist = max(distances);
    if max_dist > 0
        distances = distances / max_dist;
    end
    candidate_mask = double(distances > 0);

    mean_load = mean(load_flat);
    var_load = var(load_flat);
    triu_idx = triu(true(num_domains), 1);
    inter_loads = load_matrix(triu_idx);
    if isempty(inter_loads)
        inter_load_mean = 0;
        inter_load_max = 0;
    else
        inter_load_mean = mean(inter_loads);
        inter_load_max = max(inter_loads);
    end
    inter_load_mean_norm = inter_load_mean / (max_load + eps);
    inter_load_max_norm = inter_load_max / (max_load + eps);
    load_balance = std(load_flat) / (max_load + eps);
    avg_load = mean_load;
    smooth_max_load_norm = smooth_max_load / (max_load + eps);
    end_to_end_delay_norm = 0;

    load_variance = var(load_flat);
    max_load_feature = max_load;
    % 队列特征：如果 base 工作区存在 domain_queues，则使用其值并按容量归一化；否则返回 0
    try
        if evalin('base', 'exist(''domain_queues'', ''var'')') == 1
            dq = double(evalin('base', 'domain_queues'));
            % 确保长度与域数一致
            if numel(dq) ~= num_domains
                dq = zeros(num_domains, 1);
            end
        else
            dq = zeros(num_domains, 1);
        end
    catch
        dq = zeros(num_domains, 1);
    end
    % 获取容量信息，优先从 params.queue_model.domain_capacities，其次 params.queue_model.capacity
    try
        if evalin('base', 'exist(''params'', ''var'')') == 1
            p = evalin('base', 'params');
            if isfield(p, 'queue_model') && isfield(p.queue_model, 'domain_capacities')
                cap = double(p.queue_model.domain_capacities(:));
                if numel(cap) ~= num_domains
                    cap = repmat(double(p.queue_model.capacity), num_domains, 1);
                end
            elseif isfield(p, 'queue_model') && isfield(p.queue_model, 'capacity')
                cap = repmat(double(p.queue_model.capacity), num_domains, 1);
            else
                cap = ones(num_domains, 1);
            end
        else
            cap = ones(num_domains, 1);
        end
    catch
        cap = ones(num_domains, 1);
    end

    % 归一化并构造队列特征
    if all(cap > 0)
        mean_queue_norm = mean(dq) / mean(cap);
        max_queue_norm = max(dq) / max(cap);
        curr_queue_norm = dq(curr_dom) / cap(curr_dom);
    else
        mean_queue_norm = 0;
        max_queue_norm = 0;
        curr_queue_norm = 0;
    end

    state_vec = [src_dom_norm, dst_dom_norm, curr_dom_norm, distances, candidate_mask, load_flat_norm, ...
                 mean_load, var_load, inter_load_mean_norm, inter_load_max_norm, ...
                 load_balance, avg_load, smooth_max_load_norm, end_to_end_delay_norm, load_variance, max_load_feature, ...
                 mean_queue_norm, max_queue_norm, curr_queue_norm];
end