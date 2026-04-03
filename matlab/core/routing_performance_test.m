function step_results = routing_performance_test(satellites, topology, domains, params, step, domain_cache, load_matrix, smooth_max_load, use_parallel)
    if nargin < 9
        use_parallel = false;
    end
    num_tests = params.num_routing_tests;
    algorithm_names = params.routing_algorithms;
    
    step_results = struct();
    for i = 1:length(algorithm_names)
        algo_name = algorithm_names{i};
        step_results.(algo_name) = struct(...
            'success_rate', 0, 'avg_hops', 0, 'avg_delay', 0, ...
            'cross_domain_rate', 0, 'routing_overhead', 0);
    end
    
    [test_pairs, traffic_info] = generate_test_pairs(satellites, domains, num_tests);
    adjacency = full(topology.adjacency_matrix);
    % 域数量（domain_graph 的维度）
    num_domains = size(domain_cache.domain_graph, 1);

    % 队列模型参数（若存在于 params）
    queue_enabled = false;
    if isfield(params, 'queue_model') && isstruct(params.queue_model)
        try
            qm = params.queue_model;
            queue_enabled = isfield(qm, 'enable') && qm.enable;
            if isfield(qm, 'domain_capacities')
                q_capacity = double(qm.domain_capacities(:));
            else
                q_capacity = repmat(double(qm.capacity), num_domains, 1);
            end
            if isfield(qm, 'domain_service_rate')
                q_service_rate = double(qm.domain_service_rate(:));
            else
                q_service_rate = repmat(double(qm.service_rate), num_domains, 1);
            end
            q_time_per_step_ms = double(qm.time_per_step_ms);
            q_delay_weight = double(qm.delay_weight);
            q_drop_penalty = double(qm.drop_penalty);
        catch
            queue_enabled = false;
        end
    end

    % 读取 base 中的初始 domain_queues（若存在）
    if queue_enabled
        try
            if evalin('base', 'exist(''domain_queues'', ''var'')') == 1
                base_domain_queues = double(evalin('base', 'domain_queues'));
                if numel(base_domain_queues) ~= num_domains
                    base_domain_queues = zeros(num_domains, 1);
                end
            else
                base_domain_queues = zeros(num_domains, 1);
            end
        catch
            base_domain_queues = zeros(num_domains, 1);
        end
    else
        base_domain_queues = zeros(num_domains, 1);
    end

    if use_parallel
        if exist('parpool', 'file') == 2
            pool = gcp('nocreate');
            if isempty(pool)
                try
                    parpool('local');
                catch
                    warning('无法启动并行池，改为串行执行。');
                    use_parallel = false;
                end
            end
        else
            warning('未安装 Parallel Computing Toolbox，改为串行执行。');
            use_parallel = false;
        end
    end
    
    for algo_idx = 1:length(algorithm_names)
        algo_name = algorithm_names{algo_idx};
        fprintf('     测试算法: %s... ', algo_name);

        success_flags = false(num_tests, 1);
        hops = zeros(num_tests, 1);
        delays = zeros(num_tests, 1);
        overheads = zeros(num_tests, 1);
        cross_domain_flags = false(num_tests, 1);
        % 每个算法维护自己的负载状态，确保负载感知路由在同一算法内部生效
        algo_load_matrix = load_matrix;

        if use_parallel
            parfor test_idx = 1:num_tests
                src = test_pairs(test_idx, 1);
                dst = test_pairs(test_idx, 2);
                switch algo_name
                    case 'dijkstra'
                        [path, distance] = dijkstra_algorithm(src, dst, adjacency);
                        if isempty(path)
                            hops(test_idx) = Inf;
                            overhead = 0;
                        else
                            hops(test_idx) = length(path) - 1;
                            overhead = size(adjacency, 1) * log(size(adjacency, 1));
                        end
                    case 'minhop'
                        [path, distance, overhead] = minhop(src, dst, adjacency);
                        if isempty(path)
                            hops(test_idx) = Inf;
                        else
                            hops(test_idx) = length(path) - 1;
                        end
                    case 'cross_domain'
                        [path, distance, overhead] = cross_domain_routing(src, dst, adjacency, domains, params, domain_cache, load_matrix);
                        if isempty(path)
                            hops(test_idx) = Inf;
                        else
                            hops(test_idx) = length(path) - 1;
                        end
                    case 'rl'
                        [path, distance, overhead] = rl_cross_domain_routing(src, dst, adjacency, domains, params, domain_cache, load_matrix, smooth_max_load);
                        if isempty(path)
                            hops(test_idx) = Inf;
                        else
                            hops(test_idx) = length(path) - 1;
                        end
                    otherwise
                        error('未知算法: %s', algo_name);
                end
                overheads(test_idx) = overhead;
                if ~isinf(distance)
                    success_flags(test_idx) = true;
                    delays(test_idx) = calculate_delay(distance, hops(test_idx), params);
                    if domains.domain_assignment(src) ~= domains.domain_assignment(dst)
                        cross_domain_flags(test_idx) = true;
                    end
                end
            end
        else
            % 每个算法运行时使用 base_domain_queues 的副本，保证不同算法间互不干扰
            if queue_enabled
                algo_domain_queues = base_domain_queues;
            else
                algo_domain_queues = zeros(num_domains,1);
            end

            for test_idx = 1:num_tests
                src = test_pairs(test_idx, 1);
                dst = test_pairs(test_idx, 2);
                % 在每个请求前先让域按服务速率出队（每次请求代表时间推进的一个小刻度）
                if queue_enabled
                    % 支持标量或向量 service rate
                    if isscalar(q_service_rate)
                        sr = repmat(q_service_rate, num_domains, 1);
                    else
                        sr = q_service_rate;
                    end
                    algo_domain_queues = max(0, algo_domain_queues - sr);
                    % 把当前算法的队列状态写回 base，确保 build_state_vector 能读取到最新队列信息
                    try
                        assignin('base', 'domain_queues', algo_domain_queues);
                    catch
                    end
                end
                switch algo_name
                    case 'dijkstra'
                        [path, distance] = dijkstra_algorithm(src, dst, adjacency);
                        if isempty(path)
                            hops(test_idx) = Inf;
                            overhead = 0;
                        else
                            hops(test_idx) = length(path) - 1;
                            overhead = size(adjacency, 1) * log(size(adjacency, 1));
                        end
                    case 'minhop'
                        [path, distance, overhead] = minhop(src, dst, adjacency);
                        if isempty(path)
                            hops(test_idx) = Inf;
                        else
                            hops(test_idx) = length(path) - 1;
                        end
                    case 'cross_domain'
                        [path, distance, overhead] = cross_domain_routing(src, dst, adjacency, domains, params, domain_cache, algo_load_matrix);
                        if isempty(path)
                            hops(test_idx) = Inf;
                        else
                            hops(test_idx) = length(path) - 1;
                            algo_load_matrix = update_load_matrix_by_path(path, domains, algo_load_matrix);
                        end
                    case 'rl'
                        [path, distance, overhead] = rl_cross_domain_routing(src, dst, adjacency, domains, params, domain_cache, algo_load_matrix, smooth_max_load);
                        if isempty(path)
                            hops(test_idx) = Inf;
                        else
                            hops(test_idx) = length(path) - 1;
                            algo_load_matrix = update_load_matrix_by_path(path, domains, algo_load_matrix);
                        end
                    otherwise
                        error('未知算法: %s', algo_name);
                end
                overheads(test_idx) = overhead;
                if ~isinf(distance)
                    % 若启用队列模型，则对到达域进行入队并计算额外延时或丢包
                    dropped = false;
                    extra_delay_ms = 0;
                    if queue_enabled && ~isempty(path)
                        try
                            arrival_domain = domains.domain_assignment(path(end));
                            % 入队
                            algo_domain_queues(arrival_domain) = algo_domain_queues(arrival_domain) + 1;
                            % 检查是否溢出
                            if algo_domain_queues(arrival_domain) > q_capacity(arrival_domain)
                                dropped = true;
                            else
                                pos = algo_domain_queues(arrival_domain);
                                wait_steps = max(0, (pos - 1) / max(1e-6, q_service_rate(arrival_domain)));
                                proc_steps = 1 / max(1e-6, q_service_rate(arrival_domain));
                                delay_steps = wait_steps + proc_steps;
                                extra_delay_ms = delay_steps * q_time_per_step_ms;
                            end
                        catch
                            dropped = false;
                            extra_delay_ms = 0;
                        end
                    end

                    if dropped
                        % 若丢包则视为失败
                        success_flags(test_idx) = false;
                        hops(test_idx) = Inf;
                        delays(test_idx) = Inf;
                        % 记录失败请求
                        step_results.failed_requests = [step_results.failed_requests; struct('src', src, 'dst', dst, 'reason', 'queue_drop')];
                    else
                        success_flags(test_idx) = true;
                        delays(test_idx) = calculate_delay(distance, hops(test_idx), params) + extra_delay_ms;
                        if domains.domain_assignment(src) ~= domains.domain_assignment(dst)
                            cross_domain_flags(test_idx) = true;
                        end
                    end
                    % 把更新后的队列状态写回 base 以便后续状态构建使用
                    if queue_enabled
                        try
                            assignin('base', 'domain_queues', algo_domain_queues);
                        catch
                        end
                    end
                end
            end
        end

        success_count = sum(success_flags);
        if success_count > 0
            valid_idx = find(success_flags);
            step_results.(algo_name).success_rate = 100 * success_count / num_tests;
            step_results.(algo_name).avg_hops = sum(hops(valid_idx)) / success_count;
            step_results.(algo_name).avg_delay = sum(delays(valid_idx)) / success_count;
            step_results.(algo_name).cross_domain_rate = 100 * sum(cross_domain_flags(valid_idx)) / success_count;
            step_results.(algo_name).routing_overhead = sum(overheads(valid_idx)) / success_count;
        else
            step_results.(algo_name).success_rate = 0;
            step_results.(algo_name).avg_hops = 0;
            step_results.(algo_name).avg_delay = 0;
            step_results.(algo_name).cross_domain_rate = 0;
            step_results.(algo_name).routing_overhead = 0;
        end
        fprintf('完成\n');
    end
    
    step_results.traffic_info = traffic_info;
    step_results.step = step;
    step_results.failed_requests = [];
end

function [test_pairs, traffic_info] = generate_test_pairs(satellites, domains, num_tests)
    num_sats = length(satellites);
    test_pairs = zeros(num_tests, 2);
    traffic_info.total_pairs = num_tests;
    traffic_info.cross_domain_pairs = 0;
    traffic_info.intra_domain_pairs = 0;
    for i = 1:num_tests
        src = randi(num_sats);
        dst = randi(num_sats);
        while dst == src
            dst = randi(num_sats);
        end
        test_pairs(i, :) = [src, dst];
        if domains.domain_assignment(src) ~= domains.domain_assignment(dst)
            traffic_info.cross_domain_pairs = traffic_info.cross_domain_pairs + 1;
        else
            traffic_info.intra_domain_pairs = traffic_info.intra_domain_pairs + 1;
        end
    end
    traffic_info.cross_domain_ratio = traffic_info.cross_domain_pairs / num_tests;
end

function load_matrix = update_load_matrix_by_path(path, domains, load_matrix)
    if isempty(path)
        return;
    end

    prev_domain = domains.domain_assignment(path(1));
    for idx = 2:length(path)
        curr_domain = domains.domain_assignment(path(idx));
        if curr_domain ~= prev_domain
            load_matrix(prev_domain, curr_domain) = load_matrix(prev_domain, curr_domain) + 1;
            load_matrix(curr_domain, prev_domain) = load_matrix(curr_domain, prev_domain) + 1;
            prev_domain = curr_domain;
        end
    end
end