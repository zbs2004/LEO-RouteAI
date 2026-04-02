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
        distances = inf(num_tests, 1);
        hops = zeros(num_tests, 1);
        delays = zeros(num_tests, 1);
        overheads = zeros(num_tests, 1);
        cross_domain_flags = false(num_tests, 1);

        if use_parallel
            parfor test_idx = 1:num_tests
                src = test_pairs(test_idx, 1);
                dst = test_pairs(test_idx, 2);
                distance = inf;
                overhead = 0;
                switch algo_name
                    case 'dijkstra'
                        [path, distance] = dijkstra_algorithm(src, dst, adjacency);
                        if isempty(path)
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
                distances(test_idx) = distance;
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
            for test_idx = 1:num_tests
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
                distances(test_idx) = distance;
                overheads(test_idx) = overhead;
                if ~isinf(distance)
                    success_flags(test_idx) = true;
                    delays(test_idx) = calculate_delay(distance, hops(test_idx), params);
                    if domains.domain_assignment(src) ~= domains.domain_assignment(dst)
                        cross_domain_flags(test_idx) = true;
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