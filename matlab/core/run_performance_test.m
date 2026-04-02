% ========================================
% LEO卫星网络跨域路由性能测试 - 单独入口脚本
% 该脚本仅运行路由算法性能测试，不执行完整仿真流程结果保存。
% ========================================

clear; clc; close all;

% 添加项目根目录到 MATLAB 路径
addpath(genpath('.'));

fprintf('========================================\n');
fprintf('LEO卫星网络跨域路由性能测试\n');
fprintf('单独运行性能测试，仿真与性能评估已经分离\n');
fprintf('========================================\n\n');

% 0. 固定随机种子，保证性能测试可复现
rng(140);

% 1. 加载配置参数
fprintf('1. 加载参数...\n');
params = load_config_params();

% 2. 生成卫星星座
fprintf('2. 生成卫星星座...\n');
[satellites, params] = generate_constellation(params);

% 3. 初始化性能测试结果结构
num_algorithms = length(params.routing_algorithms);
algorithm_names = params.routing_algorithms;
results = struct();
for i = 1:num_algorithms
    algo_name = algorithm_names{i};
    results.(algo_name) = struct(...
        'success_rates', zeros(params.num_time_steps, 1), ...
        'avg_hops', zeros(params.num_time_steps, 1), ...
        'avg_delays', zeros(params.num_time_steps, 1), ...
        'cross_domain_rates', zeros(params.num_time_steps, 1), ...
        'routing_overheads', zeros(params.num_time_steps, 1));
end
all_failed_requests = [];

% 4. 动态网络初始化和缓存变量
first_step = true;
prev_satellites = [];
prev_topology = [];
prev_domains = [];
prev_domain_nodes = [];
prev_intra_dists = [];
prev_inter_links = [];
prev_domain_graph = [];
prev_k_shortest_paths = [];

% 使用空负载矩阵作为性能测试基础输入
num_domains = params.cross_domain_params.num_domains;
load_matrix = zeros(num_domains);
smooth_max_load = 0;

fprintf('3. 开始性能测试（%d个时间步长）...\n', params.num_time_steps);
for step = 1:params.num_time_steps
    fprintf('\n   时间步长 %d/%d\n', step, params.num_time_steps);
    satellites = update_satellite_positions(satellites, params, step);

    if first_step
        topology = create_topology(satellites, params);
        domains = perform_domain_partition(satellites, topology, params);
        domain_cache = precompute_domain_cache(full(topology.adjacency_matrix), domains, params);
        domain_nodes = domain_cache.domain_nodes;
        intra_dists = domain_cache.intra_dists;
        inter_links = domain_cache.inter_links;
        domain_graph = full(domain_cache.domain_graph);
        k_shortest_paths = domain_cache.k_shortest_paths;

        prev_satellites = satellites;
        prev_topology = topology;
        prev_domains = domains;
        prev_domain_nodes = domain_nodes;
        prev_intra_dists = intra_dists;
        prev_inter_links = inter_links;
        prev_domain_graph = domain_graph;
        prev_k_shortest_paths = k_shortest_paths;
        first_step = false;
    else
        [topology, domains, domain_nodes, intra_dists, inter_links, domain_graph, k_shortest_paths] = ...
            incremental_update_network(satellites, params, ...
                prev_satellites, prev_topology, ...
                prev_domains, prev_domain_nodes, ...
                prev_intra_dists, prev_inter_links, ...
                prev_domain_graph, prev_k_shortest_paths);

        % 同步 domain_cache，以便后续测试使用最新缓存信息
        domain_cache.domain_nodes = domain_nodes;
        domain_cache.intra_dists = intra_dists;
        domain_cache.inter_links = inter_links;
        domain_cache.domain_graph = domain_graph;
        domain_cache.k_shortest_paths = k_shortest_paths;

        prev_satellites = satellites;
        prev_topology = topology;
        prev_domains = domains;
        prev_domain_nodes = domain_nodes;
        prev_intra_dists = intra_dists;
        prev_inter_links = inter_links;
        prev_domain_graph = domain_graph;
        prev_k_shortest_paths = k_shortest_paths;
    end

    % 5. 本步性能测试
    step_results = routing_performance_test(satellites, topology, domains, params, step, domain_cache, load_matrix, smooth_max_load, true);
    if isfield(step_results, 'failed_requests')
        all_failed_requests = [all_failed_requests; step_results.failed_requests];
    end

    for algo_idx = 1:num_algorithms
        algo_name = algorithm_names{algo_idx};
        if isfield(step_results, algo_name)
            algo_result = step_results.(algo_name);
            results.(algo_name).success_rates(step) = algo_result.success_rate;
            results.(algo_name).avg_hops(step) = algo_result.avg_hops;
            results.(algo_name).avg_delays(step) = algo_result.avg_delay;
            results.(algo_name).cross_domain_rates(step) = algo_result.cross_domain_rate;
            results.(algo_name).routing_overheads(step) = algo_result.routing_overhead;
        end
    end

    display_step_summary(step_results, algorithm_names, step);
end

% 6. 综合性能分析
fprintf('\n4. 综合性能分析...\n');
analyze_overall_performance(results, params);

% 7. 结果可视化
fprintf('5. 生成可视化结果...\n');
visualize_all_results(results, satellites, params, domains, topology, all_failed_requests);

fprintf('\n========================================\n');
fprintf('性能测试完成！结果已保存至: %s\n', params.output_dir);
fprintf('========================================\n');
