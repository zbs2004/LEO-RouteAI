% ========================================
% LEO卫星网络跨域路由仿真平台 - 整理优化版
% 主程序文件
% ========================================

clear; clc; close all;

% 添加项目根目录到 MATLAB 路径，保证子目录函数可用
scriptDir = fileparts(mfilename('fullpath'));
projectRoot = fullfile(scriptDir, '..');
addpath(genpath(projectRoot));

% 设置随机种子保证可复现
rng(140);

fprintf('========================================\n');
fprintf('LEO卫星网络跨域路由仿真平台\n');
fprintf('北京邮电大学 - 赵宝硕\n');
fprintf('========================================\n\n');

% 1. 加载配置参数
fprintf('1. 加载仿真参数...\n');
params = load_config_params();

% 2. 生成卫星星座
fprintf('2. 生成卫星星座...\n');
[satellites, params] = generate_constellation(params);

% 3. 初始化结果存储
num_algorithms = length(params.routing_algorithms);
algorithm_names = params.routing_algorithms;
all_failed_requests = [];

% 为每种算法初始化结果存储
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


% 4. 主仿真循环
fprintf('3. 开始动态仿真（%d个时间步长）...\n', params.num_time_steps);

% 保存最后一次的域划分信息用于可视化
last_domains = [];

% 初始化持久状态变量
first_step = true;
% 缓存上一时刻的数据
prev_satellites = [];
prev_topology = [];
prev_domains = [];
prev_domain_nodes = [];
prev_intra_dists = [];
prev_inter_links = [];
prev_domain_graph = [];
prev_k_shortest_paths = [];
% 在 main.m 中，初始化负载矩阵并存入 base
num_domains = params.cross_domain_params.num_domains;
load_matrix = zeros(num_domains);
assignin('base', 'load_matrix', load_matrix);
init_rl_variables();
    % 初始化域级队列状态与参数，供 routing_performance_test 和 build_state_vector 使用
    if isfield(params, 'queue_model')
        try
            dq = zeros(num_domains, 1);
            assignin('base', 'domain_queues', dq);
            % domain capacities 和 service_rate 可在 params.queue_model 中以向量或标量提供
            if isfield(params.queue_model, 'domain_capacities')
                assignin('base', 'domain_queue_capacity', params.queue_model.domain_capacities(:));
            else
                assignin('base', 'domain_queue_capacity', repmat(params.queue_model.capacity, num_domains, 1));
            end
            if isfield(params.queue_model, 'domain_service_rate')
                assignin('base', 'domain_service_rate', params.queue_model.domain_service_rate(:));
            else
                assignin('base', 'domain_service_rate', repmat(params.queue_model.service_rate, num_domains, 1));
            end
        catch
            % 忽略任何赋值错误，保持健壮性
        end
    else
        assignin('base', 'domain_queues', zeros(num_domains, 1));
        assignin('base', 'domain_queue_capacity', ones(num_domains, 1));
        assignin('base', 'domain_service_rate', ones(num_domains, 1));
    end
    
for step = 1:params.num_time_steps
    fprintf('\n   时间步长 %d/%d (%.1f分钟)\n', ...
        step, params.num_time_steps, (step-1)*params.time_step_duration/60);
    
    % 4.1 更新卫星位置
    satellites = update_satellite_positions(satellites, params, step);
    
    if first_step
        % 第一步：完全生成拓扑和所有缓存
        topology = create_topology(satellites, params);
        domains = perform_domain_partition(satellites, topology, params);
        % 预计算所有缓存（复用 precompute_domain_cache 但需要返回全部）
        domain_cache = precompute_domain_cache(full(topology.adjacency_matrix), domains, params);
        domain_nodes = domain_cache.domain_nodes;
        intra_dists = domain_cache.intra_dists;
        inter_links = domain_cache.inter_links;
        domain_graph = full(domain_cache.domain_graph); % 需要修改 precompute_domain_cache 返回 domain_graph
        k_shortest_paths = domain_cache.k_shortest_paths;
        % 将缓存存入 base，供其他函数使用
        assignin('base', 'domain_cache', domain_cache);
        assignin('base', 'domain_nodes', domain_nodes);
        assignin('base', 'intra_dists', intra_dists);
        assignin('base', 'inter_links', inter_links);
        assignin('base', 'domain_graph', domain_graph);
        assignin('base', 'k_shortest_paths', k_shortest_paths);
        
        % 保存到缓存变量
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
        % 增量更新
        [topology, domains, domain_nodes, intra_dists, inter_links, domain_graph, k_shortest_paths] = ...
            incremental_update_network(satellites, params, ...
                prev_satellites, prev_topology, ...
                prev_domains, prev_domain_nodes, ...
                prev_intra_dists, prev_inter_links, ...
                prev_domain_graph, prev_k_shortest_paths);
        % 更新 base 中的缓存
        domain_cache.domain_nodes = domain_nodes;
        domain_cache.intra_dists = intra_dists;
        domain_cache.inter_links = inter_links;
        domain_cache.k_shortest_paths = k_shortest_paths;
        assignin('base', 'domain_cache', domain_cache);
        assignin('base', 'domain_nodes', domain_nodes);
        assignin('base', 'intra_dists', intra_dists);
        assignin('base', 'inter_links', inter_links);
        assignin('base', 'domain_graph', domain_graph);
        assignin('base', 'k_shortest_paths', k_shortest_paths);
        
        % 更新缓存变量
        prev_satellites = satellites;
        prev_topology = topology;
        prev_domains = domains;
        prev_domain_nodes = domain_nodes;
        prev_intra_dists = intra_dists;
        prev_inter_links = inter_links;
        prev_domain_graph = domain_graph;
        prev_k_shortest_paths = k_shortest_paths;
    end
    
    % 保存最后一次的域划分
    if step == params.num_time_steps
        last_domains = domains;
    end
    
    % 4.4 路由性能测试
    % 在调用前从 base 获取所需变量
    domain_cache = evalin('base', 'domain_cache');
    load_matrix = evalin('base', 'load_matrix');
    smooth_max_load = evalin('base', 'smooth_max_load');
    step_results = routing_performance_test(satellites, topology, domains, params, step, domain_cache, load_matrix, smooth_max_load);
    if isfield(step_results, 'failed_requests')
        all_failed_requests = [all_failed_requests; step_results.failed_requests];
    end
    % 4.5 存储结果
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
    
    % 显示当前步长结果
    display_step_summary(step_results, algorithm_names, step);
end

% 5. 综合性能分析
fprintf('\n4. 综合性能分析...\n');
analyze_overall_performance(results, params);

% 在 main.m 的时间步长循环结束后，保存最后一次的拓扑和域信息
last_topology = topology;
last_domains = domains;

% 6. 结果可视化
fprintf('5. 生成可视化结果...\n');
visualize_all_results(results, satellites, params, last_domains, last_topology, all_failed_requests);

fprintf('\n========================================\n');
fprintf('仿真完成！结果已保存至: %s\n', params.output_dir);
fprintf('========================================\n');



function init_rl_variables()
    % 初始化 RL 相关的全局变量（用于仿真测试）
    num_domains = evalin('base', 'params.cross_domain_params.num_domains');
    load_matrix = zeros(num_domains);
    assignin('base', 'load_matrix', load_matrix);
    assignin('base', 'smooth_max_load', 0);
    assignin('base', 'current_src', []);
    assignin('base', 'current_dst', []);
    assignin('base', 'current_satellite', []);
    assignin('base', 'current_domain', []);
    fprintf('RL 变量初始化完成\n');
end