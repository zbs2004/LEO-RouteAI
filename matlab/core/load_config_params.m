function params = load_config_params()
    % 加载仿真配置参数
    
    params = struct();
    
    % ========== 基础参数 ==========
    params.simulation_id = datestr(now, 'yyyymmdd_HHMMSS');
    params.earth_radius = 6371e3;           % 地球半径 (m)
    
    % ========== 卫星星座参数 ==========
    params.num_satellites = 648;             % 卫星总数
    params.altitude = 1200e3;                % 轨道高度 (m)
    params.inclination = 87.9;                % 轨道倾角 (°)
    params.num_planes = 18;                  % 轨道面数
    
    % ========== 网络参数 ==========
    params.max_isl_distance = 6000e3;       % 最大星间链路距离 (m)
    params.min_elevation = 5;               % 最小仰角 (°)
    params.link_availability = 0.95;        % 链路可用率
    
    % ========== 时延计算参数 ==========
    params.light_speed = 3e8;               % 光速 (m/s)
    params.processing_delay_per_hop = 0.1;  % 每跳处理时延 (ms)
    params.queueing_delay_per_hop = 0.05;   % 每跳排队时延 (ms)
    
    % ========== 动态仿真参数 ==========
    params.num_time_steps = 12;             % 时间步长数量
    params.time_step_duration = 300;        % 每个步长持续时间 (s) = 5分钟
    params.total_simulation_time = params.num_time_steps * params.time_step_duration;
    
    % ========== 路由算法参数 ==========
    params.routing_algorithms = {'dijkstra', 'minhop', 'cross_domain', 'rl'};
    
    params.dijkstra_params = struct(...
        'name', 'Dijkstra最短路径算法 (距离)', ...
        'weight_type', 'distance');
    
    params.minhop_params = struct(...
        'name', '最小跳数路由', ...
        'weight_type', 'hop');
    
    params.cross_domain_params = struct(...
        'name', '跨域路由算法', ...
        'num_domains', 16, ...
        'domain_partition_method', 'latitude', ...
        'load_balancing', true, ...
        'load_balancing_factor', 0.1, ...          % 减小基础因子
        'load_weight_alpha', 0.5, ...           % EWMA平滑因子
        'max_detour_ratio', 1.3, ...                 % 最大绕行倍数
        'adaptive_factor', true);                    % 是否启用自适应因子
    params.cross_domain_params.K_paths = 4;   % 候选域路径数量
    params.cross_domain_params.rl_load_penalty = 1.0;   % 负载惩罚系数
    params.cross_domain_params.load_smoothing_alpha = 0.2;  % 保留备用

    % ========== 测试参数 ==========
    params.num_routing_tests = 50;          % 每个时间步长的路由测试次数
    params.max_hops = 20;                   % 最大允许跳数
    
    % ========== 输出参数 ==========
    params.output_dir = 'simulation_results/';
    params.save_figures = true;
    params.save_data = true;
    params.verbose = true;
    
    % 创建输出目录
    if ~exist(params.output_dir, 'dir')
        mkdir(params.output_dir);
        mkdir(fullfile(params.output_dir, 'figures'));
        mkdir(fullfile(params.output_dir, 'data'));
        mkdir(fullfile(params.output_dir, 'logs'));
    end
    
    % 保存配置文件
    if params.save_data
        config_file = fullfile(params.output_dir, 'config.mat');
        save(config_file, 'params');
    end
    
    fprintf('   参数配置完成，仿真ID: %s\n', params.simulation_id);
end