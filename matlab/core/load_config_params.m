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
        'load_balancing_factor', 0.2, ...         % 负载惩罚强度，使候选路径生成考虑当前负载
        'hop_penalty_factor', 0.1, ...           % hop 惩罚，使候选路径更倾向少跳
        'load_weight_alpha', 0.5, ...           % EWMA平滑因子
        'max_detour_ratio', 1.0, ...                 % 最大绕行倍数（候选路径优先最短）
        'adaptive_factor', false);                    % 是否启用自适应因子
    params.cross_domain_params.K_paths = 4;   % 候选域路径数量
    params.cross_domain_params.rl_load_penalty = 1.0;   % 负载惩罚系数
    params.cross_domain_params.load_smoothing_alpha = 0.2;  % 保留备用

    % ========== 队列与处理模型（用于压力场景） ==========
    % 这些参数用于模拟域级别的排队与处理延迟，单位说明：容量(包)，速率(包/步长)，时间(ms)
    params.queue_model = struct();
    params.queue_model.enable = true;                  % 是否启用队列模型
    params.queue_model.capacity = 500;                 % 每个域默认队列容量（包）
    params.queue_model.service_rate = 1.0;             % 每个时间步长每域处理包数（包/步）
    params.queue_model.time_per_step_ms = 10.0;        % 每个队列时间步对应的额外延迟（ms）
    params.queue_model.delay_weight = 0.01;            % 将队列延迟转换为奖励/惩罚的权重
    params.queue_model.drop_penalty = 20.0;            % 丢包/拒绝服务的惩罚分值
    % 可选的域级别配置（数组长度为域数），默认使用统一值
    params.queue_model.domain_capacities = ones(params.cross_domain_params.num_domains, 1) * params.queue_model.capacity;
    params.queue_model.domain_service_rate = ones(params.cross_domain_params.num_domains, 1) * params.queue_model.service_rate;

    % ========== RL 与 reward 归一化参数 ==========
    % 这些参数主要影响与 Python 端 RL 训练与 reward 处理的交互与一致性
    params.rl_params = struct();
    params.rl_params.enabled = true;                   % 是否启用 RL 算法比较
    params.rl_params.state_dim = 13;                   % 状态维度（含新增队列特征）
    params.rl_params.learning_rate = 1e-4;             % DQN 学习率（仅供记录）
    params.rl_params.batch_size = 64;                  % 训练批次大小（仅供记录）
    params.rl_params.tau = 0.001;                      % 软更新系数（Polyak）
    params.rl_params.gamma = 0.99;                     % 折扣因子
    params.rl_params.n_step = 3;                       % n-step returns

    % reward 归一化/稳定化参数（用于与 Python 环境同步）
    params.reward_norm = struct();
    params.reward_norm.enabled = true;
    params.reward_norm.ema_alpha = 0.005;              % EMA 平滑系数
    params.reward_norm.min_count = 200;                % warm-up 最小样本数
    params.reward_norm.clip = 5.0;                     % 归一化后剪切阈值（绝对值）


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
        % 同时导出为 JSON 格式，便于 Python 端直接读取和复现实验配置
        try
            % 使用 jsonencode 将 struct 转为 JSON 字符串
            json_txt = jsonencode(params);
            config_json_file = fullfile(params.output_dir, 'config.json');
            fid = fopen(config_json_file, 'w');
            if fid ~= -1
                fprintf(fid, '%s', json_txt);
                fclose(fid);
            else
                warning('无法打开 %s 写入 config.json', config_json_file);
            end
        catch ME
            warning('保存 config.json 失败: %s', ME.message);
        end
    end
    
    fprintf('   参数配置完成，仿真ID: %s\n', params.simulation_id);
end