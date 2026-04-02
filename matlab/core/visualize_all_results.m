function visualize_all_results(results, satellites, params, domains, topology, failed_requests)
    % 综合可视化模块
    % 输入：
    %   results - 仿真结果数据
    %   satellites - 卫星结构数组
    %   params - 参数结构
    %   domains - 域划分信息
    %   topology - 网络拓扑信息
    
    fprintf('   生成可视化图表...\n');
    
    algorithm_names = params.routing_algorithms;
    num_algorithms = length(algorithm_names);
    num_steps = params.num_time_steps;
    
    % 时间向量（分钟）
    time_vector = (0:num_steps-1) * params.time_step_duration / 60;
    
    % ========== 图1：路由算法性能对比图 ==========
    plot_performance_comparison(results, algorithm_names, time_vector, params);
    
    % ========== 图2：综合性能雷达图 ==========
    plot_performance_radar(results, algorithm_names, params);
    
    % ========== 图3：卫星网络拓扑图 ==========
    if nargin >= 5 && ~isempty(topology) && ~isempty(domains)
        %plot_satellite_topology(satellites, topology, domains, params, num_steps);
    end
    
    % ========== 图4：卫星分布与域划分图 ==========
    %plot_satellite_distribution(satellites, domains, params);
    
    % ========== 保存数据 ==========
    if params.save_data
        data_file = fullfile(params.output_dir, 'data', 'simulation_results.mat');
        save(data_file, 'results', 'params', 'satellites', 'algorithm_names', 'time_vector');
        
        % 导出为CSV
        export_to_csv(results, algorithm_names, time_vector, params);
        
        fprintf('     仿真数据已保存: %s\n', data_file);
    end

    if nargin >= 6 && ~isempty(failed_requests)
        %plot_failed_requests(satellites, failed_requests, params);
    end
    
end

%% ========== 子函数1：路由算法性能对比图 ==========
function plot_performance_comparison(results, algorithm_names, time_vector, params)
    % 绘制路由算法性能对比图
    
    num_algorithms = length(algorithm_names);
    
    figure('Position', [100, 100, 1400, 800], 'Name', '路由算法性能对比');
    
    colors = lines(num_algorithms);
    markers = {'o-', 's-', '^-', 'd-', '*-', 'x-', '+-', 'v-'};
    
    % 1.1 成功率对比
    subplot(2, 3, 1);
    hold on;
    
    for algo_idx = 1:num_algorithms
        algo_name = algorithm_names{algo_idx};
        success_rates = results.(algo_name).success_rates;
        
        plot(time_vector, success_rates, markers{mod(algo_idx-1, length(markers))+1}, ...
            'Color', colors(algo_idx, :), 'LineWidth', 2, 'MarkerSize', 6, ...
            'DisplayName', algo_name);
    end
    
    xlabel('仿真时间 (分钟)');
    ylabel('路由成功率 (%)');
    title('(a) 路由成功率随时间变化');
    grid on;
    legend('Location', 'best', 'Box', 'off');
    ylim([0, 105]);
    
    % 1.2 平均跳数对比
    subplot(2, 3, 2);
    hold on;
    
    for algo_idx = 1:num_algorithms
        algo_name = algorithm_names{algo_idx};
        avg_hops = results.(algo_name).avg_hops;
        
        plot(time_vector, avg_hops, markers{mod(algo_idx-1, length(markers))+1}, ...
            'Color', colors(algo_idx, :), 'LineWidth', 2, 'MarkerSize', 6, ...
            'DisplayName', algo_name);
    end
    
    xlabel('仿真时间 (分钟)');
    ylabel('平均跳数');
    title('(b) 平均跳数随时间变化');
    grid on;
    legend('Location', 'best', 'Box', 'off');
    
    % 1.3 平均时延对比
    subplot(2, 3, 3);
    hold on;
    
    for algo_idx = 1:num_algorithms
        algo_name = algorithm_names{algo_idx};
        avg_delays = results.(algo_name).avg_delays;
        
        plot(time_vector, avg_delays, markers{mod(algo_idx-1, length(markers))+1}, ...
            'Color', colors(algo_idx, :), 'LineWidth', 2, 'MarkerSize', 6, ...
            'DisplayName', algo_name);
    end
    
    xlabel('仿真时间 (分钟)');
    ylabel('平均时延 (ms)');
    title('(c) 平均时延随时间变化');
    grid on;
    legend('Location', 'best', 'Box', 'off');
    
    % 1.4 跨域路由比例
    subplot(2, 3, 4);
    hold on;
    
    for algo_idx = 1:num_algorithms
        algo_name = algorithm_names{algo_idx};
        cross_domain_rates = results.(algo_name).cross_domain_rates;
        
        plot(time_vector, cross_domain_rates, markers{mod(algo_idx-1, length(markers))+1}, ...
            'Color', colors(algo_idx, :), 'LineWidth', 2, 'MarkerSize', 6, ...
            'DisplayName', algo_name);
    end
    
    xlabel('仿真时间 (分钟)');
    ylabel('跨域路由比例 (%)');
    title('(d) 跨域路由比例随时间变化');
    grid on;
    legend('Location', 'best', 'Box', 'off');
    ylim([0, 105]);
    
    % 1.5 路由开销对比
    subplot(2, 3, 5);
    hold on;
    
    for algo_idx = 1:num_algorithms
        algo_name = algorithm_names{algo_idx};
        routing_overheads = results.(algo_name).routing_overheads;
        
        plot(time_vector, routing_overheads, markers{mod(algo_idx-1, length(markers))+1}, ...
            'Color', colors(algo_idx, :), 'LineWidth', 2, 'MarkerSize', 6, ...
            'DisplayName', algo_name);
    end
    
    xlabel('仿真时间 (分钟)');
    ylabel('路由开销');
    title('(e) 路由开销随时间变化');
    grid on;
    legend('Location', 'best', 'Box', 'off');
    
    % 1.6 算法综合性能条形图
    subplot(2, 3, 6);
    
    % 计算各算法的平均性能
    performance_data = zeros(num_algorithms, 3);
    for algo_idx = 1:num_algorithms
        algo_name = algorithm_names{algo_idx};
        
        % 成功率（归一化到0-1）
        avg_success = mean(results.(algo_name).success_rates) / 100;
        
        % 跳数优化（跳数越少越好）
        avg_hops = mean(results.(algo_name).avg_hops);
        norm_hops = max(0, 1 - avg_hops / 15);
        
        % 时延优化（时延越少越好）
        avg_delay = mean(results.(algo_name).avg_delays);
        norm_delay = max(0, 1 - avg_delay / 200);
        
        performance_data(algo_idx, :) = [avg_success, norm_hops, norm_delay];
    end
    
    bar_handles = bar(performance_data);
    
    % 设置颜色
    bar_handles(1).FaceColor = [0.2, 0.6, 0.8];
    bar_handles(2).FaceColor = [0.8, 0.4, 0.2];
    bar_handles(3).FaceColor = [0.4, 0.8, 0.2];
    
    xlabel('算法');
    ylabel('性能指标（归一化）');
    title('(f) 算法综合性能对比');
    legend({'成功率', '跳数优化', '时延优化'}, 'Location', 'best', 'Box', 'off');
    grid on;
    ylim([0, 1.1]);
    set(gca, 'XTickLabel', algorithm_names);
    
    % 保存图像
    if params.save_figures
        fig_file = fullfile(params.output_dir, 'figures', 'performance_comparison.png');
        saveas(gcf, fig_file);
        fprintf('     性能对比图已保存: %s\n', fig_file);
    end
end

%% ========== 子函数2：综合性能雷达图 ==========
function plot_performance_radar(results, algorithm_names, params)
    % 绘制综合性能雷达图
    
    num_algorithms = length(algorithm_names);
    
    figure('Position', [200, 200, 800, 600], 'Name', '算法综合性能雷达图');
    
    % 计算各算法的平均性能（归一化）
    performance_matrix = zeros(num_algorithms, 4);
    
    for algo_idx = 1:num_algorithms
        algo_name = algorithm_names{algo_idx};
        
        % 成功率（0-1）
        avg_success = mean(results.(algo_name).success_rates) / 100;
        
        % 跳数优化（跳数越少越好）
        avg_hops = mean(results.(algo_name).avg_hops);
        norm_hops = max(0, 1 - avg_hops / 15);
        
        % 时延优化（时延越少越好）
        avg_delay = mean(results.(algo_name).avg_delays);
        norm_delay = max(0, 1 - avg_delay / 200);
        
        % 跨域能力
        avg_cross_domain = mean(results.(algo_name).cross_domain_rates) / 100;
        
        performance_matrix(algo_idx, :) = [avg_success, norm_hops, norm_delay, avg_cross_domain];
    end
    
    % 绘制雷达图
    categories = {'成功率', '跳数优化', '时延优化', '跨域能力'};
    
    % 为雷达图创建闭合数据
    perf_closed = [performance_matrix, performance_matrix(:, 1)];
    cats_closed = [categories, categories(1)];
    
    % 创建极坐标图
    theta = linspace(0, 2*pi, length(cats_closed));
    
    polaraxes;
    hold on;
    
    for algo_idx = 1:num_algorithms
        rho = perf_closed(algo_idx, :);
        
        polarplot(theta, rho, '-o', 'LineWidth', 2, 'MarkerSize', 8, ...
            'DisplayName', algorithm_names{algo_idx});
    end
    
    % 设置极坐标属性
    thetaticks(rad2deg(theta(1:end-1)));
    thetaticklabels(categories);
    rlim([0, 1]);
    rticks(0:0.2:1);
    rticklabels({'0', '0.2', '0.4', '0.6', '0.8', '1.0'});
    
    title('路由算法综合性能雷达图');
    legend('Location', 'bestoutside', 'Box', 'off');
    grid on;
    
    % 保存图像
    if params.save_figures
        fig_file = fullfile(params.output_dir, 'figures', 'performance_radar.png');
        saveas(gcf, fig_file);
        fprintf('     性能雷达图已保存: %s\n', fig_file);
    end
end

%% ========== 子函数3：卫星网络拓扑图（域内/域间链路区分）==========
function plot_satellite_topology(satellites, topology, domains, params, step)
    % 绘制卫星网络拓扑图（区分域内和域间链路）
    
    fprintf('     绘制卫星网络拓扑图...\n');
    
    % 创建图形
    figure('Position', [100, 100, 1200, 400], 'Name', '卫星网络拓扑图');
    
    % 获取卫星经纬度
    longitudes = [satellites.longitude];
    latitudes = [satellites.latitude];
    
    % 获取邻接矩阵
    adjacency = full(topology.adjacency_matrix);
    num_sats = length(satellites);
    
    % ========== 子图1：完整的卫星网络拓扑 ==========
    subplot(1, 3, 1);
    hold on;
    
    % 统计域内和域间链路数量
    intra_link_count = 0;
    inter_link_count = 0;
    
    % 先绘制域内链路（灰色细线）
    for i = 1:num_sats
        for j = i+1:num_sats
            if adjacency(i, j) > 0
                domain_i = domains.domain_assignment(i);
                domain_j = domains.domain_assignment(j);
                
                if domain_i == domain_j
                    % 域内链路：灰色细线
                    plot([longitudes(i), longitudes(j)], ...
                         [latitudes(i), latitudes(j)], ...
                         'Color', [0.7, 0.7, 0.7], 'LineWidth', 0.5);
                    intra_link_count = intra_link_count + 1;
                end
            end
        end
    end
    
    % 再绘制域间链路（红色粗线）
    for i = 1:num_sats
        for j = i+1:num_sats
            if adjacency(i, j) > 0
                domain_i = domains.domain_assignment(i);
                domain_j = domains.domain_assignment(j);
                
                if domain_i ~= domain_j
                    % 域间链路：红色粗线
                    plot([longitudes(i), longitudes(j)], ...
                         [latitudes(i), latitudes(j)], ...
                         'Color', [1, 0, 0], 'LineWidth', 1.0);
                    inter_link_count = inter_link_count + 1;
                end
            end
        end
    end
    
    fprintf('       绘制了 %d 条域内链路（灰色），%d 条域间链路（红色）\n', intra_link_count, inter_link_count);
    
    % 绘制卫星节点（按域着色）
    num_domains = domains.num_domains;
    domain_colors = lines(num_domains);
    
    for d = 1:num_domains
        domain_sats = find(domains.domain_assignment == d);
        
        if ~isempty(domain_sats)
            % 绘制卫星节点
            scatter(longitudes(domain_sats), latitudes(domain_sats), ...
                    30, domain_colors(d, :), 'filled', ...
                    'MarkerEdgeColor', 'k', 'LineWidth', 0.5);
        end
    end
    
    % 添加赤道和本初子午线作为参考
    plot([-180, 180], [0, 0], 'k-', 'LineWidth', 0.5, 'DisplayName', '赤道'); % 赤道
    plot([0, 0], [-90, 90], 'k-', 'LineWidth', 0.5, 'DisplayName', '本初子午线'); % 本初子午线
    
    % 创建自定义图例（只显示2种链路类型和卫星节点）
    h_intra = plot(NaN, NaN, 'Color', [0.7, 0.7, 0.7], 'LineWidth', 1.5, 'DisplayName', '域内链路');
    h_inter = plot(NaN, NaN, 'Color', [1, 0, 0], 'LineWidth', 1.5, 'DisplayName', '域间链路');
    h_sat = scatter(NaN, NaN, 30, [0.2, 0.6, 0.8], 'filled', 'MarkerEdgeColor', 'k', 'DisplayName', '卫星节点');
    
    legend([h_intra, h_inter, h_sat], 'Location', 'best', 'Box', 'off', 'FontSize', 9);
    
    grid on;
    xlabel('经度 (°)');
    ylabel('纬度 (°)');
    title(sprintf('卫星网络拓扑\n总卫星: %d, 总链路: %d', ...
        num_sats, intra_link_count + inter_link_count));
    axis([-180 180 -90 90]);
    
    % ========== 子图2：域间链路特写 ==========
    subplot(1, 3, 2);
    hold on;
    
    % 只绘制域间链路（突出显示）
    for i = 1:num_sats
        for j = i+1:num_sats
            if adjacency(i, j) > 0
                domain_i = domains.domain_assignment(i);
                domain_j = domains.domain_assignment(j);
                
                if domain_i ~= domain_j
                    % 域间链路：红色粗线
                    plot([longitudes(i), longitudes(j)], ...
                         [latitudes(i), latitudes(j)], ...
                         'Color', [1, 0, 0], 'LineWidth', 1.2);
                    
                    % 标记边界节点（较大圆点）
                    scatter(longitudes(i), latitudes(i), 40, ...
                           domain_colors(domain_i, :), 'filled', ...
                           'MarkerEdgeColor', 'k', 'LineWidth', 0.8);
                    scatter(longitudes(j), latitudes(j), 40, ...
                           domain_colors(domain_j, :), 'filled', ...
                           'MarkerEdgeColor', 'k', 'LineWidth', 0.8);
                end
            end
        end
    end
    
    % 添加域边界（简化表示）
    if strcmp(domains.partition_method, 'latitude')
        % 基于纬度划分的域边界
        latitudes_all = [satellites.latitude];
        lat_min = min(latitudes_all);
        lat_max = max(latitudes_all);
        lat_range = lat_max - lat_min;
        domain_height = lat_range / num_domains;
        
        for d = 1:num_domains-1
            boundary_lat = lat_min + d * domain_height;
            plot([-180, 180], [boundary_lat, boundary_lat], 'k--', ...
                 'LineWidth', 1.0, 'DisplayName', '域边界');
        end
    end
    
    grid on;
    xlabel('经度 (°)');
    ylabel('纬度 (°)');
    title(sprintf('域间链路特写\n域间链路: %d条 (%.1f%%)', ...
        inter_link_count, 100*inter_link_count/(intra_link_count+inter_link_count)));
    axis([-180 180 -90 90]);
    
    % 添加图例
    h_inter2 = plot(NaN, NaN, 'Color', [1, 0, 0], 'LineWidth', 1.5, 'DisplayName', '域间链路');
    h_boundary = plot(NaN, NaN, 'k--', 'LineWidth', 1.0, 'DisplayName', '域边界');
    legend([h_inter2, h_boundary], 'Location', 'best', 'Box', 'off', 'FontSize', 9);
    
    % ========== 子图3：链路类型统计 ==========
    subplot(1, 3, 3);
    
    % 准备数据
    link_types = {'域内链路', '域间链路'};
    link_counts = [intra_link_count, inter_link_count];
    link_percentages = 100 * link_counts / sum(link_counts);
    
    % 绘制饼图
    colors = [0.7, 0.7, 0.7; 1.0, 0.0, 0.0]; % 灰色和红色
    pie(link_counts, link_types);
    colormap(gca, colors);
    
    % 添加数值标签
    text_labels = cell(length(link_types), 1);
    for i = 1:length(link_types)
        text_labels{i} = sprintf('%s\n%d条 (%.1f%%)', link_types{i}, link_counts(i), link_percentages(i));
    end
    
    % 清除原有图例，使用自定义标签
    legend(text_labels, 'Location', 'best', 'Box', 'off', 'FontSize', 9);
    title('链路类型统计');
    
    % 添加整体统计信息
    annotation_text = sprintf('整体统计:\n卫星总数: %d\n域划分数: %d\n平均每域: %.1f颗\n网络密度: %.2f%%', ...
        num_sats, num_domains, num_sats/num_domains, topology.connection_density*100);
    
    annotation('textbox', [0.75, 0.15, 0.2, 0.1], 'String', annotation_text, ...
        'FontSize', 9, 'BackgroundColor', [0.95, 0.95, 0.95], 'EdgeColor', 'k');
    
    % 保存图像
    if params.save_figures
        fig_file = fullfile(params.output_dir, 'figures', 'satellite_topology.png');
        saveas(gcf, fig_file);
        fprintf('     卫星拓扑图已保存: %s\n', fig_file);
    end
end

%% ========== 子函数4：卫星分布与域划分图 ==========
function plot_satellite_distribution(satellites, domains, params)
    % 绘制卫星分布与域划分图
    
    fprintf('     绘制卫星分布与域划分图...\n');
    
    figure('Position', [100, 100, 1200, 400], 'Name', '卫星分布与域划分');
    
    % ========== 子图1：卫星地理分布 ==========
    subplot(1, 3, 1);
    
    % 获取卫星经纬度
    longitudes = [satellites.longitude];
    latitudes = [satellites.latitude];
    
    % 检查是否有域信息
    if ~isempty(domains) && isfield(domains, 'domain_assignment')
        % 使用domains中的域信息
        domain_assignment = domains.domain_assignment;
        num_domains = domains.num_domains;
        domain_colors = hsv(num_domains);
        
        hold on;
        for d = 1:num_domains
            domain_sats = find(domain_assignment == d);
            
            if ~isempty(domain_sats)
                % 绘制该域的卫星
                scatter(longitudes(domain_sats), latitudes(domain_sats), ...
                        40, domain_colors(d, :), 'filled', ...
                        'DisplayName', sprintf('域 %d (%d颗)', d, length(domain_sats)));
            end
        end
        
        % 绘制地球大陆轮廓（简化）
        load  coastlines.mat;
        plot(coastlon, coastlat, 'k-', 'LineWidth', 0.5);
        
        xlabel('经度 (°)');
        ylabel('纬度 (°)');
        title('卫星地理分布与域划分');
        legend('Location', 'best', 'Box', 'off');
        grid on;
        axis([-180 180 -90 90]);
    else
        % 如果没有域信息，仅显示卫星分布
        scatter(longitudes, latitudes, 40, 'b', 'filled');
        xlabel('经度 (°)');
        ylabel('纬度 (°)');
        title('卫星地理分布');
        grid on;
        axis([-180 180 -90 90]);
    end
    
    % ========== 子图2：卫星高度分布 ==========
    subplot(1, 3, 2);
    
    % 绘制卫星高度分布
    altitudes = [satellites.altitude] / 1000; % 转换为km
    
    if ~isempty(domains) && isfield(domains, 'domain_assignment')
        % 按域着色
        domain_assignment = domains.domain_assignment;
        num_domains = domains.num_domains;
        domain_colors = hsv(num_domains);
        
        hold on;
        for d = 1:num_domains
            domain_sats = find(domain_assignment == d);
            
            if ~isempty(domain_sats)
                scatter(longitudes(domain_sats), altitudes(domain_sats), ...
                        40, domain_colors(d, :), 'filled', ...
                        'DisplayName', sprintf('域 %d', d));
            end
        end
        
        xlabel('经度 (°)');
        ylabel('高度 (km)');
        title('卫星高度分布');
        legend('Location', 'best', 'Box', 'off');
        grid on;
        axis([-180 180 params.altitude/1000-100 params.altitude/1000+100]);
    else
        scatter(longitudes, altitudes, 40, 'b', 'filled');
        xlabel('经度 (°)');
        ylabel('高度 (km)');
        title('卫星高度分布');
        grid on;
        axis([-180 180 params.altitude/1000-100 params.altitude/1000+100]);
    end
    
    % ========== 子图3：卫星数量统计 ==========
    subplot(1, 3, 3);
    
    if ~isempty(domains) && isfield(domains, 'domain_assignment')
        % 统计各域卫星数量
        domain_assignment = domains.domain_assignment;
        num_domains = domains.num_domains;
        
        domain_counts = zeros(num_domains, 1);
        for d = 1:num_domains
            domain_counts(d) = sum(domain_assignment == d);
        end
        
        % 绘制饼图
        labels = cell(num_domains, 1);
        for d = 1:num_domains
            labels{d} = sprintf('域 %d\n%d颗 (%.1f%%)', ...
                d, domain_counts(d), 100*domain_counts(d)/sum(domain_counts));
        end
        
        pie(domain_counts, labels);
        title('各域卫星数量分布');
    else
        % 如果没有域信息，显示简单的卫星总数
        text(0.5, 0.5, sprintf('卫星总数: %d\n轨道高度: %.0f km\n轨道倾角: %.0f°', ...
            length(satellites), params.altitude/1000, params.inclination), ...
            'HorizontalAlignment', 'center', 'FontSize', 12);
        axis off;
        title('星座参数');
    end
    
    % 保存图像
    if params.save_figures
        fig_file = fullfile(params.output_dir, 'figures', 'satellite_distribution.png');
        saveas(gcf, fig_file);
        fprintf('     卫星分布图已保存: %s\n', fig_file);
    end
end

%% ========== 子函数5：导出数据到CSV ==========
function export_to_csv(results, algorithm_names, time_vector, params)
    % 导出数据为CSV格式
    
    num_steps = length(time_vector);
    num_algorithms = length(algorithm_names);
    
    % 创建数据表格
    data_table = table();
    data_table.Time_Minutes = time_vector';
    
    % 添加各算法的性能数据
    for algo_idx = 1:num_algorithms
        algo_name = algorithm_names{algo_idx};
        
        % 避免字段名冲突
        field_prefix = strrep(algo_name, '_', '');
        
        data_table.(sprintf('%s_SuccessRate', field_prefix)) = results.(algo_name).success_rates;
        data_table.(sprintf('%s_AvgHops', field_prefix)) = results.(algo_name).avg_hops;
        data_table.(sprintf('%s_AvgDelay', field_prefix)) = results.(algo_name).avg_delays;
        data_table.(sprintf('%s_CrossDomainRate', field_prefix)) = results.(algo_name).cross_domain_rates;
        data_table.(sprintf('%s_RoutingOverhead', field_prefix)) = results.(algo_name).routing_overheads;
    end
    
    % 写入CSV文件
    csv_file = fullfile(params.output_dir, 'data', 'performance_data.csv');
    writetable(data_table, csv_file);
    
    fprintf('     性能数据已导出为CSV: %s\n', csv_file);
end
%% 失败请求分布函数
function plot_failed_requests(satellites, failed_requests, params)
    if isempty(failed_requests)
        return;
    end
    figure('Name', '失败请求分布', 'Position', [100, 100, 800, 400]);
    hold on;
    
    % 背景：所有卫星（灰色小点）
    longitudes = [satellites.longitude];
    latitudes = [satellites.latitude];
    scatter(longitudes, latitudes, 10, [0.7 0.7 0.7], 'filled', 'DisplayName', '卫星节点');
    
    % 绘制失败请求连线（红色虚线）
    for k = 1:length(failed_requests)
        src = failed_requests(k).src;
        dst = failed_requests(k).dst;
        plot([satellites(src).longitude, satellites(dst).longitude], ...
             [satellites(src).latitude, satellites(dst).latitude], ...
             'r--', 'LineWidth', 1, 'DisplayName', '失败路径');
    end
    % 避免图例重复
    h = findobj(gca, 'Type', 'line', 'LineStyle', '--');
    if ~isempty(h)
        legend(h(1), '失败路径', 'Location', 'best');
    end
    xlabel('经度 (°)');
    ylabel('纬度 (°)');
    title('失败请求分布（红色虚线表示无法建立路径的源-目的对）');
    axis([-180 180 -90 90]);
    grid on;
    
    % 保存图像
    if params.save_figures
        fig_file = fullfile(params.output_dir, 'figures', 'failed_requests.png');
        saveas(gcf, fig_file);
        fprintf('     失败请求分布图已保存: %s\n', fig_file);
    end
end