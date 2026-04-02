function display_step_summary(step_results, algorithm_names, step)
    % 显示当前步长的结果摘要
    
    fprintf('\n     ===== 时间步长 %d 性能摘要 =====\n', step);
    fprintf('     算法名称       成功率  平均跳数  平均时延 跨域比例  路由开销\n');
    fprintf('     -----------------------------------------------------------\n');
    
    for i = 1:length(algorithm_names)
        algo_name = algorithm_names{i};
        results = step_results.(algo_name);
        
        % 格式化为更易读的时延
        if results.avg_delay > 1000
            delay_str = sprintf('%.2fs', results.avg_delay/1000);
        else
            delay_str = sprintf('%.1fms', results.avg_delay);
        end
        
        fprintf('     %-12s   %5.1f%%   %7.2f   %10s  %6.1f%%   %.0f\n', ...
            algo_name, ...
            results.success_rate, ...
            results.avg_hops, ...
            delay_str, ...
            results.cross_domain_rate, ...
            results.routing_overhead);
    end
    
    fprintf('     -----------------------------------------------------------\n');
    
    % 显示流量信息
    if isfield(step_results, 'traffic_info')
        info = step_results.traffic_info;
        fprintf('     流量分布: 跨域路由 %.1f%%, 域内路由 %.1f%%\n', ...
            100 * info.cross_domain_ratio, 100 * (1 - info.cross_domain_ratio));
    end
end