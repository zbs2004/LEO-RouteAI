function analyze_overall_performance(results, params)
    % 综合性能分析
    
    algorithm_names = params.routing_algorithms;
    
    fprintf('\n    ====== 算法综合性能分析 ======\n\n');
    
    % 计算各算法的平均性能
    performance_table = cell(length(algorithm_names) + 1, 6);
    performance_table{1, 1} = '算法名称';
    performance_table{1, 2} = '平均成功率(%)';
    performance_table{1, 3} = '平均跳数';
    performance_table{1, 4} = '平均时延(ms)';
    performance_table{1, 5} = '跨域成功率(%)';
    performance_table{1, 6} = '相对开销';
    
    for algo_idx = 1:length(algorithm_names)
        algo_name = algorithm_names{algo_idx};
        algo_results = results.(algo_name);
        
        performance_table{algo_idx+1, 1} = algo_name;
        performance_table{algo_idx+1, 2} = sprintf('%.1f', mean(algo_results.success_rates));
        performance_table{algo_idx+1, 3} = sprintf('%.2f', mean(algo_results.avg_hops));
        performance_table{algo_idx+1, 4} = sprintf('%.1f', mean(algo_results.avg_delays));
        performance_table{algo_idx+1, 5} = sprintf('%.1f', mean(algo_results.cross_domain_rates));
        
        % 计算相对开销（归一化）
        all_overheads = [];
        for i = 1:length(algorithm_names)
            all_overheads = [all_overheads; results.(algorithm_names{i}).routing_overheads];
        end
        
        avg_overhead = mean(algo_results.routing_overheads);
        norm_overhead = avg_overhead / max(mean(all_overheads), 1);
        performance_table{algo_idx+1, 6} = sprintf('%.2f', norm_overhead);
    end
    
    % 显示性能表格
    fprintf('     %-12s %12s %10s %12s %12s %12s\n', ...
        performance_table{1, :});
    fprintf('     -------------------------------------------------------------\n');
    
    for i = 2:size(performance_table, 1)
        fprintf('     %-12s %12s %10s %12s %12s %12s\n', ...
            performance_table{i, :});
    end
    
    fprintf('\n    ====== 性能对比分析 ======\n');
    
    % 进行统计显著性测试
    for i = 1:length(algorithm_names)
        algo_i = algorithm_names{i};
        rates_i = results.(algo_i).success_rates;
        for j = i+1:length(algorithm_names)
            algo_j = algorithm_names{j};
            rates_j = results.(algo_j).success_rates;
            
            avg_i = mean(rates_i);
            avg_j = mean(rates_j);
            n = length(rates_i);
            diff = avg_i - avg_j;
            pooled_std = sqrt((std(rates_i)^2 + std(rates_j)^2) / 2);
            se = pooled_std * sqrt(2/n);
            ci_low = diff - 1.96 * se;
            ci_high = diff + 1.96 * se;
            
            if ci_low > 0
                fprintf('     %s 在成功率上显著优于 %s (%.1f%% vs %.1f%%, 95%% CI [%.2f, %.2f])\n', ...
                    algo_i, algo_j, avg_i, avg_j, ci_low, ci_high);
            elseif ci_high < 0
                fprintf('     %s 在成功率上显著优于 %s (%.1f%% vs %.1f%%, 95%% CI [%.2f, %.2f])\n', ...
                    algo_j, algo_i, avg_j, avg_i, ci_low, ci_high);
            else
                fprintf('     %s 和 %s 成功率无显著差异 (%.1f%% vs %.1f%%)\n', ...
                    algo_i, algo_j, avg_i, avg_j);
            end
        end
    end
end