% ========================================
% LEO卫星网络跨域路由仿真 - 运行脚本
% 运行此脚本开始仿真
% ========================================

% 清空环境
clear; clc; close all;

% 添加当前目录到路径
addpath(genpath('.'));

% 运行主程序
main;

% 显示完成信息
fprintf('\n========================================\n');
fprintf('所有文件已保存在当前目录\n');
fprintf('主程序: main.m\n');
fprintf('运行脚本: run_simulation.m\n');
fprintf('结果目录: %s\n', 'simulation_results/');
fprintf('如需仅运行性能测试，请使用 matlab/core/run_performance_test.m\n');
fprintf('========================================\n');