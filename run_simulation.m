% 启动项目入口脚本
% 将项目的 matlab 子目录加入 MATLAB 路径并运行主程序

projectRoot = fileparts(mfilename('fullpath'));
matlabDir = fullfile(projectRoot, 'matlab');
if exist(matlabDir, 'dir')
    addpath(genpath(matlabDir));
else
    error('未找到 matlab 目录：%s', matlabDir);
end

main;
