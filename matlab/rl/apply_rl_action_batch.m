function [next_states, rewards, dones] = apply_rl_action_batch(actions)
    % actions: 行向量，元素为 0-15（Python 0-index）
    n = length(actions);
    next_states = cell(n, 1);
    rewards = zeros(n, 1);
    dones = false(n, 1);
    for i = 1:n
        [next_states{i}, rewards(i), dones(i)] = apply_rl_action(actions(i));
    end
end