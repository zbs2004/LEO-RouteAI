function action = dqn_predict(state_vec, action_mask, temperature)
    persistent feature_w1 feature_b1 feature_w2 feature_b2 ...
               value_w1 value_b1 value_w2 value_b2 ...
               advantage_w1 advantage_b1 advantage_w2 advantage_b2 ...
               weights_file weights_file_mtime

    needs_reload = isempty(feature_w1);
    if ~needs_reload && (~exist(weights_file, 'file') || isempty(weights_file))
        needs_reload = true;
    end
    if ~needs_reload
        info = dir(weights_file);
        if isempty(info) || info.datenum ~= weights_file_mtime
            needs_reload = true;
        end
    end

    if needs_reload
        script_dir = fileparts(mfilename('fullpath'));
        project_root = fileparts(fileparts(script_dir));
        candidate_files = {
            fullfile(project_root, 'dqn_weights.mat');
            fullfile(script_dir, 'dqn_weights.mat');
            'dqn_weights.mat'};
        weights_file = '';
        for i = 1:numel(candidate_files)
            if exist(candidate_files{i}, 'file') == 2
                weights_file = candidate_files{i};
                break;
            end
        end
        if isempty(weights_file)
            error('dqn_predict: cannot find dqn_weights.mat in project root or rl folder.');
        end
        data = load(weights_file, 'feature_w1', 'feature_b1', 'feature_w2', 'feature_b2', ...
             'value_w1', 'value_b1', 'value_w2', 'value_b2', ...
             'advantage_w1', 'advantage_b1', 'advantage_w2', 'advantage_b2');
        feature_w1 = data.feature_w1';
        feature_w2 = data.feature_w2';
        value_w1 = data.value_w1';
        value_w2 = data.value_w2';
        advantage_w1 = data.advantage_w1';
        advantage_w2 = data.advantage_w2';
        feature_b1 = reshape(data.feature_b1, 1, []);
        feature_b2 = reshape(data.feature_b2, 1, []);
        value_b1 = reshape(data.value_b1, 1, []);
        value_b2 = reshape(data.value_b2, 1, []);
        advantage_b1 = reshape(data.advantage_b1, 1, []);
        advantage_b2 = reshape(data.advantage_b2, 1, []);
        if size(feature_b1, 2) ~= size(feature_w1, 2) || size(feature_b2, 2) ~= size(feature_w2, 2) || ...
           size(value_b1, 2) ~= size(value_w1, 2) || size(value_b2, 2) ~= size(value_w2, 2) || ...
           size(advantage_b1, 2) ~= size(advantage_w1, 2) || size(advantage_b2, 2) ~= size(advantage_w2, 2)
            error('dqn_predict: loaded bias sizes do not match corresponding weight matrix output dimensions.');
        end
        info = dir(weights_file);
        if isempty(info)
            error('dqn_predict: failed to read file info for %s', weights_file);
        end
        weights_file_mtime = info.datenum;
    end
    x = state_vec(:)';
    if size(x, 2) ~= size(feature_w1, 1)
        error('dqn_predict: state vector length %d does not match feature_w1 input dim %d', size(x,2), size(feature_w1,1));
    end
    % 特征层
    x = max(0, x * feature_w1 + feature_b1);
    x = max(0, x * feature_w2 + feature_b2);
    % 价值流
    v = x * value_w1 + value_b1;
    v = max(0, v);
    v = v * value_w2 + value_b2;
    % 优势流
    a = x * advantage_w1 + advantage_b1;
    a = max(0, a);
    a = a * advantage_w2 + advantage_b2;
    % Q = V + (A - mean(A))
    q = v + a - mean(a, 2);
    % 应用掩码
    if numel(action_mask) ~= numel(q)
        error('dqn_predict: action_mask length %d does not match Q output length %d', numel(action_mask), numel(q));
    end
    q(~action_mask) = -inf;
    if all(~action_mask)
        error('dqn_predict: no valid actions available after masking');
    end

    if nargin < 3 || isempty(temperature) || temperature <= 0
        [~, idx] = max(q);
        action = idx - 1;
        return;
    end

    tau = temperature;
    prob = exp(q / tau);
    prob(~isfinite(prob)) = 0;
    total_prob = sum(prob);
    if total_prob <= 0
        [~, idx] = max(q);
        action = idx - 1;
        return;
    end
    prob = prob / total_prob;
    r = rand();
    cum = cumsum(prob);
    idx = find(cum >= r, 1);
    if isempty(idx)
        idx = length(prob);
    end
    action = idx - 1;
end