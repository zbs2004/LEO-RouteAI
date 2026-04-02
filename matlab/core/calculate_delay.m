function delay = calculate_delay(distance, hops, params)
    % 计算端到端时延
    
    % 光速 (m/s)
    light_speed = 3e8;
    
    % 传播时延 (s -> ms)
    propagation_delay = (distance / light_speed) * 1000;
    
    % 处理时延（每跳 0.1ms）
    processing_delay_per_hop = 0.1; % ms
    processing_delay = hops * processing_delay_per_hop;
    
    % 排队时延（简化模型，基于跳数的随机值）
    queueing_delay_per_hop = 0.05; % ms
    queueing_delay = hops * queueing_delay_per_hop;
    
    % 总时延
    delay = propagation_delay + processing_delay + queueing_delay;
end