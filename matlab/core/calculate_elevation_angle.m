function elevation = calculate_elevation_angle(pos1, pos2, earth_radius)
    % 计算从卫星1到卫星2的仰角
    
    % 卫星1到地心的向量
    vector_to_earth = -pos1;
    
    % 卫星1到卫星2的向量
    vector_to_sat2 = pos2 - pos1;
    
    % 计算夹角
    dot_product = dot(vector_to_sat2, vector_to_earth);
    cos_angle = dot_product / (norm(vector_to_sat2) * norm(vector_to_earth));
    angle = acos(min(max(cos_angle, -1), 1));
    
    % 仰角 = 90° - 夹角
    elevation = pi/2 - angle;
end