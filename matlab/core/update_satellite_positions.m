function satellites = update_satellite_positions(satellites, params, step)
    % 更新卫星位置（考虑轨道运动）
    
    % 计算当前时间
    current_time = (step - 1) * params.time_step_duration;
    
    % 轨道参数
    R = params.earth_radius + params.altitude;
    inc = deg2rad(params.inclination);
    
    for i = 1:length(satellites)
        % 计算当前相位
        delta_phase = params.mean_motion * current_time;
        current_phase = mod(satellites(i).phase + delta_phase, 2*pi);
        
        % 在轨道平面内的位置
        x_orb = R * cos(current_phase);
        y_orb = R * sin(current_phase);
        z_orb = 0;
        
        % 旋转到地心惯性坐标系
        RAAN = satellites(i).RAAN;
        
        Rz_RAAN = [cos(RAAN), -sin(RAAN), 0;
                   sin(RAAN),  cos(RAAN), 0;
                   0,          0,         1];
        
        Rx_inc = [1, 0, 0;
                  0, cos(inc), -sin(inc);
                  0, sin(inc), cos(inc)];
        
        pos_eci = Rz_RAAN * Rx_inc * [x_orb; y_orb; z_orb];
        
        % 更新卫星位置
        satellites(i).position = pos_eci';
        
        % 更新经纬度
        x = pos_eci(1); y = pos_eci(2); z = pos_eci(3);
        lat = asin(z / norm(pos_eci));
        lon = atan2(y, x);
        
        satellites(i).latitude = rad2deg(lat);
        satellites(i).longitude = rad2deg(lon);
    end
    
    fprintf('     卫星位置更新完成 (时间: %.1f分钟)\n', current_time/60);
end