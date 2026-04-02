function [satellites, params] = generate_constellation(params)
    % 生成Walker星座
    
    fprintf('   生成OneWeb星座: %d颗卫星, %d个轨道面...\n', ...
        params.num_satellites, params.num_planes);
    
    N = params.num_satellites;
    h = params.altitude;
    inc = deg2rad(params.inclination);
    
    % 计算轨道参数
    R = params.earth_radius + h;
    mu = 3.986004418e14;
    orbital_period = 2 * pi * sqrt(R^3 / mu);
    mean_motion = 2 * pi / orbital_period;
    
    params.orbital_period = orbital_period;
    params.mean_motion = mean_motion;
    
    % 初始化卫星结构数组
    satellites = repmat(struct(...
        'id', 0, ...
        'plane', 0, ...
        'position', [0, 0, 0], ...
        'latitude', 0, ...
        'longitude', 0, ...
        'altitude', h, ...
        'phase', 0, ...
        'RAAN', 0, ...
        'neighbors', [], ...
        'domain', 0), N, 1);
    
    % Walker星座生成
    sats_per_plane = N / params.num_planes;
    sat_idx = 1;
    
    for plane = 0:params.num_planes-1
        RAAN = 2 * pi * plane / params.num_planes;
        
        for sat_in_plane = 0:sats_per_plane-1
            % 计算相位
            phase = 2 * pi * (sat_in_plane/sats_per_plane + plane/N);
            
            % 在轨道平面内的位置
            x_orb = R * cos(phase);
            y_orb = R * sin(phase);
            z_orb = 0;
            
            % 旋转到地心惯性坐标系
            Rz_RAAN = [cos(RAAN), -sin(RAAN), 0;
                       sin(RAAN),  cos(RAAN), 0;
                       0,          0,         1];
            
            Rx_inc = [1, 0, 0;
                      0, cos(inc), -sin(inc);
                      0, sin(inc), cos(inc)];
            
            pos_eci = Rz_RAAN * Rx_inc * [x_orb; y_orb; z_orb];
            
            % 转换为经纬度
            x = pos_eci(1); y = pos_eci(2); z = pos_eci(3);
            lat = asin(z / norm(pos_eci));
            lon = atan2(y, x);
            
            % 存储卫星信息
            satellites(sat_idx).id = sat_idx;
            satellites(sat_idx).plane = plane + 1;
            satellites(sat_idx).position = pos_eci';
            satellites(sat_idx).latitude = rad2deg(lat);
            satellites(sat_idx).longitude = rad2deg(lon);
            satellites(sat_idx).phase = phase;
            satellites(sat_idx).RAAN = RAAN;
            
            sat_idx = sat_idx + 1;
        end
    end
    
    fprintf('   星座生成完成\n');
    fprintf('   轨道周期: %.2f 分钟\n', orbital_period / 60);
end