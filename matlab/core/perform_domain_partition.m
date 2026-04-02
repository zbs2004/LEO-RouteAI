function domains = perform_domain_partition(satellites, topology, params)
    % 执行域划分
    
    num_sats = length(satellites);
    num_domains = params.cross_domain_params.num_domains;
    partition_method = params.cross_domain_params.domain_partition_method;
    
    % 初始化域分配
    domain_assignment = zeros(num_sats, 1);
    
    switch partition_method
        case 'latitude'
            % 基于纬度划分
            latitudes = [satellites.latitude];
            lat_range = max(latitudes) - min(latitudes);
            if lat_range == 0
                warning('所有卫星纬度相同，无法基于纬度划分，将采用均匀分配');
                for i = 1:num_sats
                    domain_assignment(i) = mod(i-1, num_domains) + 1;
                end
            else
                domain_height = lat_range / num_domains;
                for i = 1:num_sats
                    domain_num = ceil((latitudes(i) - min(latitudes)) / domain_height);
                    domain_assignment(i) = max(1, min(domain_num, num_domains));
                end
            end
            
        case 'longitude'
            % 基于经度划分
            longitudes = [satellites.longitude];
            lon_range = 360;
            domain_width = lon_range / num_domains;
            adj_lons = mod(longitudes + 180, 360);
            for i = 1:num_sats
                domain_num = ceil(adj_lons(i) / domain_width);
                domain_assignment(i) = max(1, min(domain_num, num_domains));
            end
            
        case 'clustering'
            % 基于聚类划分
            features = [[satellites.latitude]', [satellites.longitude]'];
            [domain_assignment, ~] = kmeans(features, num_domains, ...
                'MaxIter', 100, 'Replicates', 3);
            
        case 'orbital_plane'
            % 基于轨道面划分（简化实现）
            planes = [satellites.plane];
            unique_planes = unique(planes);
            if length(unique_planes) >= num_domains
                planes_per_domain = ceil(length(unique_planes) / num_domains);
                for i = 1:num_sats
                    plane_idx = find(unique_planes == satellites(i).plane, 1);
                    domain_num = ceil(plane_idx / planes_per_domain);
                    domain_assignment(i) = max(1, min(domain_num, num_domains));
                end
            else
                warning('轨道面数量少于域数，使用纬度划分替代');
                latitudes = [satellites.latitude];
                lat_range = max(latitudes) - min(latitudes);
                if lat_range == 0
                    for i = 1:num_sats
                        domain_assignment(i) = mod(i-1, num_domains) + 1;
                    end
                else
                    domain_height = lat_range / num_domains;
                    for i = 1:num_sats
                        domain_num = ceil((latitudes(i) - min(latitudes)) / domain_height);
                        domain_assignment(i) = max(1, min(domain_num, num_domains));
                    end
                end
            end
            
        case 'hybrid'
            % 混合划分：先按轨道面分组，再按纬度细分（简易回退）
            warning('混合划分方法未完全实现，使用均匀分配替代');
            for i = 1:num_sats
                domain_assignment(i) = mod(i-1, num_domains) + 1;
            end
            
        otherwise
            error('未知的域划分方法: %s', partition_method);
    end
    
    % 最终安全检查：如果仍有0值，强制均匀分配
    if any(domain_assignment == 0)
        warning('域分配中存在0，强制使用均匀分配');
        for i = 1:num_sats
            domain_assignment(i) = mod(i-1, num_domains) + 1;
        end
    end
    
    % 构建域结构
    domains.num_domains = num_domains;
    domains.domain_assignment = domain_assignment;
    domains.partition_method = partition_method;
    
    % 统计域信息
    fprintf('     域划分完成: %d个域\n', num_domains);
    for d = 1:num_domains
        domain_sats = find(domain_assignment == d);
        fprintf('       域 %d: %d颗卫星\n', d, length(domain_sats));
    end
    
    % 将域信息添加到卫星结构体中（不影响外部，仅用于本函数内部）
    for i = 1:num_sats
        satellites(i).domain = domain_assignment(i);
    end
end