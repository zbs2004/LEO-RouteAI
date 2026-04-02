function state = get_state_for_rl(src, dst, current_satellite, domains, load_matrix, domain_graph)
    state.src_domain = domains.domain_assignment(src);
    state.dst_domain = domains.domain_assignment(dst);
    state.current_domain = domains.domain_assignment(current_satellite);
    state.load_matrix = load_matrix;
    state.domain_graph = domain_graph;
end