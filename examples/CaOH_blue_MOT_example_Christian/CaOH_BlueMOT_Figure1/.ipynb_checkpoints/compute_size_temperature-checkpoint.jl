function compute_size_temperature_figure2(prob, scan_func, scan_values)
    """
    1) Compute 
    2) 
    3) 
    """

    # 1)
    @everywhere begin
        prob.p.diffusion_constant[1] = 0.
        prob.p.diffusion_constant[2] = 0.
        prob.p.diffusion_constant[3] = 0.
        prob.p.add_spontaneous_decay_kick = true
        
        T_initial = 35e-6
        prob.p.params.x_dist = Normal(0, 700e-6)
        prob.p.params.y_dist = Normal(0, 700e-6)
        prob.p.params.z_dist = Normal(0, 400e-6)
        prob.p.params.vx_dist = Normal(0, sqrt(kB*T_initial/2m))
        prob.p.params.vy_dist = Normal(0, sqrt(kB*T_initial/2m))
        prob.p.params.vz_dist = Normal(0, sqrt(kB*T_initial/2m))

    end
    all_sols_no_diffusion = distributed_solve(100, prob, prob_func!, scan_func, scan_values)
    
    σxs = σx_fit.(all_sols_no_diffusion)
    σys = σy_fit.(all_sols_no_diffusion)
    σzs = σz_fit.(all_sols_no_diffusion)
    
    Txs = Tx_fit.(all_sols_no_diffusion)
    Tys = Ty_fit.(all_sols_no_diffusion)
    Tzs = Tz_fit.(all_sols_no_diffusion)
    
    # 2) 
    @everywhere begin
        cb = ContinuousCallback(condition_new, stochastic_collapse_new!, save_positions=(false,false))
        prob_diffusion = remake(prob, callback=cb)
        
        prob_diffusion.p.add_spontaneous_decay_kick = false
        prob_diffusion.p.params.B_ramp_time = 1e-10 / (1/Γ)
        prob_diffusion.p.params.s_ramp_time = 1e-10 / (1/Γ)
    end
    
    n_avgs  = 50000
    t_end   = 1e-6
    τ_total = 1e-6
    n_times = 100
    
    scan_values_with_σ_and_T = zip(scan_values, zip(σxs, σys, σzs, Txs, Tys, Tzs))
    (diffusions, diffusion_errors, diffusions_over_time) = distributed_compute_diffusion(
        prob_diffusion, prob_func!, scan_func_with_initial_conditions!(scan_func), scan_values_with_σ_and_T, n_avgs, t_end, τ_total, n_times
    )

    # 3)
    @everywhere begin
        prob.p.params.B_ramp_time = 4e-3 / (1/Γ)
        prob.p.params.s_ramp_time = 4e-3 / (1/Γ)

        T_initial = 35e-6
        prob.p.params.x_dist = Normal(0, 700e-6)
        prob.p.params.y_dist = Normal(0, 700e-6)
        prob.p.params.z_dist = Normal(0, 400e-6)
        prob.p.params.vx_dist = Normal(0, sqrt(kB*T_initial/2m))
        prob.p.params.vy_dist = Normal(0, sqrt(kB*T_initial/2m))
        prob.p.params.vz_dist = Normal(0, sqrt(kB*T_initial/2m))
    end

    scan_values_with_diffusion = zip(scan_values, diffusions)
    all_sols_with_diffusion = distributed_solve(200, prob, prob_func!, scan_func_with_diffusion!(scan_func), scan_values_with_diffusion)
    
    return (all_sols_no_diffusion, all_sols_with_diffusion, diffusions, diffusion_errors, diffusions_over_time)
end

function scan_func_with_initial_conditions!(scan_func)
    (prob, scan_value) -> begin
        scan_func(prob, scan_value[1])
        σx, σy, σz, Tx, Ty, Tz = scan_value[2]
        prob.p.params.x_dist = Normal(0, σx)
        prob.p.params.y_dist = Normal(0, σy)
        prob.p.params.z_dist = Normal(0, σz)
        prob.p.params.vx_dist = Normal(0, Tx)
        prob.p.params.vy_dist = Normal(0, Ty)
        prob.p.params.vz_dist = Normal(0, Tz)
        return nothing
    end
end

function scan_func_with_diffusion!(scan_func)
    (prob, scan_value) -> begin
        scan_func(prob, scan_value[1])
        diffusion = scan_value[2]
        prob.p.diffusion_constant[1] = diffusion
        prob.p.diffusion_constant[2] = diffusion
        prob.p.diffusion_constant[3] = diffusion
        return nothing
    end
end

    
    