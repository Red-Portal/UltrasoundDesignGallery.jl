
function append_choice!(r::Real, x1::Vector, x1_idx::Int, x2::Vector, gp_state::Dict)
    # This part is pretty ugly, but can't help it
    data_x = gp_state[:data_x]
    data_c = gp_state[:data_c]

    choices  = nothing
    xc = r*x1 + (1-r)*x2
    xm = (x1 + x2) / 2
    
    start_idx = size(data_x, 2) + 1
    data_new, choices = begin
        if(abs(r - 1.0) < 0.02)
            # Corner solution: x1 is favored
            # [x1 x_mid x2] 
            x_new   = hcat(xm, x2)
            choices = [x1_idx, start_idx, start_idx+1]
            x_new, choices
        elseif(abs(r - 0.0) < 0.02)
            # Corner solution: x2 is favored
            # [x2 x_mid x1] 
            x_new   = hcat(xm, x2)
            choices = [start_idx+1, start_idx, x1_idx]
            x_new, choices
        else
            # [x_c x1 x2] 
            x_new   = hcat(x2, xc)
            choices = [start_idx+1, x1_idx, start_idx]
            x_new, choices
        end
    end
    data_x = hcat(data_x, data_new)
    data_c = vcat(data_c, reshape(choices, (1, 3)))
    gp_state[:data_x] = data_x
    gp_state[:data_c] = data_c
end

function append_choice!(r::Real, x1::Vector, x2::Vector, gp_state::Dict)
    # This part is pretty ugly, but can't help it
    data_x = gp_state[:data_x]
    data_c = gp_state[:data_c]

    choices  = nothing
    xc = r*x1 + (1-r)*x2
    xm = (x1 + x2) / 2
    
    start_idx = size(data_x, 2) + 1
    data_new, choices = begin
        if(abs(r - 1.0) < 0.02)
            # Corner solution: x1 is favored
            x_new   = hcat(xm, x2, x1)
            choices = [start_idx+2, start_idx+1, start_idx]
            x_new, choices
        elseif(abs(r - 0.0) < 0.02)
            # Corner solution: x2 is favored
            x_new   = hcat(xm, x1, x2)
            choices = [start_idx+2, start_idx+1, start_idx]
            x_new, choices
        else
            # [x_c x1 x2] 
            x_new   = hcat(x1, x2, xc)
            choices = [start_idx+2, start_idx+1, start_idx]
            x_new, choices
        end
    end
    data_x = hcat(data_x, data_new)
    data_c = vcat(data_c, reshape(choices, (1, 3)))
    gp_state[:data_x] = data_x
    gp_state[:data_c] = data_c
end

function train_gp!(prng,
                   gp_state::Dict,
                   mcmc_samples::Int,
                   mcmc_burnin::Int,
                   mcmc_thin::Int,
                   scale::Real,
                   marginalize::Bool)

    dims   = size(gp_state[:data_x], 1)
    priors = Product(
        vcat([Normal(0, 2), Normal(-2, 2)],
             [Normal(log(0.5), log(2)) for i = 1:dims]))

    if(marginalize)
        θ_init = begin
            if(haskey(gp_state, :θ))
                mean(gp_state[:θ], dims=2)[:,1]
            else
                rand(prng, priors)
            end
        end

        latent_init = zeros(size(gp_state[:data_x], 2))
        θ, f, a, K = pm_ess(prng,
                            mcmc_samples,
                            mcmc_burnin,
                            mcmc_thin,
                            θ_init,
                            latent_init,
                            priors,
                            scale,
                            gp_state[:data_x],
                            gp_state[:data_c])
        k = precompute_lgp(θ)
        gp_state[:θ] = θ
        gp_state[:f] = f
        gp_state[:a] = a
        gp_state[:K] = K
        gp_state[:k] = k
    else
        θ_init = begin
            if(haskey(gp_state, :θ))
                θ_init
            else
                rand(prng, priors)
            end
        end

        θ, f, _, a, _, K = map_laplace(gp_state[:data_x],
                                       gp_state[:data_c],
                                       θ_init,
                                       scale,
                                       priors;
                                       verbose=true)
        k = construct_kernel(θ...)
        gp_state[:θ] = θ
        gp_state[:f] = f
        gp_state[:a] = a
        gp_state[:K] = K
        gp_state[:k] = k
    end
end

function prefbo_next_query(prng,
                           gp_state::Dict,
                           search_budget::Real;
                           verbose::Bool=false)
    dims = size(gp_state[:data_x],1)
    K    = gp_state[:K]
    a    = gp_state[:a]
    k    = gp_state[:k]
    X    = gp_state[:data_x]

    x_opt, y_opt, opt_idx = optimize_mean(dims, search_budget, X, K, a, k)
    x_sol, y_sol          = optimize_acquisition(
        dims, search_budget, x_opt, y_opt, X, K, a, k;
        verbose=verbose, prng=prng)

    x1 = x_opt 
    x2 = x_sol
    x1, opt_idx, x2
end
