
function prefbo_linesearch(objective_linesearch,
                           iters::Int,
                           dims::Int,
                           scale::Real,
                           marginalize::Bool;
                           prng=MersenneTwister(),
                           warmup_samples::Int=4,
                           search_budget::Int=4000,
                           mcmc_burnin::Int=32,
                           mcmc_samples::Int=32)
    warmup_steps = 4
    ℓ² = 1.0
    σ² = 1.0  
    ϵ² = 1.0

    data_x = zeros(dims, 0)
    data_c = zeros(Int64, warmup_steps, 3)
    for i = 1:3:warmup_samples*3
        x1 = rand(prng, dims)
        x2 = rand(prng, dims)
        c1, c2, c3 = objective_linesearch(x1, x2)
        i_nduel = ceil(Int64, i / 3)
        data_c[i_nduel, 1] = i
        data_c[i_nduel, 2] = i+1
        data_c[i_nduel, 3] = i+2
        data_x = hcat(data_x, reshape(c1, (:,1)))
        data_x = hcat(data_x, reshape(c2, (:,1)))
        data_x = hcat(data_x, reshape(c3, (:,1)))
    end
    
    priors = Product([Normal(0, 1),
                      Normal(0, 1),
                      Normal(-2, 2)])

    initial_latent = zeros(size(data_x, 2))
    θ_init  = [ℓ², σ², ϵ²]

    θ, f, a, K, k = begin
        if(marginalize)
            θ, f, a, K = pm_ess(
                prng, mcmc_samples, mcmc_burnin, θ_init,
                initial_latent, priors, scale, data_x, data_c)
            k = precompute_lgp(θ)
            θ, f, a, K, k
        else
            θ, f, _, a, _, K = map_laplace(
                data_x, data_c, θ_init, scale, priors;
                verbose=true)
            k = construct_kernel(θ...)
            θ, f, a, K, k
        end
    end
    
    #status = bo_status_window!()
    for i = 1:10
        #bo_status_update!(status, "Finding current optimum")
        x_opt, y_opt = optimize_mean(dims, search_budget, data_x, K, a, k)

        #bo_status_update!(status, "Optimizing acquisition")
        x_query, _   = optimize_acquisition(
            dims, search_budget, y_opt, data_x, K, a, k)

        #bo_status_destroy!(status)
        x1, x2, x3 = objective_linesearch(x_opt, x_query)
        #status = bo_status_window!()

        data_x = hcat(data_x, reshape(x1, (:,1)))
        data_x = hcat(data_x, reshape(x2, (:,1)))
        data_x = hcat(data_x, reshape(x3, (:,1)))

        choices = size(data_c, 1) .+ [1,2,3]
        data_c  = vcat(data_c, reshape(choices, (:,3)))

        @info("BO iteration stat",
              iteration = i,
              x_query   = x_query,
              x_optimum = x_opt,
              x_selected = x1)
        initial_latent = zeros(size(data_x, 2))
        #bo_status_update!(status, "Infering preference model")
        θ, f, a, K, k = begin
            if(marginalize)
                θ, f, a, K = pm_ess(
                    prng, mcmc_samples, mcmc_burnin, θ_init,
                    initial_latent, priors, scale, data_x, data_c)
                k = precompute_lgp(θ)
                θ, f, a, K, k
            else
                θ, f, _, a, _, K = map_laplace(
                    data_x, data_c, θ_init, scale, priors;
                    verbose=true)
                k = construct_kernel(θ...)
                θ, f, a, K, k
            end
        end
    end
    x_opt, y_opt = optimize_mean(dims, 30000, data_x, K, a, k)
    x_opt
end
