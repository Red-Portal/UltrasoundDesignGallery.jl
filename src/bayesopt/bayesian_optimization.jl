
function append_choice(r::Real, x_prev::Vector, x_query::Vector,
                       data_x::Matrix{<:Real}, data_c::Matrix{<:Int})
    # This part is pretty ugly, but can't help it
    x_choice = r*x_prev + (1-r)*x_query
    if(abs(r - 1.0) < 0.02)
        x_mid  = (x_prev + x_query) / 2
        data_x = hcat(data_x, reshape(x_mid,    (:,1)))
        data_x = hcat(data_x, reshape(x_query,  (:,1)))
    elseif(abs(r - 0.0) < 0.02)
        x_mid  = (x_prev + x_query) / 2
        data_x = hcat(data_x, reshape(x_mid,    (:,1)))
        data_x = hcat(data_x, reshape(x_query,    (:,1)))
    else
        data_x = hcat(data_x, reshape(x_query,  (:,1)))
        data_x = hcat(data_x, reshape(x_choice, (:,1)))
    end
    choices = begin
        if(r == 1.0)
            size(data_x, 2) .+ [-2, 0, -1]
        else
            size(data_x, 2) .+ [0, -1, -2]
        end
    end
    data_c  = vcat(data_c, reshape(choices, (1,3)))
    data_x, data_c, x_choice
end

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
    ℓ² = 1.0
    σ² = 1.0  
    ϵ² = 1.0

    data_x = zeros(Float64, dims, 0)
    data_c = zeros(Int64, 0, 3)
    for i = 1:warmup_samples
        x1 = rand(prng, dims)
        x2 = rand(prng, dims)
        r  = objective_linesearch(x1, x2)
        if(abs(r - 1.0) >= 0.02)
            data_x = hcat(data_x, reshape(x1, (:,1)))
        end
        data_x, data_c, x_prev = append_choice(r, x1, x2, data_x, data_c)
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

    x_prev = data_x[:,end]
    for i = 1:10
        x_opt, y_opt = optimize_mean(
            dims, search_budget, data_x, K, a, k;
            x_hints=data_x[:,argmin(mean(f, dims=2)[:,1])])
        x_query, _   = optimize_acquisition(
            dims, search_budget, y_opt, data_x, K, a, k;
            x_hints=data_x[:,argmin(mean(f, dims=2)[:,1])])
            #x_hints=[data_x[:,i] for i = 1:size(data_x, 2)])

        r = objective_linesearch(x_prev, x_query)
        data_x, data_c, x_prev = append_choice(r, x_prev, x_query, data_x, data_c)

        @info "" i size(data_x)
        println(data_c)
        @info("BO iteration stat",
              iteration  = i,
              x_query    = x_query,
              x_optimum  = x_opt,
              x_selected = x_prev)
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
