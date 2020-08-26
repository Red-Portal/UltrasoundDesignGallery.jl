
function expected_improvement(x::Vector,
                              y_opt::Real,
                              X::Matrix,
                              K::Array{PDMats.PDMat},
                              a::Matrix,
                              k::Array{KernelFunctions.Kernel})
    N  = length(K)
    ei = zeros(N)
    @simd for i = 1:N
        μ, σ² = gp_predict(x, X, K[i], a[:,i], k[i])
        σ     = sqrt(σ²)
        Δy    = μ - y_opt
        z     = Δy / σ
        ei[i] = Δy * normcdf(z) + σ*normpdf(z)
    end
    mean(ei)
end

function expected_improvement(x::Vector,
                              y_opt::Real,
                              X::Matrix,
                              K::PDMats.PDMat,
                              a::Vector,
                              k::KernelFunctions.Kernel)
    μ, σ² = gp_predict(x, X, K, a, k)
    σ     = sqrt(σ²)
    Δy    = μ - y_opt
    z     = Δy / σ
    ei    = Δy * normcdf(z) + σ*normpdf(z)
    ei
end

function optimize_acquisition(dim::Int64, max_iter::Int64, y_opt::Real, X::Matrix,
                              K, a, k; verbose::Bool=true)
    # The below Optim API calling part is succeptible to API breaks.
    # Optim's constained optimization API is not stable right now.
    f(x, g)  = expected_improvement(x, y_opt, X, K, a, k)

    opt = NLopt.Opt(:GN_DIRECT, dim)
    opt.lower_bounds  = zeros(dim)
    opt.upper_bounds  = ones(dim)
    #opt.ftol_abs      = 1e-5 
    #opt.xtol_abs      = 1e-5 
    opt.maxeval       = max_iter
    opt.max_objective = f

    res, time = @timed NLopt.optimize(opt, rand(dim))
    optimum, solution, status = res
    solution = clamp.(solution, 0, 1)
    if(verbose)
        @info "Inner Optimization Stat" status time solution
    end
    return solution, optimum
end

function optimize_mean(dim::Int64, max_iter::Int64, X::Matrix,
                       K, a, k; verbose::Bool=true)
    # The below Optim API calling part is succeptible to API breaks.
    # Optim's constained optimization API is not stable right now.
    f(x, g)  = gp_predict(x, X, K, a, k)[1]

    opt = NLopt.Opt(:GN_DIRECT, dim)
    opt.lower_bounds  = zeros(dim)
    opt.upper_bounds  = ones(dim)
    opt.maxeval       = max_iter
    opt.max_objective = f

    res, time = @timed NLopt.optimize(opt, rand(dim))
    optimum, solution, status = res
    solution = clamp.(solution, 0, 1)
    if(verbose)
        @info "Optima Finding Stat" status time solution
    end
    return solution, optimum
end


function pairwise_prefbo(objective, dims, warmup_steps)
    img      = TestImages.testimage("lena_gray_256.tif")
    prng     = MersenneTwister(1)
    scale    = 1.0

    ℓ² = 1.0
    σ² = 1.0  
    ϵ² = 1.0

    data_x = rand(prng, dims, warmup_steps*2)
    data_c = zeros(Int64, warmup_steps, 2)
    for i = 1:2:warmup_steps*2
        choice  = objective(data_x[:,i], data_x[:,i+1])
        i_nduel = ceil(Int64, i / 2)
        data_c[i_nduel, 1] = choice == 1 ? i   : i+1
        data_c[i_nduel, 2] = choice == 1 ? i+1 : i
    end
    
    priors = Product([Normal(0, 1),
                      Normal(0, 1),
                      Normal(-2, 2)])

    initial_latent = zeros(size(data_x, 2))
    θ_init  = [ℓ², σ², ϵ²]
    samples = 64
    warmup  = 64

    θs, fs, as, Ks = pm_ess(
        prng, samples, warmup, θ_init, initial_latent,
        priors, scale, data_x, data_c)
    ks    = precompute_lgp(θs)
    for i = 1:10
        x_opt, y_opt = optimize_mean(dims, 10000, data_x, Ks, as, ks)
        x_query, _   = optimize_acquisition(dims, 10000, y_opt, data_x, Ks, as, ks)

        choice  = objective(x_opt, x_query)

        data_x = hcat(data_x, reshape(x_opt,   (:,1)))
        data_x = hcat(data_x, reshape(x_query, (:,1)))
        choice  = begin
            if(choice == 1)
                [size(data_x, 2) - 1, size(data_x, 2)]
            else
                [size(data_x, 2), size(data_x, 2) - 1]
            end
        end
        data_c = vcat(data_c, reshape(choice, 1, :))

        @info("BO iteration stat",
              iteration = i,
              x_query   = x_query,
              x_optimum = x_opt)

        θ_init         = mean(θs, dims=2)[:,1]
        initial_latent = zeros(size(data_x, 2))
        θs, fs, as, Ks = pm_ess(
            prng, samples, warmup, θ_init, initial_latent,
            priors, scale, data_x, data_c)
        ks    = precompute_lgp(θs)
    end
    x_opt, y_opt = optimize_mean(dims, 30000, data_x, Ks, as, ks)
    x_opt
end

function lerp(x::Real, low::Real, high::Real)
    return x * (high - low) + low;
end

