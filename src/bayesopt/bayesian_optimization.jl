
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
    ϵ = 1e-10

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

    if(verbose)
        @info "Inner Optimization Stat" status time solution
    end
    return solution, optimum
end
