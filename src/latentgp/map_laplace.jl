
function map_laplace(data::Matrix{<:Real},
                     choices::Matrix{<:Int},
                     initial_θ::Vector{<:Real},
                     scale::Real,
                     θ_prior;
                     verbose::Bool=true)
    μ = nothing
    Σ = nothing
    a = nothing
    B = nothing
    K = nothing
    function f(x)
        θ = exp.(x)
        K = compute_gram_matrix(data, θ[1], θ[2], θ[3])   
        try
            K          = PDMats.PDMat(K)
            μ, Σ, a, B = laplace_approximation(K, choices, zeros(size(data, 2)), scale, verbose=false)
            loglike    = logbtl(choices, μ, scale)
            mll        = loglike + dot(a, μ)/-2 + logdet(B)/-2 + logpdf(θ_prior, x)
            return mll
        catch err
            if(isa(err, LinearAlgebra.PosDefException))
                @warn "Cholesky failed. Rejecting proposal"
                return -Inf
            else
                throw(err)
            end
        end
    end

    opt_res = Optim.maximize(f, log.(initial_θ),
                             Optim.BFGS(linesearch=LineSearches.BackTracking()),
                             Optim.Options(g_tol = 1e-4,
                                           x_tol = 1e-4,
                                           iterations=100))
    if(verbose)
        @info(opt_res)
    end
    θ_opt = Optim.maximizer(opt_res)
    exp.(θ_opt), μ, Σ, a, B, K
end
