
function optimize_acquisition(dim::Int64,
                              max_feval::Int64,
                              x_hint::AbstractVector,
                              y_opt::Real,
                              X::Matrix,
                              K, a, k;
                              verbose::Bool=true,
                              prng=Random.GLOBAL_RNG)
    α(x) = -expected_improvement(x, y_opt, X, K, a, k)
    dims = length(x_hint)

    res = CMAEvolutionStrategy.minimize(
        α,
        x_hint,
        0.5*√(dim);
        lower=zeros(dim),
        upper=ones(dim),
        seed=prng.seed[1],
        xtol=1e-4,
        ftol=1e-5,
        maxfevals=max_feval,
        multi_threading=true,
        verbosity=verbose ? 3 : 0
    )

    x_sol = CMAEvolutionStrategy.xbest(res)
    y_sol = α(x_sol)
    x_sol = clamp.(x_sol, 0, 1)
    return x_sol, y_sol
end

function optimize_mean(dim::Int64,
                       max_feval::Int64,
                       X::Matrix,
                       K, a, k)
    data_μs = [ gp_predict(X[:,i], X, K, a, k)[1] for i = 1:size(X,2) ]
    opt_idx    = argmax(data_μs)
    x_opt      = X[:,opt_idx]
    y_opt      = data_μs[opt_idx]
    x_opt, y_opt, opt_idx
end
