
# function optimize_acquisition(dim::Int64, max_iter::Int64, y_opt::Real, X::Matrix,
#                               K, a, k; verbose::Bool=true)
#     f(x, g)  = expected_improvement(x, y_opt, X, K, a, k)

#     opt = NLopt.Opt(:GN_DIRECT, dim)
#     opt.lower_bounds  = zeros(dim)
#     opt.upper_bounds  = ones(dim)
#     #opt.ftol_abs      = 1e-5 
#     #opt.xtol_abs      = 1e-5 
#     opt.maxeval       = max_iter
#     opt.max_objective = f

#     res, time = @timed NLopt.optimize(opt, rand(dim))
#     optimum, solution, status = res
#     solution = clamp.(solution, 0, 1)
#     if(verbose)
#         @info "Inner Optimization Stat" status time solution
#     end
#     return solution, optimum
# end

# function optimize_mean(dim::Int64, max_iter::Int64, X::Matrix,
#                        K, a, k; verbose::Bool=true)
#     f(x, g)  = gp_predict(x, X, K, a, k)[1]

#     opt = NLopt.Opt(:GN_DIRECT, dim)
#     opt.lower_bounds  = zeros(dim)
#     opt.upper_bounds  = ones(dim)
#     opt.maxeval       = max_iter
#     opt.max_objective = f

#     res, time = @timed NLopt.optimize(opt, rand(dim))
#     optimum, solution, status = res
#     solution = clamp.(solution, 0, 1)
#     if(verbose)
#         @info "Optima Finding Stat" status time solution
#     end
#     return solution, optimum
# end

function optimize_acquisition(dim::Int64, max_feval::Int64, y_opt::Real, X::Matrix,
                              K, a, k; verbose::Bool=true)
    f(x)  = expected_improvement(x, y_opt, X, K, a, k)

    res = BlackBoxOptim.bboptimize(
        f;
        FitnessScheme=BlackBoxOptim.MaximizingFitnessScheme,
        SearchRange=(0.0, 1.0),
        NumDimensions=dim,
        Method=:xnes,
        MaxFuncEvals=max_feval,
        NThreads=Threads.nthreads()-1,
        lambda=32,
        TraceMode= verbose ? :verbose : :compact)
    
    solution = BlackBoxOptim.best_candidate(res)
    optimum  = BlackBoxOptim.best_fitness(res)
    solution = clamp.(solution, 0, 1)
    return solution, optimum
end

function optimize_mean(dim::Int64, max_feval::Int64, X::Matrix,
                       K, a, k; verbose::Bool=true)
    f(x)  = gp_predict(x, X, K, a, k)[1]

    res = BlackBoxOptim.bboptimize(
        f;
        FitnessScheme=BlackBoxOptim.MaximizingFitnessScheme,
        SearchRange=(0.0, 1.0),
        NumDimensions=dim,
        Method=:xnes,
        MaxFuncEvals=max_feval,
        NThreads=Threads.nthreads()-1,
        lambda=32,
        TraceMode= verbose ? :verbose : :compact)
    
    solution = BlackBoxOptim.best_candidate(res)
    optimum  = BlackBoxOptim.best_fitness(res)
    solution = clamp.(solution, 0, 1)
    return solution, optimum
end
