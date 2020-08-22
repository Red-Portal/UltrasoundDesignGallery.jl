
using Test

import ForwardDiff
import Calculus
import PDMats
import MCMCChains
import ProgressMeter

include("../src/UltrasoundVisualGallery.jl")

function test_laplace_approx()
    npoints  = 3
    ncand    = 5
    prng     = MersenneTwister(1)
    scale    = 1.0
    dims     = 1

    ℓ²  = 0.1
    σ²  = 1.0  
    ϵ²  = 0.1

    testpoints = rand(prng, dims, npoints, ncand)
    goodness   = sinc.((testpoints .- 0.5)*4*π)[1,:,:] + randn(prng, size(testpoints)[2:3]) * 0.1
    orders     = [sortperm(goodness[i,:], rev=true) for i = 1:npoints]

    for i = 1:npoints
        goodness[i,:]     = goodness[i, orders[i]] 
        testpoints[:,i,:] = testpoints[:, i, orders[i]] 
    end

    initial_latent = randn(prng, npoints, ncand)

    println("initial likelihood = ", logbtl(initial_latent, scale))
    println("true    likelihood = ", logbtl(goodness, scale))

    begin
        K, Kinv = compute_gram_matrix(testpoints, ℓ², σ², ϵ²)
        @time latent, Σ = laplace_approximation(K, Kinv, scale, initial_latent)

        display(Plots.scatter(reshape(testpoints, :), reshape(latent, :)))
        display(Plots.scatter!(reshape(testpoints, :), reshape(goodness, :)))
    end
end

function test_derivative()
    goodness = randn(MersenneTwister(1), 10, 4)
    scale    = 0.1

    begin
        logpref = logbtl_full(goodness, scale)
        g       = ∇logbtl(logpref, scale)
        g_num   = Calculus.gradient(x->begin
                                    x = reshape(x, (10, 4))
                                    l = logbtl(x, scale)
                                    end, reshape(goodness, 40))
        @test norm(g - g_num, Inf) < 0.001
    end

    begin
        logpref = logbtl_full(goodness, scale)
        g       = ∇logbtl(logpref, scale)
        g       = reshape(g, size(logpref))
        H       = ∇²logbtl(logpref, g, scale)
        H_num   = Calculus.hessian(x->begin
                                   x = reshape(x, (10, 4))
                                   l = logbtl(x, scale)
                                   end, reshape(goodness, 40))
        display(Plots.heatmap(H - H_num))
        @test norm(H - H_num, Inf) < 0.001
    end
end

function test_bo()
    npoints  = 5
    ncand    = 2
    prng     = MersenneTwister(1)
    scale    = 1.0
    dim      = 1

    ℓ²  = 1.0
    σ²  = 1.0  
    ϵ²  = 1.0

    f(x) = sin((x[1] .- 0.5)*4*π) + randn(prng) * 0.1
    
    testpoints = rand(prng, dim, npoints * ncand)
    goodness   = hcat([f(testpoints[:,i]) for i = 1:npoints * ncand]...)
    goodness   = reshape(goodness, (npoints, ncand))
    testpoints = reshape(testpoints, (dim, npoints, ncand))
    orders     = [sortperm(goodness[i,:], rev=true) for i = 1:npoints]

    for i = 1:npoints
        goodness[i,:]     = goodness[i, orders[i]] 
        testpoints[:,i,:] = testpoints[:, i, orders[i]] 
    end
    initial_latent = zeros( npoints, ncand)
    priors = Product([Normal(2, 2),
                      Normal(2, 2),
                      Normal(2, 2)])

    θ_init  = [ℓ², σ², ϵ²]
    hist    = Float64[]
    samples = 64
    warmup  = 64
    for i = 1:20
        opt_idx = argmax(goodness)
        x_opt   = testpoints[:,opt_idx]
        y_opt   = goodness[opt_idx]
        push!(hist, y_opt)

        θs, fs, as, Ks = pm_ess(
            prng, samples, warmup, θ_init, initial_latent,
            priors, scale, testpoints)
        ks = precompute_lgp(θs)

        X = reshape(testpoints, (dim,:))
        x_query, _ = optimize_acquisition(dim, 1024, y_opt, X, Ks, as, ks)
        y_query    = f(x_query)

        new_testpoint, new_goodness = begin
            if(y_query > y_opt)
                hcat(x_query, x_opt), [y_query, y_opt]
            else
                hcat(x_opt, x_query), [y_opt, y_query]
            end
        end
        goodness   = cat(goodness,   reshape(new_goodness,  (1,2)),     dims=1)
        testpoints = cat(testpoints, reshape(new_testpoint, (dim,1,2)), dims=2)

        @info(iteration=i,
              x_query = x_query,
              y_query = y_query,
              y_optimal = y_opt)

        println(testpoints)
        println(goodness)
        initial_latent = zeros(npoints+i, ncand)
        #initial_latent = cat(initial_latent, randn(prng, 1, 2), dims=1)
        #θ_init         = mean(θs, dims=2)[:,1]
        θ_init  = [ℓ², σ², ϵ²]
    end
    hist
end

function test_pmmh()
    npoints  = 30
    ncand    = 2
    prng     = MersenneTwister(1)
    scale    = 1.0
    dims     = 1

    ℓ²  = 0.1
    σ²  = 1.0  
    ϵ²  = 0.1

    testpoints = rand(prng, dims, npoints, ncand)
    goodness   = sin.((testpoints .- 0.5)*4*π)[1,:,:] + randn(prng, size(testpoints)[2:3]) * 0.1
    orders     = [sortperm(goodness[i,:], rev=true) for i = 1:npoints]

    for i = 1:npoints
        goodness[i,:]     = goodness[i, orders[i]] 
        testpoints[:,i,:] = testpoints[:, i, orders[i]] 
    end

    initial_latent = zeros(npoints, ncand)

    priors = Product([Normal(0, 2),
                      Normal(0, 2),
                      Normal(-2, 1)])
    samples = 1024
    warmup  = 1024
    θ_samples, f_samples, a_samples, Σ_samples = pm_ess(
        prng, samples, warmup, [ℓ², σ², ϵ²],
        initial_latent, priors, scale, testpoints)

    θ_perm = permutedims(θ_samples, (2, 1))
    θ_perm = reshape(θ_perm, (size(θ_perm, 1), size(θ_perm, 2), 1))
    chain  = MCMCChains.Chains(θ_perm, ["ℓ2", "σ2", "ϵ2"])

    # X  = reshape(testpoints, (dims, :)) 
    # y  = reshape(goodness, :)
    # ks = precompute_lgp(θ_samples)
    # θ_samples, f_samples, X, y, ks
    chain
end

