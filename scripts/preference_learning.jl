
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
    prng      = MersenneTwister(3)
    goodness = randn(prng, 10)
    choices  = rand(prng, DiscreteUniform(1, 10), 4, 3)
    scale    = 0.1
    println(choices)
    println(goodness)
    println(logbtl_matrix(choices, goodness, scale))
    println(logbtl(choices, goodness, scale))

    begin
        btl_matrix = logbtl_matrix(choices, goodness, scale)
        g          = ∇logbtl(btl_matrix, goodness, choices, scale)
        g_num      = Calculus.gradient(x->logbtl(choices, x, scale), goodness)
        @test norm(g - g_num, Inf) < 0.001
    end

    begin
        btl_matrix = logbtl_matrix(choices, goodness, scale)
        grad       = ∇logbtl(btl_matrix, goodness, choices, scale)

        H      = ∇²logbtl(btl_matrix, goodness, choices, scale)
        H_num  = Calculus.hessian(x->logbtl(choices, x, scale), goodness)

        #display(Plots.heatmap(H - H_num))
        display(Plots.heatmap(tanh.(H * 1e+4)))

        #display(Plots.heatmap(H_num))
        #println(H_num)
        #display(Plots.heatmap(tanh.(H_num * 1e+4)))
        #display(Plots.heatmap(H_num - H))
        @test norm(H - H_num, Inf) < 0.001
    end
end

function test_bo()
    npoints  = 3
    ncand    = 2
    prng     = MersenneTwister(1)
    scale    = 1.0
    dims     = 1

    ℓ²  = 1.0
    σ²  = 1.0  
    ϵ²  = 1.0

    f(x) = sin((x[1] .- 0.5)*4*π) + randn(prng) * 0.1
    
    testpoints = rand(prng, dims, npoints * ncand)
    goodness   = hcat([f(testpoints[:,i]) for i = 1:npoints * ncand]...)
    goodness   = reshape(goodness, (npoints, ncand))
    testpoints = reshape(testpoints, (dims, npoints, ncand))
    orders     = [sortperm(goodness[i,:], rev=true) for i = 1:npoints]

    for i = 1:npoints
        goodness[i,:]     = goodness[i, orders[i]] 
        testpoints[:,i,:] = testpoints[:, i, orders[i]] 
    end
    goodness   = reshape(goodness, :) 
    testpoints = reshape(testpoints, (dims, :))
    choices    = reshape(collect(1:npoints*ncand), npoints, ncand) 

    priors = Product([Normal(2, 2),
                      Normal(2, 2),
                      Normal(2, 2)])

    initial_latent = zeros(length(goodness))
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
            priors, scale, testpoints, choices)
        ks = precompute_lgp(θs)

        x_query, _ = optimize_acquisition(dims, 1024, y_opt, testpoints, Ks, as, ks)
        y_query    = f(x_query)

        choices    = begin
            new_idx    = length(goodness)+1
            new_choice = reshape([opt_idx, new_idx], (1,2))
            if(y_query > y_opt)
                vcat(choices, new_choice)
            else
                vcat(choices, new_choice)
            end
        end
        goodness   = push!(goodness, y_query)
        testpoints = hcat(testpoints, reshape(x_query, (:,1)))

        display(plot(hist))

        @info(iteration=i,
              x_query = x_query,
              y_query = y_query,
              y_optimal = y_opt)

        initial_latent = zeros(length(goodness))
        #initial_latent = cat(initial_latent, randn(prng, 1, 2), dims=1)
        θ_init         = mean(θs, dims=2)[:,1]
        #θ_init  = [ℓ², σ², ϵ²]
    end
    hist
end

function test_pmmh()
    npoints  = 10
    ncand    = 2
    prng     = MersenneTwister(3)
    scale    = 1.0
    dims     = 1

    ℓ²  = 10
    σ²  = 1 
    ϵ²  = 10

    testpoints = rand(prng, dims, npoints-2, ncand)
    goodness   = sin.((testpoints .- 0.5)*4*π)[1,:,:] + randn(prng, size(testpoints)[2:3]) * 0.1
    orders     = [sortperm(goodness[i,:], rev=true) for i = 1:(npoints-2)]

    for i = 1:npoints-2
        goodness[i,:]     = goodness[i, orders[i]] 
        testpoints[:,i,:] = testpoints[:, i, orders[i]] 
    end
    testpoints = reshape(testpoints, (dims, :))
    goodness   = reshape(goodness, :)
    choices    = reshape(collect(1:(npoints-2)*ncand), (npoints-2, ncand)) 
    choices[end-1,:] = choices[end-2,:]
    choices[end,:]   = choices[end-2,:]

    initial_latent = zeros(size(testpoints, 2))

    priors = Product([Normal(0, 2),
                      Normal(0, 2),
                      Normal(-3, 1)])
    samples = 1024
    warmup  = 1024
    θ_samples, f_samples, a_samples, K_samples = pm_ess(
        prng, samples, warmup, [ℓ², σ², ϵ²],
        initial_latent, priors, scale, testpoints, choices)

    θ_perm = permutedims(θ_samples, (2, 1))
    θ_perm = reshape(θ_perm, (size(θ_perm, 1), size(θ_perm, 2), 1))
    chain  = MCMCChains.Chains(θ_perm, ["ℓ2", "σ2", "ϵ2"])

    ks = precompute_lgp(θ_samples)
    μ, σ² = mgp_predict(reshape(collect(0.0:0.01:1.0), (1,:)),
                        testpoints, K_samples, a_samples, ks)

    Plots.plot(0.0:0.01:1.0, μ, ribbon=1.96*sqrt.(σ²))
    #Plots.scatter!(testpoints[1,:], goodness)

    ei = [ expected_improvement([x], 1.0, testpoints, K_samples, a_samples, ks) for x = 0.0:0.01:1.0 ]

    Plots.plot!(0.0:0.01:1.0, ei)

    # X  = reshape(testpoints, (dims, :)) 
    # y  = reshape(goodness, :)
    # ks = precompute_lgp(θ_samples)
    # θ_samples, f_samples, X, y, ks
    # chain

    # θ_init  = [ℓ², σ², ϵ²]
    # map_laplace(testpoints, θ_init, scale)
end

