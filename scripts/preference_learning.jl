
using Test

import ForwardDiff
import Calculus
import PDMats
import MCMCChains
import ProgressMeter

include("../src/UltrasoundVisualGallery.jl")

function compute_gram_matrix(data, ℓ², σ², ϵ²)
    kernel = construct_kernel(ℓ², σ²)
    K    = KernelFunctions.kernelmatrix(
        kernel, reshape(data, (size(data,1),:)), obsdim=2)
    K    = K + ϵ²*I
    K    = PDMats.PDMat(K)
    Kinv = inv(K)
    K, Kinv
end

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

function test_pmmh()
    npoints  = 30
    ncand    = 5
    prng     = MersenneTwister(1)
    scale    = 1.0
    dims     = 1

    ℓ²  = 0.1
    σ²  = 1.0  
    ϵ²  = 0.1

    testpoints = rand(prng, dims, npoints, ncand)
    goodness   = sinc.((testpoints .- 0.5)*4*π)[1,:,:] + randn(prng, size(testpoints)[2:3]) * 0.01
    orders     = [sortperm(goodness[i,:], rev=true) for i = 1:npoints]

    for i = 1:npoints
        goodness[i,:]     = goodness[i, orders[i]] 
        testpoints[:,i,:] = testpoints[:, i, orders[i]] 
    end

    initial_latent = randn(prng, npoints, ncand)

    priors = Product([Normal(0, 1),
                      Normal(0, 2),
                      Normal(0, 1)])
    samples = 1024
    warmup  = 1024
    θ_samples, f_samples, Kinvs = pm_ess(
        prng, samples, warmup, [ℓ², σ², ϵ²], initial_latent, priors, scale, testpoints)

    θ_perm = permutedims(θ_samples, (2, 1))
    θ_perm = reshape(θ_perm, (size(θ_perm, 1), size(θ_perm, 2), 1))
    chain = MCMCChains.Chains(θ_perm, ["ℓ2", "σ2", "ϵ2"])
    show(chain)

    X = reshape(testpoints, (dims, :)) 
    y = reshape(goodness, :)
    αs = precompute_lgp(f_samples, Kinvs)
    θ_samples, f_samples, Kinvs, αs, X, y
end
