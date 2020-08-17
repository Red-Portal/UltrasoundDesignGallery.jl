
using Test

import ForwardDiff
import Calculus
import PDMats

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

function pseudo_marginal(prng, Nimp, scale, ℓ², σ², ϵ², initial_latent)
    K, Kinv  = compute_gram_matrix(data, ℓ², σ², ϵ²)
    μ_f, Σ_f = laplace_approximation(K, scale, initial_latent)

    q         = MvNormal(μ_f, Σ_f)
    f_samples = rand(prng, q, Nimp)

    joint   = [logjoint_prob(K, Kinv, latent, scale) for latent in f_samples]
    logpml  = logsumexp(joint - logpdf.(Ref(q), f_samples)) - log(Nimp)
    logpml, f_samples
end

function pmmh(prng, priors)
    samples = 1024
    latent  = randn()
    σ       = 1.0
    Nimp    = 1

    θ_samples     = zeros(4, samples) 
    f_samples     = zeros(length(latent), samples)
    prev_logprior = -Inf
    prev_logpml   = -Inf
    prev_latent   = deepcopy(latent)
    for i = 1:samples
        prop_θ = prev_θ + randn(prng, size(θ)) * σ
        prop_logpml, prop_latents = pseudo_marginal(prng, Nimp, prop_θ[1],
                                                    prop_θ[2], prop_θ[3],
                                                    prop_θ[4], latent)
        prop_logprior = sum(logpdf.(priors, θ))
        α = min(prop_logpml + prop_logprior - prev_logpml - prev_logprior, 0)

        if(log(rand(prng)) < α)
            prev_logprior  = prop_logprior
            prev_logpml    = prop_logpml
            θ_samples[:,i] = prop_θ
            f_samples[:,i] = latents[1]
        else
            θ_samples[:,i] = prev_θ
            f_samples[:,i] = prev_latents[1]
        end
    end
end

function test_laplace_approx()
    npoints  = 30
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
        kernel = construct_kernel(ℓ², σ²)
        K = KernelFunctions.kernelmatrix(
            kernel, reshape(testpoints, (size(testpoints,1),:)), obsdim=2)
        K = K + ϵ²*I
        #display(Plots.heatmap(K))

        @time latent, Σ = laplace_approximation(K, scale, initial_latent)

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
