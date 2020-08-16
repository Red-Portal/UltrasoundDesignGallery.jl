
using Test
using Statistics
using Distributions
using LinearAlgebra
using Random
using StatsFuns

import Calculus
import PDMats
import Optim

function logbtl(goodness::Array{Float64, 2},
                scale::Float64)
    # goodness ∈ R^{npoints, ncandidates} 
    # Only compute the preference of the "choices"
    scaled  = goodness / scale
    logpref = scaled[:,1] - logsumexp(scaled, dims=2)[:,1]
    sum(logpref)
end

function logbtl_full(goodness::Array{Float64, 2},
                     scale::Float64)
    # goodness ∈ R^{npoints, ncandidates} 
    # Compute the preferences of all entries
    scaled  = goodness / scale
    logpref = scaled .- logsumexp(scaled, dims=2)[:,1]
    return logpref
end

@inline function ∇logbtl(logpref::Array{Float64, 2},
                 scale::Float64)
    pref   = exp.(logpref)
    result = zeros(size(logpref))
    
    # first column derivative
    choices = pref[:,1]
    ∇choice = (1 .- choices) / scale
    result[:,1] = ∇choice

    # first column derivative
    comps  = pref[:,2:end]
    ∇comp  = comps / -scale
    result[:,2:end] = ∇comp
    return reshape(result, :)
end

@inline function ∇²logbtl(logpref::Array{Float64, 2},
                          scale::Float64)
    # logpref ∈ R^{npoints, ncandidates} 
    # logpref[:,1]     are the choices
    # logpref[:,2:end] are the compared candidates
    pref    = exp.(logpref)
    ∇btl    = ∇logbtl(logpref, scale)
    ∇btl    = reshape(∇btl, size(logpref))

    # hessian
    ndata  = size(logpref, 1)
    ncand  = size(logpref, 2)
    total  = prod(size(logpref))
    result = zeros(total, total)
    @inbounds for entry_i = 1:ncand
        @inbounds for entry_j = entry_i:ncand
            @simd for block = 1:ndata
                i = (entry_i - 1) * ndata + block
                j = (entry_j - 1) * ndata + block

                ∇btl_i_j = begin
                    if(entry_i != 1 && entry_i == entry_j)
                        ∇btl[block, entry_j] + 1 / scale
                    else
                        ∇btl[block, entry_j]
                    end
                end

                result[i, j] = pref[block, entry_i] * ∇btl_i_j / -scale
            end
        end
    end
    return Symmetric(result)
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
        H       = ∇²logbtl(logpref, scale)
        H_num   = Calculus.hessian(x->begin
                                   x = reshape(x, (10, 4))
                                   l = logbtl(x, scale)
                                   end, reshape(goodness, 40))
        display(Plots.heatmap(H - H_num))
        @test norm(H - H_num, Inf) < 0.001
    end
end

function marginal_log_likelihood(data, latent, scale, param)
    data    = randn(10, 3)
    latent  = reshape(latent, size(data))
    loglike = btl(data, latent, scale)
    ∇²btl(data, goodness, scale)
end

# function laplace_approximation(f, ∇L, K, W)
#     K    = PDMats.PDMat(K)
#     Kinv = inv(K)
#     Hq   = (-W - Linv.mat) 
#     ∇q   = W*f + ∇L

#     latent = 
#     loglike, ∇loglike, ∇²loglike = loglikelihood(latent, scale)

#     function f(latent)
# end
