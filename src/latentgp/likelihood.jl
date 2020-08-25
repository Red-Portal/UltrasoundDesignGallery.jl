
# @inline function safe_cholesky(K::Matrix)
#     try
#         PDMats.PDMat(K)
#     catch
#         Kmax = maximum(K)
#         α    = eps(eltype(K))
#         while !isposdef(K+α*I) && α < 0.01*Kmax
#             α *= 2.0
#         end
#         if α >= 0.01*Kmax
#             throw(ErrorException("Adding noise on the diagonal was not sufficient to build a positive-definite matrix:\n\t- Check that your kernel parameters are not extreme\n\t- Check that your data is sufficiently sparse\n\t- Maybe use a different kernel"))
#         end
#         PDMats.PDMat(K+α*I)
#     end
# end

@inline function logbtl(choices::Array{<:Int, 2},
                        goodness::Array{<:Real},
                        scale::Real)
    goodness_matrix = goodness[choices]
    scaled  = goodness_matrix / scale
    logpref = scaled[:,1] - logsumexp(scaled, dims=2)[:,1]
    sum(logpref)
end

@inline function logbtl_matrix(choices::Array{<:Int, 2},
                               goodness::Array{<:Real},
                               scale::Real)
    # goodness ∈ R^{npoints, ncandidates} 
    # Compute the preferences of all entries
    goodness_matrix = goodness[choices]
    scaled  = goodness_matrix / scale
    Z       = logsumexp(scaled, dims=2)[:,1]
    @simd for i = 1:size(scaled, 1)
        scaled[i,:] = scaled[i,:] .- Z[i]
    end
    return scaled
end

@inline function ∇logbtl(logbtl_matrix::Array{<:Real, 2},
                         latent::Array{<:Real, 1},
                         choices::Array{<:Int, 2},
                         scale::Real)
    pref         = exp.(logbtl_matrix)
    partial_grad = zeros(size(logbtl_matrix))

    # first column derivative
    pick  = pref[:,1]
    ∇pick = (1 .- pick) / scale
    partial_grad[:,1] = ∇pick

    # other column derivative
    comps  = pref[:,2:end]
    ∇comp  = comps / -scale
    partial_grad[:,2:end] = ∇comp

    partial_grad = reshape(partial_grad, :)
    choices      = reshape(choices, :)
    full_grad    = zeros(length(latent))
    @simd for i = 1:length(partial_grad)
        full_grad[choices[i]] += partial_grad[i]
    end
    return full_grad
end

function ∇²logbtl(logbtl_matrix::Array{<:Real, 2},
                  latent::Array{<:Real},
                  choices::Array{<:Int, 2},
                  scale::Real)
    pref    = exp.(logbtl_matrix)

    # hessian
    nlatent = length(latent)
    ndata   = size(logbtl_matrix, 1)
    ncand   = size(logbtl_matrix, 2)
    result  = zeros(nlatent, nlatent)

    @inbounds for entry_j = 1:ncand
        @inbounds for entry_i = 1:ncand
            @simd for block = 1:ndata
                H_i_j = pref[block, entry_j] / (scale*scale)
                if(entry_i == entry_j)
                    H_i_j *= pref[block, entry_j] - 1
                else
                    H_i_j *= pref[block, entry_i] 
                end
                latent_i = choices[block, entry_i]
                latent_j = choices[block, entry_j]
                result[latent_i, latent_j] += H_i_j 
            end
        end
    end
    return Symmetric(result)
end

@inline function approx_marginallike(a::Vector,
                                     B::LU,
                                     choices::Array{<:Int, 2},
                                     latent::Array{<:Real},
                                     scale::Real)
    # GPML 3.32
    loglike = logbtl(choices, latent, scale)
    t1 = loglike 
    t2 = dot(latent, a) / -2 
    t3 = logdet(B) / -2
    t1 + t2 + t3
end

@inline function approx_marginallike(K::Matrix,
                                     choices::Array{<:Int, 2},
                                     latent::Array{<:Real},
                                     scale::Real)
    # GPML 3.32
    btl_mat = logbtl_matrix(choices, latent, scale)
    ∇ll     = ∇logbtl(btl_mat, latent, choices, scale)
    W       = -∇²logbtl(btl_mat, latent, choices, scale)

    f   = reshape(latent, :)
    WK  = W*K
    b   = W*f + ∇ll
    B   = I + WK
    Blu = lu(B)
    a   = (b - Blu \ (WK*b))

    loglike = sum(btl_mat[:,1])
    t1 = loglike 
    t2 = dot(f, a) / -2 
    t3 = logdet(Blu) / -2

    # GPML 3.32
    t1 + t2 + t3
end

@inline function logjointlike(K::PDMats.PDMat,
                              choices::Array{<:Int, 2},
                              latent::Array{<:Real},
                              scale::Real)
    # GPML 3.12
    t3 = logdet(K) / -2
    t4 = size(K, 1) * log(2*π) / -2
    loglike = logbtl(choices, latent, scale)

    t1 = loglike 
    t2 = PDMats.invquad(K, latent) / -2

    t1 + t2 + t3 + t4
end

