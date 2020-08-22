
@inline function safe_cholesky(K::Matrix)
    try
        PDMats.PDMat(K)
    catch
        Kmax = maximum(K)
        α    = eps(eltype(K))
        while !isposdef(K+α*I) && α < 0.01*Kmax
            α *= 2.0
        end
        if α >= 0.01*Kmax
            throw(ErrorException("Adding noise on the diagonal was not sufficient to build a positive-definite matrix:\n\t- Check that your kernel parameters are not extreme\n\t- Check that your data is sufficiently sparse\n\t- Maybe use a different kernel"))
        end
        PDMats.PDMat(K+α*I)
    end
end

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
    Z       = logsumexp(scaled, dims=2)[:,1]
    @simd for i = 1:size(scaled, 1)
        scaled[i,:] = scaled[i,:] .- Z[i]
    end
    return scaled
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
                          gradlogbtl::Array{Float64, 2},
                          scale::Float64)
    # logpref ∈ R^{npoints, ncandidates} 
    # logpref[:,1]     are the choices
    # logpref[:,2:end] are the compared candidates
    pref    = exp.(logpref)

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
                        gradlogbtl[block, entry_j] + 1 / scale
                    else
                        gradlogbtl[block, entry_j]
                    end
                end

                result[i, j] = pref[block, entry_i] * ∇btl_i_j / -scale
            end
        end
    end
    return Symmetric(result)
end

@inline function approx_marginallike(a::Vector,
                                     B::LU,
                                     latent::Array{<:Real, 2},
                                     scale::Real)
    # GPML 3.32
    loglike = logbtl(latent, scale)
    f       = reshape(latent, :)

    t1 = loglike 
    t2 = dot(f, a) / -2 
    t3 = logdet(B) / -2

    t1 + t2 + t3
end

@inline function approx_marginallike(K::Matrix,
                                     latent::Array{<:Real, 2},
                                     scale::Real)
    # GPML 3.32
    logpref = logbtl_full(latent, scale)
    ∇ll     = ∇logbtl(logpref, scale)
    W       = -∇²logbtl(logpref, reshape(∇ll, size(logpref)), scale)

    f   = reshape(latent, :)
    WK  = W*K
    b   = W*f + ∇ll
    B   = I + WK
    Blu = lu(B)
    a   = (b - Blu \ (WK*b))

    loglike = sum(logpref[:,1])

    t1 = loglike 
    t2 = dot(f, a) / -2 
    t3 = logdet(Blu) / -2

    # GPML 3.32
    t1 + t2 + t3
end

@inline function logjointlike(K::PDMats.PDMat,
                              latent::Array{<:Real, 2},
                              scale::Real)
    # GPML 3.12
    t3 = logdet(K) / -2
    t4 = size(K, 1) * log(2*π) / -2
    loglike = logbtl(latent, scale)

    t1 = loglike 
    t2 = PDMats.invquad(K, reshape(latent, :)) / -2

    t1 + t2 + t3 + t4
end

@inline function construct_kernel(ℓ², σ², ϵ²)
    # x ∈ R^{parameters, latents}
    k = KernelFunctions.Matern52Kernel()
    t = KernelFunctions.ScaleTransform(1/ℓ²)
    ϵ = ϵ²*KernelFunctions.EyeKernel()
    k = σ²*KernelFunctions.transform(k, t) + ϵ
    k
end

function compute_gram_matrix(data, ℓ², σ², ϵ²)
    kernel = construct_kernel(ℓ², σ², ϵ²)
    K = KernelFunctions.kernelmatrix(
        kernel, reshape(data, (size(data,1),:)), obsdim=2)
end

