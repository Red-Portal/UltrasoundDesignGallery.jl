
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

@inline function logjoint_prob(K::PDMats.PDMat,
                               Kinv::PDMats.PDMat,
                               latent::Array{<:Real, 2},
                               scale::Real)
    t3   = logdet(K) / 2
    t4   = size(K, 1) * log(2*π) / 2
    loglike = logbtl(latent, scale)

    t1 = loglike 
    t2 = PDMats.quad(Kinv, reshape(latent, :)) / 2

    # GPML 3.12
    t1 - t2 - t3 - t4
end

@inline function construct_kernel(ℓ², σ², ϵ²)
    # x ∈ R^{parameters, latents}
    k = KernelFunctions.Matern52Kernel()
    t = KernelFunctions.ScaleTransform(1/ℓ²)
    ϵ = ϵ²*KernelFunctions.EyeKernel()
    k = σ²*KernelFunctions.transform(k, t) + ϵ
    k
end

@inline function compute_gram_matrix(data, ℓ², σ², ϵ²)
    kernel = construct_kernel(ℓ², σ², ϵ²)
    K    = KernelFunctions.kernelmatrix(
        kernel, reshape(data, (size(data,1),:)), obsdim=2)
    K    = PDMats.PDMat(K)
    Kinv = inv(K)
    K, Kinv
end

