
function precompute_lgp(θs::Matrix)
    ks      = Array{KernelFunctions.Kernel}(undef, size(θs, 2))
    for idx = 1:size(θs, 2)
        ks[idx] = construct_kernel(θs[:,idx]...) 
    end
    ks
end

@inline function gp_predict(x::Vector,
                            X::Matrix,
                            K::PDMats.PDMat,
                            a::Vector,
                            k::KernelFunctions.Kernel)
    k_star = KernelFunctions.kernelmatrix(k, reshape(x, (:,1)), X, obsdim=2)[1,:]
    μ      = dot(k_star, a)
    σ²     = k(x, x) - PDMats.invquad(K, k_star)
    μ, σ²
end

@inline function gp_predict(x::Matrix,
                            X::Matrix,
                            K::PDMats.PDMat,
                            a::Vector,
                            k::KernelFunctions.Kernel)
    k_star = KernelFunctions.kernelmatrix(k, x, X, obsdim=2)[1,:]
    μ      = k_star * a
    σ²     = @inbounds [k(x[:,i], x[:,i]) - PDMats.invquad(K, k_star[i,:]) for i = 1:size(x,2)]
    μ, σ²
end

@inline function gp_predict(x::Vector,
                            X::Matrix,
                            K::Array{PDMats.PDMat},
                            a::Matrix,
                            ks::Array{KernelFunctions.Kernel})
    N   = length(K)
    μs  = zeros(N)
    σ²s = zeros(N)
    @simd for idx = 1:length(K)
        k        = ks[idx]
        k_star   = KernelFunctions.kernelmatrix(k, reshape(x, (:,1)), X, obsdim=2)[1,:]
        μ        = dot(k_star, a[:,idx])
        σ²       = k(x, x) - PDMats.invquad(K[idx], k_star)
        μs[idx]  = μ
        σ²s[idx] = σ²
    end
    mean(μs), mean(σ²s)
end

@inline function gp_predict(x::Matrix,
                            X::Matrix,
                            K::Array{PDMats.PDMat},
                            a::Matrix,
                            ks::Array{KernelFunctions.Kernel})
    N   = length(K)
    μs  = zeros(size(x, 2), N)
    σ²s = zeros(size(μs))
    @simd for idx = 1:N
        k      = ks[idx]
        k_star = KernelFunctions.kernelmatrix(k, x, X, obsdim=2)
        μ      = k_star * a[:,idx]
        σ²     = @inbounds [k(x[:,i], x[:,i]) - PDMats.invquad(K[idx], k_star[i,:])
                            for i = 1:size(x,2)]
        μs[:,idx]  = μ
        σ²s[:,idx] = σ²
    end
    mean(μs, dims=2)[:,1], mean(σ²s, dims=2)[:,1]
end
