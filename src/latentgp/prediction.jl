
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
    μ, sqrt(σ²)
end

# function mgp_predict(x::Vector,
#                      X::Matrix,
#                      Σ::Array{PDMats.PDMat},
#                      α::Matrix,
#                      ks::Array{KernelFunctions.Kernel})
#     ys = map(1:length(Σ)) do idx
#         k      = ks[idx]
#         k_star = KernelFunctions.kernelmatrix(k, reshape(x, (:,1)), X, obsdim=2)[1,:]
#         μ      = dot(k_star, α[:,idx])
#         σ²     = k(x, x) - PDMats.quad(Σ[idx], k_star)
#         Normal(μ, sqrt(σ²))
#     end
#     mean([y.μ for y in ys]), √(mean([(y.σ)^2 for y in ys]))
# end

# function mgp_predict(x::Matrix,
#                      X::Matrix,
#                      Σ::Array{PDMats.PDMat},
#                      α::Matrix,
#                      ks::Array{KernelFunctions.Kernel})
#     N   = length(Kinvs)
#     μs  = zeros(size(x, 2), N)
#     σ²s = zeros(size(μs))
#     for idx = 1:N
#         k      = ks[idx]
#         k_star = KernelFunctions.kernelmatrix(k, x, X, obsdim=2)
#         μ      = k_star * α[:,idx]
#         σ²     = [k(x[:,i], x[:,i]) - PDMats.quad(Σ[idx], k_star[i,:]) for i = 1:size(x,2)]
#         μs[:,idx]  = μ
#         σ²s[:,idx] = σ²
#     end
#     mean(μs, dims=2)[:,1], mean(σ²s, dims=2)[:,1]
# end
