
@inline function data_matrix(points::Array{Float64, 2},
                             choices::Array{Int64, 2})
    points[:, choices]
end

@inline function construct_kernel(σ², ϵ², ℓ)
    # x ∈ R^{parameters, latents}
    k = KernelFunctions.Matern52Kernel()
    t = KernelFunctions.ARDTransform(1 ./ ℓ)
    ϵ = ϵ²*KernelFunctions.EyeKernel()
    k = σ²*KernelFunctions.transform(k, t) + ϵ
    k
end

function compute_gram_matrix(X, σ², ϵ², ℓ)
    kernel = construct_kernel(σ², ϵ², ℓ)
    K = KernelFunctions.kernelmatrix(kernel, X, obsdim=2)
end
