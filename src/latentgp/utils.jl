
@inline function data_matrix(points::Array{Float64, 2},
                             choices::Array{Int64, 2})
    points[:, choices]
end

@inline function construct_kernel(ℓ², σ², ϵ²)
    # x ∈ R^{parameters, latents}
    k = KernelFunctions.Matern52Kernel()
    t = KernelFunctions.ScaleTransform(1/ℓ²)
    ϵ = ϵ²*KernelFunctions.EyeKernel()
    k = σ²*KernelFunctions.transform(k, t) + ϵ
    k
end

function compute_gram_matrix(X, ℓ², σ², ϵ²)
    kernel = construct_kernel(ℓ², σ², ϵ²)
    K = KernelFunctions.kernelmatrix(kernel, X, obsdim=2)
end
