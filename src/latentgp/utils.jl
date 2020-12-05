
@inline function data_matrix(points::Array{Float64, 2},
                             choices::Array{Int64, 2})
    points[:, choices]
end

@inline function construct_kernel(logσ², logϵ², logℓ)
    # x ∈ R^{parameters, latents}
    σ² = exp(logσ²)
    ϵ² = exp(logϵ²)
    ℓ  = exp.(-logℓ)
    k  = KernelFunctions.Matern52Kernel()
    t  = KernelFunctions.ARDTransform(ℓ)
    ϵ  = ϵ²*KernelFunctions.EyeKernel()
    k  = σ²*KernelFunctions.transform(k, t) + ϵ
    k
end

function compute_gram_matrix(X, logσ², logϵ², logℓ)
    kernel = construct_kernel(logσ², logϵ², logℓ)
    K = KernelFunctions.kernelmatrix(kernel, X, obsdim=2)
end
