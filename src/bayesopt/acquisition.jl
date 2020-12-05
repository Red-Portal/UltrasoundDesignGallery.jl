
function expected_improvement(x::AbstractVector,
                              y_opt::Real,
                              X::Matrix,
                              K::Array{PDMats.PDMat},
                              a::Matrix,
                              k::Array{KernelFunctions.Kernel})
    N  = length(K)
    ei = zeros(N)
    @simd for i = 1:N
        μ, σ² = gp_predict(x, X, K[i], a[:,i], k[i])
        σ     = sqrt(σ²)
        Δy    = μ - y_opt
        z     = Δy / σ
        ei[i] = Δy * normcdf(z) + σ*normpdf(z)
    end
    mean(ei)
end

function expected_improvement(x::AbstractVector,
                              y_opt::Real,
                              X::Matrix,
                              K::PDMats.PDMat,
                              a::Vector,
                              k::KernelFunctions.Kernel)
    μ, σ² = gp_predict(x, X, K, a, k)
    σ     = sqrt(σ²)
    Δy    = μ - y_opt
    z     = Δy / σ
    ei    = Δy * normcdf(z) + σ*normpdf(z)
    ei
end
