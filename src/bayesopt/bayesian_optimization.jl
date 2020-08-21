
function expected_improvement(x::Vector,
                              y_opt::Real,
                              X::Matrix,
                              Kinvs::Array{PDMats.PDMat},
                              α::Matrix,
                              k::Array{KernelFunctions.Kernel})
    N  = length(Kinvs)
    ei = zeros(N)
    @simd for i = 1:N
        μ, σ  = gp_predict(x, X, Kinvs[i], α[:,i], k[i])
        Δy    = y_opt - μ
        z     = Δy / σ
        ei[i] = Δy * normcdf(z) + σ*normpdf(z)
    end
    mean(ei)
end
