
function precompute_lgp(fs::Matrix, Kinvs::Array{PDMats.PDMat, 1})
    αs = zeros(size(fs))
    for (idx, Kinv) in enumerate(Kinvs)
        αs[:,idx] = Kinv.mat * fs[:, idx]
    end
    αs
end

function mgp_predict(x::Vector, data_X::Matrix, Kinvs::Array{PDMats.PDMat}, α::Matrix, θ::Matrix)
    ys = map(enumerate(Kinvs)) do (idx, Kinv)
        k      = construct_kernel(θ[:,idx]...) 
        #k_star = KernelFunctions.kernelmatrix(k, x), )
        μ      = dot(k_star, α[:,idx])
        σ²     = k(x, x) - PDMats.quad(Kinv, k_star)
        Normal(μ, sqrt(σ²))
    end
    mean([y.μ for y in ys]), √(mean([(y.σ)^2 for y in ys]))
end

function mgp_predict(x::Matrix, X::Matrix, Kinvs::Array{PDMats.PDMat}, α::Matrix, θ::Matrix)
    N   = length(Kinvs)
    μs  = zeros(size(x, 2), N)
    σ²s = zeros(size(μs))
    for idx = 1:N
        k      = construct_kernel(θ[:,idx]...) 
        k_star = KernelFunctions.kernelmatrix(k, x, X, obsdim=2)
        μ      = k_star * α[:,idx]
        σ²     = [k(x[:,i], x[:,i]) - PDMats.quad(Kinvs[idx], k_star[i,:]) for i = 1:size(x,2)]
        μs[:,idx]  = μ
        σ²s[:,idx] = σ²
    end
    mean(μs, dims=2)[:,1], mean(σ²s, dims=2)[:,1]
end
