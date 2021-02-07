
using Distributions
using StatsBase
using StatsFuns
using FastGaussQuadrature
using KernelFunctions
using UnPack
using LinearAlgebra
using PDMats
using Plots, StatsPlots
using FiniteDiff
using Random123
using RandomNumbers
using Roots
using Optim

struct PBODataPoint{F<:Real}
    α::F
    βs::AbstractVector{F}
    ξ::AbstractVector{F}
    x::AbstractVector{F}
end

struct PBOHyperparameters{F<:Real}
    ℓσ::F
    ℓl::AbstractVector{F}
    ℓσn::F
end

function pgp_ℓpl(Δ, hypers::PBOHyperparameters)
    n_itgr = 32
    itgr_x, itgr_w = gausshermite(n_itgr)
    σ = exp(hypers.ℓσn)
    m = size(Δ, 2)

    # convolution of a gaussian pdf and cdf
    ϕ★ϕ(x) = dot(itgr_w, normcdf.(x .- itgr_x*√2)) / √π

    -sum(ϕ★ϕ.(Δ)) / m
end

function pgp_ℓpθ(hypers::PBOHyperparameters)
    @unpack ℓσ, ℓl, ℓσn = hypers
    pℓσ  = logpdf(Normal(0.0, 2.0),          ℓσ)
    pℓl  = logpdf(MvNormal(length(ℓl), 2.0), ℓl)
    pℓσn = logpdf(Normal(0.0, 2.0),         ℓσn)
    pℓσ + pℓl + pℓσn
end

function pgp_kernel(hypers::PBOHyperparameters)
    @unpack ℓσ, ℓl, ℓσn = hypers
    k     = Matern52Kernel()
    l⁻¹   = exp.(-ℓl)
    t_ard = ARDTransform(l⁻¹)
    σ²    = exp(2*ℓσ)
    σ² * transform(k, t_ard)
end

function pgp_gram_matrix(data::Array{<:PBODataPoint},
                         hypers::PBOHyperparameters)
    n_dims = length(data[1].x)
    n_data = length(data)
    n_β    = length(data[1].βs)
    n_f    = n_data * (1 + n_β)
    X      = Array{Float64}(undef, n_dims, n_f)

    for i = 1:n_data
        @unpack α, βs, ξ, x = data[i]
        f_idx      = (i - 1)*(1 + n_β) + 1
        X[:,f_idx] = x + ξ*α
        for j = 1:n_β
            X[:,f_idx+j] = x + ξ*βs[j]
        end
    end

    k      = pgp_kernel(hypers)
    Σ      = kernelmatrix(k, X, obsdim=2)
    display(heatmap(Σ))
    PDMats.PDMat(Σ), X
end

# function pgp_pmℓl(latents::AbstractVector{<:PBODataPoint},
#                   data::AbstractVector{<:PBODataPoint},
#                   hypers::PBOHyperparameters)
#     Σ   = PDMats.PDMat(Σ)
#     ℓpf = -1/2*invquad(Σ, f)

#     ℓpθ = pgp_ℓpθ(hypers)
#     ℓq  = ℓpf + ℓpy + ℓpθ

#     Σ, ℓq
# end

function pgp_is_α(idx, Δ_shape)
    n_α = Δ_shape[1]
    n_β = Δ_shape[2]
    mod(idx, n_β+1) == 1
end

function pgp_α_idx(idx, Δ_shape)
    n_β = Δ_shape[2]
    div(idx - 1, n_β+1) + 1
end

function pgp_β_idx(idx, Δ_shape)
    n_β = Δ_shape[2]
    mod(idx - 1, n_β+1)
end

function pgp_∇T_Λ(Σ, f, Δ, hypers::PBOHyperparameters)
    σ    = exp(hypers.ℓσ)
    ϕ₂   = Normal(0, √2)
    m    = size(Δ, 2)
    n_α  = size(Δ, 1)
    n_β  = m
    n_f  = prod(size(Δ)) + size(Δ, 1)
    ϕ₂Δ  = pdf(ϕ₂, Δ)

    # Gradient
    ∇q  = zeros(n_f)
    idx = 1
    for idx = 1:n_f
        if(pgp_is_α(idx, size(Δ)))
            i = pgp_α_idx(idx, size(Δ))
            ∇q[idx] = sum(view(ϕ₂Δ,i,:)) / σ / m
        else
            i = pgp_α_idx(idx, size(Δ))
            j = pgp_β_idx(idx, size(Δ))
            ∇q[idx] = -ϕ₂Δ[i,j] / σ / m
        end
        idx += 1
    end
    ∇T = -(Σ \ f) + ∇q

    # Hessian
    Λ = zeros(n_f, n_f)
    for idx = 1:(1+n_β):n_f
        # fᵢ ∈ α, fⱼ ∈ α
        i = pgp_α_idx(idx, size(Δ))
        Λ[idx,idx] = dot(view(ϕ₂Δ,i,:), view(Δ,i,:)) / 2 / σ^2 / m
        
        # fᵢ ∈ α, fⱼ ∈ β
        for j = 1:n_β
            β_idx = idx + j
            Λ[idx, β_idx] = -Δ[i,j] * ϕ₂Δ[i,j] / 2 / σ^2 / m
        end

        # fᵢ ∈ β, fⱼ ∈ β
        for j = 1:n_β
            β_idx = idx + j
            Λ[β_idx, β_idx] = Δ[i,j] * ϕ₂Δ[i,j] / 2 / σ^2 / m
        end
    end
    Λ = Symmetric(Λ)
    ∇T, Λ
end

function pgp_Δ(f, data, hypers)
    σ   = exp(hypers.ℓσ)
    n_α = length(data)
    n_β = length(data[1].βs)
    Δ   = zeros(n_α, n_β)
    for i = 1:n_α
        for j = 1:n_β
            α_idx = (i-1) * n_β + 1
            β_idx = (i-1) * n_β + j + 1
            Δ[i,j] = (f[β_idx] - f[α_idx]) / σ
        end
    end
    Δ
end

function derivative_test()
    ℓθ  = PBOHyperparameters(0.0, zeros(2), 0.0)
    k   = pgp_kernel(ℓθ)
    n_f = (1+5)*4
    X   = rand(2, n_f)
    Σ   = PDMats.PDMat(kernelmatrix(k, X, obsdim=2))

    σ = exp(ℓθ.ℓσ)
    f = randn(n_f)*10
    Δ = zeros(4, 5)
    for i = 1:4
        for j = 1:5
            α_idx = (i-1) * 6 + 1
            β_idx = (i-1) * 6 + j + 1
            Δ[i,j] = (f[β_idx] - f[α_idx]) / σ
        end
    end

    ∇T, Λ = pgp_∇T_Λ(Σ, f, Δ, ℓθ)
    ∇²T   = -inv(Σ) + Λ
    
    T(f_in) = begin
        σ    = exp(ℓθ.ℓσ)
        Δ_in = zeros(4, 5)
        for i = 1:4
            for j = 1:5
                α_idx = (i-1) * 6 + 1
                β_idx = (i-1) * 6 + j + 1
                Δ_in[i,j] = (f_in[β_idx] - f_in[α_idx]) / σ
            end
        end
        -0.5*invquad(Σ, f_in) + pgp_ℓpl(Δ_in, ℓθ)
    end
    ∇T̂  = FiniteDiff.finite_difference_gradient(T, f)
    ∇²T̂ = FiniteDiff.finite_difference_hessian(T, f)

    println("∇²T norm: ", norm(∇²T - ∇²T̂))
    println("∇T norm:  ", norm(∇T  - ∇T̂))
end

function pgp_laplace(Σ, data, f_init, hypers)
#=
    Variant of the Newton's method based mode-locating algorithm (GPML, Algorithm 3.1)
    Utilizes the Woodburry identity for avoiding two cholesky factorizations
    per Newton iteration.
    Reduces the stepsize whenever the marginal likelhood gets stuck
    Algortihm 3.1 utilizes the fact that W is diagonal which is not for our case.
    Note: ( K^{-1} + W )^{-1} = K ( I - ( I + W K )^{-1} ) W K
=##
    α        = 0.5
    max_iter = 50
    f        = f_init
    Δ        = pgp_Δ(f, data, hypers)
    ∇T, Λ    = pgp_∇T_Λ(Σ, f, Δ, hypers)
    for i = 1:max_iter
        W     = -Λ
        WK    = W*Σ
        b     = W*f + ∇T
        B     = I + WK
        Blu   = lu(B)
        a     = (b - Blu \ (WK*b))
        p     = (Σ*a) - f

        fₖ     = deepcopy(f)
        α      = 1.0
        c₂     = 0.9
        c₂pᵀ∇T = c₂*dot(p, ∇T)
        while(true)
            f     = fₖ + α*p
            Δ     = pgp_Δ(f, data, hypers)
            ∇T, Λ = pgp_∇T_Λ(Σ, f, Δ, hypers)

            if(-dot(p, ∇T) ≤ -c₂pᵀ∇T || α < 1e-4)
                break
            end
            α /= 2.0
        end
        if(α < 1e-4)
            break
        end

        T = -0.5*invquad(Σ, f) + pgp_ℓpl(Δ, hypers)
        @info "Newton step" iter=i norm∇T=norm(∇T) T=T α=α
    end
    f
end

function ackley(x::Vector{<:Real}, a = 20, b = 0.2, c = 2*π)
    d = length(x);
    term1 = -a * exp(-b * √(sum(x.^2) / d))
    term2 = -exp(sum(cos.(c*x)) / d)
    return term1 + term2 + a + exp(1)
end

function pgp_find_bounds(x, ξ)
    # Find the upper/lower bound of α
    # where x + αξ is within the hyper-cube [0, 1]ᴰ
    # by solving the system of equations:
    # xᵢ + αξᵢ = 1, if α ≥ 0, α ∈ A+, else then α ∈ A-
    # xᵢ + αξᵢ = 0, if α ≥ 0, α ∈ A+, else then α ∈ A-
    # α upper bound = max{α | α ∈ A+}
    # α lower bound = min{α | α ∈ A-}

    lb   = -Inf
    ub   = Inf
    dims = length(x)

    for i = 1:dims
        α = (1 - x[i]) / ξ[i]
        if(α > 0)
            ub = min(α, ub)
        else
            lb = max(α, lb)
        end

        α = -x[i] / ξ[i]
        if(α > 0)
            ub = min(α, ub)
        else
            lb = max(α, lb)
        end
    end
    lb, ub
end

function laplace_test()
    dims     = 2
    n_points = 10
    f        = x->ackley((x .- 0.5) * 20)
    prng     = Philox4x(UInt64, (0x22cf80fe510ae4bd, 0x1b23a6a16922848f,), 10)
    Random123.set_counter!(prng, 1)
    
    data = PBODataPoint{Real}[]
    for i = 1:n_points
        ξ  = randn(prng, dims)
        ξ /= maximum(ξ)
        x  = rand(prng, dims)

        lb, ub = pgp_find_bounds(x, ξ)

        res = optimize(α -> f(ξ*α + x), lb, ub)
        α   = Optim.minimizer(res)[1]
        β   = rand(prng, Uniform(lb, ub), 5) 

        datum = PBODataPoint{Real}(α, β, ξ, x)
        push!(data, datum)
    end

    hypers = PBOHyperparameters(-1.0, fill(-1.5, 2), -1.0)
    Σ, X   = pgp_gram_matrix(data, hypers)
    f      = pgp_laplace(Σ, data, zeros(size(Σ, 1)), hypers)
    display(scatter(X[1,:], X[2,:], f))
end

