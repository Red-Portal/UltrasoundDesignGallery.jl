
function laplace_approximation(K::PDMats.PDMat,
                               choices::Array{<:Int, 2},
                               initial_latent::Vector{<:Real},
                               scale::Float64;
                               verbose::Bool=true)
#=
    Variant of the Newton's method based mode-locating algorithm (GPML, Algorithm 3.1)
    Utilizes the Woodburry identity for avoiding two cholesky factorizations
    per Newton iteration.

    Reduces the stepsize whenever the marginal likelhood gets stuck
    Algortihm 3.1 utilizes the fact that W is diagonal which is not for our case.

    Note: ( K^{-1} + W )^{-1} = K ( I - ( I + W K )^{-1} ) W K
=##
    max_iter = 20

    f            = initial_latent
    prev_f       = deepcopy(f)
    prev_mll     = -Inf

    WK  = nothing
    α   = 1.0
    a   = nothing
    B   = nothing
    for iteration = 1:max_iter
        btl_mat = logbtl_matrix(choices, f, scale)
        ∇ll     = ∇logbtl(btl_mat, f, choices, scale)
        W       = -∇²logbtl(btl_mat, ∇ll, choices, scale)

        WK  = W*K
        b   = W*f + ∇ll
        B   = I + WK
        Blu = lu(B)
        a   = (b - Blu \ (WK*b))

        f   = α*(K*a) + (1 - α)*prev_f
        mll = sum(btl_mat[:,1]) + dot(a, f)/-2 + logdet(Blu)/-2

        ∇mll = norm(∇ll - a)
        Δf   = norm(prev_f - f)

        if(∇mll < 1e-4 || Δf < 1e-4)
            break
        end

        if(mll <= prev_mll)
            α /= 2
        end

        if(verbose)
            @info "Laplace approximation stat" iteration ∇mll Δf
        end

        prev_f   = deepcopy(f)
        prev_mll = mll
    end
    μ  = f
    Σ  = K * (I - (I + WK) \ WK)
    # Due to numerical inaccuracies,
    # Σ often is not symmetric out of the box
    Σu = UpperTriangular(Σ)
    Σd = Diagonal(Σ)
    Σ  = Σu + Σu' - Σd
    return μ, Σ, a, B
end
