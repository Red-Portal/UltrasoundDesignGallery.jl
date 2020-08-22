
function laplace_approximation(K::PDMats.PDMat,
                               scale::Float64,
                               initial_latent::Array{Float64, 2};
                               verbose::Bool=true)
    # Variant of the Newton's method based mode-locating algorithm (GPML, Algorithm 3.1)
    # Utilizes the Woodburry identity for avoiding two cholesky factorizations
    # per Newton iteration.
    # Reduces the stepsize whenever the marginal likelhood gets stuck
    # Algortihm 3.1 utilizes the fact that W is diagonal which is not for our case.
    # Note: ( K^{-1} + W )^{-1} = K ( I - ( I + W K )^{-1} ) W K
    max_iter = 20

    latent_shape = size(initial_latent)
    f            = reshape(initial_latent, :)
    latent       = initial_latent
    prev_f       = deepcopy(f)
    prev_mll     = -Inf

    WK  = nothing
    α   = 1.0
    a   = nothing
    B   = nothing
    for iteration = 1:max_iter
        logpref = logbtl_full(latent, scale)
        ∇ll     = ∇logbtl(logpref, scale)
        W       = -∇²logbtl(logpref, reshape(∇ll, size(logpref)), scale)

        WK  = W*K
        b   = W*f + ∇ll
        B   = I + WK
        Blu = lu(B)
        a   = (b - Blu \ (WK*b))

        f   = α*(K*a) + (1 - α)*prev_f
        mll = sum(logpref[:,1]) + dot(a, f)/-2 +  - logdet(Blu)/2

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
        latent   = reshape(f, latent_shape)
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
