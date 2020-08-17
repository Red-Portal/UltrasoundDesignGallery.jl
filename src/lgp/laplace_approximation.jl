
function laplace_approximation(K::Array{Float64, 2},
                               scale::Float64,
                               initial_latent::Array{Float64, 2})
    latent_shape = size(initial_latent)
    K    = PDMats.PDMat(K)
    Kinv = inv(K)
    t3   = logdet(K) / 2
    t4   = size(K, 1) * log(2*π) / 2

    function f(x)
        latent = reshape(x, latent_shape)
        logjoint_prob(K, Kinv, latent, scale)
    end

    function g!(G, x)
        latent  = reshape(x, latent_shape)
        logpref = logbtl_full(latent, scale)
        ∇L      = ∇logbtl(logpref, scale)
        W       = -∇²logbtl(logpref, reshape(∇L, size(logpref)), scale)

        # GPML 3.18
        # Note: GPML proposes a simpler, fused Newton step in (3.18).
        #       However, to use Optim.jl, we use the more conventional Newton step.
        #       Watch for the 'sign'!
        G      .= Kinv.mat*x - ∇L
    end

    function h!(H, x)
        latent  = reshape(x, latent_shape)
        logpref = logbtl_full(latent, scale)
        ∇L      = ∇logbtl(logpref, scale)
        W       = -∇²logbtl(logpref, reshape(∇L, size(logpref)), scale)

        # GPML 3.18
        H      .= Kinv.mat + W
    end

    opt_res = Optim.optimize(f, g!, h!, reshape(initial_latent, :),
                             Optim.NewtonTrustRegion())
    @info(opt_res)

    μ_latent = Optim.minimizer(opt_res)
    logpref  = logbtl_full(reshape(μ_latent, latent_shape), scale)
    ∇L       = ∇logbtl(logpref, scale)
    W        = -∇²logbtl(logpref, reshape(∇L, size(logpref)), scale)
    Σ_latent = inv(PDMats.PDMat(Kinv.mat + W))

    μ_latent, Σ_latent
end
