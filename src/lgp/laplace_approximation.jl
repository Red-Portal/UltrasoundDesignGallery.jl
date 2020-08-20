
function laplace_approximation(K::PDMats.PDMat,
                               Kinv::PDMats.PDMat,
                               scale::Float64,
                               initial_latent::Array{Float64, 2};
                               verbose::Bool=true)
    latent_shape = size(initial_latent)
    t3   = logdet(K) / 2
    t4   = size(K, 1) * log(2*π) / 2

    function fgh!(F, G, H, x)
        latent = reshape(x, latent_shape)
        if(isnothing(G) && isnothing(H))
            -logjoint_prob(K, Kinv, latent, scale)
        else
            logpref = logbtl_full(latent, scale)
            ∇L      = ∇logbtl(logpref, scale)
            W       = -∇²logbtl(logpref, reshape(∇L, size(logpref)), scale)
            
            if(!isnothing(G))
                # GPML 3.18
                # Note: GPML proposes a simpler, fused Newton step in (3.18).
                G .= Kinv.mat*x - ∇L        
            end
            if(!isnothing(H))
                # GPML 3.18
                H .= Kinv.mat + W        
            end
            if(!isnothing(F))
                -logjoint_prob(K, Kinv, latent, scale)
            else
                nothing
            end
        end
    end

    opt_res = Optim.optimize(Optim.only_fgh!(fgh!),
                             reshape(initial_latent, :),
                             Optim.NewtonTrustRegion(),
                             Optim.Options(g_tol = 1e-4,
                                           x_tol = 1e-4))
    if(verbose)
        @info(opt_res)
    end

    μ_latent = Optim.minimizer(opt_res)
    logpref  = logbtl_full(reshape(μ_latent, latent_shape), scale)
    ∇L       = ∇logbtl(logpref, scale)
    W        = -∇²logbtl(logpref, reshape(∇L, size(logpref)), scale)
    Σ_latent = inv(PDMats.PDMat(Kinv.mat + W))

    μ_latent, Σ_latent
end
