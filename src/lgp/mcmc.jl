
@inline function pseudo_marginal_full(prng, data, scale, ℓ², σ², ϵ², latent, u)
"""
    Filippone, Maurizio, and Mark Girolami. 
    "Pseudo-marginal Bayesian inference for Gaussian processes." 
    IEEE Transactions on Pattern Analysis and Machine Intelligence (2014)
"""
    f_shape  = size(data)[2:3]
    try 
        K, Kinv  = compute_gram_matrix(data, ℓ², σ², ϵ²)
        μ_f, Σ_f = laplace_approximation(K, Kinv, scale, reshape(latent, f_shape))
        q        = MvNormal(μ_f, Σ_f)

        f_sample = PDMats.unwhiten(q.Σ, u) + q.μ
        joint    = logjoint_prob(K, Kinv, reshape(f_sample, f_shape), scale)
        logpml   = joint - logpdf(q, f_sample)
        logpml, f_sample, q, K, Kinv
    catch
        @warn "Cholesky failed. Rejecting proposal"
        -Inf, nothing, nothing, nothing, nothing
    end
end

@inline function pseudo_marginal_partial(prng, data, q, K, Kinv, scale, u)
"""
    Filippone, Maurizio, and Mark Girolami. 
    "Pseudo-marginal Bayesian inference for Gaussian processes." 
    IEEE Transactions on Pattern Analysis and Machine Intelligence (2014)
"""
    f_shape  = size(data)[2:3]
    f_sample = PDMats.unwhiten(q.Σ, u) + q.μ
    joint    = logjoint_prob(K, Kinv, reshape(f_sample, f_shape), scale)
    logpml   = joint - logpdf(q, f_sample)
    logpml
end

function ess_transition(prng, loglike, prev_θ, prev_like, prior)
"""
    Murray, Iain, Ryan Adams, and David MacKay. 
    "Elliptical slice sampling." 
    Artificial Intelligence and Statistics. 2010.
"""
    @label propose
    ν     = rand(prng, prior)
    μ     = mean(prior)
    u     = rand(prng)
    logy  = prev_like + log(u)
    ϵ     = rand(prng)*2*π
    ϵ_min = ϵ - 2*π
    ϵ_max = deepcopy(ϵ)

    while(true)
        cosϵ   = cos(ϵ) 
        sinϵ   = sin(ϵ)
        a      = 1 - (cosϵ + sinϵ)

        prop_θ    = @. cosϵ*prev_θ + sinϵ*ν + a*μ
        prop_logp = loglike(prop_θ)

        if(isinf(prop_logp))
            @goto propose
        end

        if(prop_logp > logy)
            return prop_θ, prop_logp
        else
            if(ϵ < 0)
                ϵ_min = deepcopy(ϵ)
            else
                ϵ_max = deepcopy(ϵ)
            end
            ϵ = rand(prng, Uniform(ϵ_min, ϵ_max))
        end
    end
end

function pm_ess(prng, samples, warmup, initial_θ, initial_f, prior, scale, data)
"""
    Murray, Iain, and Matthew Graham. 
    "Pseudo-marginal slice sampling." 
    Artificial Intelligence and Statistics. 2016.
"""
    θ_samples    = zeros(length(initial_θ), samples) 
    f_samples    = zeros(length(initial_f), samples)
    Kinv_samples = Array{PDMats.PDMat}(undef, samples)
    prev_logpml  = -Inf

    u_prior = MvNormal(length(initial_f), 1.0)


    θ = log.(initial_θ)
    u = rand(prng, u_prior)
    logpml, f, q, K, Kinv = pseudo_marginal_full(
        prng, data, scale, initial_θ[1], initial_θ[2], initial_θ[3], initial_f, u)
    if(isinf(logpml))
        throw(ArgumentError("Initial hyperparameters are not valid"))
    end

    ProgressMeter.@showprogress for i = 1:(samples+warmup)
        Lu = u_in->begin
            pseudo_marginal_partial(prng, data, q, K, Kinv, scale, u_in)
        end
        u, logpml = ess_transition(prng, Lu, u, logpml, u_prior)

        Lθ = θ_in->begin
            θ_lin = exp.(θ_in)
            logpml, f, q, K, Kinv = pseudo_marginal_full(
                prng, data, scale, θ_lin[1], θ_lin[2], θ_lin[3], f, u)
            logpml
        end
        θ, logpml = ess_transition(prng, Lθ, θ, logpml, prior)

        if(i > warmup)
            θ_samples[:,i-warmup]  = exp.(θ)
            f_samples[:,i-warmup]  = f
            Kinv_samples[i-warmup] = deepcopy(Kinv)
        end
    end
    θ_samples, f_samples, Kinv_samples
end
