
function pseudo_marginal_full(prng, data, choices, scale, σ², ϵ², ℓ², u)
#=
    Filippone, Maurizio, and Mark Girolami. 
    "Pseudo-marginal Bayesian inference for Gaussian processes." 
    IEEE Transactions on Pattern Analysis and Machine Intelligence (2014)
=##
    try 
        K          = compute_gram_matrix(data, σ², ϵ², ℓ²)
        K          = PDMats.PDMat(K)
        μ, Σ, a, B = laplace_approximation(K, choices, zeros(length(u)), scale;
                                           verbose=false)
        Σ          = PDMats.PDMat(Σ)
        q          = MvNormal(μ, Σ)
        f_sample   = PDMats.unwhiten(q.Σ, u) + q.μ
        joint      = logjointlike(K, choices, f_sample, scale)
        logpml     = joint - logpdf(q, f_sample)
        return logpml, f_sample, q, K, Σ, a
    catch err
        # if(isa(err, LinearAlgebra.PosDefException))
        #     @warn "Cholesky failed. Rejecting proposal"
        #     return -Inf, nothing, nothing, nothing, nothing, nothing
        # else
        #     throw(err)
        # end
        @warn "Cholesky or determinant failed. Rejecting proposal"
        return -Inf, nothing, nothing, nothing, nothing, nothing
    end
end

@inline function pseudo_marginal_partial(prng, choices, q, K, scale, u)
#=
    Filippone, Maurizio, and Mark Girolami. 
    "Pseudo-marginal Bayesian inference for Gaussian processes." 
    IEEE Transactions on Pattern Analysis and Machine Intelligence (2014)
=##
    f_sample = PDMats.unwhiten(q.Σ, u) + q.μ
    joint    = logjointlike(K, choices, f_sample, scale)
    logpml   = joint - logpdf(q, f_sample)
    logpml
end

function ess_transition(prng, loglike, prev_θ, prev_like, prior)
#=
    Murray, Iain, Ryan Adams, and David MacKay. 
    "Elliptical slice sampling." 
    Artificial Intelligence and Statistics. 2010.
=##
    @label propose
    chol_fail = 0
    ν     = rand(prng, prior)
    μ     = mean(prior)
    u     = rand(prng)
    logy  = prev_like + log(u)
    ϵ     = rand(prng)*2*π
    ϵ_min = ϵ - 2*π
    ϵ_max = deepcopy(ϵ)

    proposals = 1
    while(true)
        cosϵ   = cos(ϵ) 
        sinϵ   = sin(ϵ)
        a      = 1 - (cosϵ + sinϵ)

        prop_θ    = @. cosϵ*prev_θ + sinϵ*ν + a*μ
        prop_logp = loglike(prop_θ)

        if(isinf(prop_logp))
             chol_fail += 1
             if(chol_fail > 10)
                 throw(LinearAlgebra.PosDefException(1))
             end
             @goto propose
        end

        if(prop_logp > logy)
            return prop_θ, prop_logp, proposals
        else
            if(ϵ < 0)
                ϵ_min = deepcopy(ϵ)
            else
                ϵ_max = deepcopy(ϵ)
            end
            ϵ = rand(prng, Uniform(ϵ_min, ϵ_max))

            proposals += 1
        end
    end
end

function pm_ess(prng,
                samples::Int,
                warmup::Int,
                thinning::Int,
                initial_θ::Vector{<:Real},
                initial_f::Vector{<:Real},
                prior,
                scale::Real,
                data::Matrix{<:Real},
                choices::Matrix{<:Int})
#=
    Murray, Iain, and Matthew Graham. 
    Pseudo-marginal slice sampling. 
    Artificial Intelligence and Statistics. 2016.
=##
    θ_samples = zeros(length(initial_θ), samples) 
    f_samples = zeros(length(initial_f), samples)
    a_samples = zeros(length(initial_f), samples)
    K_samples = Array{PDMats.PDMat}(undef, samples)
    prev_logpml = -Inf

    u_prior = MvNormal(length(initial_f), 1.0)

    θ_map, f, Σ, a, B, K = map_laplace(data, choices, initial_θ, scale, prior;
                                       verbose=true)
    θ      = θ_map
    u      = rand(prng, u_prior)
    q      = MvNormal(f, Σ)
    logpml = pseudo_marginal_partial(prng, choices, q, K, scale, u)

    u_acc_mavg = OnlineStats.Mean()
    θ_acc_mavg = OnlineStats.Mean()
    prog       = ProgressMeter.Progress(samples+warmup)
    for i = 1:(samples+warmup)
        Lu = u_in->begin
            pseudo_marginal_partial(prng, choices, q, K, scale, u_in)
        end
        u, logpml, u_nprop = ess_transition(prng, Lu, u, logpml, u_prior)

        Lθ = θ_in->begin
            logpml, f, q, K, Σ, a = pseudo_marginal_full(
                prng, data, choices, scale,
                θ_in[1], θ_in[2], θ_in[3:end], u)
            logpml
        end
        θ, logpml, θ_nprop = ess_transition(prng, Lθ, θ, logpml, prior)

        if(i > warmup)
            θ_samples[:,i-warmup] = θ
            f_samples[:,i-warmup] = f
            a_samples[:,i-warmup] = a
            K_samples[i-warmup]   = K
        end
        u_acc = 1/u_nprop
        θ_acc = 1/θ_nprop
        OnlineStats.fit!(u_acc_mavg, u_acc)
        OnlineStats.fit!(θ_acc_mavg, θ_acc)
        ProgressMeter.next!(
            prog; showvalues=[(:iteration, i),
                              (:pseudo_marginal_loglikelihood, logpml),
                              (:u_acceptance, u_acc),
                              (:θ_acceptance, θ_acc),
                              (:u_average_acceptance, u_acc_mavg.μ),
                              (:θ_average_acceptance, θ_acc_mavg.μ)
                              ])
    end
    θ_samples = θ_samples[:,1:thinning:end]
    f_samples = f_samples[:,1:thinning:end]
    a_samples = a_samples[:,1:thinning:end]
    K_samples = K_samples[1:thinning:end]

    θ_samples, f_samples, a_samples, K_samples
end
