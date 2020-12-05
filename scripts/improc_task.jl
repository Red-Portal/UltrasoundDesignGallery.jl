
function lerp(x::Real, lo::Real, hi::Real)
    return hi*x + (1-x)*lo
end

function transform_domain(x::Vector)
    ϵ   = eps(Float64)
    res = clamp.(x, ϵ, 1-ϵ)

    i = 1
    res[i]   = 2^lerp(x[i],   -6, 2.0)  # Δt 
    res[i+1] = 2^lerp(x[i+1], -7, 0.0)  # ρ
    res[i+2] = lerp(x[i+2],   2, 8)    # niters
    i += 3

    res[i]   = 2^lerp(x[i],   -6, 2.0) # Δt
    res[i+1] = 2^lerp(x[i+1], -7, 0.0) # ρ
    res[i+2] = lerp(x[i+2],    2, 8)   # niters
    res[i+3] = 2^lerp(x[i+3], -6, 4)   # σ
    res[i+4] = lerp(x[i+4],    0, 1)   # α
    res[i+5] = lerp(x[i+5],    1, 2)   # β
    i += 6

    res[i]   = 2^lerp(x[i],   -6, 2.0) # Δt 
    res[i+1] = 2^lerp(x[i+1], -7, 0.0) # ρ
    res[i+2] = lerp(x[i+2],    2, 9)   # niters
    res[i+3] = 2^lerp(x[i+3], -6, 4)   # σ
    res[i+4] = lerp(x[i+4],    0, 1)   # α
    res[i+5] = lerp(x[i+5],    1, 2)   # β
    i += 6

    res[i]   = 2^lerp(x[i],   -6, 2.0) # Δt 
    res[i+1] = 2^lerp(x[i+1], -7, 0.0) # ρ
    res[i+2] = lerp(x[i+2],    2, 8)   # niters
    res[i+3] = 2^lerp(x[i+3], -6, 4)   # σ
    res[i+4] = lerp(x[i+4],    0, 1)   # α
    res[i+5] = lerp(x[i+5],    1, 2)   # β
    i += 6

    res
end

function reduce_speckle(img, θ)
    M = size(img, 1)
    N = size(img, 2)

    M_pad = 2^4 * ceil(Int, M / 2^4)
    N_pad = 2^4 * ceil(Int, N / 2^4)

    img_pad           = zeros(Float32, M_pad, N_pad)
    img_pad[1:M, 1:N] = img

    device = CUDA.CuDevice(0)
    factor = 2
    pyr    = Images.gaussian_pyramid(img_pad, 4, factor, Float32(16.0))
    pyr    = laplacian_pyramid(pyr, factor)
    i      = 1 

    x      = srad(CUDA.CuArray(pyr[1]), θ[i], θ[i+1], floor(Int, θ[i+2]); device=device)
    i     += 3

    x      = srad(CUDA.CuArray(pyr[2]), θ[i], θ[i+1], floor(Int, θ[i+2]); device=device)
    pyr[2] = homo_contrast(x, θ[i+3], θ[i+4], θ[i+5]; device=device)
    i     += 6

    x      = srad(CUDA.CuArray(pyr[3]), θ[i], θ[i+1], floor(Int, θ[i+2]); device=device)
    pyr[3] = homo_contrast(x, θ[i+3], θ[i+4], θ[i+5]; device=device)
    i     += 6

    x      = srad(CUDA.CuArray(pyr[4]), θ[i], θ[i+1], floor(Int, θ[i+2]); device=device)
    pyr[4] = homo_contrast(x, θ[i+3], θ[i+4], θ[i+5]; device=device)

    img_pad = synthesize_pyramid(pyr, 2)
    img     = img_pad[1:M, 1:N]
    img     = img / maximum(img)
    img     = clamp.(img, Float32(0.0), Float32(1.0))

    img[isnan.(img)] .= 0.0
    img
end
