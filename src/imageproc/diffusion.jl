
macro swap!(a::Symbol,b::Symbol)
    blk = quote
        c = $(esc(a))
        $(esc(a)) = $(esc(b))
        $(esc(b)) = c
    end
    return blk 
end

function pmad_kernel!(image, output, g, λ::Real)
    M = size(image, 1)
    N = size(image, 2)

    @inbounds for j = 1:N
        @simd for i = 1:M
            w = image[max(i - 1, 1), j]
            n = image[i, max(j - 1, 1)]
            c = image[i, j]
            s = image[i, min(j + 1, N)]
            e = image[min(i + 1, M), j]
            
            ∇n = n - c
            ∇s = s - c
            ∇w = w - c
            ∇e = e - c

            Cn = g(abs(∇n))
            Cs = g(abs(∇s))
            Cw = g(abs(∇w))
            Ce = g(abs(∇e))

            output[i, j] = c + λ*(Cn * ∇n + Cs * ∇s + Ce * ∇e + Cw * ∇w)
        end
    end
end

function diffusion(image, λ::Real, K::Real, niters::Int)
#=
    Perona, Pietro, and Jitendra Malik. 
    "Scale-space and edge detection using anisotropic diffusion." 
    IEEE Transactions on Pattern Analysis and Machine Intelligence (PAMI), 1990.
=##
    @assert 0 <= λ && λ <= 0.25

    @inline function g(norm∇I)
        coef = (norm∇I / K)
        1/(1 + coef*coef)
    end

    output = deepcopy(image)
    image  = deepcopy(image)
    for i = 1:niters
        pmad_kernel!(image, output, g, λ)
        @swap!(image, output)
    end
    return output
end

function srad_kernel_cpu!(image, output, C, M, N, t, Δt::Real, ρ::Real)
    ϵ       = eps(Float32)
    q0²     = exp(-ρ*t*2)
    q0const = q0² * (1 + q0²)
    @inbounds for j = 1:N
        @simd for i = 1:M
            Iw = image[max(i - 1, 1), j]
            In = image[i, min(j + 1, N)]
            Ic = image[i, j]
            Is = image[i, max(j - 1, 1)]
            Ie = image[min(i + 1, M), j]

            ∇eI = Ie - Ic
            ∇wI = Iw - Ic
            ∇nI = In - Ic
            ∇sI = Is - Ic

            ∇²I    = (∇eI + ∇wI + ∇nI + ∇sI) / (Ic + ϵ)
            ∇²I²   = ∇²I * ∇²I
            ∇I²    = (∇eI*∇eI + ∇wI*∇wI + ∇nI*∇nI + ∇sI*∇sI) / (Ic*Ic + ϵ)
            denom  = (1 + ∇²I / 4) 
            q²     = max(((∇I² / 2) - (∇²I² / 16)) / (denom*denom + ϵ), 0.0)
            C[i,j] = exp(-(q² - q0²)/(q0const + ϵ))
        end
    end

    @inbounds for j = 1:N
        @simd for i = 1:M
            Iw = image[max(i - 1, 1), j]
            In = image[i, min(j + 1, N)]
            Ic = image[i, j]
            Is = image[i, max(j - 1, 1)]
            Ie = image[min(i + 1, M), j]

            ∇eI = Ie - Ic
            ∇wI = Iw - Ic
            ∇nI = In - Ic
            ∇sI = Is - Ic

            Cw  = C[max(i - 1, 1), j]
            Cn  = C[i, min(j + 1, N)]
            Cs  = C[i, max(j - 1, 1)]
            Ce  = C[min(i + 1, M), j]

            d = Cw*∇wI + Ce*∇eI + Cn*∇nI + Cs*∇sI
            output[i, j] = Ic + (Δt/4)*d
        end
    end
end

function srad_gpu_phase1!(image, C, M, N, q0², q0const)
    ϵ = eps(Float32)
    i = (CUDA.blockIdx().x-1) * CUDA.blockDim().x + CUDA.threadIdx().x
    j = (CUDA.blockIdx().y-1) * CUDA.blockDim().y + CUDA.threadIdx().y

    if(i > M || j > N)
        return
    end
    
    @inbounds Iw = image[max(i - 1, 1), j]
    @inbounds In = image[i, min(j + 1, N)]
    @inbounds Ic = image[i, j]
    @inbounds Is = image[i, max(j - 1, 1)]
    @inbounds Ie = image[min(i + 1, M), j]

    ∇eI = Ie - Ic
    ∇wI = Iw - Ic
    ∇nI = In - Ic
    ∇sI = Is - Ic
    ∇²I   = (∇eI + ∇wI + ∇nI + ∇sI) / (Ic + ϵ)
    ∇²I²  = ∇²I * ∇²I
    ∇I²   = (∇eI*∇eI + ∇wI*∇wI + ∇nI*∇nI + ∇sI*∇sI) / (Ic*Ic + ϵ)
    denom = (1 + ∇²I / 4) 
    q²    = max(((∇I² / 2) - (∇²I² / 16)) / (denom*denom + ϵ), Float32(0.0))
    @inbounds C[i,j] = exp(-(q² - q0²)/q0const)
    return
end
    
function srad_gpu_phase2!(image, output, C, M, N, CΔt)
    i = (CUDA.blockIdx().x-1) * CUDA.blockDim().x + CUDA.threadIdx().x
    j = (CUDA.blockIdx().y-1) * CUDA.blockDim().y + CUDA.threadIdx().y

    if(i > M || j > N)
        return
    end

    @inbounds Iw = image[max(i - 1, 1), j]
    @inbounds In = image[i, min(j + 1, N)]
    @inbounds Ic = image[i, j]
    @inbounds Is = image[i, max(j - 1, 1)]
    @inbounds Ie = image[min(i + 1, M), j]

    ∇eI = Ie - Ic
    ∇wI = Iw - Ic
    ∇nI = In - Ic
    ∇sI = Is - Ic

    @inbounds Cw  = C[max(i - 1, 1), j]
    @inbounds Cn  = C[i, min(j + 1, N)]
    @inbounds Cs  = C[i, max(j - 1, 1)]
    @inbounds Ce  = C[min(i + 1, M), j]

    d = Cw*∇wI + Ce*∇eI + Cn*∇nI + Cs*∇sI
    @inbounds output[i, j] = Ic + CΔt*d
    return
end

function srad(image::AbstractArray{Float32},
              Δt::Real,
              ρ::Real,
              niters::Int;
              device=CUDA.CuDevice(0))
#=
    Yu, Yongjian and Acton, S. T.
    "Speckle reducing anisotropic diffusion." 
    IEEE Transactions on Image Processing (TIP), 2002.
=##
    @assert ρ < 1.0
    M      = size(image, 1)
    N      = size(image, 2)

    image  = convert(Array{Float32}, image)
     
    output = Array{Float32}(undef, M, N)

    # Diffusion coefficient buffer
    C      = Array{Float32}(undef, M, N)

    if((device isa CUDA.CuDevice) && CUDA.functional())
        image_dev  = CUDA.CuArray(image)
        output_dev = CUDA.CuArray(output)
        C_dev      = CUDA.CuArray(C)

        thread_x = 8
        thread_y = 8
        block_x  = ceil(Int, M / thread_x)
        block_y  = ceil(Int, N / thread_y)
        threads  = (thread_x, thread_y)
        blocks   = (block_x,  block_y)

        ρ   = Float32(ρ)
        Δt  = Float32(Δt)
        CΔt = Float32(Δt/4)

        CUDA.device!(device) do
            for t = 1:niters
                q0²     = exp(-ρ*(t-1)*Δt*2)
                q0const = q0² * (1 + q0²) + eps(Float32)
                CUDA.@cuda(threads=threads, blocks=blocks,
                           srad_gpu_phase1!(image_dev, C_dev, M, N, q0², q0const))
                CUDA.@cuda(threads=threads, blocks=blocks,
                           srad_gpu_phase2!(image_dev, output_dev, C_dev, M, N, CΔt))
                @swap!(image_dev, output_dev)
            end
        end
        return Array{Float32}(output_dev)
    else
        for t = 1:niters
            srad_kernel_cpu!(image, output, C, M, N, Δt*(t-1), Δt, ρ)
            @swap!(image, output)
        end
        return output
    end
end

