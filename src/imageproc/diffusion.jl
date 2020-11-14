
macro swap!(a::Symbol,b::Symbol)
    blk = quote
        c = $(esc(a))
        $(esc(a)) = $(esc(b))
        $(esc(b)) = c
    end
    return blk 
end

function duffision_kernel!(image, output, g, λ::Real)
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
        duffision_kernel!(image, output, g, λ)
        @swap!(image, output)
    end
    return output
end

# function lee_diffusion(image, Cu::Real, niters::Int)
#     result     = image
#     div_kernel = centered([0 1 0; 1 -4 1; 0 1 0])
#     for i = 1:niters
#         ḡ   = mapwindow(mean, result, (3,3))
#         σ²g = mapwindow(var,  result, (3,3))
#         C²  = σ²g ./ ḡ.^2
#         C   = (Cu.^4 + Cu.^2)./(Cu.^4 .+ C²) 
#         ∇²u = ImageFiltering.imfilter(result, div_kernel)
#         result = result + C.*∇²u
#     end
#     return result
# end
