
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

    @inline @inbounds function op(buf)
        west   = Float32(buf[2,1])
        north  = Float32(buf[1,2])
        center = Float32(buf[2,2])
        south  = Float32(buf[3,2])
        east   = Float32(buf[2,3])

        ∇n = north - center
        ∇s = south - center
        ∇w = west  - center
        ∇e = east  - center
        Cn = g(abs(∇n))
        Cs = g(abs(∇s))
        Cw = g(abs(∇w))
        Ce = g(abs(∇e))
        Float64(center + λ*(Cn * ∇n + Cs * ∇s + Ce * ∇e + Cw * ∇w))
    end

    result     = image
    div_kernel = Images.centered([0 1 0; 1 -4 1; 0 1 0])
    for i = 1:niters
        result = ImageFiltering.mapwindow(op, result, (3,3))
    end
    return result
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
