
function diffusion(image, λ::Real, K::Real, niters::Int)
#=
    Perona, Pietro, and Jitendra Malik. 
    "Scale-space and edge detection using anisotropic diffusion." 
    IEEE Transactions on Pattern Analysis and Machine Intelligence (PAMI), 1990.
=##
    @assert 0 <= λ && λ <= 0.25

    @inline function g(norm∇I)
        1/(1 + (norm∇I / K).^2)
    end

    @inline @inbounds function op(buf)
        north  = buf[1,2]
        south  = buf[3,2]
        west   = buf[2,1]
        east   = buf[2,3]
        center = buf[2,2]
        ∇n = north - center
        ∇s = south - center
        ∇w = west  - center
        ∇e = east  - center

        Cn = g(abs(∇n))
        Cs = g(abs(∇s))
        Cw = g(abs(∇w))
        Ce = g(abs(∇e))
        center + λ*(Cn * ∇n + Cs * ∇s + Ce * ∇e + Cw * ∇w)
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
