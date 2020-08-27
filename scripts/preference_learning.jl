
import TestImages
import Images
import Noise
include("../src/UltrasoundVisualGallery.jl")

import .UltrasoundVisualGallery

const USVG = UltrasoundVisualGallery

function lerp(x::Real, lo::Real, hi::Real)
    return hi*x + (1-x)*lo
end

@inline @inbounds function diffusion_transform_domain(x::Vector)
    res    = clamp.(x, 0, 1)
    res[1] = 2^lerp(x[1], 0, 5)

    res[2] = lerp(x[2], 0.0001, 0.25)
    res[3] = 2^lerp(x[3], -8, 0)
    res[4] = floor(2^lerp(x[4], 0, 8))
    res
end

function main()
    base_img = TestImages.testimage("lena_gray_512.tif")
    base_img = Noise.mult_gauss(base_img)
    base_img = clamp.(base_img, 0, 1)

    function process_diffusion(params::Vector)
        params = diffusion_transform_domain(params)
        img    = USVG.diffusion(base_img, params[2:3]..., Int64(params[4]))
        img    = Images.imadjustintensity(img)
    end

    USVG.linesearch_main(base_img, process_diffusion)
end

