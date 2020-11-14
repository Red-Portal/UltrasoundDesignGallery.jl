
using DrWatson
@quickactivate "UltrasoundDesignGallery"

import TestImages
import Images
import ImageView
import Colors
import Noise
import FileIO
include("../src/UltrasoundDesignGallery.jl")

#import .UltrasoundVisualGallery

#const USVG = UltrasoundVisualGallery

function lerp(x::Real, lo::Real, hi::Real)
    return hi*x + (1-x)*lo
end

@inline @inbounds function diffusion_transform_domain(x::Vector)
    res    = clamp.(x, 0, 1)
    res[1] = 2^lerp(x[1], 0, 5)
    i = 2

    res[i]   = lerp(x[i], 0.0001, 0.25)
    res[i+1] = 2^lerp(x[i+1], -8, 0)
    res[i+2] = floor(2^lerp(x[i+2], 0, 8))
    res[i+3] = 2^lerp(x[i+3], -3, 2)
    res[i+4] = 2^lerp(x[i+4], -3, 2)
    i += 5

    res[i]   = lerp(x[i], 0.0001, 0.25)
    res[i+1] = 2^lerp(x[i+1], -8, 0)
    res[i+2] = floor(2^lerp(x[i+2], 0, 8))
    res[i+3] = 2^lerp(x[i+3], -3, 2)
    res[i+4] = 2^lerp(x[i+4], -3, 2)
    i += 5

    res[i]   = lerp(x[i], 0.0001, 0.25)
    res[i+1] = 2^lerp(x[i+1], -8, 0)
    res[i+2] = floor(2^lerp(x[i+2], 0, 8))
    res[i+3] = 2^lerp(x[i+3], -3, 2)
    res[i+4] = 2^lerp(x[i+4], -3, 2)
    i += 5


    res[i]   = 2^lerp(x[i],   -3, 2)
    res[i+1] = 2^lerp(x[i+1], -3, 2)
    #res[5] = 2^lerp(x[5], -1, 1)
    res
end

function cgl(image, pos_gain, neg_gain)
    pos_image  = max.(image, 0)
    neg_image  = min.(image, 0)
    pos_image *= pos_gain
    neg_image *= neg_gain
    pos_image + neg_image
    #pos_image  = max.(image, pos_limit)
    #neg_image  = min.(image, neg_limit)
end

function main()
    #base_img = TestImages.testimage("lena_gray_512.tif")
    base_img = FileIO.load(DrWatson.datadir("image", "forearm.png"))
    base_img = Float64.(Colors.Gray.(base_img))

    new_size  = ceil.(Int64, size(base_img) ./ 16) .* 16
    pad_size  = (new_size .- size(base_img)) ./ 2
    top_pad   = floor(Int64, pad_size[1])
    bot_pad   = ceil(Int64,  pad_size[1]) 
    right_pad = floor(Int64, pad_size[2])
    left_pad  = ceil(Int64,  pad_size[2])
    base_img  = ImageFiltering.padarray(
        base_img, ImageFiltering.Fill(
            0.0, (top_pad, left_pad), (bot_pad, right_pad)))
    base_img  = parent(base_img)
    #base_img  = Noise.add_gauss(base_img, clip=true)

    function process_diffusion(params::Vector)
        params = diffusion_transform_domain(params)
        pyr = Images.gaussian_pyramid(base_img, 4, 2, params[1])
        pyr = laplacian_pyramid(pyr, 2)
        i = 2

        dif    = diffusion(pyr[1], params[i:i+1]..., Int64(params[i+2]))
        lap    = ImageFiltering.imfilter(dif, ImageFiltering.Laplacian())
        pyr[1] = dif - cgl(lap, params[i+3], params[i+4])
        i += 5

        dif    = diffusion(pyr[2], params[i:i+1]..., Int64(params[i+2]))
        lap    = ImageFiltering.imfilter(dif, ImageFiltering.Laplacian())
        pyr[2] = dif - cgl(lap, params[i+3], params[i+4])
        i += 5

        dif    = diffusion(pyr[3], params[i:i+1]..., Int64(params[i+2]))
        lap    = ImageFiltering.imfilter(dif, ImageFiltering.Laplacian())
        pyr[3] = dif - cgl(lap, params[i+3], params[i+4])
        i += 5

        lap    = ImageFiltering.imfilter(pyr[4], ImageFiltering.Laplacian())
        pyr[4] = pyr[4] - cgl(lap, params[i], params[i+2])

        img = synthesize_pyramid(pyr, 2)
        img
    end

    x   = linesearch_main(base_img, process_diffusion, 20, 20)
    @info "" x
    x   = diffusion_transform_domain(x)
    @info "" x
    img = process_diffusion(x)

    gui = ImageView.imshow_gui(size(img), (1, 2))
    cv  = gui["canvas"]
    ImageView.imshow(cv[1,1], base_img)
    ImageView.imshow(cv[1,2], img)
    Gtk.showall(gui["window"])
end

