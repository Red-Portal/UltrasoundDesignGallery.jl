
using TestImages
using ImageView
#using Random
#using Distributions

include("../src/UltrasoundVisualGallery.jl")

img  = testimage("lena_gray_256.tif")
img  = clamp.(img + rand(eltype(img), size(img))*0.2, 0, 1)

imshow(img)
factor = 2
pyr    = Images.gaussian_pyramid(img, 4, factor, 16.0)
pyr    = laplacian_pyramid(pyr, factor)

pyr[1] = diffusion(pyr[1], 0.1,  0.03, 32)
pyr[2] = diffusion(pyr[2], 0.1,  0.03, 32)
pyr[3] = diffusion(pyr[3], 0.01, 0.01, 32)
pyr[4] = diffusion(pyr[4], 0.01, 0.01, 32)

img = synthesize_pyramid(pyr, factor)
imshow(img)
