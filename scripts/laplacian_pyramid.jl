
using DrWatson
@quickactivate "UltrasoundDesignGallery"
include("../src/UltrasoundDesignGallery.jl")

import FileIO
import Colors
import CUDA

using TestImages
using ImageView
using ImageCore

f   = ImageCore.scaleminmax(0.0, 1.0)

img = FileIO.load(DrWatson.datadir("image", "forearm.png"))
#img = FileIO.load(DrWatson.datadir("phantom", "cyst_phantom.png"))
img = Float32.(Colors.Gray.(img))

M = size(img, 1)
N = size(img, 2)

M_pad = 2^4 * ceil(Int, M / 2^4)
N_pad = 2^4 * ceil(Int, N / 2^4)

img_pad = zeros(Float32, M_pad, N_pad)
img_pad[1:M, 1:N] = img
img = img_pad

factor = 2
device = CUDA.CuDevice(0)
#device = :cpu

pyr    = Images.gaussian_pyramid(img, 4, factor, Float32(16.0))
pyr    = laplacian_pyramid(pyr, factor)

niter  = 32
σ      = 0.1
α      = 0.8
β      = 1.5
#x      = srad(CUDA.CuArray(pyr[1]), 0.3, 0.5, niter; device=device)
pyr[1]  = diffusion(pyr[1], 0.1, 2.0, niter)
#pyr[1] = homo_contrast(x, σ, α, β; device=device)

#x      = srad(CUDA.CuArray(pyr[2]), 0.5, 0.5, niter; device=device)
pyr[2] = diffusion(pyr[2], 0.1, 1.5, niter)
#pyr[2] = homo_contrast(x, σ, α, β; device=device)

#x      = srad(CUDA.CuArray(pyr[3]), 1.0, 0.5, niter; device=device)
pyr[3] = diffusion(pyr[3], 0.1, 1.5, niter)
#pyr[3] = homo_contrast(x, σ, α, β; device=device)

#x      = srad(CUDA.CuArray(pyr[4]), 1.0, 0.3, niter; device=device)
pyr[4] = diffusion(pyr[4], 0.1, 1.5, niter)
#pyr[4] = homo_contrast(x, σ, α, β; device=device)

img  = synthesize_pyramid(pyr, factor)

img = f.(img)
imshow(img)
FileIO.save("pmad.png", img)
