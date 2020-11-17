
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
img = Float32.(Colors.Gray.(img))
img = img[1:976, 1:1024]
img = f.(img)
img = convert(Array{Float32}, img)

factor = 2
device = CUDA.CuDevice(0)
#device = :cpu

pyr    = Images.gaussian_pyramid(img, 4, factor, Float32(16.0))
pyr    = laplacian_pyramid(pyr, factor)

pyr[1] = srad(pyr[1], 0.3, 0.2, 16; device=device)
pyr[2] = srad(pyr[2], 1.0, 0.1, 32; device=device)
pyr[3] = srad(pyr[3], 1.0, 0.5, 128; device=device)
pyr[4] = srad(pyr[4], 1.0, 0.3, 512; device=device)

img = synthesize_pyramid(pyr, factor)

img = f.(img)
imshow(img)
