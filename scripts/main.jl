
using DrWatson
@quickactivate "UltrasoundDesignGallery"

include("../src/UltrasoundDesignGallery.jl")
include("improc_task.jl")

function main()
    f = (image, x)->begin
        reduce_speckle(image, transform_domain(x))
    end
    create_ui(MersenneTwister(1), 21, f)
end
