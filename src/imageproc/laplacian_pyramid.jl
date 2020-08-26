
#=
```
pyramid = laplacian_pyramid(img, n_scales, downsample, sigma)
```
Returns a laplacian pyramid
=##

function laplacian_pyramid(gaussian_pyramid::Array, upsample::Real)
    laplacian_pyramid = typeof(gaussian_pyramid[1])[]
    for i = 1:length(gaussian_pyramid)-1
        x      = gaussian_pyramid[i]  
        x_next = gaussian_pyramid[i+1]  
        x_next = ImageTransformations.imresize(x_next, ratio=upsample)
        Δimg   = x - x_next
        push!(laplacian_pyramid, Δimg)
    end
    laplacian_pyramid
end

function synthesize_pyramid(pyramid::Array, upsample::Real)
    result = pyramid[end]
    for i = (length(pyramid)-1):-1:1
        x      = pyramid[i]
        result = ImageTransformations.imresize(result, ratio=upsample)
        result += x
    end
    result
end
