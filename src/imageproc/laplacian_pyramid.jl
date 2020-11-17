
#=
```
pyramid = laplacian_pyramid(img, n_scales, downsample, sigma)
```
Returns a laplacian pyramid
=##

function laplacian_pyramid(gaussian_pyramid::AbstractArray, upsample::Real)
    laplacian_pyramid = copy(gaussian_pyramid)
    for i = 1:length(laplacian_pyramid)-1
        x      = laplacian_pyramid[i]  
        x_next = laplacian_pyramid[i+1]  
        x_next = ImageTransformations.imresize(x_next, ratio=upsample)
        Δimg   = x - x_next
        laplacian_pyramid[i] = Δimg
    end
    laplacian_pyramid
end

function synthesize_pyramid(pyramid::AbstractArray, upsample::Real)
    result = pyramid[end]
    for i = (length(pyramid)-1):-1:1
        x       = pyramid[i]
        result  = ImageTransformations.imresize(result, ratio=upsample)
        result += x
    end
    result
end
