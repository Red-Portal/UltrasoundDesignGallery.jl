
function render_duel(img1, img2)
    gui = ImageView.imshow_gui((300, 300), (1, 2))
    canvases = gui["canvas"]
    ImageView.imshow(canvases[1,1], img1)
    ImageView.imshow(canvases[1,2], img2)
    Gtk.showall(gui["window"])

    options = ["right image", "left image"]
    menu    = TerminalMenus.RadioMenu(options, pagesize=2)
    choice  = TerminalMenus.request("Choose image with better quality:", menu)
    if(choice == -1)
        ErrorException("Cancelled choice")
    end
    ImageView.closeall() 
    return choice
end

function pyramid_diffusion(img, x::Vector)
    factor = 2
    pyr    = Images.gaussian_pyramid(img, 4, factor, x[1])
    pyr    = laplacian_pyramid(pyr, factor)
    pyr[1] = diffusion(pyr[1], x[2:3]..., Int64(x[4]))
    pyr[2] = diffusion(pyr[2], x[5:6]..., Int64(x[7]))
    pyr[3] = diffusion(pyr[3], x[8:9]..., Int64(x[10]))
    pyr[4] = diffusion(pyr[4], x[11:12]..., Int64(x[13]))
    img    = synthesize_pyramid(pyr, factor)
end

function lerp(x::Real, low::Real, high::Real)
    return x * (high - low) + low;
end


function pyramid_diffusion_duel(x1::Vector, x2::Vector)
    function transform_domain(x::Vector)
        res    = clamp.(x, 0, 1)
        res[1] = 2^lerp(x[1], 0, 5)

        res[2] = lerp(x[2], 0.0001, 0.25)
        res[3] = 2^lerp(x[3], -8, 0)
        res[4] = floor(2^lerp(x[4], 0, 8))

        res[5] = lerp(x[5], 0.0001, 0.25)
        res[6] = 2^lerp(x[6], -8, 0)
        res[7] = floor(2^lerp(x[7], 0, 8))

        res[8] = lerp(x[2], 0.0001, 0.25)
        res[9] = 2^lerp(x[3], -8, 0)
        res[10] = floor(2^lerp(x[4], 0, 8))

        res[11] = lerp(x[11], 0.0001, 0.25)
        res[12] = 2^lerp(x[12], -8, 0)
        res[13] = floor(2^lerp(x[13], 0, 8))
        res
    end

    img = TestImages.testimage("lena_gray_256.tif")
    img = clamp.(img + rand(eltype(img), size(img))*0.2, 0, 1)

    x1 = transform_domain(x1)
    x2 = transform_domain(x2)
    img1 = pyramid_diffusion(img, x1)
    img2 = pyramid_diffusion(img, x2)
    render_duel(img1, img2)
end

function diffusion_duel(x1::Vector, x2::Vector)
    function transform_domain(x::Vector)
        res    = deepcopy(x)
        res[1] = 2^lerp(x[1], 0, 5)
        res[2] = lerp(x[2], 0.0001, 0.249)
        res[3] = 2^lerp(x[3], -8, 0)
        res[4] = floor(2^lerp(x[4], 0, 8))
        res
    end

    img = TestImages.testimage("lena_gray_256.tif")
    img = clamp.(img + rand(eltype(img), size(img))*0.2, 0, 1)

    x1 = transform_domain(x1)
    x2 = transform_domain(x2)
    img1 = diffusion(img, x1[2:3]..., Int64(x1[4]))
    img2 = diffusion(img, x2[2:3]..., Int64(x2[4]))
    render_duel(img1, img2)
end

#pairwise_prefbo(f, dims, warmup_steps)
