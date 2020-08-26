
function render_linesearch(x1::Vector,x2::Vector)
    function transform_domain(x::Vector)
        res    = deepcopy(x)
        res[1] = 2^lerp(x[1], 0, 5)
        res[2] = lerp(x[2], 0.0001, 0.249)
        res[3] = 2^lerp(x[3], -8, 0)
        res[4] = ceil(2^lerp(x[4], 0, 6))
        res
    end

    img = TestImages.testimage("lena_gray_256.tif")
    img = clamp.(img + rand(eltype(img), size(img))*0.2, 0, 1)

    c     = GtkReactive.canvas(GtkReactive.UserUnit, size(img)...);
    sldr  = GtkReactive.slider(0.0:0.01:1.0)
    butn  = GtkReactive.button("done")

    win  = Gtk.Window("Testing")
    vbox = Gtk.Box(:v)
    hbox = Gtk.Box(:h)
    push!(hbox, sldr)
    push!(hbox, butn)

    push!(vbox, c);
    push!(vbox, hbox);
    push!(win, vbox)

    point   = nothing
    sldrsig = map(GtkReactive.value, sldr)
    redraw  = GtkReactive.draw(c, sldrsig) do cnvs, r
        #point = lerp(r, x1, x2)
        point = r*x1 + (1-r)*x2
        param = transform_domain(point)

        res   = diffusion(img, param[2:3]..., Int64(param[4]))   
        copy!(cnvs, res)
    end

    block = Gtk.Condition()
    # butnsig = map(butn) do b
    #     println("pressed!")
    #     Gtk.notify(block)
    # end

    Gtk.signal_connect(win, :destroy) do widget
        Gtk.notify(block)
    end
    Gtk.showall(win);
    Gtk.wait(block)
    return point
end

function objective_linesearch(x1::Vector, x2::Vector)
    x3 = render_linesearch(x1, x2)
    x3, x1, x2
end

function slide_prefbo()
    prng     = MersenneTwister(1)
    scale    = 1.0
    warmup_steps = 4
    dims     = 13

    ℓ² = 1.0
    σ² = 1.0  
    ϵ² = 1.0

    data_x = zeros(dims, 0)
    data_c = zeros(Int64, warmup_steps, 3)
    for i = 1:3:warmup_steps*3
        x1 = rand(prng, dims)
        x2 = rand(prng, dims)
        c1, c2, c3 = objective_linesearch(x1, x2)
        i_nduel = ceil(Int64, i / 3)
        data_c[i_nduel, 1] = i
        data_c[i_nduel, 2] = i+1
        data_c[i_nduel, 3] = i+2
        data_x = hcat(data_x, reshape(c1, (:,1)))
        data_x = hcat(data_x, reshape(c2, (:,1)))
        data_x = hcat(data_x, reshape(c3, (:,1)))
    end
    
    priors = Product([Normal(0, 1),
                      Normal(0, 1),
                      Normal(-2, 2)])

    initial_latent = zeros(size(data_x, 2))
    θ_init  = [ℓ², σ², ϵ²]
    samples = 64
    warmup  = 64

    θs, fs, as, Ks = pm_ess(
        prng, samples, warmup, θ_init, initial_latent,
        priors, scale, data_x, data_c)
    ks    = precompute_lgp(θs)
    for i = 1:10
        x_opt, y_opt = optimize_mean(dims, 10000, data_x, Ks, as, ks)
        x_query, _   = optimize_acquisition(dims, 10000, y_opt, data_x, Ks, as, ks)

        x1, x2, x3 = objective_linesearch(x_opt, x_query)

        data_x = hcat(data_x, reshape(x1, (:,1)))
        data_x = hcat(data_x, reshape(x2, (:,1)))
        data_x = hcat(data_x, reshape(x3, (:,1)))

        choices = size(data_c, 1) .+ [1,2,3]
        data_c  = vcat(data_c, reshape(choices, (:,3)))

        @info("BO iteration stat",
              iteration = i,
              x_query   = x_query,
              x_optimum = x_opt,
              x_selected = x1)

        θ_init         = mean(θs, dims=2)[:,1]
        initial_latent = zeros(size(data_x, 2))
        θs, fs, as, Ks = pm_ess(
            prng, samples, warmup, θ_init, initial_latent,
            priors, scale, data_x, data_c)
        ks    = precompute_lgp(θs)
    end
    x_opt, y_opt = optimize_mean(dims, 30000, data_x, Ks, as, ks)
    x_opt
end
#pairwise_prefbo(f, dims, warmup_steps)
