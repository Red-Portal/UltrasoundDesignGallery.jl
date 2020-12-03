
function render_loop!(x1, x2, image, widgets, signals, transform_op)
    # Gtk.signal_connect(win, :destroy) do w
    #     return nothing
    # end

    block        = Gtk.Condition()
    render_image = GtkReactive.Signal(image[:,:,1])
    sig_draw = GtkReactive.draw(widgets[:canvas], render_image) do canvas, ren
        copy!(canvas, ren)
        Gtk.notify(block)
        nothing
    end


    render_sig = GtkReactive.Signal(true)
    Gtk.signal_connect(widgets[:button].widget, :clicked) do widget
        push!(render_sig, false)
    end

    # sig_butn = map(signals[:button_done]; init=false) do sig
    #     @info("press button! ", sig)
    #     if(sig)
    #         render_flag[:] = false
    #         nothing
    #     end
    # end

    while(GtkReactive.value(render_sig))
        sleep(0.01)
        slider_val  = GtkReactive.value(widgets[:slider])
        θ           = slider_val*x1 + (1 - slider_val)*x2
        idx         = GtkReactive.value(widgets[:player])
        img_trans   = transform_op(image[:,:,idx], θ)
        push!(render_image, img_trans)
        Gtk.wait(block)
    end
end

function opt_loop!(prng, image, dims, widgets, signals, transform_op)
    settings = Dict{Symbol, Any}()
    settings[:n_warmup]         = 4
    settings[:n_mcmc_samples]   = 200
    settings[:n_mcmc_burnin]    = 100
    settings[:marginalize]      = true
    settings[:preference_scale] = 1.0
    settings[:verbose]          = true

    state              = Dict{Symbol, Any}()
    state[:best_param] = zeros(0)
    state[:data_x]     = zeros(Float64, dims, 0)
    state[:data_c]     = zeros(Int64, 0, 3)

    x1 = rand(prng, dims)
    x2 = rand(prng, dims)

    sig_exp = map(signals[:menu_export_image]; init=false) do sig
        #
    end

    restart_opt = false
    sig_restart = map(signals[:restart_opt]; init=false) do sig
        @info("restart op")
        if(sig)
            restart_opt = true
            nothing
        end
    end

    # block = Gtk.Condition()
    # sig_start = map(signals[:action_start]; init=false) do sig
    #     @info("pressed button start")
    #     if(sig)
    #         Gtk.notify(block)
    #         nothing
    #     end
    # end
    # Gtk.wait(block)

    for i = 1:settings[:n_warmup]
        @info("safe 1")
        x1[:] = rand(prng, dims)
        x2[:] = rand(prng, dims)

        render_loop!(x1, x2, image, widgets, signals, transform_op)
        r = GtkReactive.value(widgets[:slider])
        @info("safe 2")
        append_choice!(r, x1, x2, state)
    end

    @info("training gp")
    train_gp!(prng,
              state,
              settings[:n_mcmc_samples],
              settings[:n_mcmc_burnin],
              settings[:preference_scale],
              settings[:marginalize])

    while(!restart_opt)
        @info("finding query")
        x1, x2 = prefbo_next_query(prng,
                                   state,
                                   settings[:search_budget];
                                   verbose=settings[:verbose])
        @info("waiting")
        render_loop!(x1, x2, image, widgets, signals, transform_op)
        r = GtkReactive.value(widgets[:slider])
        append_choice!(r, x1, x2, state)

        @info("train GP")
        train_gp!(prng,
                  state,
                  settings[:n_mcmc_samples],
                  settings[:n_mcmc_burnin],
                  settings[:preference_scale],
                  settings[:marginalize])
        @info("done")
    end
end

function create_ui(prng, dims, transform_op)
    widgets, signals = create_window("Ultrasound Visual Gallery", (512, 360))
    dct  = Dict(:gui=>widgets, :signals=>signals)
    win  = widgets[:window]
    GtkReactive.gc_preserve(win, dct)
    Gtk.showall(win)

    #image = zeros(300, 200)
    #image = rand(200,300,100)

    f   = ImageCore.scaleminmax(0.0, 1.0)
    image = FileIO.load(DrWatson.datadir("image", "forearm.png"))
    image = Float32.(Colors.Gray.(image))
    image = image[1:976, 1:1024]
    image = f.(image)
    image = convert(Array{Float32}, image)
    image = reshape(image, (size(image,1), size(image,2), 1))

    sig_menu = map(signals[:menu_open]; init=false) do fname
        image = FileIO.load(fname)

        video_frames = size(image, 3)
        Gtk.empty!(widgets[:grid])

        widgets[:player]      = GtkReactive.player(signals[:player],
                                                    1:video_frames)
        widgets[:grid][1:4,1] = widgets[:player]
        widgets[:grid][1:3,2] = widgets[:slider]
        widgets[:grid][4,  2] = widgets[:button]
        nothing
    end

    settings = Dict{Symbol, Any}()
    settings[:initial_values] = 4
    settings[:marginalize]    = true
    settings[:search_budget]  = 1024

    # foreach(signals[:menu_open]) do fname
    # end

    while(true)
        opt_loop!(prng, image, dims, widgets, signals, transform_op)
    end
end
