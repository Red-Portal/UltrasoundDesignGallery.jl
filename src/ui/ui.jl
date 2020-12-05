
function render_image_op(x1, x2, image, transform_op, widgets)
    slider_val  = GtkReactive.value(widgets[:slider])
    θ           = slider_val*x1 + (1 - slider_val)*x2
    idx         = GtkReactive.value(widgets[:player])
    img_trans   = transform_op(image[:,:,idx], θ)
end

function render_video_op(x1, x2, image, transform_op, widgets)
    slider_val  = GtkReactive.value(widgets[:slider])
    θ           = slider_val*x1 + (1 - slider_val)*x2
    video_trans = [transform_op(image[:,:,idx], θ) for idx = 1:size(image,3)]
end

function render_loop!(x1, x2, image, widgets, signals, transform_op)
    block        = Gtk.Condition()
    render_image = GtkReactive.Signal(image[:,:,1])
    sig_draw = GtkReactive.draw(widgets[:canvas], render_image) do canvas, ren
        copy!(canvas, ren)
        Gtk.notify(block)
        nothing
    end

    signals[:file_export_best_der] = map(signals[:file_export_best]) do sig
        if(sig)
            idx       = GtkReactive.value(widgets[:player])
            img_trans = transform_op(image[:,:,idx], x1)
            menu_export_dialog(img_trans)
            nothing
        end
    end

    signals[:file_export_der] = map(signals[:file_export_image]) do sig
        if(sig)
            data = render_image_op(x1, x2, image, transform_op, widgets)
            menu_export_dialog(data)
            nothing
        end
    end

    signals[:button_done].value = false
    while(!GtkReactive.value(signals[:button_done]))
        sleep(0.005)
        img_trans = render_image_op(x1, x2, image, transform_op, widgets)
        push!(render_image, img_trans)
        Gtk.wait(block)
    end
end

function opt_loop!(prng, image, dims, widgets, signals, transform_op, settings)
    state              = Dict{Symbol, Any}()
    state[:best_param] = Array{Float64}[]
    state[:data_x]     = zeros(Float64, dims, 0)
    state[:data_c]     = zeros(Int64, 0, 3)

    signals[:action_restart].value = false

    x1 = rand(prng, dims)
    x2 = rand(prng, dims)

    # signals[:file_export_der] = map(signals[:file_export_best]) do sig
    #     if(sig)
    #         fname = Gtk.open_dialog(
    #             "Load an image or video",
    #             Gtk.GtkNullContainer(),
    #             ("*.png",
    #              "*.jpg",
    #              Gtk.GtkFileFilter("*.png, *.jpg",
    #                                name="All supported formats")))
    #         if(fname[end-3:end] == "png")
    #             idx = GtkReactive.value(widgets[:player])
    #             FileIO.save(image[:,:,idx], fname)
    #         end
    #     end
    # end

    signals[:action_start_der] = map(signals[:action_start]) do sig
        @info("pressed button start", sig)
        if(sig)
            @info("pressed button")
        end
    end

    signals[:file_export_gp_der] = map(signals[:file_export_gp]) do sig
        @info("pressed button export gp", sig)
        if(sig)
            JLD.save("gp_state.jld", "gp", state)
        end
    end

    for i = 1:settings[:n_initial]
        x1[:] = rand(prng, dims)
        x2[:] = rand(prng, dims)

        render_loop!(x1, x2, image, widgets, signals, transform_op)
        r = GtkReactive.value(widgets[:slider])
        append_choice!(r, x1, x2, state)
        push!(widgets[:slider], 0.5)
    end

    @info("training gp")
    train_gp!(prng,
              state,
              settings[:n_mcmc_samples],
              settings[:n_mcmc_burnin],
              settings[:n_mcmc_thin],
              settings[:preference_scale],
              settings[:marginalize])

    while(!GtkReactive.value(signals[:action_restart]))
        @info("finding query")
        x1, x1_idx, x2 = prefbo_next_query(prng,
                                           state,
                                           settings[:search_budget];
                                           verbose=settings[:verbose])
        push!(state[:best_param], x1)
        @info("waiting")
        push!(widgets[:slider], 0.5)


        signals[:file_export_der] = map(signals[:file_export_best]) do sig
            if(sig)
                fname = Gtk.open_dialog(
                    "Load an image or video",
                    Gtk.GtkNullContainer(),
                    ("*.png",
                     "*.jpg",
                     Gtk.GtkFileFilter("*.png, *.jpg",
                                       name="All supported formats")))
                if(fname[end-3:end] == "png")
                    idx = GtkReactive.value(widgets[:player])
                    FileIO.save(image[:,:,idx], fname)
                end
            end
        end

        render_loop!(x1, x2, image, widgets, signals, transform_op)
        r = GtkReactive.value(widgets[:slider])
        append_choice!(r, x1, x1_idx, x2, state)

        @info("train GP")
        train_gp!(prng,
                  state,
                  settings[:n_mcmc_samples],
                  settings[:n_mcmc_burnin],
                  settings[:n_mcmc_thin],
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

    sig_menu = map(signals[:file_open]; init=false) do fname
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
    settings[:n_initial]   = 4
    settings[:marginalize]      = true
    settings[:search_budget]    = 512
    settings[:n_mcmc_samples]   = 2000
    settings[:n_mcmc_burnin]    = 1000
    settings[:n_mcmc_thin]      = 20
    settings[:preference_scale] = 1.0
    settings[:verbose]          = true

    # foreach(signals[:file_open]) do fname
    # end

    while(true)
        opt_loop!(prng, image, dims, widgets, signals, transform_op, settings)
    end
end
