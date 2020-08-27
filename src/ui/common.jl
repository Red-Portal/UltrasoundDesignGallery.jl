
const window_wrefs = WeakKeyDict{Gtk.GtkWindowLeaf,Nothing}()

function create_window(name::String, canvassize::Tuple{Int,Int})
    gridsize = (1, 1)
    winsize  = ImageView.canvas_size(ImageView.screen_size(),
                                     map(*, canvassize, gridsize))
    win     = Gtk.Window(name, winsize...)
    window_wrefs[win] = nothing
    Gtk.signal_connect(win, :destroy) do w
        delete!(window_wrefs, win)
    end

    vbox = Gtk.Box(:v)
    push!(win, vbox)

    frames, canvases = ImageView.frame_canvas(:auto)
    push!(vbox, frames)

    grid = Gtk.Grid()
    Gtk.set_gtk_property!(grid, :column_homogeneous, true)
    sldr = GtkReactive.slider(0.0:0.01:1.0)
    butn = GtkReactive.button("done")
    grid[1:3,1] = sldr
    grid[4  ,1] = butn

    push!(vbox, grid)

    guidict = Dict("window"=>win,
                   "vbox"=>vbox,
                   "grid"=>grid,
                   "frame"=>frames,
                   "slider"=>sldr,
                   "button"=>butn,
                   "canvas"=>canvases)
    # Add the player controls
    # if !isempty(slicedata)
    #     players = [player(slicedata.signals[i], axisvalues(slicedata.axs[i])[1]; id=i) for i = 1:length(slicedata)]
    #     guidict["players"] = players
    #     hbox = Box(:h)
    #     for p in players
    #         push!(hbox, frame(p))
    #     end
    #     push!(guidict["vbox"], hbox)
    # end
    guidict
end

function create_ui(img, redraw_op, block_cond::Gtk.Condition)
    axes  = ImageView.default_axes(img)
    ps    = map(abs, ImageView.pixelspacing(img))
    zr, _ = ImageView.roi(img, axes)

    canvas_size = ImageView.default_canvas_size(
        ImageView.fullsize(GtkReactive.value(zr)), ps[2]/ps[1])
    guidict     = create_window("Ultrasound Visual Gallery", canvas_size)

    win    = guidict["window"]
    canvas = guidict["canvas"]
    butn   = guidict["button"]

    sldrsig = map(GtkReactive.value, guidict["slider"])
    redraw  = GtkReactive.draw(redraw_op, canvas, sldrsig)

    Gtk.signal_connect(butn.widget, :clicked) do widget
        Gtk.notify(block_cond)
    end

    sigs = Dict("slider"=>sldrsig, "redraw"=>redraw)
    dct  = Dict("gui"=>guidict, "signals"=>sigs)
    GtkReactive.gc_preserve(win, dct)

    # GtkReactive.draw(canvas) do cnvs
    #     copy!(canvas, img)
    #     Gtk.set_coordinates(canvas, ImageView.axes(img))
    # end

    Gtk.showall(win)
    return dct
end

function bo_status_window!()
    win    = Gtk.Window("Gallery Status")
    vbox   = Gtk.Box(:v)
    txtbox = GtkReactive.textarea("Starting Bayesian Optimization")
    push!(win, vbox)
    push!(vbox, txtbox)
    Gtk.showall(win)
    dict   = Dict("window"=>win, "vbox"=>vbox)
    GtkReactive.gc_preserve(win, dict)
    dict
end

function bo_status_update!(dict::Dict, message::String)
    txtbox = GtkReactive.textarea(message)
    push!(dict["vbox"], txtbox)
end

function bo_status_destroy!(dict::Dict)
    GtkReactive.destroy(dict["window"])
end
