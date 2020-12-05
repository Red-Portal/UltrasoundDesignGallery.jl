
function window_signals!(widgets, signals)
    signals[:slider] = map(GtkReactive.value, widgets[:slider])

    button_signal = GtkReactive.Signal(false)
    Gtk.signal_connect(widgets[:button].widget, :clicked) do widget
        push!(button_signal, true)
    end
    signals[:button_done] = button_signal
end

function create_window(name::String, canvassize::Tuple{Int,Int})
    gridsize = (1, 1)
    winsize  = ImageView.canvas_size(ImageView.screen_size() ./ 2,
                                     map(*, canvassize, gridsize))
    win     = Gtk.Window(name, winsize...)
    ImageView.window_wrefs[win] = nothing
    Gtk.signal_connect(win, :destroy) do w
        exit()
    end

    vbox = Gtk.Box(:v)
    push!(win, vbox)

    menubar, menu_signals = create_menubar(win)
    push!(vbox, menubar)

    frames, canvases = ImageView.frame_canvas(:auto)
    push!(vbox, frames)

    video_range   = 1:100
    player_signal = GtkReactive.Signal(1)
    player_widget = GtkReactive.player(player_signal, video_range)

    grid = Gtk.Grid()
    Gtk.set_gtk_property!(grid, :column_homogeneous, true)
    slider_widget = GtkReactive.slider(0.0:0.01:1.0)
    button_widget = GtkReactive.button("done")
    grid[1:4,1] = player_widget
    grid[1:3,2] = slider_widget
    grid[4  ,2] = button_widget

    push!(vbox, grid)

    widgets = Dict(:window=>win,
                   :vbox=>vbox,
                   :grid=>grid,
                   :frame=>frames,
                   :slider=>slider_widget,
                   :button=>button_widget,
                   :player=>player_widget,
                   :canvas=>canvases,
                   :player_range=>video_range)
    window_signals = Dict{Symbol, GtkReactive.Signal}(:player=>player_signal)
    signals        = merge(window_signals, menu_signals)

    window_signals!(widgets, signals)

    widgets, signals
end
