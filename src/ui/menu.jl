
function menu_signals!(signals)
    menu_open_impl = map(signals[:menu_open], init=false) do sig
        if(sig)
            Gtk.open_dialog(
                "Load an image or video",
                Gtk.GtkNullContainer(),
                ("*.png",
                 "*.jpg",
                 Gtk.GtkFileFilter("*.png, *.jpg",
                                   name="All supported formats")))
        end
        ""
    end
    signals[:menu_open] = filter(fname-> fname != "", menu_open_impl)
end

function add_item!(menu,
                   signals::Dict,
                   menu_name::String,
                   signal_name::Symbol)
    signal = GtkReactive.Signal(false)
    item   = Gtk.MenuItem(menu_name)
    push!(menu, item)
    Gtk.signal_connect(item, :activate) do widget
        push!(signal, true)
        push!(signal, false)
        nothing
    end
    signals[signal_name] = signal
end

function create_menubar(window)
    signals = Dict{Symbol, GtkReactive.Signal}()

    file     = Gtk.MenuItem("_File")
    filemenu = Gtk.Menu(file)
    add_item!(filemenu, signals, "Open",              :menu_open) 
    add_item!(filemenu, signals, "Export Image",      :menu_export_image)
    add_item!(filemenu, signals, "Export Parameters", :menu_export_params)

    action     = Gtk.MenuItem("_Action")
    actionmenu = Gtk.Menu(action)
    add_item!(actionmenu, signals, "Start",                :action_start) 
    add_item!(actionmenu, signals, "Show Best",            :action_best) 
    add_item!(actionmenu, signals, "Restart Optimization", :restart_opt) 

    menubar = Gtk.MenuBar()
    push!(menubar, file)
    push!(menubar, action)

    menu_signals!(signals)

    menubar, signals
end
