
function menu_export_dialog(data)
    fname = Gtk.save_dialog("Export as...",
                            Gtk.Null(),
                            (Gtk.GtkFileFilter("*.png, *.jpg",
                                               name="All supported formats"),
                             "*.png",
                             "*.jpg"))
    if(length(fname) < 4)
        return
    end

    println(fname)

    if(fname[end-2:end] == "png")
        println(fname)
        FileIO.save(fname, data)
    end
end

function menu_signals!(signals)
    menu_open_impl = map(signals[:file_open], init=false) do sig
        if(sig)
            Gtk.save_dialog(
                "Load an image or video",
                Gtk.GtkNullContainer(),
                ("*.png",
                 "*.jpg",
                 Gtk.GtkFileFilter("*.png, *.jpg",
                                   name="All supported formats")))
        end
        ""
    end
    signals[:file_open] = filter(fname-> fname != "", menu_open_impl)
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
    end
    signals[signal_name] = signal
end

function create_menubar(window)
    signals = Dict{Symbol, GtkReactive.Signal}()

    file     = Gtk.MenuItem("_File")
    filemenu = Gtk.Menu(file)
    add_item!(filemenu, signals, "Open",                    :file_open) 
    add_item!(filemenu, signals, "Export current image",    :file_export_image)
    add_item!(filemenu, signals, "Export best image",       :file_export_best)
    add_item!(filemenu, signals, "Export best parameter",   :file_export_params)
    add_item!(filemenu, signals, "Export Gaussian process", :file_export_gp)

    action     = Gtk.MenuItem("_Action")
    actionmenu = Gtk.Menu(action)
    add_item!(actionmenu, signals, "Start",                :action_start) 
    #add_item!(actionmenu, signals, "Show Best",            :action_best) 
    add_item!(actionmenu, signals, "Restart", :action_restart) 

    menubar = Gtk.MenuBar()
    push!(menubar, file)
    push!(menubar, action)

    menu_signals!(signals)

    menubar, signals
end
