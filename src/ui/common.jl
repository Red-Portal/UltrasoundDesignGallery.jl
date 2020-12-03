
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
