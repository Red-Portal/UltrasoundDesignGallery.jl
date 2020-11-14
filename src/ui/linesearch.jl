
function linesearch_main(img, process_image, dims, iters)
    prefscale   = 1.0
    marginalize = false

    slider_out = nothing 
    x1         = nothing
    x2         = nothing
    function redraw_op!(canvas, slider)
        slider_out = deepcopy(slider)
        param = slider*x1 + (1-slider)*x2
        if(isempty(Reactive._messages.data))
            res   = process_image(param)
            res   = clamp.(res, 0, 1) 
            copy!(canvas, res)
        end
    end

    block = Gtk.Condition()
    dct = create_ui(img, redraw_op!, block)

    function propose_image(x1_in::Vector, x2_in::Vector)
        push!(dct["gui"]["slider"], 0.5)
        x1 = x1_in
        x2 = x2_in
        Gtk.wait(block)
        slider_out
    end
    prefbo_linesearch(propose_image, iters, dims, prefscale, marginalize;
                      search_budget=100000)
end
