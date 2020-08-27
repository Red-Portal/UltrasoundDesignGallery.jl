
function transform_output(r::Real, x1::Vector, x2::Vector)
    x3 = r*x1 + (1-r)*x2

    if(abs(r - 1.0) < 0.1)
        x1 = (x1 + x2) / 2
    end
    if(abs(r - 0.0) < 0.1)
        x2 = (x1 + x2) / 2
    end
    x3, x1, x2
end

function linesearch_main(img, process_image)
    warmups     = 4
    prefscale   = 1.0
    marginalize = true
    iters       = 10

    slider_out = nothing 
    x1         = nothing
    x2         = nothing
    function redraw_op!(canvas, slider)
        slider_out = deepcopy(slider)

        param = slider*x1 + (1-slider)*x2
        res   = process_image(param)
        res   = clamp.(res, 0, 1) 
        copy!(canvas, res)
    end

    block = Gtk.Condition()
    dct = create_ui(img, redraw_op!, block)

    function propose_image(x1_in::Vector, x2_in::Vector)
        push!(dct["gui"]["slider"], 0.5)
        x1 = x1_in
        x2 = x2_in
        Gtk.wait(block)
        transform_output(slider_out, x1, x2)
    end
    prefbo_linesearch(propose_image, iters, warmups, prefscale, marginalize)
end
