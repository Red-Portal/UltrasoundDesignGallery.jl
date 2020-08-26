
include("../src/UltrasoundVisualGallery.jl")

image = TestImages.testimage("lighthouse");
c     = GtkReactive.canvas(GtkReactive.UserUnit, size(image)...);
sldr  = GtkReactive.slider(0.0:0.01:1.0)
butn  = GtkReactive.button("done")

win = Gtk.Window("Testing")
vbox = Gtk.Box(:v)
hbox = Gtk.Box(:h)
push!(hbox, sldr)
push!(hbox, butn)

push!(vbox, c);
push!(vbox, hbox);

push!(win, vbox)

#sldrsig = GtkReactive.Signal(sldr)

sldrsig = map(GtkReactive.value, sldr)
butnsig = map(butn) do b
    println("pressed!")
end
redraw = GtkReactive.draw(c, sldrsig) do cnvs, γ
    # Copy the pixel data to the canvas. Because `img` is the value of `imgsig`,
    # this will only copy the region that was selected by the `view` call above.
    img = Images.adjust_histogram(image, Images.GammaCorrection(;gamma=γ))
    copy!(cnvs, img)

    println("what fuck")

    # Here we set the coordinates of the canvas to correspond
    # to the selected region of the image. This ensures that
    # every point on the canvas has coordinates that correspond
    # to the same position in the image.
    #set_coordinates(cnvs, r)
end

Gtk.showall(win);
