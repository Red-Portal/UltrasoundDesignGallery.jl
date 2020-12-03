
#module UltrasoundDesignGallery

using Statistics
using Distributions
using LinearAlgebra
using Random
using StatsFuns

import Base.Threads
import CMAEvolutionStrategy
import CUDA
import Colors
import DSP
import FileIO
import Gtk
import Gtk.GConstants
import Gtk.ShortNames
import Reactive
import GtkReactive
import ImageCore
import ImageFiltering
import ImageTransformations
import ImageView
import Images
import KernelFunctions
import LineSearches
import NLopt
import OffsetArrays
import OnlineStats
import Optim
import PDMats
import ProgressMeter
import REPL.TerminalMenus

include("latentgp/likelihood.jl")
include("latentgp/laplace.jl")
include("latentgp/map_laplace.jl")
include("latentgp/mcmc.jl")
include("latentgp/prediction.jl")
include("latentgp/utils.jl")
include("bayesopt/bayesian_optimization.jl")
include("bayesopt/optimization.jl")
include("bayesopt/acquisition.jl")
include("imageproc/diffusion.jl")
include("imageproc/filter.jl")
include("imageproc/laplacian_pyramid.jl")
include("ui/pairwise.jl")
include("ui/linesearch.jl")
include("ui/menu.jl")
include("ui/ui.jl")
include("ui/window.jl")
#include("ui/common.jl")
include("simulation/ultrasound.jl")

#end
