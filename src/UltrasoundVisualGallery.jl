
using Statistics
using Distributions
using LinearAlgebra
using Random
using StatsFuns

import PDMats
import OnlineStats
import KernelFunctions
import Optim
import LineSearches
import NLopt

include("latentgp/likelihood.jl")
include("latentgp/laplace.jl")
include("latentgp/map_laplace.jl")
include("latentgp/mcmc.jl")
include("latentgp/prediction.jl")
include("latentgp/utils.jl")
include("bayesopt/bayesian_optimization.jl")

