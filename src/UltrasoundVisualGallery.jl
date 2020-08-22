
using Statistics
using Distributions
using LinearAlgebra
using Random
using StatsFuns

import PDMats
import OnlineStats
import KernelFunctions
import Optim
import NLopt

include("latentgp/likelihood.jl")
include("latentgp/laplace.jl")
include("latentgp/mcmc.jl")
include("latentgp/prediction.jl")
include("bayesopt/bayesian_optimization.jl")

