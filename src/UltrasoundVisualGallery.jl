
using Statistics
using Distributions
using LinearAlgebra
using Random
using StatsFuns

import PDMats
import OnlineStats
import KernelFunctions
import Optim

include("latentgp/likelihood.jl")
include("latentgp/laplace_approximation.jl")
include("latentgp/mcmc.jl")
include("latentgp/prediction.jl")
include("bayesopt/bayesian_optimization.jl")

