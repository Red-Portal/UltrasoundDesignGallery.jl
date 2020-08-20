
using Statistics
using Distributions
using LinearAlgebra
using Random
using StatsFuns

import PDMats
import OnlineStats
import KernelFunctions
import Optim

include("lgp/likelihood.jl")
include("lgp/laplace_approximation.jl")
include("lgp/mcmc.jl")
include("lgp/prediction.jl")
