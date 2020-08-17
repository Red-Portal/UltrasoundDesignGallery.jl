
using Statistics
using Distributions
using LinearAlgebra
using Random
using StatsFuns

import KernelFunctions
import Optim

include("lgp/likelihood.jl")
include("lgp/laplace_approximation.jl")
