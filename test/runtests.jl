using DiffEqMonteCarlo
using Base.Test

@time @testset "Monte Carlo Simulations" begin include("monte.jl") end
