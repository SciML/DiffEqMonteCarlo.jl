using DiffEqMonteCarlo
using Test

@time @testset "Monte Carlo Simulations" begin include("monte.jl") end
@time @testset "Monte Carlo Analysis" begin include("analysis.jl") end
