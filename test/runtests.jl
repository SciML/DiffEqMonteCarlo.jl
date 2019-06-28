using DiffEqMonteCarlo
using Test

@time @testset "Ensemble Simulations" begin include("monte.jl") end
@time @testset "Ensemble Analysis" begin include("analysis.jl") end
