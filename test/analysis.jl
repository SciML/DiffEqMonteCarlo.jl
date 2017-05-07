using DiffEqMonteCarlo, StochasticDiffEq, DiffEqBase,
      DiffEqProblemLibrary, OrdinaryDiffEq
using Base.Test

prob = prob_sde_2Dlinear
prob2 = MonteCarloProblem(prob)
sim = solve(prob2,SRIW1(),dt=1//2^(3),num_monte=10,adaptive=false)

m = timestep_mean(sim,3)
m2,v = timestep_meanvar(sim,3)
@test m ≈ m2
m3,m4,c = timestep_meancov(sim,3,3)
@test m ≈ m3
@test v ≈ c
m3,m4,c = timestep_meancor(sim,3,3)
@test c ≈ ones(c)
timeseries_steps_mean(sim)
m_series,v_series = timeseries_steps_meanvar(sim)
m4,v4 = m_series[3],v_series[3]
covar_mat = timeseries_steps_meancov(sim)[3,3]
@test m ≈ m4
@test v ≈ v4
@test m ≈ covar_mat[1]
@test m ≈ covar_mat[2]
@test v ≈ covar_mat[3]

@test (get_timestep(sim,1)...) == (get_timepoint(sim,0.0)...)
@test (get_timestep(sim,2)...) == (get_timepoint(sim,1//2^(3))...)
@test (get_timestep(sim,3)...) == (get_timepoint(sim,1//2^(2))...)

sim = solve(prob2,SRIW1(),dt=1//2^(3),num_monte=10)
m = timepoint_mean(sim,0.5)
m2,v = timepoint_meanvar(sim,0.5)
@test m ≈ m2
m3,m4,c = timepoint_meancov(sim,0.5,0.5)
@test m ≈ m3
@test v ≈ c
m3,m4,c = timepoint_meancor(sim,0.5,0.5)
@test c ≈ ones(c)
m_series,v_series = timeseries_point_meanvar(sim,0:1//2^(3):1)
m5,v5 = m_series[5],v_series[5]
@test m ≈ m5
@test v ≈ v5
m6,m7,v6 = timeseries_point_meancov(sim,0:1//2^(3):1,0:1//2^(3):1)[5,5]
@test m ≈ m6
@test m ≈ m7
@test v ≈ v6
