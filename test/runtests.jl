using DiffEqMonteCarlo, StochasticDiffEq, DiffEqBase, DiffEqProblemLibrary, OrdinaryDiffEq
using Base.Test


prob = prob_sde_2Dlinear
sim = monte_carlo_simulation(prob,SRIW1(),dt=1//2^(3),num_monte=10)
calculate_sim_errors(sim)

prob = prob_sde_additivesystem
sim = monte_carlo_simulation(prob,SRA1(),dt=1//2^(3),num_monte=10)
calculate_sim_errors(sim)

prob = prob_sde_lorenz
sim = monte_carlo_simulation(prob,SRIW1(),dt=1//2^(3),num_monte=10)

prob = prob_ode_2Dlinear
function prob_func(prob)
  prob.u0 = randn()*prob.u0
end
sim = monte_carlo_simulation(prob,Tsit5(),u0_func,num_monte=10)
errs = calculate_sim_errors(sim)


using Plots; plot(sim)

plotly()
reshape(sim.solutions,1,length(sim))
