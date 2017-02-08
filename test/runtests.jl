using DiffEqMonteCarlo, StochasticDiffEq, DiffEqBase, DiffEqProblemLibrary, OrdinaryDiffEq
using Base.Test


prob = prob_sde_2Dlinear
sim = monte_carlo_simulation(prob,SRIW1(),dt=1//2^(3),num_monte=10)
calculate_sim_errors(sim)

prob = prob_sde_additivesystem
sim = monte_carlo_simulation(prob,SRA1(),dt=1//2^(3),num_monte=10)
calculate_sim_errors(sim)

output_func = function (x)
  last(last(x))^2
end
sim = monte_carlo_simulation(prob,SRA1(),output_func=output_func,dt=1//2^(3),num_monte=10)

prob = prob_sde_lorenz
sim = monte_carlo_simulation(prob,SRIW1(),dt=1//2^(3),num_monte=10)

prob = prob_ode_linear
prob_func = function (prob)
  prob.u0 = rand()*prob.u0
  prob
end
sim = monte_carlo_simulation(prob,Tsit5(),prob_func = prob_func,num_monte=100)
