__precompile__()

module DiffEqMonteCarlo

using DiffEqBase

import DiffEqBase: solve

function solve(prob::AbstractMonteCarloProblem,alg::DEAlgorithm;num_monte=10000,kwargs...)
  elapsedTime = @elapsed solution_data = pmap((i)-> begin
    new_prob = prob.prob_func(deepcopy(prob.prob),i)
    prob.output_func(solve(new_prob,alg;kwargs...))
  end,1:num_monte)
  solution_data = convert(Array{typeof(solution_data[1])},solution_data)
  return(MonteCarloSolution(solution_data,elapsedTime))
end

end # module
