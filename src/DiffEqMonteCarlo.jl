__precompile__()

module DiffEqMonteCarlo

using DiffEqBase

import DiffEqBase: solve

function solve(prob::AbstractMonteCarloProblem,alg::DEAlgorithm;num_monte=10000,parallel_type=:pmap,kwargs...)
  
  if parallel_type == :pmap
    elapsedTime = @elapsed solution_data = pmap((i)-> begin
      new_prob = prob.prob_func(deepcopy(prob.prob),i)
      prob.output_func(solve(new_prob,alg;kwargs...),i)
    end,1:num_monte)
    solution_data = convert(Array{typeof(solution_data[1])},solution_data)

  elseif parallel_type == :parfor
    elapsedTime = @elapsed solution_data = @sync @parallel (vcat) for i in 1:num_monte
      new_prob = prob.prob_func(deepcopy(prob.prob),i)
      prob.output_func(solve(new_prob,alg;kwargs...),i)
    end

  elseif parallel_type == :threads
    solution_data = Vector{Any}()
    elapsedTime = @elapsed Threads.@threads for i in 1:num_monte
      new_prob = prob.prob_func(deepcopy(prob.prob),i)
      push!(solution_data,prob.output_func(solve(new_prob,alg;kwargs...),i))
    end
    solution_data = convert(Array{typeof(solution_data[1])},solution_data)

  elseif parallel_type == :none
    solution_data = Vector{Any}()
    elapsedTime = @elapsed for i in 1:num_monte
      new_prob = prob.prob_func(deepcopy(prob.prob),i)
      push!(solution_data,prob.output_func(solve(new_prob,alg;kwargs...),i))
    end
    solution_data = convert(Array{typeof(solution_data[1])},solution_data)

  else
    error("Method $parallel_type is not a valid parallelism method.")
  end

  return(MonteCarloSolution(solution_data,elapsedTime))
end

end # module
