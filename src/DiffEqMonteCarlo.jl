module DiffEqMonteCarlo

using DiffEqBase, RecipesBase

type MonteCarloTestSimulation{S,T} <: AbstractMonteCarloSimulation
  solutions::S
  errors::Dict{Symbol,Vector{T}}
  error_means::Dict{Symbol,T}
  error_medians::Dict{Symbol,T}
  elapsedTime::Float64
end

type MonteCarloSimulation{T} <: AbstractMonteCarloSimulation
  solutions::Vector{T}
  elapsedTime::Float64
end

type ReductionMonteCarloSimulation{T} <: AbstractMonteCarloSimulation
  solution_data::Vector{T}
  elapsedTime::Float64
end

"""
`monte_carlo_simulation(prob::DEProblem,alg)`

Performs a parallel Monte-Carlo simulation to solve the DEProblem numMonte times.
Returns a vector of solution objects.

### Keyword Arguments
* `numMonte` - Number of Monte-Carlo simulations to run. Default is 10000
* `save_timeseries` - Denotes whether save_timeseries should be turned on in each run. Default is false.
* `kwargs...` - These are common solver arguments which are then passed to the solve method
"""
function calculate_sim_errors(sim::MonteCarloSimulation)
  solutions = sim.solutions
  errors = Dict{Symbol,Vector{eltype(solutions[1].u[1])}}() #Should add type information
  error_means  = Dict{Symbol,eltype(solutions[1].u[1])}()
  error_medians= Dict{Symbol,eltype(solutions[1].u[1])}()
  for k in keys(solutions[1].errors)
    errors[k] = [sol.errors[k] for sol in solutions]
    error_means[k] = mean(errors[k])
    error_medians[k]=median(errors[k])
  end
  return MonteCarloTestSimulation(solutions,errors,error_means,error_medians,sim.elapsedTime)
end



function monte_carlo_simulation(prob::DEProblem,alg,output_func;prob_func=identity,num_monte=10000,kwargs...)
  elapsedTime = @elapsed solutions = pmap((i)-> begin
    new_prob = prob_func(deepcopy(prob))
    sol = solve(new_prob,alg;kwargs...)
    output_func(last(sol))
  end,1:num_monte)
  return(ReductionMonteCarloSimulation(solutions,elapsedTime))
end

function monte_carlo_simulation(prob::DEProblem,alg;prob_func=identity,num_monte=10000,kwargs...)
  elapsedTime = @elapsed solutions = pmap((i)-> begin
    new_prob = prob_func(deepcopy(prob))
    solve(new_prob,alg;kwargs...)
  end,1:num_monte)
  solutions = convert(Array{typeof(solutions[1])},solutions)
  return(MonteCarloSimulation(solutions,elapsedTime))
end


Base.length(sim::AbstractMonteCarloSimulation) = length(sim.solutions)
Base.endof( sim::AbstractMonteCarloSimulation) = length(sim)
Base.getindex(sim::AbstractMonteCarloSimulation,i::Int) = sim.solutions[i]
Base.getindex(sim::AbstractMonteCarloSimulation,i::Int,I::Int...) = sim.solutions[i][I...]
Base.size(sim::AbstractMonteCarloSimulation) = (length(sim),)
Base.start(sim::AbstractMonteCarloSimulation) = 1
function Base.next(sim::AbstractMonteCarloSimulation,state)
  state += 1
  (sim[state],state)
end
Base.done(sim::AbstractMonteCarloSimulation,state) = state >= length(sim)

@recipe function f(sim::AbstractMonteCarloSimulation)

  for sol in sim
    @series begin
      legend := false
      sol
    end
  end
end

export monte_carlo_simulation, calculate_sim_errors

export MonteCarloSimulation, MonteCarloTestSimulation,ReductionMonteCarloSimulation

end # module
