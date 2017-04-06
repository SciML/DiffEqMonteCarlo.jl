__precompile__()

module DiffEqMonteCarlo

using DiffEqBase, RecipesBase

type MonteCarloTestSimulation{S,T} <: AbstractMonteCarloSimulation
  solution_data::S
  errors::Dict{Symbol,Vector{T}}
  error_means::Dict{Symbol,T}
  error_medians::Dict{Symbol,T}
  elapsedTime::Float64
end

type MonteCarloSimulation{T} <: AbstractMonteCarloSimulation
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
  solution_data = sim.solution_data
  errors = Dict{Symbol,Vector{eltype(solution_data[1].u[1])}}() #Should add type information
  error_means  = Dict{Symbol,eltype(solution_data[1].u[1])}()
  error_medians= Dict{Symbol,eltype(solution_data[1].u[1])}()
  for k in keys(solution_data[1].errors)
    errors[k] = [sol.errors[k] for sol in solution_data]
    error_means[k] = mean(errors[k])
    error_medians[k]=median(errors[k])
  end
  return MonteCarloTestSimulation(solution_data,errors,error_means,error_medians,sim.elapsedTime)
end

function monte_carlo_simulation(prob::DEProblem,alg;output_func = identity,prob_func= (prob,i)->prob,num_monte=10000,kwargs...)
  elapsedTime = @elapsed solution_data = pmap((i)-> begin
    new_prob = prob_func(deepcopy(prob),i)
    output_func(solve(new_prob,alg;kwargs...))
  end,1:num_monte)
  solution_data = convert(Array{typeof(solution_data[1])},solution_data)
  return(MonteCarloSimulation(solution_data,elapsedTime))
end

Base.length(sim::AbstractMonteCarloSimulation) = length(sim.solution_data)
Base.endof( sim::AbstractMonteCarloSimulation) = length(sim)
Base.getindex(sim::AbstractMonteCarloSimulation,i::Int) = sim.solution_data[i]
Base.getindex(sim::AbstractMonteCarloSimulation,i::Int,I::Int...) = sim.solution_data[i][I...]
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

export MonteCarloSimulation, MonteCarloTestSimulation

end # module
