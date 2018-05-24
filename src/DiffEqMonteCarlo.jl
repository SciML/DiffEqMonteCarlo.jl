__precompile__()

module DiffEqMonteCarlo

using DiffEqBase, RecursiveArrayTools, StaticArrays

using DistributedArrays

import DiffEqBase: solve, MonteCarloSummary

include("solve.jl")
include("analysis.jl")

export get_timestep, get_timepoint, apply_timestep, apply_timepoint,
       componentwise_vectors_timestep, componentwise_vectors_timepoint

export componentwise_mean, componentwise_meanvar

export timestep_mean, timestep_median, timestep_quantile, timestep_meanvar,
       timestep_meancov, timestep_meancor, timestep_weighted_meancov

export timeseries_steps_mean, timeseries_steps_median, timeseries_steps_quantile,
       timeseries_steps_meanvar, timeseries_steps_meancov,
       timeseries_steps_meancor, timeseries_steps_weighted_meancov

export timepoint_mean, timepoint_median, timepoint_quantile,
       timepoint_meanvar, timepoint_meancov,
       timepoint_meancor, timepoint_weighted_meancov

export timeseries_point_mean, timeseries_point_median, timeseries_point_quantile,
       timeseries_point_meanvar, timeseries_point_meancov,
       timeseries_point_meancor, timeseries_point_weighted_meancov

end # module
