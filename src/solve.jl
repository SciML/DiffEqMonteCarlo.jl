abstract type BasicMonteCarloAlgorithm <: DiffEqBase.MonteCarloAlgorithm end
struct MonteThreads <: BasicMonteCarloAlgorithm end
struct MonteDistributed <: BasicMonteCarloAlgorithm end
struct MonteSplitThreads <: BasicMonteCarloAlgorithm end
struct MonteSerial <: BasicMonteCarloAlgorithm end

#=
if (kwargs[:parallel_type] == != :none && kwargs[:parallel_type] == != :threads)
  error("Distributed arrays cannot be generated via none or threads")
end
(batch_size != num_monte) && warn("batch_size and reductions are ignored when !collect_result")

elapsed_time = @elapsed u = DArray((num_monte,)) do I
    solve_batch(prob,alg,kwargs[:parallel_type] ==,I[1],pmap_batch_size,kwargs...)
end
return MonteCarloSolution(u,elapsed_time,false)
=#

function DiffEqBase.__solve(prob::DiffEqBase.AbstractMonteCarloProblem,
                            alg::Union{DiffEqBase.DEAlgorithm,Nothing};
                            kwargs...)
    if alg isa DiffEqBase.MonteCarloAlgorithm
      @error "You forgot to pass a DE solver algorithm! Only a MonteCarloAlgorithm has been supplied. Exiting"
    end
    if :parallel_type ∈ keys(kwargs)
      if kwargs[:parallel_type] == :none
        montealg = MonteSerial()
      elseif kwargs[:parallel_type] == :pmap || kwargs[:parallel_type] == :parfor
        montealg = MonteDistributed()
      elseif kwargs[:parallel_type] == :threads
        montealg = MonteThreads()
      elseif kwargs[:parallel_type] == :split_threads
        montealg = MonteSplitThreads()
      else
        @error "parallel_type value not recognized"
      end
    else
      montealg = MonteSerial()
    end
    if :parallel_type ∈ keys(kwargs)
      @warn "parallel_type has been deprecated. Please refer to the docs for the new dispatch-based system."
    end
    DiffEqBase.__solve(prob,alg,montealg;kwargs...)
end

function DiffEqBase.__solve(prob::DiffEqBase.AbstractMonteCarloProblem,
                 alg::Union{DiffEqBase.DEAlgorithm,Nothing},
                 montealg::BasicMonteCarloAlgorithm;
                 num_monte, batch_size = num_monte,
                 pmap_batch_size = batch_size÷100 > 0 ? batch_size÷100 : 1, kwargs...)

  num_batches = num_monte ÷ batch_size
  num_batches * batch_size != num_monte && (num_batches += 1)

  u = deepcopy(prob.u_init)
  converged = false
  elapsed_time = @elapsed for i in 1:num_batches
    if i == num_batches
      I = (batch_size*(i-1)+1):num_monte
    else
      I = (batch_size*(i-1)+1):batch_size*i
    end
    batch_data = solve_batch(prob,alg,montealg,I,pmap_batch_size,kwargs...)
    u,converged = prob.reduction(u,batch_data,I)
    converged && break
  end
  if typeof(u) <: Vector{Any}
    _u = convert(Array{typeof(u[1])},u)
  else
    _u = u
  end
  return MonteCarloSolution(_u,elapsed_time,converged)
end

function batch_func(i,prob,alg,I,kwargs...)
  iter = 1
  new_prob = prob.prob_func(deepcopy(prob.prob),i,iter)
  rerun = true
  x = prob.output_func(solve(new_prob,alg;kwargs...),i)
  if !(typeof(x) <: Tuple)
      @warn("output_func should return (out,rerun). See docs for updated details")
      _x = (x,false)
  else
    _x = x
  end
  rerun = _x[2]
  while rerun
      iter += 1
      new_prob = prob.prob_func(deepcopy(prob.prob),i,iter)
      x = prob.output_func(solve(new_prob,alg;kwargs...),i)
      if !(typeof(x) <: Tuple)
          @warn("output_func should return (out,rerun). See docs for updated details")
          _x = (x,false)
      else
        _x = x
      end
      rerun = _x[2]
  end
  _x[1]
end

function solve_batch(prob,alg,::MonteDistributed,I,pmap_batch_size,kwargs...)
  wp=CachingPool(workers())
  batch_data = let
    pmap(wp,I,batch_size=pmap_batch_size) do i
      batch_func(i,prob,alg,I,kwargs...)
    end
  end
  _batch_data = convert(Array{typeof(batch_data[1])},batch_data)
end

function solve_batch(prob,alg,::MonteSerial,I,pmap_batch_size,kwargs...)
  batch_data = let
    map(I) do i
      batch_func(i,prob,alg,I,kwargs...)
    end
  end
  _batch_data = convert(Array{typeof(batch_data[1])},batch_data)
end

function solve_batch(prob,alg,::MonteThreads,I,pmap_batch_size,kwargs...)
  batch_data = Vector{Any}(undef,length(I))
  let
    Threads.@threads for batch_idx in axes(batch_data, 1)
        i = I[batch_idx]
        iter = 1
        new_prob = prob.prob_func(deepcopy(prob.prob),i,iter)
        x = prob.output_func(solve(new_prob,alg;kwargs...),i)
        if !(typeof(x) <: Tuple)
            warn("output_func should return (out,rerun). See docs for updated details")
            _x = (x,false)
        else
          _x = x
        end
        rerun = _x[2]

        while rerun
            iter += 1
            new_prob = prob.prob_func(deepcopy(prob.prob),i,iter)
            x = prob.output_func(solve(new_prob,alg;kwargs...),i)
            if !(typeof(x) <: Tuple)
                warn("output_func should return (out,rerun). See docs for updated details")
                _x = (x,false)
            else
              _x = x
            end
            rerun = _x[2]
        end
        batch_data[batch_idx] = _x[1]
    end
  end
  _batch_data = convert(Array{typeof(batch_data[1])},batch_data)
end

function solve_batch(prob,alg,::MonteSplitThreads,I,pmap_batch_size,kwargs...)
  wp=CachingPool(workers())
  batch_data = let
    pmap(wp,1:nprocs(),batch_size=pmap_batch_size) do i
      thread_monte(prob,I,alg,i,kwargs...)
    end
  end
  _batch_data = vector_batch_data_to_arr(batch_data)
end

function thread_monte(prob,I,alg,procid,kwargs...)
  start = I[1]+(procid-1)*length(I)
  stop = I[1]+procid*length(I)-1
  portion = start:stop
  batch_data = Vector{Any}(undef,length(portion))
  let
    Threads.@threads for i in portion
      iter = 1
      new_prob = prob.prob_func(deepcopy(prob.prob),i,iter)
      rerun = true
      x = prob.output_func(solve(new_prob,alg;kwargs...),i)
      if !(typeof(x) <: Tuple)
          warn("output_func should return (out,rerun). See docs for updated details")
          _x = (x,false)
      else
        _x = x
      end
      rerun = _x[2]
      while rerun
          iter += 1
          new_prob = prob.prob_func(deepcopy(prob.prob),i,iter)
          x = prob.output_func(solve(new_prob,alg;kwargs...),i)
          if !(typeof(x) <: Tuple)
              warn("output_func should return (out,rerun). See docs for updated details")
              _x = (x,false)
          else
            _x = x
          end
          rerun = _x[2]
      end
      batch_data[i - start + 1] = _x[1]
    end
  end
  batch_data
end

function vector_batch_data_to_arr(batch_data)
  _batch_data = Vector{typeof(batch_data[1][1])}(undef,sum((length(x) for x in batch_data)))
  idx = 0
  @inbounds for a in batch_data
    for x in a
      idx += 1
      _batch_data[idx] = x
    end
  end
  _batch_data
end
