function solve(prob::DiffEqBase.AbstractMonteCarloProblem,
               alg::Union{DiffEqBase.DEAlgorithm,Void}=nothing,
               collect_result::Type{Val{T}} = Val{true};
               num_monte=10000,batch_size = num_monte,
               pmap_batch_size= batch_size÷100 > 0 ? batch_size÷100 : 1,
               parallel_type= T ? :pmap : :none,kwargs...) where {T}
  #=
  if !T
    if (parallel_type != :none && parallel_type != :threads)
      error("Distributed arrays cannot be generated via none or threads")
    end
    (batch_size != num_monte) && warn("batch_size and reductions are ignored when !collect_result")

    elapsed_time = @elapsed u = DArray((num_monte,)) do I
        solve_batch(prob,alg,parallel_type,I[1],pmap_batch_size,kwargs...)
    end
    return MonteCarloSolution(u,elapsed_time,false)

  else
  =#
  !T && warn("Distributed collection is currently disabled in v0.7/v1.0")
  num_batches = num_monte ÷ batch_size
  u = deepcopy(prob.u_init)
  converged = false
  elapsed_time = @elapsed for i in 1:num_batches
    if i == num_batches
      I = (batch_size*(i-1)+1):num_monte
    else
      I = (batch_size*(i-1)+1):batch_size*i
    end
    batch_data = solve_batch(prob,alg,parallel_type,I,pmap_batch_size,kwargs...)
    u,converged = prob.reduction(u,batch_data,I)
    converged && break
  end
  if T && typeof(u) <: Vector{Any}
    _u = convert(Array{typeof(u[1])},u)
  else
    _u = u
  end
  return MonteCarloSolution(_u,elapsed_time,converged)
  #end
end

function solve_batch(prob,alg,parallel_type,I,pmap_batch_size,kwargs...)
  if parallel_type == :pmap
      wp=CachingPool(workers())
      batch_data = pmap(wp,(i)-> begin
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
      _x[1]
    end,I,batch_size=pmap_batch_size)
    _batch_data = convert(Array{typeof(batch_data[1])},batch_data)
  elseif parallel_type == :none

    batch_data = map((i)-> begin
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
    _x[1]
  end,I)
  _batch_data = convert(Array{typeof(batch_data[1])},batch_data)

  elseif parallel_type == :parfor
    _batch_data = @sync @parallel (vcat) for i in I
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
      [_x[1]]
    end

  elseif parallel_type == :threads
    batch_data = Vector{Any}(length(I))
    Threads.@threads for i in I
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
        batch_data[i] = _x[1]
    end
    _batch_data = convert(Array{typeof(batch_data[1])},batch_data)

  elseif parallel_type == :split_threads
    wp=CachingPool(workers())
    batch_data = pmap(wp,(i) -> begin
      thread_monte(prob,I,alg,i,kwargs...)
    end,1:nprocs(),batch_size=pmap_batch_size)
    _batch_data = vector_batch_data_to_arr(batch_data)
  else
    error("Method $parallel_type is not a valid parallelism method.")
  end
  _batch_data
end

function thread_monte(prob,I,alg,procid,kwargs...)
  start = I[1]+(procid-1)*length(I)
  stop = I[1]+procid*length(I)-1
  portion = start:stop
  batch_data = Vector{Any}(length(portion))
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
  batch_data
end

function vector_batch_data_to_arr(batch_data)
  _batch_data = Vector{typeof(batch_data[1][1])}(sum((length(x) for x in batch_data)))
  idx = 0
  @inbounds for a in batch_data
    for x in a
      idx += 1
      _batch_data[idx] = x
    end
  end
  _batch_data
end
