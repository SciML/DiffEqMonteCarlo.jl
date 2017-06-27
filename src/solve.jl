function solve{T}(prob::AbstractMonteCarloProblem,alg::Union{DEAlgorithm,Void}=nothing,
               collect_result::Type{Val{T}} = Val{true};
               num_monte=10000,batch_size = num_monte,
               pmap_batch_size= batch_size÷100 > 0 ? batch_size÷100 : 1,
               parallel_type= T ? :pmap : :none,kwargs...)

  if !T
    if (parallel_type != :none && parallel_type != :threads)
      error("Distributed arrays can only be generated via none or threads")
    end
    (batch_size != num_monte) && warn("batch_size and reductions are ignored when !collect_result")

    elapsed_time = @elapsed u = DArray((num_monte,)) do I
        solve_batch(prob,alg,parallel_type,I[1],pmap_batch_size,kwargs...)
    end
    return MonteCarloSolution(u,elapsed_time,false)

  else
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
  end
end

function solve_batch(prob,alg,parallel_type,I,pmap_batch_size,kwargs...)
  if parallel_type == :pmap
      wp=CachingPool(workers())
      batch_data = pmap(wp,(i)-> begin
      new_prob = prob.prob_func(deepcopy(prob.prob),i)
      prob.output_func(solve(new_prob,alg;kwargs...),i)
    end,I,batch_size=pmap_batch_size)
    batch_data = convert(Array{typeof(batch_data[1])},batch_data)

  elseif parallel_type == :parfor
    batch_data = @parallel (vcat) for i in I
      new_prob = prob.prob_func(deepcopy(prob.prob),i)
      [prob.output_func(solve(new_prob,alg;kwargs...),i)]
    end

  elseif parallel_type == :threads
    batch_data = Vector{Any}()
    for i in 1:Threads.nthreads()
      push!(batch_data,[])
    end
    Threads.@threads for i in I
      new_prob = prob.prob_func(deepcopy(prob.prob),i)
      push!(batch_data[Threads.threadid()],prob.output_func(solve(new_prob,alg;kwargs...),i))
    end
    batch_data = vcat(batch_data...)
    batch_data = convert(Array{typeof(batch_data[1])},batch_data)

  elseif parallel_type == :split_threads
    wp=CachingPool(workers())
    batch_data = pmap(wp,(i) -> begin
      thread_monte(prob,I,alg,i,kwargs...)
    end,1:nprocs(),batch_size=pmap_batch_size)
    batch_data = vcat(batch_data...)
    batch_data = convert(Array{typeof(batch_data[1])},batch_data)

  elseif parallel_type == :none
    batch_data = Vector{Any}()
    for i in I
      new_prob = prob.prob_func(deepcopy(prob.prob),i)
      push!(batch_data,prob.output_func(solve(new_prob,alg;kwargs...),i))
    end
    batch_data = convert(Array{typeof(batch_data[1])},batch_data)

  else
    error("Method $parallel_type is not a valid parallelism method.")
  end
  batch_data
end

function thread_monte(prob,I,alg,procid,kwargs...)
  batch_data = Vector{Any}()
  for i in 1:Threads.nthreads()
    push!(batch_data,[])
  end
  Threads.@threads for i in (I[1]+(procid-1)*length(I)+1):(I[1]+procid*length(I))
    new_prob = prob.prob_func(deepcopy(prob.prob),i)
    push!(batch_data[Threads.threadid()],prob.output_func(solve(new_prob,alg;kwargs...),i))
  end
  batch_data = vcat(batch_data...)
  batch_data = convert(Array{typeof(batch_data[1])},batch_data)
end
