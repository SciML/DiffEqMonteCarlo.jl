function solve(prob::AbstractMonteCarloProblem,alg::Union{DEAlgorithm,Void}=nothing;num_monte=10000,batch_size = 1,parallel_type=:pmap,reduction = (u,data,I)->(append!(u,data),false),u_init = Vector{Any}(),kwargs...)
  num_batches = div(num_monte, batch_size)
  u = u_init
  elapsedTime = @elapsed for i in 1:num_batches
    I = (batch_size*(i-1)+1):batch_size*i
    batch_data = solve_batch(prob,alg,parallel_type,I,kwargs...)
    u,converged = reduction(u,batch_data,I)
    converged ? break : nothing
  end
  MonteCarloSolution(u,elapsedTime)
end

function solve_batch(prob,alg,parallel_type,I,kwargs...)
  batch_data = Vector{Any}()
  if parallel_type == :pmap
      batch_data = pmap((i)-> begin
      new_prob = prob.prob_func(deepcopy(prob.prob),i)
      prob.output_func(solve(new_prob,alg;kwargs...),i)
    end,I)
    batch_data = convert(Array{typeof(batch_data[1])},batch_data)

  elseif parallel_type == :parfor
    batch_data = @sync @parallel (vcat) for i in I
      new_prob = prob.prob_func(deepcopy(prob.prob),i)
      prob.output_func(solve(new_prob,alg;kwargs...),i)
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
    batch_data = @sync @parallel (vcat) for procid in 1:nprocs()
      _num_monte = length(I)÷nprocs() # probably can be made more even?
      if procid == nprocs()
        _num_monte = length(I)-_num_monte*(nprocs()-1)
      end
      thread_monte(prob,I,alg,procid,kwargs...)
    end

  elseif parallel_type == :none
    batch_data = Vector{Any}()
    for i in I
      new_prob = prob.prob_func(deepcopy(prob.prob),i)
      prob.output_func(solve(new_prob,alg;kwargs...),i)
    end
    batch_data = convert(Array{typeof(batch_data[1])},batch_data)

  else
    error("Method $parallel_type is not a valid parallelism method.")
  end
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
  batch_data = convert(Array{typeof(solution_data[1])},batch_data)
end
