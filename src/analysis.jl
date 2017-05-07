get_timestep(sim,i) = (getindex(sol,i) for sol in sim)
get_timepoint(sim,t) = (sol(t) for sol in sim)
apply_timestep(f,sim,i) = f(get_timestep(sim,i))
apply_timepoint(f,sim,t) = f(get_timepoint(sim,t))

timestep_mean(sim,i) = componentwise_mean(get_timestep(sim,i))
timestep_mean(sim,::Colon) = timeseries_steps_mean(sim)
timestep_meanvar(sim,i) = componentwise_meanvar(get_timestep(sim,i))
timestep_meanvar(sim,::Colon) = timeseries_steps_meanvar(sim)
timestep_meancov(sim,i,j) = componentwise_meancov(get_timestep(sim,i),get_timestep(sim,j))
timestep_meancov(sim,::Colon,::Colon) = timeseries_steps_meancov(sim)
timestep_meancor(sim,i,j) = componentwise_meancor(get_timestep(sim,i),get_timestep(sim,j))
timestep_meancor(sim,::Colon,::Colon) = timeseries_steps_meancor(sim)
timestep_weighted_meancov(sim,W,i,j) = componentwise_weighted_meancov(get_timestep(sim,i),get_timestep(sim,j),W)
timestep_weighted_meancov(sim,W,::Colon,::Colon) = timeseries_steps_weighted_meancov(sim,W)

function MonteCarloSummary{T,N}(sim::AbstractMonteCarloSolution{T,N})
  t = sim[1].t
  m,v = timeseries_steps_meanvar(sim)
  MonteCarloSummary{T,N,typeof(t),typeof(m),typeof(v)}(t,m,v,sim.elapsedTime)
end

function timeseries_steps_mean(sim)
  DiffEqArray([timestep_mean(sim,i) for i in 1:length(sim[1])],sim[1].t)
end
function timeseries_steps_meanvar(sim)
  means = typeof(sim[1][1])[]
  vars = typeof(sim[1][1])[]
  for i in 1:length(sim[1])
    m,v = timestep_meanvar(sim,i)
    push!(means,m)
    push!(vars,v)
  end
  DiffEqArray(means,sim[1].t),DiffEqArray(vars,sim[1].t)
end
function timeseries_steps_meancov(sim)
  reshape([timestep_meancov(sim,i,j) for i in 1:length(sim[1]) for j in 1:length(sim[1])],length(sim[1]),length(sim[1]))
end
function timeseries_steps_meancor(sim)
  reshape([timestep_meancor(sim,i,j) for i in 1:length(sim[1]) for j in 1:length(sim[1])],length(sim[1]),length(sim[1]))
end
function timeseries_steps_weighted_meancov(sim,W)
  reshape([timestep_meancov(sim,W,i,j) for i in 1:length(sim[1]) for j in 1:length(sim[1])],length(sim[1]),length(sim[1]))
end

timepoint_mean(sim,t) = componentwise_mean(get_timepoint(sim,t))
timepoint_meanvar(sim,t) = componentwise_meanvar(get_timepoint(sim,t))
timepoint_meancov(sim,t1,t2) = componentwise_meancov(get_timepoint(sim,t1),get_timepoint(sim,t2))
timepoint_meancor(sim,t1,t2) = componentwise_meancor(get_timepoint(sim,t1),get_timepoint(sim,t2))
timepoint_weighted_meancov(sim,W,t1,t2) = componentwise_weighted_meancov(get_timepoint(sim,t1),get_timepoint(sim,t2),W)

function MonteCarloSummary{T,N}(sim::AbstractMonteCarloSolution{T,N},t)
  m,v = timeseries_point_meanvar(sim,t)
  MonteCarloSummary{T,N,typeof(t),typeof(m),typeof(v)}(t,m,v,sim.elapsedTime)
end

function timeseries_point_mean(sim,ts)
  DiffEqArray([timepoint_mean(sim,t) for t in ts],ts)
end
function timeseries_point_meanvar(sim,ts)
  means = typeof(sim[1][1])[]
  vars = typeof(sim[1][1])[]
  for t in ts
    m,v = timepoint_meanvar(sim,t)
    push!(means,m)
    push!(vars,v)
  end
  DiffEqArray(means,ts),DiffEqArray(vars,ts)
end
function timeseries_point_meancov(sim,ts1,ts2)
  reshape([timepoint_meancov(sim,t1,t2) for t1 in ts1 for t2 in ts2],length(ts1),length(ts2))
end
function timeseries_point_meancor(sim,ts1,ts2)
  reshape([timepoint_meancor(sim,t1,t2) for t1 in ts1 for t2 in ts2],length(ts1),length(ts2))
end
function timeseries_point_weighted_meancov(sim,W,ts1,ts2)
  reshape([timepoint_meancov(sim,W,t1,t2) for t1 in ts1 for t2 in ts2],length(ts1),length(ts2))
end

function componentwise_mean(A)
  x0 = first(A)
  n = 0
  mean = zero(x0)
  for x in A
    n += 1
    mean += x
  end
  mean ./= n
end

# Welford algorithm
# https://en.wikipedia.org/wiki/Algorithms_for_calculating_variance
function componentwise_meanvar(A;bessel=true)
  x0 = first(A)
  n = 0
  mean = zero(x0)
  M2 = zero(x0)
  delta = zero(x0)
  delta2 = zero(x0)
  for x in A
    n += 1
    delta .= x .- mean
    mean .+= delta./n
    delta2 .= x .- mean
    M2 .+= delta.*delta2
  end
  if n < 2
    return NaN
  else
    if bessel
      M2 .= M2 ./ (n .- 1)
    else
      M2 .= M2 ./ n
    end
    return mean,M2
  end
end

function componentwise_meancov(A,B;bessel=true)
  x0 = first(A)
  y0 = first(B)
  n = 0
  meanx = zero(x0)
  meany = zero(y0)
  C = zero(x0)
  dx = zero(x0)
  for (x,y) in zip(A,B)
    n += 1
    dx .= x .- meanx
    meanx .+= dx./n
    meany .+= (y.-meany)./n
    C .+= dx .* (y .- meany)
  end
  if n < 2
    return NaN
  else
    if bessel
      C .= C ./ (n .- 1)
    else
      C .= C ./ n
    end
    return meanx,meany,C
  end
end

function componentwise_meancor(A,B;bessel=true)
  mx,my,cov = componentwise_meancov(A,B;bessel=bessel)
  mx,vx = componentwise_meanvar(A;bessel=bessel)
  my,vy = componentwise_meanvar(B;bessel=bessel)
  vx .= sqrt.(vx)
  vy .= sqrt.(vy)
  mx,my,cov./(vx.*vy)
end

function componentwise_weighted_meancov(A,B,W;weight_type=:reliability)
  x0 = first(A)
  y0 = first(B)
  w0 = first(W)
  n = 0
  meanx = zero(x0)
  meany = zero(y0)
  wsum = zero(w0)
  wsum2 = zero(w0)
  C = zero(x0)
  dx = zero(x0)
  for (x,y,w) in zip(A,B,W)
    n += 1
    wsum .+= w
    wsum2 .+= w.*w
    dx .= x .- meanx
    meanx .+= (w ./ wsum) .* dx
    meany .+= (w ./ wsum) .* (y .- meany)
    C .+= w .* dx .* (y .- meany)
  end
  if n < 2
    return NaN
  else
    if weight_type == :population
      C .= C ./ wsum
    elseif weight_type == :reliability
      C .= C ./ (wsum .- wsum2 ./ wsum)
    elseif weight_type == :frequency
      C .= C ./ (wsum .- 1)
    else
      error("The weight_type which was chosen is not allowed.")
    end
    return meanx,meany,C
  end
end
