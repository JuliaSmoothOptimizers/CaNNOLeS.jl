# CaNNOLeS - Constrained and NoNlinear Optimizer of LEast-Squares
module CaNNOLeS

# stdlib
using LinearAlgebra, Logging

# JSO packages
using HSL, Krylov, LDLFactorizations, LinearOperators, NLPModels, SolverCore, SparseMatricesCOO
using SolverCore: eval_fun

function __init__()
  global available_linsolvers = [:ldlfactorizations]
  if isdefined(HSL, :libhsl_ma57)
    push!(available_linsolvers, :ma57)
  end
  # if isdefined(HSL, :libhsl_ma97)
  #   push!(available_linsolvers, :ma97)
  # end
end

const avail_mtds = Symbol[:Newton, :LM, :Newton_noFHess, :Newton_vanishing]

"""
    _check_available_method(method::Symbol)

Return an error if `method` is not in `CaNNOLeS.avail_mtds`
"""
function _check_available_method(method::Symbol)
  if !(method in avail_mtds)
    s = "`method` must be one of these: "
    for x in avail_mtds
      s *= "`$x`, "
    end
    error(s[1:(end - 2)])
  end
end

"""
    ParamCaNNOLeS(eig_tol,δmin,κdec,κinc,κlargeinc,ρ0,ρmax,ρmin,γA)
    ParamCaNNOLeS(::Type{T})

Structure containing all the parameters used in the [`cannoles`](@ref) call.
"""
mutable struct ParamCaNNOLeS{T}
  eig_tol::T
  δmin::T
  κdec::T
  κinc::T
  κlargeinc::T
  ρ0::T
  ρmax::T
  ρmin::T
  γA::T
end

function ParamCaNNOLeS(
  ::Type{T};
  ϵM::T = eps(T),
  eig_tol::T = ϵM,
  δmin::T = √ϵM,
  κdec::T = T(1 // 3),
  κinc::T = T(8),
  κlargeinc::T = min(T(100), sizeof(T) * 16),
  ρ0::T = ϵM^T(1 / 3),
  ρmax::T = min(ϵM^T(-2.0), prevfloat(T(Inf))),
  ρmin::T = √ϵM,
  γA::T = ϵM^T(1 / 4),
) where {T}
  ParamCaNNOLeS(eig_tol, δmin, κdec, κinc, κlargeinc, ρ0, ρmax, ρmin, γA)
end

function update!(
  params::ParamCaNNOLeS{T},
  ϵM::T;
  eig_tol::T = ϵM,
  δmin::T = √ϵM,
  κdec::T = T(1 // 3),
  κinc::T = T(8),
  κlargeinc::T = min(T(100), sizeof(T) * 16),
  ρ0::T = ϵM^T(1 / 3),
  ρmax::T = min(ϵM^T(-2.0), prevfloat(T(Inf))),
  ρmin::T = √ϵM,
  γA::T = ϵM^T(1 / 4),
) where {T}
  params.eig_tol = eig_tol
  params.δmin = δmin
  params.κdec = κdec
  params.κinc = κinc
  params.κlargeinc = κlargeinc
  params.ρ0 = ρ0
  params.ρmax = ρmax
  params.ρmin = ρmin
  params.γA = γA
  return params
end

import SolverCore.solve!
export cannoles, CaNNOLeSSolver, solve!

include("solver_types.jl")

"""
    cannoles(nls)

Implementation of a solver for Nonlinear Least Squares with nonlinear constraints.

``
  min   f(x) = ¹/₂‖F(x)‖²   s.t.  c(x) = 0
``

For advanced usage, first define a `CaNNOLeSSolver` to preallocate the memory used in the algorithm, and then call `solve!`:

    solver = CaNNOLeSSolver(nls; linsolve = :ma57)
    solve!(solver, nls; kwargs...)

or even pre-allocate the output:

    stats = GenericExecutionStats(nls)
    solve!(solver, nls, stats; kwargs...)

# Arguments
- `nls :: AbstractNLSModel{T, V}`: nonlinear least-squares model created using `NLPModels`.

# Keyword arguments 
- `x::AbstractVector = nls.meta.x0`: the initial guess;
- `λ::AbstractVector = T[]`: the initial Lagrange multiplier;
- `method::Symbol = :Newton`: available methods `:Newton, :LM, :Newton_noFHess`, and `:Newton_vanishing`;
- `linsolve::Symbol = :ma57`: solver to compute LDLt factorization. Available methods are: `:ma57`, `:ldlfactorizations`;
- `max_iter::Int = -1`: maximum number of iterations;
- `max_eval::Real = 100000`: maximum number of evaluations computed by `neval_residual(nls) + neval_cons(nls)`;
- `max_time::Float64 = 30.0`: maximum time limit in seconds;
- `max_inner::Int = 10000`: maximum number of inner iterations (return stalled if this limit is reached);
- `atol::T = √eps(T)`: absolute tolerance;
- `rtol::T = √eps(T)`: relative tolerance: the algorithm uses `ϵtol := atol + rtol * ‖∇F(x⁰)ᵀF(x⁰) - ∇c(x⁰)ᵀλ⁰‖`;
- `Fatol::T = √eps(T)`: absolute tolerance on the residual;
- `Frtol::T = eps(T)`: relative tolerance on the residual, the algorithm stops when ‖F(xᵏ)‖ ≤ Fatol + Frtol * ‖F(x⁰)‖  and ‖c(xᵏ)‖∞ ≤ √ϵtol;
- `verbose::Int = 0`: if > 0, display iteration details every `verbose` iteration;
- `always_accept_extrapolation::Bool = false`: if `true`, run even if the extrapolation step fails;
- `δdec::Real = T(0.1)`: reducing factor on the parameter `δ`.

The algorithm stops when ``‖c(xᵏ)‖∞ ≤ ϵtol`` and ``‖∇F(xᵏ)ᵀF(xᵏ) - ∇c(xᵏ)ᵀλᵏ‖ ≤ ϵtol * max(1, ‖λᵏ‖ / 100ncon)``.

# Output
The value returned is a `GenericExecutionStats`, see `SolverCore.jl`.

# Callback
The callback is called at each iteration.
The expected signature of the callback is `callback(nls, solver, stats)`, and its output is ignored.
Changing any of the input arguments will affect the subsequent iterations.
In particular, setting `stats.status = :user` will stop the algorithm.
All relevant information should be available in `nlp` and `solver`.
Notably, you can access, and modify, the following:
- `solver.x`: current iterate;
- `solver.cx`: current value of the constraints at `x`;
- `stats`: structure holding the output of the algorithm (`GenericExecutionStats`), which contains, among other things:
  - `stats.solution`: current iterate;
  - `stats.multipliers`: current Lagrange multipliers wrt to the constraints;
  - `stats.primal_feas`:the primal feasibility norm at `solution`;
  - `stats.dual_feas`: the dual feasibility norm at `solution`;
  - `stats.iter`: current iteration counter;
  - `stats.objective`: current objective function value;
  - `stats.status`: current status of the algorithm. Should be `:unknown` unless the algorithm has attained a stopping criterion. Changing this to anything will stop the algorithm, but you should use `:user` to properly indicate the intention.
  - `stats.elapsed_time`: elapsed time in seconds.

# Examples
```jldoctest
using CaNNOLeS, ADNLPModels
nls = ADNLSModel(x -> x, ones(3), 3)
stats = cannoles(nls, linsolve = :ldlfactorizations, verbose = 0)
stats

# output

"Execution stats: first-order stationary"
```
```jldoctest
using CaNNOLeS, ADNLPModels
nls = ADNLSModel(x -> x, ones(3), 3)
solver = CaNNOLeSSolver(nls, linsolve = :ldlfactorizations)
stats = solve!(solver, nls, verbose = 0)
stats

# output

"Execution stats: first-order stationary"
```
"""
mutable struct CaNNOLeSSolver{Ti, T, V, F} <: AbstractOptimizationSolver
  x::V
  cx::V
  r::V
  Fx::V
  d::V
  dλ::V
  rhs::V

  xt::V
  rt::V
  λt::V
  Ft::V
  ct::V

  Jxtr::V
  Jcxtλ::V
  dual::V
  primal::V

  rows::Vector{Ti}
  cols::Vector{Ti}
  vals::V
  hsr_rows::Vector{Ti}
  hsr_cols::Vector{Ti}
  Jx_rows::Vector{Ti}
  Jx_cols::Vector{Ti}
  Jx_vals::V
  Jx::SparseMatrixCOO{T, Ti}
  Jt_vals::V
  Jt::SparseMatrixCOO{T, Ti}
  Jcx_rows::Vector{Ti}
  Jcx_cols::Vector{Ti}
  Jcx_vals::V
  Jcx::SparseMatrixCOO{T, Ti}
  Jct_vals::V
  Jct::SparseMatrixCOO{T, Ti}

  LDLT::F
  cgls_solver::CglsSolver{T, T, V}

  params::ParamCaNNOLeS{T}
end

function CaNNOLeSSolver(nls::AbstractNLSModel{T, V}; linsolve::Symbol = :ma57) where {T, V}
  nvar = nls.meta.nvar
  nequ = nls_meta(nls).nequ
  ncon = nls.meta.ncon

  x = similar(nls.meta.x0)
  cx = zeros(T, ncon)
  r = V(undef, nequ)
  Fx = V(undef, nequ)
  d = zeros(T, nvar + nequ + ncon)
  dλ = zeros(T, ncon)
  rhs = zeros(T, nvar + nequ + ncon)

  xt = copy(x)
  rt = V(undef, nequ)
  λt = V(undef, ncon)
  Ft = V(undef, nequ)
  ct = V(undef, ncon)

  Jxtr = V(undef, nvar)
  Jcxtλ = V(undef, nvar)
  dual = V(undef, nvar)
  primal = V(undef, nequ + ncon)

  nnzhF, nnzhc = nls.nls_meta.nnzh, ncon > 0 ? nls.meta.nnzh : 0
  nnzjF, nnzjc = nls.nls_meta.nnzj, nls.meta.nnzj
  nnzNS = nnzhF + nnzhc + nnzjF + nnzjc + nvar + nequ + ncon

  hsr_rows, hsr_cols = hess_structure_residual(nls)
  Ti = eltype(hsr_rows)
  rows = Vector{Ti}(undef, nnzNS)
  cols = Vector{Ti}(undef, nnzNS)
  vals = V(undef, nnzNS)
  Jx_rows, Jx_cols = jac_structure_residual(nls)
  Jx_vals = V(undef, nls.nls_meta.nnzj)
  Jx = SparseMatrixCOO(nequ, nvar, Jx_rows, Jx_cols, Jx_vals)
  Jt_vals = V(undef, nls.nls_meta.nnzj)
  Jt = SparseMatrixCOO(nequ, nvar, Jx_rows, Jx_cols, Jt_vals)
  Jcx_rows, Jcx_cols = jac_structure(nls)
  Jcx_vals = V(undef, nls.meta.nnzj)
  Jcx = SparseMatrixCOO(ncon, nvar, Jcx_rows, Jcx_cols, Jcx_vals)
  Jct_vals = V(undef, nls.meta.nnzj)
  Jct = SparseMatrixCOO(ncon, nvar, Jcx_rows, Jcx_cols, Jct_vals)

  # Allocation and structure of Newton system matrix
  # G = [Hx + ρI; Jx -I; Jcx 0 -δI]
  # Hx
  sI = 1:nnzhF
  rows[sI] .= hsr_rows
  cols[sI] .= hsr_cols
  if ncon > 0
    sI = nnzhF .+ (1:nnzhc)
    rows[sI], cols[sI] = hess_structure(nls)
  end
  # Jx
  sI = nnzhF + nnzhc .+ (1:nnzjF)
  rows[sI] .= Jx_rows .+ nvar
  cols[sI] .= Jx_cols
  # Jcx
  if ncon > 0
    sI = nnzhF + nnzhc + nnzjF .+ (1:nnzjc)
    rows[sI] .= Jcx_rows .+ (nvar + nequ)
    cols[sI] .= Jcx_cols
  end
  # -I
  sI = nnzhF + nnzhc + nnzjF + nnzjc .+ (1:nequ)
  rows[sI], cols[sI] = (nvar + 1):(nvar + nequ), (nvar + 1):(nvar + nequ)
  vals[sI] .= -one(T)
  # -δI
  if ncon > 0
    sI = nnzhF + nnzhc + nnzjF + nnzjc + nequ .+ (1:ncon)
    rows[sI], cols[sI] =
      (nvar + nequ + 1):(nvar + nequ + ncon), (nvar + nequ + 1):(nvar + nequ + ncon)
  end
  # ρI
  sI = nnzhF + nnzhc + nnzjF + nnzjc + nequ + ncon .+ (1:nvar)
  rows[sI], cols[sI] = 1:nvar, 1:nvar

  if !(linsolve in available_linsolvers)
    @warn("linsolve $linsolve not available. Using :ldlfactorizations instead")
    linsolve = :ldlfactorizations
  end

  LDLT = if linsolve == :ma57
    LDLT = MA57Struct(nvar + nequ + ncon, rows, cols, vals)
    vals = LDLT.factor.vals
    LDLT
  elseif linsolve == :ldlfactorizations
    LDLT = LDLFactStruct(rows, cols, vals)
    vals = LDLT.vals
    LDLT
  else
    error("Can't handle $linsolve")
  end
  F = typeof(LDLT)

  cgls_solver = CglsSolver(nvar, ncon, V)

  params = ParamCaNNOLeS(T)

  return CaNNOLeSSolver{Ti, T, V, F}(
    x,
    cx,
    r,
    Fx,
    d,
    dλ,
    rhs,
    xt,
    rt,
    λt,
    Ft,
    ct,
    Jxtr,
    Jcxtλ,
    dual,
    primal,
    rows,
    cols,
    vals,
    hsr_rows,
    hsr_cols,
    Jx_rows,
    Jx_cols,
    Jx_vals,
    Jx,
    Jt_vals,
    Jt,
    Jcx_rows,
    Jcx_cols,
    Jcx_vals,
    Jcx,
    Jct_vals,
    Jct,
    LDLT,
    cgls_solver,
    params,
  )
end

function SolverCore.reset!(solver::CaNNOLeSSolver)
  solver
end
function SolverCore.reset!(solver::CaNNOLeSSolver, nls::AbstractNLSModel)
  ncon = nls.meta.ncon
  hess_structure_residual!(nls, solver.hsr_rows, solver.hsr_cols)
  jac_structure_residual!(nls, solver.Jx_rows, solver.Jx_cols)
  jac_structure!(nls, solver.Jcx_rows, solver.Jcx_cols)
  if ncon > 0
    nnzhF, nnzhc = nls.nls_meta.nnzh, ncon > 0 ? nls.meta.nnzh : 0
    sI = nnzhF .+ (1:nnzhc)
    solver.rows[sI], solver.cols[sI] = hess_structure(nls)
  end
  solver
end

@doc (@doc CaNNOLeSSolver) function cannoles(
  nls::AbstractNLSModel;
  linsolve::Symbol = :ma57,
  kwargs...,
)
  if has_bounds(nls) || inequality_constrained(nls)
    error("Problem has inequalities, can't solve it")
  end
  if !(nls.meta.minimize)
    error("CaNNOLeS only works for minimization problem")
  end
  solver = CaNNOLeSSolver(nls, linsolve = linsolve)
  return SolverCore.solve!(solver, nls; kwargs...)
end

function SolverCore.solve!(
  solver::CaNNOLeSSolver{Ti, T, V, F},
  nls::AbstractNLSModel{T, V},
  stats::GenericExecutionStats{T, V, V};
  callback = (args...) -> nothing,
  x::V = nls.meta.x0,
  λ::V = T[],
  method::Symbol = :Newton,
  max_iter::Int = -1,
  max_eval::Real = 100000,
  max_time::Real = 30.0,
  max_inner::Int = 10000,
  atol::T = √eps(T),
  rtol::T = √eps(T),
  Fatol::Real = √eps(T),
  Frtol::Real = eps(T),
  verbose::Integer = 0,
  always_accept_extrapolation::Bool = false,
  δdec::T = T(0.1),
) where {Ti, T, V, F}
  reset!(stats)
  start_time = time()

  _check_available_method(method)
  merit = :auglag

  nvar = nls.meta.nvar
  nequ = nls_meta(nls).nequ
  ncon = nls.meta.ncon

  x = solver.x .= x

  # Parameters
  params = update!(solver.params, eps(T))
  ρ = ρold = zero(T)
  δ = one(T)

  nnzhF, nnzhc = nls.nls_meta.nnzh, ncon > 0 ? nls.meta.nnzh : 0
  nnzjF, nnzjc = nls.nls_meta.nnzj, nls.meta.nnzj
  Jx_rows, Jx_cols = solver.Jx_rows, solver.Jx_cols
  Jx_vals, Jt_vals = solver.Jx_vals, solver.Jt_vals
  Jcx_rows, Jcx_cols = solver.Jcx_rows, solver.Jcx_cols
  Jcx_vals, Jct_vals = solver.Jcx_vals, solver.Jct_vals
  Jt, Jct = solver.Jt, solver.Jct
  vals = solver.vals
  LDLT = solver.LDLT
  cgls_solver = solver.cgls_solver

  # Shorter function definitions
  F!(x, Fx) = residual!(nls, x, Fx)
  crhs = nls.meta.lcon
  c!(x, cx) =
    if ncon == 0
      crhs
    else
      begin
        cons!(nls, x, cx)
        cx .-= crhs
      end
    end

  ϕ(x, λ, Fx, cx, η) = begin
    dot(Fx, Fx) / 2 - dot(λ, cx) + η * dot(cx, cx) / 2
  end

  # Initial values
  Fx = F!(x, solver.Fx)
  if check_nan_inf(Fx)
    error("Initial point gives Inf or Nan")
  end
  fx = dot(Fx, Fx) / 2

  jac_coord_residual!(nls, x, Jx_vals)
  Jx = solver.Jx
  Jx.vals .= Jx_vals

  cx, Jcx = solver.cx, solver.Jcx
  c!(x, cx)
  if ncon > 0
    jac_coord!(nls, x, Jcx_vals)
    Jcx.vals .= Jcx_vals
  end

  r = solver.r .= Fx
  d = solver.d
  dx = view(d, 1:nvar)
  dr = view(d, nvar .+ (1:nequ))
  dλ = solver.dλ

  Jxtr = solver.Jxtr
  mul!(Jxtr, Jx', r)

  elapsed_time = 0.0

  if length(λ) == 0
    Krylov.solve!(cgls_solver, Jcx', Jxtr) # Armand 2012
    λ = cgls_solver.x
    if norm(λ) == 0
      λ = ones(T, ncon)
    end
  end

  Jcxtλ = solver.Jcxtλ
  mul!(Jcxtλ, Jcx', λ)
  dual = solver.dual .= Jxtr .- Jcxtλ
  primal = solver.primal
  primal[1:nequ] .= Fx .- r
  primal[(nequ + 1):end] .= cx

  rhs = solver.rhs

  normdualhat = normdual = norm(dual, Inf)
  normprimalhat = normprimal = norm(primal, Inf)

  smax = T(100.0)
  ϵF = Fatol + Frtol * 2 * √fx # fx = 0.5‖F(x)‖²
  ϵtol = atol + rtol * normdual
  ϵc = sqrt(ϵtol)

  # Small residual
  small_residual = (2 * √fx <= ϵF) && norm(cx) ≤ ϵc
  sd = dual_scaling(λ, smax)
  first_order = max(normdual / sd, normprimal) <= ϵtol
  if small_residual && !first_order
    normprimal, normdual = optimality_check_small_residual!(
      cgls_solver,
      r,
      λ,
      dual,
      primal,
      Fx,
      cx,
      Jx,
      Jcx,
      Jxtr,
      Jcxtλ,
    )
    sd = dual_scaling(λ, smax)
    first_order = max(normdual / sd, normprimal) <= ϵtol
  end
  elapsed_time = time() - start_time
  tired = eval_fun(nls) > max_eval || elapsed_time > max_time
  broken = false
  internal_msg = ""

  # Trial point pre-allocation
  xt, rt, λt, Ft, ct = solver.xt, solver.rt, solver.λt, solver.Ft, solver.ct

  η = one(T)
  if ncon == 0
    η = zero(T)
  end
  set_iter!(stats, 0)
  inner_iter = 0
  nbk = nfact = nlinsolve = 0

  ϵk = 1e3

  status = SolverCore.get_status(
    nls;
    elapsed_time = elapsed_time,
    optimal = first_order,
    small_residual = small_residual,
    exception = broken,
    max_eval = max_eval,
    max_time = max_time,
    max_iter = max_iter,
  )
  set_status!(stats, status)

  if verbose > 0
    @info log_header(
      [:I, :nF, :fx, :Δt, :dual, :Fxminusr, :primal, :α, :η, :ρ, :δ, :in_it, :nbk],
      [Int, Int, T, Float64, T, T, T, T, T, T, T, Int, Int],
      hdr_override = Dict(
        :nF => "#F",
        :dual => "‖∇L‖",
        :Fxminusr => "‖Fx - r‖",
        :primal => "‖c(x)‖",
      ),
    )
    @info log_row(
      Any[0, eval_fun(nls), fx, 0.0, normdual, norm(primal[1:nequ]), norm(primal[(nequ + 1):end])],
    )
  end

  set_objective!(stats, dot(Fx, Fx) / 2)
  set_residuals!(stats, norm(cx), normdual)
  set_solution!(stats, x)
  set_constraint_multipliers!(stats, λ)
  callback(nls, solver, stats)

  done = stats.status != :unknown

  while !done
    # |G(w) - μe|
    combined_optimality = normdual + normprimal
    δ = max(params.δmin, min(δdec * δ, combined_optimality))

    damp = one(T)

    inner_iter = 0
    combined_optimality_hat = T(Inf)
    first_iteration = true
    while first_iteration ||
      !(combined_optimality_hat <= T(0.99) * combined_optimality + ϵk || tired)
      first_iteration = false

      ### System solution
      if inner_iter != 1 || always_accept_extrapolation # If = 1, then extrapolation step failed, and x is not updated
        if method in [:Newton, :Newton_noFHess, :Newton_vanishing]
          if method == :Newton || (method == :Newton_vanishing && dot(Fx, Fx) > 1e-8)
            sI = 1:nnzhF
            @views hess_coord_residual!(nls, x, r, vals[sI])
          end
          sI = nnzhF + nnzhc .+ (1:nnzjF)
          vals[sI] .= Jx_vals
          if ncon > 0
            sI = nnzhF .+ (1:nnzhc)
            @views hess_coord!(nls, x, -λ, vals[sI], obj_weight = zero(T))
            sI = nnzhF + nnzhc + nnzjF .+ (1:nnzjc)
            vals[sI] .= Jcx_vals
            sI = nnzhF + nnzhc + nnzjF + nnzjc + nequ .+ (1:ncon)
            vals[sI] = -δ * ones(ncon)
          end
          sI = nnzhF + nnzhc + nnzjF + nnzjc + nequ + ncon .+ (1:nvar)
          vals[sI] .= zero(T)
          #=
          elseif method == :LM
          #Hx = spzeros(nvar, nvar)
          Λ = [norm(Jx[:,j])^2 for j = 1:nvar] * max(1e-10, min(1e8, damp))
          #Λ = ones(nvar) * max(1e-10, min(1e8, damp))
          Hx = spdiagm(0 => Λ)
          =#
        else
          error("No method $method")
        end

        # on first time, μnew = μ⁺
        rhs .= [dual; primal]
        d, newton_success, ρ, ρold, nfacti =
          newton_system!(d, nvar, nequ, ncon, rhs, LDLT, ρold, params)
        nfact += nfacti
        nlinsolve += 1

        if ρ > params.ρmax || !newton_success || any(isinf.(d)) || any(isnan.(d)) || fx ≥ T(1e60) # Error on hs70
          internal_msg = if ρ > params.ρmax
            "ρ → ∞"
          elseif !newton_success
            "Failure in Newton step computation"
          elseif any(isinf.(d))
            "d → ∞"
          elseif any(isnan.(d))
            "d is NaN"
          elseif fx ≥ T(1e60)
            "f → ∞"
          end
          broken = true
          break
        end

        dλ .= -d[nvar .+ nequ .+ (1:ncon)]
      end # inner_iter != 1
      ### End of System solution

      α = zero(T) # For the log
      if inner_iter == 0
        ϵk = max(min(1e3 * δ, 99 * ϵk / 100), 9 * ϵk / 10)
        xt .= x + dx
        rt .= r + dr

        Mdλ = T(1e4)
        if norm(dλ) > Mdλ
          dλ .= dλ * Mdλ / norm(dλ)
        end
        λt .= λ + dλ
        F!(xt, Ft)
        c!(xt, ct)
      else
        η, α, ϕx, Dϕ, nbki = line_search(
          x,
          r,
          λ,
          dx,
          dr,
          dλ,
          Fx,
          cx,
          Jx,
          Jcx,
          ϕ,
          xt,
          rt,
          λt,
          Ft,
          ct,
          F!,
          c!,
          ρ,
          δ,
          η,
          false,
          merit,
          params,
        )

        nbk += nbki

        rt .= Ft
        λt .= λ - cx ./ δ       # Safe if ncon = 0 and δ = 0.0
      end

      if method == :LM
        Ared = norm(Fx)^2 - norm(Ft)^2
        Pred = norm(Fx)^2 - (α == 0 ? norm(Fx + Jx * dx)^2 : norm(Fx + α * Jx * dx)^2)
        if Ared / Pred > 0.75
          damp /= 10
        elseif Ared / Pred < 0.25
          damp *= 10
        end
      end

      jac_coord_residual!(nls, xt, Jt_vals)
      Jt.vals .= Jt_vals
      if ncon > 0
        jac_coord!(nls, xt, Jct_vals)
        Jct.vals .= Jct_vals
      end

      dual .= Jt' * rt - Jct' * λt
      primal .= [Ft - rt; ct]

      # dual, primal and comp overwrite the previous vectors, but normdualhat doesn't
      normdualhat = norm(dual, Inf)
      normprimalhat = norm(primal, Inf)

      combined_optimality_hat = normdualhat + normprimalhat

      if inner_iter > 0 ||
         always_accept_extrapolation ||
         combined_optimality_hat <= T(0.99) * combined_optimality + ϵk
        x .= xt
        r .= rt
        Fx .= Ft
        fx = dot(Fx, Fx) / 2
        cx .= ct
        Jx.vals .= Jt.vals
        Jx_vals .= Jt_vals
        if ncon > 0
          Jcx.vals .= Jct.vals
          Jcx_vals .= Jct_vals
        end
      end

      if combined_optimality_hat <= T(0.99) * combined_optimality + ϵk
        λ .= λt
      else
        dual .= Jx' * r - Jcx' * λ
      end

      if ncon > 0 &&
         inner_iter > 0 &&
         normdualhat ≤ T(0.99) * normdual + ϵk / 2 &&
         normprimalhat > T(0.99) * normprimal + ϵk / 2
        δ = max(δ / 10, params.δmin)
      end

      inner_iter += 1
      elapsed_time = time() - start_time
      tired = eval_fun(nls) > max_eval || elapsed_time > max_time || inner_iter > max_inner

      verbose > 0 &&
        mod(stats.iter, verbose) == 0 &&
        @info log_row(
          Any[
            stats.iter,
            eval_fun(nls),
            fx,
            elapsed_time,
            normdualhat,
            norm(primal[1:nequ]),
            norm(primal[(nequ + 1):end]),
            α,
            η,
            ρ,
            δ,
            inner_iter,
            nbk,
          ],
        )
    end

    normdual = normdualhat
    normprimal = normprimalhat

    elapsed_time = time() - start_time
    sd = dual_scaling(λ, smax)
    first_order = max(normdual / sd, normprimal) <= ϵtol
    small_residual = (2 * √fx <= ϵF) && norm(cx) ≤ ϵc
    if small_residual && !first_order
      normprimal, normdual = optimality_check_small_residual!(
        cgls_solver,
        r,
        λ,
        dual,
        primal,
        Fx,
        cx,
        Jx,
        Jcx,
        Jxtr,
        Jcxtλ,
      )
      sd = dual_scaling(λ, smax)
      first_order = max(normdual / sd, normprimal) <= ϵtol
    end

    verbose > 0 &&
      mod(stats.iter, verbose) == 0 &&
      @info log_row(
        Any[
          stats.iter,
          eval_fun(nls),
          fx,
          elapsed_time,
          normdual,
          norm(primal[1:nequ]),
          norm(primal[(nequ + 1):end]),
          0.0,
          η,
          ρ,
          δ,
          stats.iter,
          nbk,
        ],
      )
    set_iter!(stats, stats.iter + 1)
    set_time!(stats, elapsed_time)
    status = SolverCore.get_status(
      nls;
      elapsed_time = elapsed_time,
      iter = stats.iter,
      optimal = first_order,
      small_residual = small_residual,
      exception = broken,
      max_eval = max_eval,
      max_time = max_time,
      max_iter = max_iter,
      stalled = inner_iter > max_inner ≥ 0,
    )
    set_status!(stats, status)

    set_objective!(stats, dot(Fx, Fx) / 2)
    set_residuals!(stats, norm(primal[(nequ + 1):end]), normdual)
    set_constraint_multipliers!(stats, λ)
    set_solution!(stats, x)
    callback(nls, solver, stats)

    done = stats.status != :unknown
  end

  set_solver_specific!(stats, :nbk, nbk)
  set_solver_specific!(stats, :nfact, nfact)
  set_solver_specific!(stats, :nlinsolve, nlinsolve)
  set_solver_specific!(stats, :internal_msg, internal_msg)
  return stats
end

"""
    normprimal, normdual = optimality_check_small_residual!(cgls_solver, r, λ, dual, primal, Fx, cx, Jx, Jcx, Jxtr, Jcxtλ)

Compute the norm of the primal and dual residuals.
The values of `r`, `Jxtr`, `λ`, `primal` and `dual` are updated.
"""
function optimality_check_small_residual!(
  cgls_solver::CglsSolver{T, T, V},
  r::V,
  λ::V,
  dual::V,
  primal::V,
  Fx::V,
  cx::V,
  Jx,
  Jcx,
  Jxtr::V,
  Jcxtλ::V,
) where {T, V}
  r .= Fx
  mul!(Jxtr, Jx', r)
  Krylov.solve!(cgls_solver, Jcx', Jxtr)
  λ .= cgls_solver.x # Armand 2012
  mul!(Jcxtλ, Jcx', λ)
  dual .= Jxtr .- Jcxtλ
  normdual = norm(dual, Inf)
  nequ = length(r)
  primal[1:nequ] .= zero(T)
  primal[(nequ + 1):end] .= cx
  normprimal = norm(cx, Inf)
  return normprimal, normdual
end

"""
Return `true` if `x` contains `NaN` or `Inf`.
"""
function check_nan_inf(x)
  for i in eachindex(x)
    if isnan(x[i]) || isinf(x[i])
      return true
    end
  end
  return false
end

"""
    sd = dual_scaling(λ::AbstractVector{T}, smax::T)

Return the dual scaling on the residual, so that the algorithm stops when `max(normdual / sd, normprimal) <= ϵtol`.
Return 1 if the problem has no constraints.
"""
function dual_scaling(λ::AbstractVector{T}, smax::T) where {T}
  ncon = length(λ)
  return ncon > 0 ? max(smax, norm(λ, 1) / ncon) / smax : one(T)
end

@deprecate newton_system(x, r, λ, Fx, rhs, LDLT, ρold, params, method, linsolve) newton_system(
  length(x),
  length(r),
  length(λ),
  rhs,
  LDLT,
  ρold,
  params,
)

"""
    newton_system!(d, nvar, nequ, ncon, rhs, LDLT, ρold, params)

Compute an LDLt factorization of the (`nvar + nequ + ncon`)-square matrix for the Newton system contained in `LDLT`, i.e., `sparse(LDLT.rows, LDLT.cols, LDLT.vals, N, N)`.
If the factorization fails, a new factorization is attempted with an increased value for the regularization ρ as long as it is smaller than `params.ρmax`.
The factorization is then used to solve the linear system whose right-hand side is `rhs`.

# Output

- `d`: the solution of the linear system;
- `solve_success`: `true` if the usage of the LDLt factorization is successful;
- `ρ`: the value of the regularization parameter used in the factorization;
- `ρold`: the value of the regularization parameter used in the previous successful factorization, or 0 if this is the first one;
- `nfact`: the number of factorization attempts.
"""
function newton_system!(
  d::AbstractVector{T},
  nvar::Integer,
  nequ::Integer,
  ncon::Integer,
  rhs::AbstractVector{T},
  LDLT::LinearSolverStruct,
  ρold::T,
  params::ParamCaNNOLeS{T},
) where {T}
  nfact = 0

  ρ = zero(T)

  success = try_to_factorize(LDLT, nvar, nequ, ncon, params.eig_tol)
  nfact += 1

  vals = get_vals(LDLT)
  sI = (length(vals) - nvar + 1):length(vals)

  if !success
    ρ = ρold == 0 ? params.ρ0 : max(params.ρmin, params.κdec * ρold)
    vals[sI] .= ρ
    success = try_to_factorize(LDLT, nvar, nequ, ncon, params.eig_tol)
    nfact += 1
    ρiter = 0
    while !success && ρ <= params.ρmax
      ρ = ρold == 0 ? params.κlargeinc * ρ : params.κinc * ρ
      if ρ <= params.ρmax
        vals[sI] .= ρ
        success = try_to_factorize(LDLT, nvar, nequ, ncon, params.eig_tol)
        nfact += 1
      end
      ρiter += 1
    end
    if ρ <= params.ρmax
      ρold = ρ
    end
  end

  if success
    d, solve_success = solve_ldl!(rhs, LDLT.factor, d)
  else
    solve_success = false
  end

  return d, solve_success, ρ, ρold, nfact
end

function line_search(
  x::AbstractVector{T},
  r,
  λ,
  dx,
  dr,
  dλ,
  Fx,
  cx,
  Jx,
  Jcx,
  ϕ,
  xt,
  rt,
  λt,
  Ft,
  ct,
  F!,
  c!,
  ρ,
  δ,
  η,
  trial_computed,
  merit,
  params,
) where {T}
  Dϕ = dot(Jx' * Fx, dx) - dot(dx, Jcx' * (λ - cx / δ))

  if length(λ) > 0
    η = 1 / δ
  end
  @assert Dϕ < 0

  if !trial_computed
    xt .= x .+ dx
    F!(xt, Ft)
    c!(xt, ct)
  end

  ϕx = ϕ(x, λ, Fx, cx, η)
  ϕt = ϕ(xt, λ, Ft, ct, η)

  α = one(T)
  armijo_param = params.γA
  nbk = 0
  while !(ϕt <= ϕx + armijo_param * α * Dϕ)
    nbk += 1
    α /= 4
    xt .= x + α * dx
    F!(xt, Ft)
    c!(xt, ct)
    ϕt = ϕ(xt, λ, Ft, ct, η)
    if α < eps(T)^2
      error("α too small")
    end
  end

  return η, α, ϕx, Dϕ, nbk
end

end # module
