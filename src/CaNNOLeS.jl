# CaNNOLeS - Constrained and NoNlinear Optimizer of LEast-Squares
module CaNNOLeS

# stdlib
using LinearAlgebra, Logging, SparseArrays

# JSO packages
using HSL, Krylov, LDLFactorizations, LinearOperators, NLPModels, SolverTools

function __init__()
  global available_linsolvers = [:ldlfactorizations]
  if isdefined(HSL, :libhsl_ma57)
    push!(available_linsolvers, :ma57)
  end
  if isdefined(HSL, :libhsl_ma97)
    push!(available_linsolvers, :ma97)
  end
end

export cannoles

include("solver_types.jl")

"""
    cannoles(nls)

Implementation of a solver for Nonlinear Least Squares with nonlinear constraints.

  min   f(x) = ¹/₂‖F(x)‖²   s.t.  c(x) = 0

Input:
- `nls :: AbstractNLSModel`: Nonlinear least-squares model created using `NLPModels`.
"""
function cannoles(nls :: AbstractNLSModel;
                  x :: AbstractVector = copy(nls.meta.x0),
                  λ :: AbstractVector = eltype(x)[],
                  method :: Symbol = :Newton,
                  merit :: Symbol = :auglag, # :norm1, :auglag
                  linsolve :: Symbol = :ma57, # :ma57, :ma97, :ldlfactorizations
                  max_f :: Real = 100000,
                  max_time :: Real = 30.0,
                  max_inner :: Int = 10000,
                  ϵtol :: Real = √eps(eltype(x)),
                  check_small_residual :: Bool = true,
                  always_accept_extrapolation :: Bool = false,
                  ϵkchoice = :delta, # :delta or :slow
                  δdec :: Real = eltype(x)(0.1)
  )

  start_time = time()
  avail_mtds = [:Newton, :LM, :Newton_noFHess, :Newton_vanishing]
  if !(method in avail_mtds)
    s = "`method` must be one of these: "
    s *= join(["`$x`" for x in avail_mtds], ", ")
    error(s)
  end
  merit in [:auglag] || error("Wrong merit function $merit")
  T = eltype(x)
  if !(linsolve in available_linsolvers)
    @warn("linsolve $linsolve not available. Using :ldlfactorizations instead")
    linsolve = :ldlfactorizations
  end
  if has_bounds(nls) || inequality_constrained(nls)
    error("Problem has inequalities, can't solve it")
  end
  ϵM = eps(T)
  nvar = nls.meta.nvar
  nequ = nls_meta(nls).nequ
  ncon = nls.meta.ncon

  # Parameters
  params = Dict{Symbol, T}()
  ρmin, ρ0, ρmax = √ϵM, ϵM^T(1/3), min(ϵM^T(-2.0), prevfloat(T(Inf)))
  ρ = ρold = zero(T)
  δ, δmin = one(T), √ϵM
  κdec, κinc, κlargeinc = T(1/3), T(8.0), min(T(100.0), T.size * 16)
  eig_tol = ϵM
  params[:eig_tol] = eig_tol
  params[:δmin] = δmin
  params[:κdec] = κdec
  params[:κinc] = κinc
  params[:κlargeinc] = κlargeinc
  params[:ρ0] = ρ0
  params[:ρmax] = ρmax
  params[:ρmin] = ρmin
  params[:γA] = ϵM^T(1/4)

  # Allocation and structure of Newton system matrix
  # G = [Hx + ρI; Jx -I; Jcx 0 -δI]
  nnzhF, nnzhc, nnzjF, nnzjc = nls.nls_meta.nnzh, ncon > 0 ? nls.meta.nnzh : 0, nls.nls_meta.nnzj, nls.meta.nnzj
  nnzNS = nnzhF + nnzhc + nnzjF + nnzjc + nvar + nequ + ncon
  # Hx
  hsr_rows, hsr_cols = hess_structure_residual(nls)
  Ti = eltype(hsr_rows)
  rows = zeros(Ti, nnzNS)
  cols = zeros(Ti, nnzNS)
  vals = zeros(T, nnzNS)
  sI = 1:nnzhF
  rows[sI] .= hsr_rows
  cols[sI] .= hsr_cols
  if ncon > 0
    sI = nnzhF .+ (1:nnzhc)
    rows[sI], cols[sI] = hess_structure(nls)
  end
  # Jx
  sI = nnzhF + nnzhc .+ (1:nnzjF)
  Jx_rows, Jx_cols = jac_structure_residual(nls)
  rows[sI] .= Jx_rows .+ nvar
  cols[sI] .= Jx_cols
  Jx_vals = zeros(T, nls.nls_meta.nnzj)
  Jt_vals = similar(Jx_vals)
  # Jcx
  local Jcx_rows, Jcx_cols, Jcx_vals, Jct_vals
  if ncon > 0
    sI = nnzhF + nnzhc + nnzjF .+ (1:nnzjc)
    Jcx_rows, Jcx_cols = jac_structure(nls)
    rows[sI] .= Jcx_rows .+ (nvar + nequ)
    cols[sI] .= Jcx_cols
    Jcx_vals = zeros(T, nls.meta.nnzj)
    Jct_vals = similar(Jcx_vals)
  end
  # -I
  sI = nnzhF + nnzhc + nnzjF + nnzjc .+ (1:nequ)
  rows[sI], cols[sI], vals[sI] = nvar+1:nvar+nequ, nvar+1:nvar+nequ, -ones(nequ)
  # -δI
  if ncon > 0
    sI = nnzhF + nnzhc + nnzjF + nnzjc + nequ .+ (1:ncon)
    rows[sI], cols[sI] = nvar+nequ+1:nvar+nequ+ncon, nvar+nequ+1:nvar+nequ+ncon
  end
  # ρI
  sI = nnzhF + nnzhc + nnzjF + nnzjc + nequ + ncon .+ (1:nvar)
  rows[sI], cols[sI] = 1:nvar, 1:nvar

  # Shorter function definitions
  F!(x, Fx) = residual!(nls, x, Fx)
  crhs = T.(nls.meta.lcon)
  c!(x, cx) = if ncon == 0 crhs; else
      begin
        cons!(nls, x, cx)
        cx .-= crhs
      end
    end

  ϕ(x, λ, Fx, cx, η) = begin
    dot(Fx, Fx) / 2 - dot(λ, cx) + η * dot(cx, cx) / 2
  end

  # Initial values
  Fx = residual(nls, x)
  if any(isnan.(Fx)) || any(isinf.(Fx))
    error("Initial point gives Inf or Nan")
  end
  fx = dot(Fx, Fx) / 2

  jac_coord_residual!(nls, x, Jx_vals)
  Jx = sparse(Jx_rows, Jx_cols, Jx_vals, nequ, nvar)

  cx = zeros(T, ncon)
  c!(x, cx)
  if ncon > 0
    jac_coord!(nls, x, Jcx_vals)
  end
  Jcx = ncon > 0 ? sparse(Jcx_rows, Jcx_cols, Jcx_vals, ncon, nvar) : spzeros(ncon, nvar)

  r = copy(Fx)
  dx = zeros(T, nvar)
  dr = zeros(T, nequ)
  dλ = zeros(T, ncon)

  Jxtr = Jx' * r

  elapsed_time = 0.0

  if length(λ) == 0
    λ = T.(cgls(Jcx', Jxtr)[1]) # Armand 2012
    if norm(λ) == 0
      λ = ones(T, ncon)
    end
  end
  @debug("Starting values", x, Fx, λ)

  dual    = Jxtr - Jcx' * λ
  primal  = [Fx - r; cx]

  rhs = zeros(T, nvar + nequ + ncon)

  normdualhat   = normdual   = norm(dual, Inf)
  normprimalhat = normprimal = norm(primal, Inf)

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

  smax = T(100.0)
  ϵf = ϵtol / 2 # fx = 0.5‖F(x)‖² ≤ ϵf
  ϵc = sqrt(ϵtol)

  # Small residual
  small_residual = check_small_residual && fx ≤ ϵf && norm(cx) ≤ ϵc
  sd = ncon > 0 ? max(smax, norm(λ, 1) / ncon) / smax : one(T)
  first_order = max(normdual / sd, normprimal) <= ϵtol
  if small_residual && !first_order
    r .= Fx
    Jxtr = Jx' * r
    λ = T.(cgls(Jcx', Jxtr)[1]) # Armand 2012
    dual = Jxtr - Jcx' * λ
    normdual = norm(dual, Inf)
    primal = [zeros(nequ); cx]
    normprimal = norm(cx, Inf)
    sd = ncon > 0 ? max(smax, norm(λ, 1) / ncon) / smax : one(T)
    first_order = max(normdual / sd, normprimal) <= ϵtol
  end
  solved = first_order
  elapsed_time = time() - start_time
  tired = sum_counters(nls) > max_f || elapsed_time > max_time
  broken = false
  internal_msg = ""

  # Trial point pre-allocation
  xt = copy(x)
  rt = copy(r)
  λt = copy(λ)
  Ft = copy(Fx)
  ct = copy(cx)

  η = one(T)
  if ncon == 0
    η = zero(T)
  end
  iter = 1
  inner_iter = 0
  nbk = nfact = nlinsolve = 0

  ϵk = 1e3

  @info log_header([:I, :nF, :fx, :Δt, :dual, :Fxminusr, :primal, :α, :η, :ρ, :δ, :in_it, :nbk],
                   [Int, Int, T, Float64, T, T, T, T, T, T, T, Int, Int],
                   hdr_override=Dict(:nF=>"#F", :dual=>"‖∇L‖", :Fxminusr=>"‖Fx - r‖", :primal=>"‖c(x)‖"))
  @info log_row(Any[0, sum_counters(nls), fx, 0.0, normdual, norm(primal[1:nequ]), norm(primal[nequ+1:end])])

  while !(solved || tired || broken)
    # |G(w) - μe|
    combined_optimality = normdual + normprimal
    δ = max(δmin, min(δdec * δ, combined_optimality))

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
            @views hess_coord!(nls, x, -λ, vals[sI], obj_weight=zero(T))
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
        d, ρ, ρold, nfacti = newton_system(x, r, λ, Fx, rhs, LDLT, ρold, params, method, linsolve)
        nfact += nfacti
        nlinsolve += 1

        if ρ > params[:ρmax] || any(isinf.(d)) || any(isnan.(d)) || fx ≥ T(1e60) # Error on hs70
          internal_msg = if ρ > params[:ρmax]
            "ρ → ∞"
          elseif any(isinf.(d))
            "d → ∞"
          elseif any(isnan.(d))
            "d is NaN"
          elseif fx ≥ T(1e60)
            "f → ∞"
          end
          broken = true
          continue
        end

        dx .= d[1:nvar]
        dr .= d[nvar .+ (1:nequ)]
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
        η, α, ϕx, Dϕ, nbki = line_search(x, r, λ, dx, dr, dλ,
                                        Fx, cx, Jx, Jcx, ϕ, xt, rt, λt,
                                        Ft, ct, F!, c!, ρ, δ, η,
                                        false, merit, params)

        nbk += nbki

        rt .= Ft
        λt .= λ - cx ./ δ       # Safe if ncon = 0 and δ = 0.0
      end

      if method == :LM
        Ared = norm(Fx)^2 - norm(Ft)^2
        Pred = norm(Fx)^2 - (α == 0.0 ? norm(Fx + Jx * dx)^2 : norm(Fx + α * Jx * dx)^2)
        if Ared / Pred > 0.75
          damp /= 10
        elseif Ared / Pred < 0.25
          damp *= 10
        end
      end

      jac_coord_residual!(nls, xt, Jt_vals)
      Jt = sparse(Jx_rows, Jx_cols, Jt_vals, nequ, nvar)
      if ncon > 0
        jac_coord!(nls, xt, Jct_vals)
      end
      Jct = ncon > 0 ? sparse(Jcx_rows, Jcx_cols, Jct_vals, ncon, nvar) : spzeros(ncon, nvar)

      dual   .= Jt' * rt - Jct' * λt
      primal .= [Ft - rt; ct]

      # dual, primal and comp overwrite the previous vectors, but normdualhat doesn't
      normdualhat   = norm(dual, Inf)
      normprimalhat = norm(primal, Inf)

      combined_optimality_hat = normdualhat + normprimalhat

      if inner_iter > 0 || always_accept_extrapolation ||
        combined_optimality_hat <= T(0.99) * combined_optimality + ϵk
        x .= xt
        r .= rt
        Fx .= Ft
        fx = dot(Fx, Fx) / 2
        cx .= ct
        Jx .= Jt
        Jx_vals .= Jt_vals
        if ncon > 0
          Jcx .= Jct
          Jcx_vals .= Jct_vals
        end
      end

      if combined_optimality_hat <= T(0.99) * combined_optimality + ϵk
        λ .= λt
      else
        dual .= Jx' * r - Jcx' * λ
      end

      if ncon > 0 && inner_iter > 0 && normdualhat ≤ T(0.99) * normdual + ϵk / 2 &&
          normprimalhat > T(0.99) * normprimal + ϵk / 2
        δ = max(δ / 10, δmin)
      end

      inner_iter += 1
      elapsed_time = time() - start_time
      tired = sum_counters(nls) > max_f || elapsed_time > max_time ||
              inner_iter > max_inner

      @info log_row(Any[iter, sum_counters(nls), fx, elapsed_time, normdualhat, norm(primal[1:nequ]),
                        norm(primal[nequ+1:end]), α, η, ρ, δ, inner_iter, nbk])
    end

    normdual   = normdualhat
    normprimal = normprimalhat

    elapsed_time = time() - start_time
    sd = ncon > 0 ? max(smax, norm(λ, 1) / ncon) / smax : one(T)
    first_order = max(normdual / sd, normprimal) <= ϵtol
    small_residual = check_small_residual && fx ≤ ϵf && norm(cx) ≤ ϵc
    if small_residual && !first_order
      r .= Fx
      Jxtr = Jx' * r
      λ = T.(cgls(Jcx', Jxtr)[1]) # Armand 2012
      dual = Jxtr - Jcx' * λ
      normdual = norm(dual, Inf)
      primal = [zeros(nequ); cx]
      normprimal = norm(cx, Inf)
      sd = ncon > 0 ? max(smax, norm(λ, 1) / ncon) / smax : one(T)
      first_order = max(normdual / sd, normprimal) <= ϵtol
    end
    solved = first_order
    tired = sum_counters(nls) > max_f || elapsed_time > max_time ||
            inner_iter > max_inner

    @info log_row(Any[iter, sum_counters(nls), fx, elapsed_time, normdual, norm(primal[1:nequ]),
                      norm(primal[nequ+1:end]), 0.0, η, ρ, δ, inner_iter, nbk])
    iter += 1

  end

  status = if first_order
    :first_order
  #elseif small_residual
  #  :small_residual
  elseif tired
    if sum_counters(nls) > max_f
      :max_eval
    elseif elapsed_time > max_time
      :max_time
    else
      :max_iter
    end
  elseif broken
    :exception
  end

  elapsed_time = time() - start_time

  return GenericExecutionStats(status, nls, solution=x, objective=dot(Fx, Fx) / 2,
                               dual_feas=normdual, elapsed_time=elapsed_time, primal_feas=norm(primal[nequ+1:end]),
                               solver_specific=Dict(:nbk => nbk,
                                                    :nfact => nfact,
                                                    :nlinsolve => nlinsolve,
                                                    :multipliers => λ,
                                                    :internal_msg => internal_msg,
                                                   )
                              )
end

function newton_system(x, r, λ, Fx, rhs, LDLT, ρold, params, method, linsolve)
  nvar = length(x)
  nequ = length(r)
  ncon = length(λ)

  T = eltype(x)
  nfact = 0

  ρ = zero(eltype(x))

  function try_to_factorize(LDLT)
    if linsolve == :ma57
      ma57_factorize(LDLT.factor)
      success = LDLT.factor.info.info[1] == 0 && LDLT.factor.info.num_negative_eigs == nequ + ncon
      return success
    elseif linsolve == :ldlfactorizations
      try
        N = nvar + nequ + ncon
        A = sparse(LDLT.rows, LDLT.cols, LDLT.vals, N, N)
        A = Matrix(Symmetric(A, :L))
        M = ldl(A)
        pos_eig = count(M.D .> params[:eig_tol])
        zer_eig = count(abs.(M.D) .≤ params[:eig_tol])
        success = pos_eig == nvar && zer_eig == 0
        LDLT.factor = M
        return success
      catch
        return false
      end
    else
      error("Can't handle $linsolve")
      #=
    elseif linsolve == :ma97
      LDLT = Ma97(B)
      ma97_factorize!(LDLT, matrix_type=:real_indef)
      success = LDLT.info.flag == 0 && LDLT.info.num_neg == nequ + ncon
      return LDLT, success
      =#
    end
  end

  success = try_to_factorize(LDLT)
  nfact += 1

  vals = if linsolve == :ma57
    LDLT.factor.vals
  elseif linsolve == :ldlfactorizations
    LDLT.vals
  end
  sI = length(vals)-nvar+1:length(vals)

  if !success
    ρ = ρold == 0 ? params[:ρ0] : max(params[:ρmin], params[:κdec] * ρold)
    vals[sI] .= ρ
    success = try_to_factorize(LDLT)
    nfact += 1
    ρiter = 0
    while !success && ρ <= params[:ρmax]
      ρ = ρold == 0 ? params[:κlargeinc] * ρ : params[:κinc] * ρ
      if ρ <= params[:ρmax]
        vals[sI] .= ρ
        success = try_to_factorize(LDLT)
        nfact += 1
      end
      ρiter += 1
    end
    if ρ <= params[:ρmax]
      ρold = ρ
    end
  end
  d = if linsolve == :ma57
    ma57_solve(LDLT.factor, -rhs)
  elseif linsolve == :ldlfactorizations
    @assert LDLT.factor != nothing
    -(LDLT.factor \ rhs)
    #=
  elseif linsolve == :ma97
    ma97_solve(LDLT, -rhs)
    =#
  end

  return d, ρ, ρold, nfact
end

function line_search(x, r, λ, dx, dr, dλ, Fx,
                     cx, Jx, Jcx, ϕ, xt, rt, λt, Ft, ct, F!, c!, ρ,
                     δ, η, trial_computed, merit, params)
  T = eltype(x)
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

  ϕx = ϕ(x,  λ, Fx, cx, η)
  ϕt = ϕ(xt, λ, Ft, ct, η)

  α = one(T)
  armijo_param = params[:γA]
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
