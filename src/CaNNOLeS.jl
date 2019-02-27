# CaNNOLeS - Constrained and NoNlinear Optimizer of LEast-Squares
module CaNNOLeS

# stdlib
using LinearAlgebra, Logging, Printf, SparseArrays

# JSO packages
using HSL, Krylov, LDLFactorizations, LinearOperators, NLPModels, SolverTools

export cannoles

"""
    cannoles(nls)

Implementation of a solver for Nonlinear Least Squares with nonlinear constraints.

  min   f(x) = ¹/₂‖F(x)‖²   s.t.  c(x) = 0

Input:
- `nls :: AbstractNLSModel`: Nonlinear least-squares model created using `NLPModels`.
"""
function cannoles(nls :: AbstractNLSModel;
                  logger :: AbstractLogger = NullLogger(),
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
  avail_mtds = [:Newton, :LM]
  if !(method in avail_mtds)
    s = "`method` must be one of these: "
    s *= join(["`$x`" for x in avail_mtds], ", ")
    error(s)
  end
  merit in [:auglag] || error("Wrong merit function $merit")
  T = eltype(x)
  linsolve in [:ma57, :ma97, :ldlfactorizations] || error("Wrong linsolve value $linsolve")
  if has_bounds(nls) || inequality_constrained(nls)
    error("Problem has inequalities, can't solve it")
  end
  ϵM = eps(T)
  nvar = nls.meta.nvar
  nequ = nls_meta(nls).nequ
  ncon = nls.meta.ncon

  # TODO: Cleanup
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

  F!(x, Fx) = residual!(nls, x, Fx)
  J(x) = jac_residual(nls, x)
  crhs = T.(nls.meta.lcon)
  c!(x, cx) = if ncon == 0 crhs; else
      begin
        cons!(nls, x, cx)
        cx .-= crhs
      end
    end
  Jc(x) = ncon > 0 ? jac(nls, x) : zeros(T, 0, nvar)

  ϕ(x, λ, Fx, cx, η) = begin
    dot(Fx, Fx) / 2 - dot(λ, cx) + η * dot(cx, cx) / 2
  end

  Fx = residual(nls, x)
  if any(isnan.(Fx)) || any(isinf.(Fx))
    error("Initial point gives Inf or Nan")
  end
  fx = dot(Fx, Fx) / 2
  Jx = J(x)

  cx = zeros(T, ncon)
  c!(x, cx)
  Jcx = Jc(x)

  # TODO: Decide how to start r
  #r = ones(T, nequ)
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

  smax = T(100.0)
  ϵf = ϵtol / 2# + rtol * fx
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
  oldx = copy(x)
  oldr = copy(r)
  oldλ = copy(λ)

  η = one(T)
  if ncon == 0
    η = zero(T)
  end
  iter = 1
  inner_iter = 0
  nbk = nfact = nlinsolve = 0

  ϵk = 1e3

  with_logger(logger) do
    @info @sprintf("I    %6s  %8s  %8s  %8s  %8s  %8s  %8s  %8s  %8s  %8s  %6s  %6s\n",
                   "#F", "fx", "Δt", "‖∇L‖", "‖Fx - r‖", "‖c(x)‖", "α", "η", "ρ", "δ", "in_it", "nbk")
    @info @sprintf("∘0   %6d  %8.2e  %8.2e  %8.2e  %8.2e  %8.2e  %8s  %8s  %8s  %8s  %6s  %6d\n",
                   sum_counters(nls), fx, 0.0, normdual, norm(primal[1:nequ]),
                   norm(primal[nequ+1:end]), "-", "-", "-", "-", "-", 0)
  end

  while !(solved || tired || broken)
    oldx .= x
    oldr .= r
    oldλ .= λ
    oldJx = Jx
    oldJcx = Jcx

    # |G(w) - μe|
    combined_optimality = normdual + normprimal
    δ = max(δmin, min(δdec * δ, combined_optimality))

    damp = one(T)

    inner_iter = 0
    combined_optimality_hat = T(Inf)
    first_iteration = true
    local Hx
    while first_iteration ||
          !(combined_optimality_hat <= T(0.99) * combined_optimality + ϵk || tired)
      first_iteration = false
      # TODO: Review
      ### System solution
      if inner_iter != 1 || always_accept_extrapolation # If = 1, then extrapolation step failed, and x is not updated
        if method == :Newton
          Hx = hess_residual(nls, x, r)
          if ncon > 0
            Hx -= hess(nls, x, obj_weight=zero(T), y=λ)
          end
        elseif method == :LM
          #Hx = spzeros(nvar, nvar)
          Λ = [norm(Jx[:,j])^2 for j = 1:nvar] * max(1e-10, min(1e8, damp))
          #Λ = ones(nvar) * max(1e-10, min(1e8, damp))
          Hx = spdiagm(0 => Λ)
        else
          error("No method $method")
        end

        # on first time, μnew = μ⁺
        rhs .= [dual; primal]
        d, ρ, ρold, δ, nfacti = newton_system(x, r, λ, Fx, rhs, Jx, Jcx, Hx, ρold, δ, params, method, linsolve)
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
        # TODO: Can I do this?
        Mdλ = T(1e4)
        if norm(dλ) > Mdλ
          dλ .= dλ * Mdλ / norm(dλ)
        end
        λt .= λ + dλ
        F!(xt, Ft)
        c!(xt, ct)
      else
        curv = dot(dx, Hx * dx) + ρ * dot(dx, dx) + norm(Jx * dx)^2

        η, α, ϕx, Dϕ, nbki = line_search(x, r, λ, dx, dr, dλ,
                                        Fx, cx, Jx, Jcx, ϕ, xt, rt, λt,
                                        Ft, ct, F!, c!, ρ, δ, η,
                                        false, curv, merit, params)

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

      Jt = J(xt)
      Jct = Jc(xt)

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
        Jx = Jt
        Jcx = Jct
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

      with_logger(logger) do
        c = inner_iter == 1 ? "┌" : "│"
        #           #f   f(x)   Δt    ‖dual‖ ‖Fx-r‖ ‖c(x)‖ α      η      ρ      δ
        @info @sprintf("%s%-3d %6d  %8.2e  %8.2e  %8.2e  %8.2e  %8.2e  %8.2e  %8.2e  %8.2e  %8.2e  %6d  %6d\n",
                       c, iter, sum_counters(nls), fx, elapsed_time, normdualhat,
                      norm(primal[1:nequ]), norm(primal[nequ+1:end]),
                      α, η, ρ, δ, inner_iter, nbk)
      end
    end

    normdual   = normdualhat
    normprimal = normprimalhat

    elapsed_time = time() - start_time
    sd = ncon > 0 ? max(smax, norm(λ, 1) / ncon) / smax : one(T)
    first_order = max(normdual / sd, normprimal) <= ϵtol
    small_residual = check_small_residual && fx ≤ ϵf && norm(cx) ≤ ϵc
    if small_residual && !first_order
      # Ignoring bounds
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

    with_logger(logger) do
      #               #f   f(x)  Δt     ‖∇L‖ ‖Fx-r‖ ‖c(x)‖ α   η      ρ      δ  in_it
      @info @sprintf("└%-3d %6d  %8.2e  %8.2e  %8.2e  %8.2e  %8.2e  %8s  %8.2e  %8.2e  %8.2e  %6d  %6d\n",
              iter, sum_counters(nls), fx, elapsed_time, normdual,
              norm(primal[1:nequ]), norm(primal[nequ+1:end]),
              "-", η, ρ, δ, inner_iter, nbk)
    end
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

  #TODO: Fix infeasibility measure
  return GenericExecutionStats(status, nls, solution=x, objective=dot(Fx, Fx) / 2,
                               dual_feas=normdual, elapsed_time=elapsed_time,
                               solver_specific=Dict(:nbk => nbk,
                                                    :nfact => nfact,
                                                    :nlinsolve => nlinsolve,
                                                    :primal => norm(cx),
                                                    :multipliers => λ,
                                                    :internal_msg => internal_msg,
                                                   )
                              )
end

function newton_system(x, r, λ, Fx, rhs, Jx, Jcx, Hx, ρold, δ, params, method, linsolve)
  nvar = length(x)
  nequ = length(r)
  ncon = length(λ)

  T = eltype(x)
  nfact = 0

  #ρ = method == :LM ? max(100 * params[:ρmin], min(norm(Fx), norm(Fx)^2) ) : zero(eltype(x))
  ρ = zero(eltype(x))
  Bw = [Jx; Jcx]
  D = [ones(T, nequ);
       δ * ones(T, ncon)]
  # Only half the matrix is used
  B = [Hx  spzeros(T, nvar, nequ + ncon); # Bw';
       Bw  spdiagm(0 => -D)]

  function try_to_factorize(B)
    if linsolve == :ma57
      LDLT = Ma57(B)
      ma57_factorize(LDLT)
      success = LDLT.info.info[1] == 0 && LDLT.info.num_negative_eigs == nequ + ncon
      return LDLT, success
    elseif linsolve == :ma97
      LDLT = Ma97(B)
      ma97_factorize!(LDLT, matrix_type=:real_indef)
      success = LDLT.info.flag == 0 && LDLT.info.num_neg == nequ + ncon
      return LDLT, success
    elseif linsolve == :ldlfactorizations
      try
        LDLT = ldl(Matrix(Symmetric(B, :L)))
        λp = filter(x->x ≥ 0, LDLT.D)
        λm = filter(x->x ≤ 0, LDLT.D)
        pos_eig = count(LDLT.D .> params[:eig_tol])
        zer_eig = count(abs.(LDLT.D) .≤ params[:eig_tol])
        success = pos_eig == nvar && zer_eig == 0
        return LDLT, success
      catch
        return nothing, false
      end
    end
  end

  LDLT, success = try_to_factorize(B)
  nfact += 1

  if !success
    ρ = ρold == 0 ? params[:ρ0] : max(params[:ρmin], params[:κdec] * ρold)
    D = [ones(T, nequ);
         δ * ones(T, ncon)]
    B = [Hx + ρ * I    spzeros(T, nvar, nequ + ncon); # Bw';
         Bw  spdiagm(0 => -D)]
    LDLT, success = try_to_factorize(B)
    nfact += 1
    ρiter = 0
    while !success && ρ <= params[:ρmax]
      ρ = ρold == 0 ? params[:κlargeinc] * ρ : params[:κinc] * ρ
      if ρ <= params[:ρmax]
        B = [Hx + ρ * I    spzeros(T, nvar, nequ + ncon); # Bw';
             Bw  spdiagm(0 => -D)]
        LDLT, success = try_to_factorize(B)
        nfact += 1
      end
      ρiter += 1
    end
    if ρ <= params[:ρmax]
      ρold = ρ
    end
  end
  d = if linsolve == :ma57
    ma57_solve(LDLT, -rhs)
  elseif linsolve == :ma97
    ma97_solve(LDLT, -rhs)
  elseif linsolve == :ldlfactorizations
    @assert LDLT != nothing
    -(LDLT \ rhs)
  end

  #=
  dx = d[1:nvar]
  λv = []
  if dot(dx, Hx * dx + ρ * dx) ≤ 0
    λv = eigen(Matrix(Symmetric(B, :L))).values
  end

  @info("Newton",
        nfact,
        norm(Symmetric(B, :L) * d + rhs),
        dot(dx, Hx * dx + ρ * dx) + norm(Jx * dx)^2,
        ρ, δ,
        (nvar, nequ, ncon),
        count(λv .> 0),
        count(λv .< 0),
        LDLT.info.info[1],
        LDLT.info.num_negative_eigs
       )
  @info("eigen",
        eigen(Symmetric(Hx, :L) + Jx' * Jx).values[:]
       )
       =#

  return d, ρ, ρold, δ, nfact
end

function line_search(x, r, λ, dx, dr, dλ, Fx,
                     cx, Jx, Jcx, ϕ, xt, rt, λt, Ft, ct, F!, c!, ρ,
                     δ, η, trial_computed, curv, merit, params)
  T = eltype(x)
  # For comparison
  Dϕ₁ = -curv
  Dϕ₂ = -norm(cx + δ * dλ)^2
  #Dϕ = -curv - norm(cx + δ * dλ)^2 / δ
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
      println("ϕx = $ϕx")
      println("ϕt = $ϕt")
      println("Dϕ = $Dϕ")
      error("α too small")
    end
  end

  return η, α, ϕx, Dϕ, nbk
end

end # module
