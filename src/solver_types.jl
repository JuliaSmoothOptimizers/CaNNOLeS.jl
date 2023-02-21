using HSL, LDLFactorizations

abstract type LinearSolverStruct end

"""
    success = try_to_factorize(LDLT::LinearSolverStruct, vals::AbstractVector, nvar::Integer, nequ::Integer, ncon::Integer, eig_tol::Real)

Compute the LDLt factorization of A = sparse(LDLT.rows, LDLT.cols, LDLT.vals, N, N) where `N = nvar + ncon + nequ` and return `true` in case of success.
"""
function try_to_factorize end

"""
    success = solve_ldl!(rhs::AbstractVector, factor::Union{Ma57, LDLFactorizations.LDLFactorization}, d::AbstractVector)

Compute the solution of `LDLt d = -rhs`.
"""
function solve_ldl! end

if isdefined(HSL, :libhsl_ma57)
  mutable struct MA57Struct <: LinearSolverStruct
    factor::Ma57
  end

  function MA57Struct(N, rows, cols, vals)
    MA57Struct(ma57_coord(N, rows, cols, vals))
  end

  get_vals(LDLT::MA57Struct) = LDLT.factor.vals
  function solve_ldl!(rhs::AbstractVector, factor::Ma57, d::AbstractVector)
    d .= ma57_solve(factor, rhs)
    d .= .-d
    return true
  end

  function try_to_factorize(
    LDLT::MA57Struct,
    vals::AbstractVector,
    nvar::Integer,
    nequ::Integer,
    ncon::Integer,
    eig_tol::Real,
  )
    ma57_factorize!(LDLT.factor)
    success = LDLT.factor.info.info[1] == 0 && LDLT.factor.info.num_negative_eigs == nequ + ncon
    return success
  end
else
  function MA57Struct(N, rows, cols, vals)
    error("MA57 not installed. See HSL.jl")
  end
end

mutable struct LDLFactStruct{T, Ti <: Integer} <: LinearSolverStruct
  rows::Vector{Ti}
  cols::Vector{Ti}
  vals::Vector{T}
  A::Symmetric{T, SparseMatrixCSC{T, Ti}}
  factor::LDLFactorizations.LDLFactorization{T, Ti, Ti, Ti}
end

function set_vals!(LDLT::LDLFactStruct{T, Ti}, vals::Vector{T}) where {T, Ti}
  LDLT.A.data.nzval .= zero(T)
  for i in eachindex(vals)
    LDLT.A.data[LDLT.cols[i], LDLT.rows[i]] += vals[i]
  end
  return LDLT
end

function LDLFactStruct(N, rows, cols, vals)
  A = Symmetric(triu(sparse(cols, rows, vals, N, N)), :U)
  factor = ldl_analyze(A)
  LDLFactStruct(rows, cols, vals, A, factor)
end

get_vals(LDLT::LinearSolverStruct) = LDLT.vals

function solve_ldl!(
  rhs::AbstractVector,
  factor::LDLFactorizations.LDLFactorization,
  d::AbstractVector,
)
  ldiv!(d, factor, rhs)
  d .= .-d
  return true
end

function try_to_factorize(
  LDLT::LDLFactStruct,
  vals::AbstractVector,
  nvar::Integer,
  nequ::Integer,
  ncon::Integer,
  eig_tol::Real,
)
  N = nvar + nequ + ncon
  set_vals!(LDLT, vals)
  ldl_factorize!(LDLT.A, LDLT.factor)
  pos_eig, zer_eig = 0, 0
  for i=1:N
    di = LDLT.factor.d[i]
    pos_eig += di > eig_tol
    zer_eig += abs(di) ≤ eig_tol
  end
  success = pos_eig == nvar && zer_eig == 0
  return success
end
