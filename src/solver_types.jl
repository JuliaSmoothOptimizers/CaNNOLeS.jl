using HSL, LDLFactorizations

abstract type LinearSolverStruct end

"""
    success = try_to_factorize(LDLT::LinearSolverStruct, nvar::Integer, nequ::Integer, ncon::Integer, eig_tol::Real)

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
  A::Symmetric{T, SparseMatrixCOO{T, Ti}}
  factor::Union{LDLFactorizations.LDLFactorization, Nothing}
end

function LDLFactStruct(N, rows, cols, vals)
  A = Symmetric(SparseMatrixCOO(N, N, cols, rows, vals), :U)
  LDLFactStruct(rows, cols, vals, A, nothing)
end

get_vals(LDLT::LinearSolverStruct) = LDLT.vals

function LDLFactStruct(rows, cols, vals)
  LDLFactStruct(rows, cols, vals, nothing)
end
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
  nvar::Integer,
  nequ::Integer,
  ncon::Integer,
  eig_tol::Real,
)
  N = nvar + nequ + ncon
  LDLT.A.data.vals .= LDLT.vals
  try
    M = ldl(Matrix(LDLT.A)) # allocate
    pos_eig, zer_eig = 0, 0
    for i=1:N
      di = M.D[i, i]
      pos_eig += di > eig_tol
      zer_eig += abs(di) ≤ eig_tol
    end
    success = pos_eig == nvar && zer_eig == 0
    LDLT.factor = M # allocate
    return success
  catch
    return false
  end
end
