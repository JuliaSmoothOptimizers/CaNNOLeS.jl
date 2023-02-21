using HSL, LDLFactorizations

abstract type LinearSolverStruct end

"""
    success = try_to_factorize(LDLT::LinearSolverStruct, nvar::Integer, nequ::Integer, ncon::Integer, eig_tol::Real)

Compute the LDLt factorization of A = sparse(LDLT.rows, LDLT.cols, LDLT.vals, N, N) where `N = nvar + ncon + nequ` and return `true` in case of success.
"""
function try_to_factorize end

if isdefined(HSL, :libhsl_ma57)
  mutable struct MA57Struct <: LinearSolverStruct
    factor::Ma57
  end

  function MA57Struct(N, rows, cols, vals)
    MA57Struct(ma57_coord(N, rows, cols, vals))
  end

  get_vals(LDLT::MA57Struct) = LDLT.factor.vals
  function solve_ldl!(rhs::AbstractVector, factor::Ma57, d::AbstractVector)
    d .= ma57_solve(factor, -rhs)
    return d, true
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

mutable struct LDLFactStruct <: LinearSolverStruct
  rows::Vector{Int}
  cols::Vector{Int}
  vals::Vector
  factor::Union{LDLFactorizations.LDLFactorization, Nothing}
end

function LDLFactStruct(rows, cols, vals)
  LDLFactStruct(rows, cols, vals, nothing)
end

get_vals(LDLT::LinearSolverStruct) = LDLT.vals

solve_ldl!(::AbstractVector, factor::Nothing, ::AbstractVector) =
  error("LDLt factorization failed.")
function solve_ldl!(
  rhs::AbstractVector,
  factor::LDLFactorizations.LDLFactorization,
  d::AbstractVector,
)
  d .= -(factor \ rhs)
  return d, true
end

function try_to_factorize(
  LDLT::LDLFactStruct,
  nvar::Integer,
  nequ::Integer,
  ncon::Integer,
  eig_tol::Real,
)
  N = nvar + nequ + ncon
  A = SparseMatrixCOO(N, N, LDLT.rows, LDLT.cols, LDLT.vals)
  A = Matrix(Symmetric(A, :L))
  try
    M = ldl(A)
    D = diag(M.D)
    pos_eig = count(D .> eig_tol)
    zer_eig = count(abs.(D) .≤ eig_tol)
    success = pos_eig == nvar && zer_eig == 0
    LDLT.factor = M
    return success
  catch
    return false
  end
end
