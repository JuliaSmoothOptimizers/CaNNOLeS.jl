using HSL, LDLFactorizations

abstract type LinearSolverStruct end

mutable struct MA57Struct <: LinearSolverStruct
  factor :: Ma57
end

function MA57Struct(N, rows, cols, vals)
  MA57Struct(ma57_coord(N, rows, cols, vals))
end

mutable struct LDLFactStruct <: LinearSolverStruct
  rows :: Vector{Int}
  cols :: Vector{Int}
  vals :: Vector
  factor :: Union{LDLFactorizations.LDLFactorization,Nothing}
end

function LDLFactStruct(rows, cols, vals)
  LDLFactStruct(rows, cols, vals, nothing)
end
