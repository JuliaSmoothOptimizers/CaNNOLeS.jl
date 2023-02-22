"""
    nls = MGH01_noFHess()

## Rosenbrock function in nonlinear least squares format modified to remove second-order information

    Source: Problem 1 in
    J.J. Moré, B.S. Garbow and K.E. Hillstrom,
    "Testing Unconstrained Optimization Software",
    ACM Transactions on Mathematical Software, vol. 7(1), pp. 17-41, 1981

```math
\\begin{aligned}
\\min \\quad & \\tfrac{1}{2}\\| F(x) \\|^2
\\end{aligned}
```
where
```math
F(x) = \\begin{bmatrix}
1 - x_1 \\\\
10 (x_2 - x_1^2)
\\end{bmatrix}.
```

Starting point: `[-1.2; 1]`.
"""
mutable struct MGH01_noFHess{T, S} <: AbstractNLSModel{T, S}
  meta::NLPModelMeta{T, S}
  nls_meta::NLSMeta{T, S}
  counters::NLSCounters
end

function MGH01_noFHess(::Type{T}) where {T}
  meta = NLPModelMeta{T, Vector{T}}(2, x0 = T[-1.2; 1], name = "MGH01_noFHess_manual")
  nls_meta = NLSMeta{T, Vector{T}}(2, 2, nnzj = 3, nnzh = 1) # nnzh should be 0 but is left as 1 to force an erroring situation if used

  return MGH01_noFHess(meta, nls_meta, NLSCounters())
end
MGH01_noFHess() = MGH01_noFHess(Float64)

function NLPModels.residual!(nls::MGH01_noFHess, x::AbstractVector, Fx::AbstractVector)
  @lencheck 2 x Fx
  increment!(nls, :neval_residual)
  Fx[1] = 1 - x[1]
  Fx[2] = 10 * (x[2] - x[1]^2)
  return Fx
end

# Jx = [-1  0; -20x₁  10]
function NLPModels.jac_structure_residual!(
  nls::MGH01_noFHess,
  rows::AbstractVector{<:Integer},
  cols::AbstractVector{<:Integer},
)
  @lencheck 3 rows cols
  rows[1] = 1
  cols[1] = 1
  rows[2] = 2
  cols[2] = 1
  rows[3] = 2
  cols[3] = 2
  return rows, cols
end

function NLPModels.jac_coord_residual!(nls::MGH01_noFHess, x::AbstractVector, vals::AbstractVector)
  @lencheck 2 x
  @lencheck 3 vals
  increment!(nls, :neval_jac_residual)
  vals[1] = -1
  vals[2] = -20x[1]
  vals[3] = 10
  return vals
end

function NLPModels.jprod_residual!(
  nls::MGH01_noFHess,
  x::AbstractVector,
  v::AbstractVector,
  Jv::AbstractVector,
)
  @lencheck 2 x v Jv
  increment!(nls, :neval_jprod_residual)
  Jv[1] = -v[1]
  Jv[2] = -20 * x[1] * v[1] + 10 * v[2]
  return Jv
end

function NLPModels.jtprod_residual!(
  nls::MGH01_noFHess,
  x::AbstractVector,
  v::AbstractVector,
  Jtv::AbstractVector,
)
  @lencheck 2 x v Jtv
  increment!(nls, :neval_jtprod_residual)
  Jtv[1] = -v[1] - 20 * x[1] * v[2]
  Jtv[2] = 10 * v[2]
  return Jtv
end
