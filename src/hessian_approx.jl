abstract type HessianStruct{Ti} end

import NLPModels.get_nnzh

"""
    get_nnzh(::HessianStruct)

Return number of nonzeros in the approximatation of the Hessian.
"""
function get_nnzh end

"""
    get_structure(::HessianStruct)

Return the structure of the approximatation of the Hessian.
"""
function get_nnzh end

struct Newton_noFHess{Ti} <: HessianStruct{Ti} end
Newton_noFHess(nls, ::Type{Ti}) where {Ti} = Newton_noFHess{Ti}()
get_nnzh(::Newton_noFHess) = 0
get_structure(::Newton_noFHess{Ti}) where {Ti} = Ti[], Ti[]

struct LM{Ti} <: HessianStruct{Ti} end
LM(nls, ::Type{Ti}) where {Ti} = LM{Ti}()
get_nnzh(::LM) = 0
get_structure(::LM{Ti}) where {Ti} = Ti[], Ti[]

struct Newton{Ti} <: HessianStruct{Ti}
  hsr_rows::Vector{Ti}
  hsr_cols::Vector{Ti}
end
function Newton(nls, ::Type{Ti}) where {Ti}
  hsr_rows, hsr_cols = hess_structure_residual(nls)
  Newton{Ti}(hsr_rows, hsr_cols)
end
get_nnzh(hessian_struct::Newton) = length(hessian_struct.hsr_rows)
get_structure(hessian_struct::Newton) = (hessian_struct.hsr_rows, hessian_struct.hsr_cols)

struct Newton_vanishing{Ti} <: HessianStruct{Ti}
  hsr_rows::Vector{Ti}
  hsr_cols::Vector{Ti}
end
function Newton_vanishing(nls, ::Type{Ti}) where {Ti}
  hsr_rows, hsr_cols = hess_structure_residual(nls)
  Newton{Ti}(hsr_rows, hsr_cols)
end
get_nnzh(hessian_struct::Newton_vanishing) = length(hessian_struct.hsr_rows)
get_structure(hessian_struct::Newton_vanishing) = (hessian_struct.hsr_rows, hessian_struct.hsr_cols)

"""
    update_newton_hessian!(::HessianStruct, nls, x, r, vals, Fx)

Update, if need for `method`, the top-left block with the non-zeros values of the Hessian of the residual.
For `method=:Newton_vanishing`, this update is skipped if `‖F(xᵏ)‖ ≤ 1e-8`.
"""
function update_newton_hessian!(::HessianStruct, args...) end

function update_newton_hessian!(::Newton, nls, x, r, vals, Fx)
  sI = 1:(nls.nls_meta.nnzh)
  @views hess_coord_residual!(nls, x, r, vals[sI])
end

function update_newton_hessian!(::Newton_vanishing, nls, x, r, vals, Fx)
  if dot(Fx, Fx) > 1e-8
    sI = 1:(nls.nls_meta.nnzh)
    @views hess_coord_residual!(nls, x, r, vals[sI])
  end
end
