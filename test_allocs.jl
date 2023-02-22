using Pkg
Pkg.activate(".")
using CaNNOLeS, ADNLPModels, NLPModels, SolverCore

#=
nls = ADNLSModel(
  x -> [x[1] - 1; 10 * (x[2] - x[1]^2)],
  [-1.2; 1.0],
  2,
  x -> [x[1] * x[2] - 1],
  [0.0],
  [0.0],
)
=#

include("mgh01con.jl")
nls = MGH01CON()

stats = GenericExecutionStats(nls)
solver = CaNNOLeSSolver(nls)
stats = cannoles(nls)
x0 = zeros(2)
λ = nls.meta.y0
@allocated solve!(solver, nls, stats) # , x = x0, λ = λ)
a = @allocated solve!(solver, nls, stats) # , x = x0, λ = λ) # 6720
@show a

#=
64 without set_solver_specific !! (Keyword !!)
=#

x0 = nls.meta.x0
@allocated solve!(solver, nls, stats) # , x = x0, λ = λ)
a = @allocated solve!(solver, nls, stats) # , x = x0, λ = λ) # 45248
@show a

using Profile
Profile.Allocs.clear()
Profile.Allocs.@profile sample_rate = 1 solve!(solver, nls, stats, x = x0, λ = λ)
using PProf
PProf.Allocs.pprof(from_c = false)

#=
a = 3856
a = 34896
=#