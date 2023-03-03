using Pkg
Pkg.activate(".")
using CaNNOLeS, NLPModels, SolverCore
Pkg.status()

include("test/mgh01con.jl")
nls = MGH01CON()

stats, solver = GenericExecutionStats(nls), CaNNOLeSSolver(nls)

nls.meta.x0 .= zeros(2)
@allocated solve!(solver, nls, stats) # , x = x0, λ = λ)
@show @allocated solve!(solver, nls, stats) # , x = x0, λ = λ) # 0
reset!(nls); solve!(solver, nls, stats, verbose = 1)

nls.meta.x0 .= [-1.2; 1]
@allocated solve!(solver, nls, stats)
a = @allocated solve!(solver, nls, stats) # 0
@show a
reset!(nls); solve!(solver, nls, stats, verbose = 1)

#= TODO:
- Use updated SolverCore version (merge solver_specific PR + new release)
- Fix the initial Lagrange multiplier (right now it is [] and then we eval it)
=#

#=
using Profile
Profile.Allocs.clear()
Profile.Allocs.@profile sample_rate = 1 solve!(solver, nls, stats)
using PProf
PProf.Allocs.pprof(from_c = false)
=#