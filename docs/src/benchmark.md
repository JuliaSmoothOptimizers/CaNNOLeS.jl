With a JSO-compliant solver, such as CaNNOLeS, we can run the solver on a set of problems, explore the results, and compare to other JSO-compliant solvers using specialized benchmark tools. 
We are following here the tutorial in [SolverBenchmark.jl](https://juliasmoothoptimizers.github.io/SolverBenchmark.jl/v0.3/tutorial/) to run benchmarks on JSO-compliant solvers.
``` @example ex1
using NLSProblems, NLPModels
```

To test the implementation of CaNNOLeS, we use the package [NLSProblems.jl](https://github.com/JuliaSmoothOptimizers/NLSProblems.jl), which implements `NLSProblemsModel` an instance of `AbstractNLPModel`. 

``` @example ex1
using SolverBenchmark
```

Let us select equality-constrained problems from NLSProblems with a maximum of 10000 variables or constraints. After removing problems with fixed variables, examples with a constant objective, and infeasibility residuals, we are left with 82 problems.

``` @example ex1
problems = (NLSProblems.eval(problem)() for problem in filter(x -> x != :NLSProblems, names(NLSProblems)) )
```

We compare here CaNNOLeS with `tron` (Chih-Jen Lin and Jorge J. Moré, Newton's Method for Large Bound-Constrained Optimization Problems, SIAM J. Optim., 9(4), 1100–1127, 1999.), and `trunk` (A. R. Conn, N. I. M. Gould, and Ph. L. Toint (2000). Trust-Region Methods, volume 1 of MPS/SIAM Series on Optimization.) implemented in [JSOSolvers.jl](https://github.com/JuliaSmoothOptimizers/JSOSolvers.jl) on a subset of NLSProblems problems.
``` @example ex1
using CaNNOLeS, JSOSolvers
```
To make stopping conditions comparable, we set `tron`'s and `trunk`'s parameters `atol=0.0`, and `rtol=1e-5`.

``` @example ex1
#Same time limit for all the solvers
max_time = 1200. #20 minutes

solvers = Dict(
  :tron => nlp -> tron(
    nlp,
    atol = 0.0,
    rtol = 1e-5,
  ),
  :trunk => nlp -> trunk(
    nlp,
    atol = 0.0,
    rtol = 1e-5,
  ),
  :cannoles => nlp -> cannoles(
    nlp,
    atol = 0.0,
    rtol = 1e-5,
  ),
)

stats = bmark_solvers(solvers, problems, skipif = nls -> !NLPModels.unconstrained(nls))
```
The function `bmark_solvers` return a `Dict` of `DataFrames` with detailed information on the execution. This output can be saved in a data file.
``` @example ex1
using JLD2
@save "trunk_cannoles_$(string(length(problems))).jld2" stats
```
The result of the benchmark can be explored via tables,
```julia
pretty_stats(stats[:cannoles])
```
or it can also be used to make performance profiles.
``` @example ex1
using Plots
gr()

legend = Dict(
  :neval_obj => "number of f evals",
  :neval_residual => "number of F evals",
  :neval_cons => "number of c evals", 
  :neval_grad => "number of ∇f evals", 
  :neval_jac => "number of ∇c evals", 
  :neval_jprod => "number of ∇c*v evals", 
  :neval_jtprod  => "number of ∇cᵀ*v evals", 
  :neval_hess  => "number of ∇²f evals", 
  :elapsed_time => "elapsed time"
)
perf_title(col) = "Performance profile on NLSProblems w.r.t. $(string(legend[col]))"

styles = [:solid, :dash, :dot, :dashdot]

function print_pp_column(col::Symbol, stats)
  
  ϵ = minimum(minimum(filter(x -> x > 0, df[!, col])) for df in values(stats))
  first_order(df) = df.status .== :first_order
  unbounded(df) = df.status .== :unbounded
  solved(df) = first_order(df) .| unbounded(df)
  cost(df) = (max.(df[!, col], ϵ) + .!solved(df) .* Inf)

  p = performance_profile(
    stats, 
    cost, 
    title=perf_title(col), 
    legend=:bottomright, 
    linestyles=styles
  )
end

print_pp_column(:elapsed_time, stats) # with respect to time
```

``` @example ex1
print_pp_column(:neval_residual, stats) # with respect to number of residual function evaluations
```
