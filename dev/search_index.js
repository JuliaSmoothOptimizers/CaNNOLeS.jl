var documenterSearchIndex = {"docs":
[{"location":"reference/#Reference","page":"Reference","title":"Reference","text":"","category":"section"},{"location":"reference/","page":"Reference","title":"Reference","text":"​","category":"page"},{"location":"reference/#Contents","page":"Reference","title":"Contents","text":"","category":"section"},{"location":"reference/","page":"Reference","title":"Reference","text":"​","category":"page"},{"location":"reference/","page":"Reference","title":"Reference","text":"Pages = [\"reference.md\"]","category":"page"},{"location":"reference/","page":"Reference","title":"Reference","text":"​","category":"page"},{"location":"reference/#Index","page":"Reference","title":"Index","text":"","category":"section"},{"location":"reference/","page":"Reference","title":"Reference","text":"​","category":"page"},{"location":"reference/","page":"Reference","title":"Reference","text":"Pages = [\"reference.md\"]","category":"page"},{"location":"reference/","page":"Reference","title":"Reference","text":"​","category":"page"},{"location":"reference/","page":"Reference","title":"Reference","text":"Modules = [CaNNOLeS]","category":"page"},{"location":"reference/#CaNNOLeS.CaNNOLeSSolver","page":"Reference","title":"CaNNOLeS.CaNNOLeSSolver","text":"cannoles(nls)\n\nImplementation of a solver for Nonlinear Least Squares with nonlinear constraints.\n\nmin   f(x) = ¹₂F(x)²   st  c(x) = 0\n\nFor advanced usage, first define a CaNNOLeSSolver to preallocate the memory used in the algorithm, and then call solve!:\n\nsolver = CaNNOLeSSolver(nls; linsolve = :ma57)\nsolve!(solver, nls; kwargs...)\n\nor even pre-allocate the output:\n\nstats = GenericExecutionStats(nls)\nsolve!(solver, nls, stats; kwargs...)\n\nArguments\n\nnls :: AbstractNLSModel: nonlinear least-squares model created using NLPModels.\n\nKeyword arguments\n\nx::AbstractVector = nls.meta.x0: the initial guess;\nλ::AbstractVector = eltype(x)[]: the initial Lagrange multiplier;\nmethod::Symbol = :Newton: available methods :Newton, :LM, :Newton_noFHess, and :Newton_vanishing;\nlinsolve::Symbol = :ma57: solver to compute LDLt factorization. Available methods are: :ma57, :ldlfactorizations;\nmax_eval::Real = 100000: maximum number of evaluations computed by neval_residual(nls) + neval_cons(nls);\nmax_time::Float64 = 30.0: maximum time limit in seconds;\nmax_inner::Int = 10000: maximum number of inner iterations;\nϵtol::Real = √eps(eltype(x)): stopping tolerance;\nFatol::T = √eps(T): absolute tolerance on the residual;\nFrtol::T = eps(T): relative tolerance on the residual, the algorithm stops when ‖F(xᵏ)‖ ≤ Fatol + Frtol * ‖F(x⁰)‖  and ‖c(xᵏ)‖∞ ≤ √ϵtol;\nverbose::Int = 0: if > 0, display iteration details every verbose iteration;\nalways_accept_extrapolation::Bool = false: if true, run even if the extrapolation step fails;\nδdec::Real = eltype(x)(0.1): reducing factor on the parameter δ.\n\nThe algorithm stops when c(xᵏ)  ϵtol and F(xᵏ)ᵀF(xᵏ) - c(xᵏ)ᵀλᵏ  ϵtol * max(1 λᵏ  100ncon).\n\nOutput\n\nThe value returned is a GenericExecutionStats, see SolverCore.jl.\n\nCallback\n\nThe callback is called at each iteration. The expected signature of the callback is callback(nls, solver, stats), and its output is ignored. Changing any of the input arguments will affect the subsequent iterations. In particular, setting stats.status = :user will stop the algorithm. All relevant information should be available in nlp and solver. Notably, you can access, and modify, the following:\n\nsolver.x: current iterate;\nsolver.cx: current value of the constraints at x;\nstats: structure holding the output of the algorithm (GenericExecutionStats), which contains, among other things:\nstats.solution: current iterate;\nstats.multipliers: current Lagrange multipliers wrt to the constraints;\nstats.primal_feas:the primal feasibility norm at solution;\nstats.dual_feas: the dual feasibility norm at solution;\nstats.iter: current iteration counter;\nstats.objective: current objective function value;\nstats.status: current status of the algorithm. Should be :unknown unless the algorithm has attained a stopping criterion. Changing this to anything will stop the algorithm, but you should use :user to properly indicate the intention.\nstats.elapsed_time: elapsed time in seconds.\n\nExamples\n\nusing CaNNOLeS, ADNLPModels\nnls = ADNLSModel(x -> x, ones(3), 3)\nstats = cannoles(nls, linsolve = :ldlfactorizations, verbose = 0)\nstats\n\n# output\n\n\"Execution stats: first-order stationary\"\n\nusing CaNNOLeS, ADNLPModels\nnls = ADNLSModel(x -> x, ones(3), 3)\nsolver = CaNNOLeSSolver(nls, linsolve = :ldlfactorizations)\nstats = solve!(solver, nls, verbose = 0)\nstats\n\n# output\n\n\"Execution stats: first-order stationary\"\n\n\n\n\n\n","category":"type"},{"location":"reference/#CaNNOLeS.ParamCaNNOLeS","page":"Reference","title":"CaNNOLeS.ParamCaNNOLeS","text":"ParamCaNNOLeS(eig_tol,δmin,κdec,κinc,κlargeinc,ρ0,ρmax,ρmin,γA)\nParamCaNNOLeS(::Type{T})\n\nStructure containing all the parameters used in the cannoles call.\n\n\n\n\n\n","category":"type"},{"location":"reference/#CaNNOLeS._check_available_method-Tuple{Symbol}","page":"Reference","title":"CaNNOLeS._check_available_method","text":"_check_available_method(method::Symbol)\n\nReturn an error if method is not in CaNNOLeS.avail_mtds\n\n\n\n\n\n","category":"method"},{"location":"reference/#CaNNOLeS.cannoles-Tuple{NLPModels.AbstractNLSModel}","page":"Reference","title":"CaNNOLeS.cannoles","text":"cannoles(nls)\n\nImplementation of a solver for Nonlinear Least Squares with nonlinear constraints.\n\nmin   f(x) = ¹₂F(x)²   st  c(x) = 0\n\nFor advanced usage, first define a CaNNOLeSSolver to preallocate the memory used in the algorithm, and then call solve!:\n\nsolver = CaNNOLeSSolver(nls; linsolve = :ma57)\nsolve!(solver, nls; kwargs...)\n\nor even pre-allocate the output:\n\nstats = GenericExecutionStats(nls)\nsolve!(solver, nls, stats; kwargs...)\n\nArguments\n\nnls :: AbstractNLSModel: nonlinear least-squares model created using NLPModels.\n\nKeyword arguments\n\nx::AbstractVector = nls.meta.x0: the initial guess;\nλ::AbstractVector = eltype(x)[]: the initial Lagrange multiplier;\nmethod::Symbol = :Newton: available methods :Newton, :LM, :Newton_noFHess, and :Newton_vanishing;\nlinsolve::Symbol = :ma57: solver to compute LDLt factorization. Available methods are: :ma57, :ldlfactorizations;\nmax_eval::Real = 100000: maximum number of evaluations computed by neval_residual(nls) + neval_cons(nls);\nmax_time::Float64 = 30.0: maximum time limit in seconds;\nmax_inner::Int = 10000: maximum number of inner iterations;\nϵtol::Real = √eps(eltype(x)): stopping tolerance;\nFatol::T = √eps(T): absolute tolerance on the residual;\nFrtol::T = eps(T): relative tolerance on the residual, the algorithm stops when ‖F(xᵏ)‖ ≤ Fatol + Frtol * ‖F(x⁰)‖  and ‖c(xᵏ)‖∞ ≤ √ϵtol;\nverbose::Int = 0: if > 0, display iteration details every verbose iteration;\nalways_accept_extrapolation::Bool = false: if true, run even if the extrapolation step fails;\nδdec::Real = eltype(x)(0.1): reducing factor on the parameter δ.\n\nThe algorithm stops when c(xᵏ)  ϵtol and F(xᵏ)ᵀF(xᵏ) - c(xᵏ)ᵀλᵏ  ϵtol * max(1 λᵏ  100ncon).\n\nOutput\n\nThe value returned is a GenericExecutionStats, see SolverCore.jl.\n\nCallback\n\nThe callback is called at each iteration. The expected signature of the callback is callback(nls, solver, stats), and its output is ignored. Changing any of the input arguments will affect the subsequent iterations. In particular, setting stats.status = :user will stop the algorithm. All relevant information should be available in nlp and solver. Notably, you can access, and modify, the following:\n\nsolver.x: current iterate;\nsolver.cx: current value of the constraints at x;\nstats: structure holding the output of the algorithm (GenericExecutionStats), which contains, among other things:\nstats.solution: current iterate;\nstats.multipliers: current Lagrange multipliers wrt to the constraints;\nstats.primal_feas:the primal feasibility norm at solution;\nstats.dual_feas: the dual feasibility norm at solution;\nstats.iter: current iteration counter;\nstats.objective: current objective function value;\nstats.status: current status of the algorithm. Should be :unknown unless the algorithm has attained a stopping criterion. Changing this to anything will stop the algorithm, but you should use :user to properly indicate the intention.\nstats.elapsed_time: elapsed time in seconds.\n\nExamples\n\nusing CaNNOLeS, ADNLPModels\nnls = ADNLSModel(x -> x, ones(3), 3)\nstats = cannoles(nls, linsolve = :ldlfactorizations, verbose = 0)\nstats\n\n# output\n\n\"Execution stats: first-order stationary\"\n\nusing CaNNOLeS, ADNLPModels\nnls = ADNLSModel(x -> x, ones(3), 3)\nsolver = CaNNOLeSSolver(nls, linsolve = :ldlfactorizations)\nstats = solve!(solver, nls, verbose = 0)\nstats\n\n# output\n\n\"Execution stats: first-order stationary\"\n\n\n\n\n\n\n\n","category":"method"},{"location":"reference/#CaNNOLeS.dual_scaling-Union{Tuple{T}, Tuple{AbstractVector{T}, T}} where T","page":"Reference","title":"CaNNOLeS.dual_scaling","text":"sd = dual_scaling(λ::AbstractVector{T}, smax::T)\n\nReturn the dual scaling on the residual, so that the algorithm stops when max(normdual / sd, normprimal) <= ϵtol. Return 1 if the problem has no constraints.\n\n\n\n\n\n","category":"method"},{"location":"reference/#CaNNOLeS.newton_system!-Union{Tuple{T}, Tuple{AbstractVector{T}, Integer, Integer, Integer, AbstractVector{T}, CaNNOLeS.LinearSolverStruct, T, CaNNOLeS.ParamCaNNOLeS{T}}} where T","page":"Reference","title":"CaNNOLeS.newton_system!","text":"newton_system!(d, nvar, nequ, ncon, rhs, LDLT, ρold, params)\n\nCompute an LDLt factorization of the (nvar + nequ + ncon)-square matrix for the Newton system contained in LDLT, i.e., sparse(LDLT.rows, LDLT.cols, LDLT.vals, N, N). If the factorization fails, a new factorization is attempted with an increased value for the regularization ρ as long as it is smaller than params.ρmax. The factorization is then used to solve the linear system whose right-hand side is rhs.\n\nOutput\n\nd: the solution of the linear system;\nsolve_success: true if the usage of the LDLt factorization is successful;\nρ: the value of the regularization parameter used in the factorization;\nρold: the value of the regularization parameter used in the previous successful factorization, or 0 if this is the first one;\nnfact: the number of factorization attempts.\n\n\n\n\n\n","category":"method"},{"location":"reference/#CaNNOLeS.optimality_check_small_residual!-Union{Tuple{V}, Tuple{T}, Tuple{Krylov.CglsSolver{T, T, V}, V, V, V, V, V, V, Any, Any, V}} where {T, V}","page":"Reference","title":"CaNNOLeS.optimality_check_small_residual!","text":"normprimal, normdual = optimality_check_small_residual!(cgls_solver, r, λ, dual, primal, Fx, cx, Jx, Jcx, Jxtr)\n\nCompute the norm of the primal and dual residuals. The values of r, Jxtr, λ, primal and dual are updated.\n\n\n\n\n\n","category":"method"},{"location":"reference/#CaNNOLeS.try_to_factorize","page":"Reference","title":"CaNNOLeS.try_to_factorize","text":"success = try_to_factorize(LDLT::LinearSolverStruct, nvar::Integer, nequ::Integer, ncon::Integer, eig_tol::Real)\n\nCompute the LDLt factorization of A = sparse(LDLT.rows, LDLT.cols, LDLT.vals, N, N) where N = nvar + ncon + nequ and return true in case of success.\n\n\n\n\n\n","category":"function"},{"location":"benchmark/","page":"Benchmark","title":"Benchmark","text":"With a JSO-compliant solver, such as CaNNOLeS, we can run the solver on a set of problems, explore the results, and compare to other JSO-compliant solvers using specialized benchmark tools.  We are following here the tutorial in SolverBenchmark.jl to run benchmarks on JSO-compliant solvers.","category":"page"},{"location":"benchmark/","page":"Benchmark","title":"Benchmark","text":"using NLSProblems, NLPModels","category":"page"},{"location":"benchmark/","page":"Benchmark","title":"Benchmark","text":"To test the implementation of CaNNOLeS, we use the package NLSProblems.jl, which implements NLSProblemsModel an instance of AbstractNLPModel. ","category":"page"},{"location":"benchmark/","page":"Benchmark","title":"Benchmark","text":"using SolverBenchmark","category":"page"},{"location":"benchmark/","page":"Benchmark","title":"Benchmark","text":"Let us select equality-constrained problems from NLSProblems with a maximum of 10000 variables or constraints. After removing problems with fixed variables, examples with a constant objective, and infeasibility residuals, we are left with 82 problems.","category":"page"},{"location":"benchmark/","page":"Benchmark","title":"Benchmark","text":"problems = (NLSProblems.eval(problem)() for problem in filter(x -> x != :NLSProblems, names(NLSProblems)) )","category":"page"},{"location":"benchmark/","page":"Benchmark","title":"Benchmark","text":"We compare here CaNNOLeS with tron (Chih-Jen Lin and Jorge J. Moré, Newton's Method for Large Bound-Constrained Optimization Problems, SIAM J. Optim., 9(4), 1100–1127, 1999.), and trunk (A. R. Conn, N. I. M. Gould, and Ph. L. Toint (2000). Trust-Region Methods, volume 1 of MPS/SIAM Series on Optimization.) implemented in JSOSolvers.jl on a subset of NLSProblems problems.","category":"page"},{"location":"benchmark/","page":"Benchmark","title":"Benchmark","text":"using CaNNOLeS, JSOSolvers","category":"page"},{"location":"benchmark/","page":"Benchmark","title":"Benchmark","text":"To make stopping conditions comparable, we set tron's and trunk's parameters atol=0.0, and rtol=1e-5.","category":"page"},{"location":"benchmark/","page":"Benchmark","title":"Benchmark","text":"#Same time limit for all the solvers\nmax_time = 1200. #20 minutes\n\nsolvers = Dict(\n  :tron => nlp -> tron(\n    nlp,\n    atol = 0.0,\n    rtol = 1e-5,\n  ),\n  :trunk => nlp -> trunk(\n    nlp,\n    atol = 0.0,\n    rtol = 1e-5,\n  ),\n  :cannoles => nlp -> cannoles(\n    nlp,\n    ϵtol = 1e-5,\n  ),\n)\n\nstats = bmark_solvers(solvers, problems, skipif = nls -> !NLPModels.unconstrained(nls))","category":"page"},{"location":"benchmark/","page":"Benchmark","title":"Benchmark","text":"The function bmark_solvers return a Dict of DataFrames with detailed information on the execution. This output can be saved in a data file.","category":"page"},{"location":"benchmark/","page":"Benchmark","title":"Benchmark","text":"using JLD2\n@save \"trunk_cannoles_$(string(length(problems))).jld2\" stats","category":"page"},{"location":"benchmark/","page":"Benchmark","title":"Benchmark","text":"The result of the benchmark can be explored via tables,","category":"page"},{"location":"benchmark/","page":"Benchmark","title":"Benchmark","text":"pretty_stats(stats[:cannoles])","category":"page"},{"location":"benchmark/","page":"Benchmark","title":"Benchmark","text":"or it can also be used to make performance profiles.","category":"page"},{"location":"benchmark/","page":"Benchmark","title":"Benchmark","text":"using Plots\ngr()\n\nlegend = Dict(\n  :neval_obj => \"number of f evals\",\n  :neval_residual => \"number of F evals\",\n  :neval_cons => \"number of c evals\", \n  :neval_grad => \"number of ∇f evals\", \n  :neval_jac => \"number of ∇c evals\", \n  :neval_jprod => \"number of ∇c*v evals\", \n  :neval_jtprod  => \"number of ∇cᵀ*v evals\", \n  :neval_hess  => \"number of ∇²f evals\", \n  :elapsed_time => \"elapsed time\"\n)\nperf_title(col) = \"Performance profile on NLSProblems w.r.t. $(string(legend[col]))\"\n\nstyles = [:solid, :dash, :dot, :dashdot]\n\nfunction print_pp_column(col::Symbol, stats)\n  \n  ϵ = minimum(minimum(filter(x -> x > 0, df[!, col])) for df in values(stats))\n  first_order(df) = df.status .== :first_order\n  unbounded(df) = df.status .== :unbounded\n  solved(df) = first_order(df) .| unbounded(df)\n  cost(df) = (max.(df[!, col], ϵ) + .!solved(df) .* Inf)\n\n  p = performance_profile(\n    stats, \n    cost, \n    title=perf_title(col), \n    legend=:bottomright, \n    linestyles=styles\n  )\nend\n\nprint_pp_column(:elapsed_time, stats) # with respect to time","category":"page"},{"location":"benchmark/","page":"Benchmark","title":"Benchmark","text":"print_pp_column(:neval_residual, stats) # with respect to number of residual function evaluations","category":"page"},{"location":"#CaNNOLeS-Constrained-and-NoNlinear-Optimizer-of-Least-Squares","page":"Home","title":"CaNNOLeS - Constrained and NoNlinear Optimizer of Least Squares","text":"","category":"section"},{"location":"","page":"Home","title":"Home","text":"(Image: documentation) (Image: documentation) (Image: CI) (Image: Cirrus CI - Base Branch Build Status) (Image: codecov) (Image: GitHub)","category":"page"},{"location":"","page":"Home","title":"Home","text":"CaNNOLeS is a solver for equality-constrained nonlinear least-squares problems, i.e., optimization problems of the form","category":"page"},{"location":"","page":"Home","title":"Home","text":"min ¹/₂‖F(x)‖²      s. to     c(x) = 0.","category":"page"},{"location":"","page":"Home","title":"Home","text":"It uses other JuliaSmoothOptimizers packages for development. In particular, NLPModels.jl is used for defining the problem, and SolverCore for the output. It also uses HSL.jl's MA57 as main solver, but you can pass linsolve=:ldlfactorizations to use LDLFactorizations.jl.","category":"page"},{"location":"","page":"Home","title":"Home","text":"Cite as","category":"page"},{"location":"","page":"Home","title":"Home","text":"Orban, D., & Siqueira, A. S. A Regularization Method for Constrained Nonlinear Least Squares. Computational Optimization and Applications 76, 961–989 (2020). 10.1007/s10589-020-00201-2","category":"page"},{"location":"","page":"Home","title":"Home","text":"Check CITATION.bib for bibtex.","category":"page"},{"location":"#Installation","page":"Home","title":"Installation","text":"","category":"section"},{"location":"","page":"Home","title":"Home","text":"Follow HSL.jl's MA57 installation if possible. Otherwise LDLFactorizations.jl will be used.\npkg> add CaNNOLeS","category":"page"},{"location":"#Examples","page":"Home","title":"Examples","text":"","category":"section"},{"location":"","page":"Home","title":"Home","text":"using CaNNOLeS, ADNLPModels\n\n# Rosenbrock\nnls = ADNLSModel(x -> [x[1] - 1; 10 * (x[2] - x[1]^2)], [-1.2; 1.0], 2)\nstats = cannoles(nls)","category":"page"},{"location":"","page":"Home","title":"Home","text":"# Constrained\nnls = ADNLSModel(\n  x -> [x[1] - 1; 10 * (x[2] - x[1]^2)],\n  [-1.2; 1.0],\n  2,\n  x -> [x[1] * x[2] - 1],\n  [0.0],\n  [0.0],\n)\nstats = cannoles(nls)","category":"page"},{"location":"#Bug-reports-and-discussions","page":"Home","title":"Bug reports and discussions","text":"","category":"section"},{"location":"","page":"Home","title":"Home","text":"If you think you found a bug, feel free to open an issue. Focused suggestions and requests can also be opened as issues. Before opening a pull request, start an issue or a discussion on the topic, please.","category":"page"},{"location":"","page":"Home","title":"Home","text":"If you want to ask a question not suited for a bug report, feel free to start a discussion here. This forum is for general discussion about this repository and the JuliaSmoothOptimizers, so questions about any of our packages are welcome.","category":"page"},{"location":"tutorial/#CaNNOLeS.jl-Tutorial","page":"Tutorial","title":"CaNNOLeS.jl Tutorial","text":"","category":"section"},{"location":"tutorial/#Contents","page":"Tutorial","title":"Contents","text":"","category":"section"},{"location":"tutorial/","page":"Tutorial","title":"Tutorial","text":"Pages = [\"tutorial.md\"]","category":"page"},{"location":"tutorial/#Fine-tune-CaNNOLeS","page":"Tutorial","title":"Fine-tune CaNNOLeS","text":"","category":"section"},{"location":"tutorial/","page":"Tutorial","title":"Tutorial","text":"CaNNOLeS.jl exports the function cannoles:","category":"page"},{"location":"tutorial/","page":"Tutorial","title":"Tutorial","text":"   cannoles(nlp :: AbstractNLPModel; kwargs...)","category":"page"},{"location":"tutorial/","page":"Tutorial","title":"Tutorial","text":"Find below a list of the main options of cannoles.","category":"page"},{"location":"tutorial/#Tolerances-on-the-problem","page":"Tutorial","title":"Tolerances on the problem","text":"","category":"section"},{"location":"tutorial/","page":"Tutorial","title":"Tutorial","text":"| Parameters           | Type          | Default         | Description                                        |\n| -------------------- | ------------- | --------------- | -------------------------------------------------- |\n| ϵtol                 | AbstractFloat | √eps(eltype(x)) | tolerance.                                         |\n| unbounded_threshold  | AbstractFloat | -1e5            | below this threshold the problem is unbounded.     |\n| max_eval             | Integer       | 100000          | evaluation limit, e.g. `neval_residual(nls) + neval_cons(nls) > max_eval` |\n| max_time             | AbstractFloat | 30.             | maximum number of seconds.                         |\n| max_inner            | Integer       | 10000           | maximum number of iterations.                      |","category":"page"},{"location":"tutorial/#Algorithmic-parameters","page":"Tutorial","title":"Algorithmic parameters","text":"","category":"section"},{"location":"tutorial/","page":"Tutorial","title":"Tutorial","text":"| Parameters                  | Type           | Default           | Description                                        |\n| --------------------------- | -------------- | ----------------- | -------------------------------------------------- |\n| x                           | AbstractVector | copy(nls.meta.x0) | initial guess. |\n| λ                           | AbstractVector | eltype(x)[]       | initial guess for the Lagrange mutlipliers. |\n| method                      | Symbol         | :Newton           | method to compute direction, `:Newton`, `:LM`, `:Newton_noFHess`, or `:Newton_vanishing`. |\n| linsolve                    | Symbol         | :ma57             | solver use to compute the factorization: `:ma57`, `:ma97`, `:ldlfactorizations` |\n| check_small_residual        | Bool           | false             | |\n| always_accept_extrapolation | Bool           | false             | |\n| δdec                        | Real           | eltype(x)(0.1)    | |","category":"page"},{"location":"tutorial/#Examples","page":"Tutorial","title":"Examples","text":"","category":"section"},{"location":"tutorial/","page":"Tutorial","title":"Tutorial","text":"using CaNNOLeS, ADNLPModels\n\n# Rosenbrock\nnls = ADNLSModel(x -> [x[1] - 1; 10 * (x[2] - x[1]^2)], [-1.2; 1.0], 2)\nstats = cannoles(nls, ϵtol = 1e-5, x = ones(2))","category":"page"},{"location":"tutorial/","page":"Tutorial","title":"Tutorial","text":"# Constrained\nnls = ADNLSModel(\n  x -> [x[1] - 1; 10 * (x[2] - x[1]^2)],\n  [-1.2; 1.0],\n  2,\n  x -> [x[1] * x[2] - 1],\n  [0.0],\n  [0.0],\n)\nstats = cannoles(nls, max_time = 10.)","category":"page"}]
}
