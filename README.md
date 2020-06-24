# CaNNOLeS - Constrained and NoNlinear Optimizer of Least Squares

CaNNOLeS is a solver for equality-constrained nonlinear least-squares problems, i.e.,
optimization problems of the form

    min ¹/₂‖F(x)‖²      s. to     c(x) = 0.

It uses other JuliaSmoothOptimizers packages for development.
In particular, [NLPModels.jl](https://github.com/JuliaSmoothOptimizers/NLPModels.jl) is used for defining the problem, and [SolverTools](https://github.com/JuliaSmoothOptimizers/SolverTools.jl) for the output.
It also uses [HSL.jl](https://github.com/JuliaSmoothOptimizers/HSL.jl)'s `MA57` as main solver, but you can pass `linsolve=:ldlfactorizations` to use [LDLFactorizations.jl](https://github.com/JuliaSmoothOptimizers/LDLFactorizations.jl).

Cite as

> Orban, D., & Siqueira, A. S. (2019).
> A Regularization Method for Constrained Nonlinear Least Squares.
> Computational Optimization and Applications 76, 961–989 (2020).
> [10.1007/s10589-020-00201-2](https://doi.org/10.1007/s10589-020-00201-2)

Check [CITATION.bib](CITATION.bib) for bibtex.

## Installation

1. Follow [HSL.jl](https://github.com/JuliaSmoothOptimizers/HSL.jl)'s `MA57` installation if possible. Otherwise [LDLFactorizations.jl](https://github.com/JuliaSmoothOptimizers/LDLFactorizations.jl) will be used.
2. `pkg> add CaNNOLeS`

## Example

```julia
using CaNNOLeS, NLPModels

# Rosenbrock
nls = ADNLSModel(x -> [x[1] - 1; 10 * (x[2] - x[1]^2)], [-1.2; 1.0], 2)
stats = cannoles(nls)

# Constrained
nls = ADNLSModel(x -> [x[1] - 1; 10 * (x[2] - x[1]^2)], [-1.2; 1.0], 2
                 c=x->[x[1] * x[2] - 1], lcon=[0.0], ucon=[0.0])
stats = cannoles(nls)
```
