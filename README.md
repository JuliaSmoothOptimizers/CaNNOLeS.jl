# CaNNOLeS - Constrained and NoNlinear Optimizer of Least Squares

[![documentation](https://img.shields.io/badge/docs-stable-blue.svg)](https://JuliaSmoothOptimizers.github.io/CaNNOLeS.jl/stable)
[![documentation](https://img.shields.io/badge/docs-dev-blue.svg)](https://JuliaSmoothOptimizers.github.io/CaNNOLeS.jl/dev)
![CI](https://github.com/JuliaSmoothOptimizers/CaNNOLeS.jl/workflows/CI/badge.svg?branch=main)
[![Cirrus CI - Base Branch Build Status](https://img.shields.io/cirrus/github/JuliaSmoothOptimizers/CaNNOLeS.jl?logo=Cirrus%20CI)](https://cirrus-ci.com/github/JuliaSmoothOptimizers/CaNNOLeS.jl)
[![codecov](https://codecov.io/gh/JuliaSmoothOptimizers/CaNNOLeS.jl/branch/main/graph/badge.svg)](https://codecov.io/gh/JuliaSmoothOptimizers/CaNNOLeS.jl)
[![GitHub](https://img.shields.io/github/release/JuliaSmoothOptimizers/CaNNOLeS.svg?style=flat-square)](https://github.com/JuliaSmoothOptimizers/CaNNOLeS/releases)

CaNNOLeS is a solver for equality-constrained nonlinear least-squares problems, i.e.,
optimization problems of the form

    min ¹/₂‖F(x)‖²      s. to     c(x) = 0.

It uses other JuliaSmoothOptimizers packages for development.
In particular, [NLPModels.jl](https://github.com/JuliaSmoothOptimizers/NLPModels.jl) is used for defining the problem, and [SolverCore](https://github.com/JuliaSmoothOptimizers/SolverCore.jl) for the output.
It also uses [HSL.jl](https://github.com/JuliaSmoothOptimizers/HSL.jl)'s `MA57` as main solver, but you can pass `linsolve=:ldlfactorizations` to use [LDLFactorizations.jl](https://github.com/JuliaSmoothOptimizers/LDLFactorizations.jl).

Cite as

> Orban, D., & Siqueira, A. S.
> A Regularization Method for Constrained Nonlinear Least Squares.
> Computational Optimization and Applications 76, 961–989 (2020).
> [10.1007/s10589-020-00201-2](https://doi.org/10.1007/s10589-020-00201-2)

Check [CITATION.bib](CITATION.bib) for bibtex.

## Installation

1. Follow [HSL.jl](https://github.com/JuliaSmoothOptimizers/HSL.jl)'s `MA57` installation if possible. Otherwise [LDLFactorizations.jl](https://github.com/JuliaSmoothOptimizers/LDLFactorizations.jl) will be used.
2. `pkg> add CaNNOLeS`

## Examples

```julia
using CaNNOLeS, ADNLPModels

# Rosenbrock
nls = ADNLSModel(x -> [x[1] - 1; 10 * (x[2] - x[1]^2)], [-1.2; 1.0], 2)
stats = cannoles(nls)

# Constrained
nls = ADNLSModel(x -> [x[1] - 1; 10 * (x[2] - x[1]^2)], [-1.2; 1.0], 2
                 c=x->[x[1] * x[2] - 1], lcon=[0.0], ucon=[0.0])
stats = cannoles(nls)
```
