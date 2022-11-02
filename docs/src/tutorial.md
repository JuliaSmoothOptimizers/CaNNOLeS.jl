# CaNNOLeS.jl Tutorial

## Contents

```@contents
Pages = ["tutorial.md"]
```

## Fine-tune CaNNOLeS

CaNNOLeS.jl exports the function `cannoles`:
```
   cannoles(nlp :: AbstractNLPModel; kwargs...)
```

Find below a list of the main options of `cannoles`.

### Tolerances on the problem

```
| Parameters           | Type          | Default         | Description                                        |
| -------------------- | ------------- | --------------- | -------------------------------------------------- |
| ϵtol                 | AbstractFloat | √eps(eltype(x)) | tolerance.                                         |
| unbounded_threshold  | AbstractFloat | -1e5            | below this threshold the problem is unbounded.     |
| max_f                | Integer       | 100000          | evaluation limit, e.g. `sum_counters(nls) > max_f` |
| max_time             | AbstractFloat | 30.             | maximum number of seconds.                         |
| max_inner            | Integer       | 10000           | maximum number of iterations.                      |
```

### Algorithmic parameters

```
| Parameters                  | Type           | Default           | Description                                        |
| --------------------------- | -------------- | ----------------- | -------------------------------------------------- |
| x                           | AbstractVector | copy(nls.meta.x0) | initial guess. |
| λ                           | AbstractVector | eltype(x)[]       | initial guess for the Lagrange mutlipliers. |
| method                      | Symbol         | :Newton           | method to compute direction, `:Newton`, `:LM`, `:Newton_noFHess`, or `:Newton_vanishing`. |
| linsolve                    | Symbol         | :ma57             | solver use to compute the factorization: `:ma57`, `:ma97`, `:ldlfactorizations` |
| check_small_residual        | Bool           | true              | |
| always_accept_extrapolation | Bool           | false             | |
| δdec                        | Real           | eltype(x)(0.1)    | |
```

## Examples

```@example ex1
using CaNNOLeS, ADNLPModels

# Rosenbrock
nls = ADNLSModel(x -> [x[1] - 1; 10 * (x[2] - x[1]^2)], [-1.2; 1.0], 2)
stats = cannoles(nls, ϵtol = 1e-5, x = ones(2))
```

```@example ex1
# Constrained
nls = ADNLSModel(
  x -> [x[1] - 1; 10 * (x[2] - x[1]^2)],
  [-1.2; 1.0],
  2,
  x -> [x[1] * x[2] - 1],
  [0.0],
  [0.0],
)
stats = cannoles(nls, max_time = 10.)
```
