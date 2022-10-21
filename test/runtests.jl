# stdlib
using Logging, Test

# JSO packages
using ADNLPModels, NLPModels, SolverCore

# this package
using CaNNOLeS

mutable struct DummyModel{T, S} <: AbstractNLSModel{T, S}
  meta::NLPModelMeta{T, S}
end

nls = ADNLSModel(x -> x, zeros(5), 5, zeros(5), ones(5))
@test_throws ErrorException("Problem has inequalities, can't solve it") cannoles(nls)

nls = DummyModel(NLPModelMeta(1, minimize = false))
@test_throws ErrorException("CaNNOLeS only works for minimization problem") cannoles(nls)

function cannoles_tests()
  F_linear(x) = [x[1] - 2; x[2] - 3]
  F_Rosen(x) = [x[1] - 1; 10 * (x[2] - x[1]^2)]
  F_larger(x, n) = [[10 * (x[i + 1] - x[i]^2) for i = 1:(n - 1)]; [x[i] - 1 for i = 1:(n - 1)]]
  F_under(x, n) = [x[1] - x[i] for i = 2:n]

  c_linear(x) = [sum(x) - 1]
  c_quad(x) = [sum(x .^ 2) - 5; prod(x) - 2]

  @testset "Basic unconstrained problems" begin
    n = 10
    for (F, x0, xf) in [
      (F_linear, -ones(2), [2.0; 3.0])
      (F_Rosen, [-1.2; 1.0], ones(2))
      (x -> F_larger(x, n), 0.9 * ones(n), ones(n)) # It has other local solutions
      [(x -> F_under(x, n), i * ones(n), i * ones(n)) for i = 1:5]
    ]
      nls = ADNLSModel(F, x0, length(F(x0)))
      stats = with_logger(NullLogger()) do
        cannoles(nls, linsolve = :ldlfactorizations)
      end
      @test isapprox(stats.solution, xf, atol = 1e-4)
    end
  end

  @testset "Basic constrained problems" begin
    n = 10
    for (F, c, x0, xf) in [
      (F_linear, c_linear, -ones(2), [0.0; 1.0])
      (F_Rosen, c_linear, [-1.2; 1.0], [0.6188; 0.3812])
      (x -> F_under(x, n), c_linear, [1.0j for j = 1:n] / n, (1 / n) * ones(n))
      (F_linear, c_quad, [0.9; 1.9], [1.0; 2.0])
      (F_Rosen, c_quad, [0.9; 1.9], [1.0; 2.0])
      (x -> F_larger(x, 3), c_quad, [0.5; 1.0; 1.5], [1.0647; 1.215; 1.546])
    ]
      m = length(c(x0))
      nls = ADNLSModel(F, x0, length(F(x0)), c, zeros(m), zeros(m))
      stats = with_logger(NullLogger()) do
        cannoles(nls, linsolve = :ldlfactorizations)
      end
      @test isapprox(stats.solution, xf, atol = 1e-4)
    end
  end

  @testset "Multiprecision" begin
    for T in (Float16, Float32, Float64, BigFloat)
      nls = ADNLSModel(F_Rosen, T[-1.2; 1.0], 2, c_linear, T[0.0], T[0.0])
      stats = with_logger(NullLogger()) do
        cannoles(nls, x = T[-1.2; 1.0], linsolve = :ldlfactorizations)
      end
      @test isapprox(stats.solution, [0.6188; 0.3812], atol = max(1e-4, eps(T)^T(0.25)))
    end
  end
end

@testset "Re-solve with a different initial guess" begin
  nls = ADNLSModel(
    x -> [x[1] - 1],
    [-1.2; 1.0],
    1,
    x -> [10 * (x[2] - x[1]^2)],
    zeros(1),
    zeros(1),
    name = "HS6",
  )
  stats = GenericExecutionStats(nls)
  solver = CaNNOLeSSolver(nls)
  stats = solve!(solver, nls, stats)
  @test stats.status_reliable && stats.status == :first_order
  @test stats.solution_reliable && isapprox(stats.solution, [1.0; 1.0], atol = 1e-6)

  nls.meta.x0 .= 10.0
  reset!(solver)

  stats = solve!(solver, nls, stats)
  @test stats.status_reliable && stats.status == :first_order
  @test stats.solution_reliable && isapprox(stats.solution, [1.0; 1.0], atol = 1e-6)
end

@testset "Re-solve with a different problem" begin
  nls = ADNLSModel(
    x -> [x[1] - 1],
    [-1.2; 1.0],
    1,
    x -> [10 * (x[2] - x[1]^2)],
    zeros(1),
    zeros(1),
    name = "HS6",
  )
  stats = GenericExecutionStats(nls)
  solver = CaNNOLeSSolver(nls)
  stats = solve!(solver, nls, stats)
  @test stats.status_reliable && stats.status == :first_order
  @test stats.solution_reliable && isapprox(stats.solution, [1.0; 1.0], atol = 1e-6)

  nlp = ADNLSModel(
    x -> [x[1]],
    [-1.2; 1.0],
    1,
    x -> [10 * (x[2] - x[1]^2)],
    zeros(1),
    zeros(1),
    name = "shifted HS6",
  )
  reset!(solver, nlp)

  stats = solve!(solver, nlp, stats)
  @test stats.status_reliable && stats.status == :first_order
  @test stats.solution_reliable && isapprox(stats.solution, [0.0; 0.0], atol = 1e-6)
end

cannoles_tests()
