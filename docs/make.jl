ENV["GKSwstype"] = "100"
using Documenter, CaNNOLeS

makedocs(
  modules = [CaNNOLeS],
  doctest = true,
  # linkcheck = true,
  format = Documenter.HTML(
    assets = ["assets/style.css"],
    prettyurls = get(ENV, "CI", nothing) == "true",
  ),
  sitename = "CaNNOLeS.jl",
  pages = Any[
    "Home" => "index.md",
    "Tutorial" => "tutorial.md",
    "Benchmark" => "benchmark.md",
    "Reference" => "reference.md",
  ],
)

deploydocs(
  repo = "github.com/JuliaSmoothOptimizers/CaNNOLeS.jl.git",
  push_preview = true,
  devbranch = "main",
)
