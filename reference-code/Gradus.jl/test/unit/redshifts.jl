using Test
using Gradus

m = KerrMetric(a = 0.998)
x_start = SVector(0.0, 50.0, deg2rad(30), 0.0)
v = map_impact_parameters(m, x_start, 6.0, 1.0)

gp = tracegeodesics(m, x_start, v, 3.0) |> unpack_solution

g = energyshift(m, gp)
@test g ≈ 0.9987098281168815 rtol = 1e-3

# save values along a geodesic
sol = tracegeodesics(m, x_start, v, 80.0; save_on = true)
points = unpack_solution_full(m, sol)
gs = [energyshift(m, p) for p in points]
@test sum(gs) ≈ 159.37929499238336 rtol = 1e-3
