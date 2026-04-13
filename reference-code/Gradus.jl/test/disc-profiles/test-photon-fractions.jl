using Test
using Gradus

m = KerrMetric(1.0, 0.998)
d = ThinDisc(0.0, Inf)

corona = LampPostModel(; h = 3.0)
fracs = photon_fractions(m, d, corona)

@test fracs.disc ≈ 0.687450186142991 rtol = 1e-4
@test fracs.black_hole ≈ 0.11549677817458547 rtol = 1e-4
@test fracs.infinity ≈ 0.19705303568242344 rtol = 1e-4
