# Accretion geometry

```@meta
CurrentModule = Gradus
```

## Available accretion geometry

```@docs
ThinDisc
ThickDisc
ShakuraSunyaev
```


## Adding new accretion geometries

```@docs
AbstractAccretionGeometry
Gradus.in_nearby_region
Gradus.has_intersect
```

Specifically for accretion discs:

```@docs
AbstractAccretionDisc
distance_to_disc
AbstractThickAccretionDisc
cross_section
```

## Meshes

Gradus.jl supports the ability to implement custom accretion geometry, or even load in mesh files in any standard format using [MeshIO.jl](https://github.com/JuliaIO/MeshIO.jl). Geometry may be standard spherically symmetric accretion discs, or any other custom type.

```@docs
MeshAccretionGeometry
```
