# Point functions

```@meta
CurrentModule = Gradus
```

Point functions operate on [`AbstractGeodesicPoint`](@ref), and are used to represent calculations that can be applied to any single geodesic. They usually map to some sort of physical value (an energy shift, a proper time, and so on), but can also be used to filter geodesics based on certain predicates.

```@docs
AbstractGeodesicPoint
GeodesicPoint
unpack_solution_full
```

```@docs
AbstractPointFunction
PointFunction
FilterPointFunction
```

## Pre-defined point functions

```@docs
ConstPointFunctions
ConstPointFunctions.filter_early_term
ConstPointFunctions.filter_intersected
ConstPointFunctions.affine_time
ConstPointFunctions.shadow
ConstPointFunctions.redshift
```
